"""RagService — high-level orchestration of the retrieval-augment pipeline.

Ported from worker/pr3-rag-client-service with PR 8 memory backend additions.

``RagService`` owns three (optionally four) collaborators:

* a primary :class:`~src.rag.backend_protocol.BackendProtocol` (FaissBackend),
* an optional ``memory_backend: BackendProtocol`` for episodic mutation
  summaries (PR 8, gated by ``RAG_MEMORY_STORE_ENABLED``),
* a :class:`~src.rag.reranker.Reranker` instance (can be ``None``),
* a :class:`~src.rag.prompt_enhancer.PromptEnhancerConfig` for formatting.

All collaborators are injected at construction so every dependency is mockable
in unit tests without touching singletons or module-level state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from typing import TYPE_CHECKING, List, Optional

from .api_types import (
    AugmentRequest,
    AugmentResponse,
    RetrievedBlock,
    RetrieveRequest,
    RetrieveResponse,
)

if TYPE_CHECKING:
    from .backend_protocol import BackendProtocol
    from .reranker import Reranker


# ---------------------------------------------------------------------------
# PromptEnhancerConfig — local copy to avoid importing prompt_enhancer.py
# (which has a module-level `from .reranker import Reranker` that requires
# heavy deps at import time).
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PromptEnhancerConfig:
    """Formatting knobs for RAG prompt construction.

    Mirrors ``src.rag.prompt_enhancer.PromptEnhancerConfig`` but lives here to
    break the import chain ``service → prompt_enhancer → reranker → torch``.
    """

    top_k: int = 5
    text_candidate_k: int = 24
    text_top_k: int = 3
    text_top_k_api: int = 2
    text_top_k_pdf: int = 1


def _get_prompt_enhancer_config_class():
    """Return PromptEnhancerConfig — always the local version."""
    return PromptEnhancerConfig


def _get_constants():
    """Lazy import of cfg.constants to avoid loading torch at module import time."""
    import cfg.constants as _c  # noqa: PLC0415
    return _c


def _record_metric(*args, **kwargs):
    """Lazy wrapper around utils.rag_metrics.record_metric."""
    try:
        from utils.rag_metrics import record_metric  # noqa: PLC0415
        record_metric(*args, **kwargs)
    except Exception:
        pass


# Maximum lines of query code sent to cross-encoder to avoid CPU bottleneck
_RERANK_MAX_QUERY_LINES = 20


class RagService:
    """Orchestrates retrieval, optional reranking, and prompt formatting.

    This class is the stable seam between LLMGE and the RAG subsystem.
    Callers interact only with this class; they never import
    ``RetrievalService``, ``VectorStoreManager``, or ``EmbeddingService``
    directly.

    Args:
        backend: Primary ``BackendProtocol``-compliant instance (FAISS code/text).
            If ``None``, a FaissBackend is built with default configuration.
        memory_backend: Optional ``BackendProtocol`` for episodic memory
            (PR 8).  When not ``None`` and ``RAG_MEMORY_STORE_ENABLED=true``,
            ``augment()`` calls this backend after code/text retrieval and
            merges the returned blocks with dedup-by-gene-id (code blocks win
            when the same gene appears in both channels).
        reranker: A ``Reranker`` instance.  ``None`` disables reranking.
        config: A ``PromptEnhancerConfig``.  Defaults to ``PromptEnhancerConfig()``.
        _reranker_sentinel: Internal flag — do not pass from user code.
    """

    _SENTINEL = object()

    def __init__(
        self,
        backend: Optional["BackendProtocol"] = None,
        memory_backend: Optional["BackendProtocol"] = None,
        reranker: Optional["Reranker"] = _SENTINEL,  # type: ignore[assignment]
        config: Optional["PromptEnhancerConfig"] = None,
    ) -> None:
        self._backend: "BackendProtocol" = backend or self._build_default_backend()
        self._memory_backend: Optional["BackendProtocol"] = memory_backend
        _pec_cls = _get_prompt_enhancer_config_class()
        self._config = config or _pec_cls()

        if reranker is RagService._SENTINEL:
            self._reranker: Optional["Reranker"] = self._build_default_reranker()
        else:
            self._reranker = reranker  # type: ignore[assignment]

    # ---------------------------------------------------------------------- #
    # Default factory helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _build_default_backend() -> "BackendProtocol":
        from .backends.faiss_backend import FaissBackend
        return FaissBackend()

    @staticmethod
    def _build_default_reranker() -> Optional["Reranker"]:
        if not _get_constants().RAG_RERANKER_ENABLED:
            return None
        from .reranker import Reranker
        return Reranker()

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Pass-through to the primary backend."""
        return self._backend.retrieve(request)

    def augment(self, request: AugmentRequest) -> AugmentResponse:
        """Run the full retrieve → rerank → format pipeline.

        When a ``memory_backend`` is present and ``RAG_MEMORY_STORE_ENABLED``
        is true, episodic memory summaries are retrieved after code/text blocks
        and merged into the prompt.  Dedup: if the same ``gene_id`` appears in
        both memory bullets and code blocks, only the code block is kept (it
        carries richer signal).
        """
        t0 = time.monotonic()
        _c = _get_constants()

        code_blocks: List[RetrievedBlock] = []
        text_blocks: List[RetrievedBlock] = []
        memory_blocks: List[RetrievedBlock] = []
        reranker_used = False

        # --- Code namespace retrieval -------------------------------------- #
        if _c.RAG_USE_CODE_CONTEXT and request.query_code:
            code_req = RetrieveRequest(
                query=request.query_code,
                namespace="code",
                top_k=self._config.top_k,
                filters=None,
                run_id=request.run_id,
                request_id=request.request_id,
            )
            code_resp = self._backend.retrieve(code_req)
            code_blocks = list(code_resp.blocks)

        # --- Text namespace retrieval -------------------------------------- #
        if _c.RAG_USE_TEXT_CONTEXT:
            text_query_parts = self._build_text_query_parts(
                template=request.template,
                query_code=request.query_code,
                mutation_type=request.mutation_type,
            )
            if text_query_parts:
                text_query = " ".join(text_query_parts)
                text_req = RetrieveRequest(
                    query=text_query,
                    namespace="text",
                    top_k=self._config.text_candidate_k,
                    filters=None,
                    run_id=request.run_id,
                    request_id=request.request_id,
                )
                text_resp = self._backend.retrieve(text_req)
                text_blocks = list(text_resp.blocks)

        # --- Memory namespace retrieval (PR 8, opt-in) -------------------- #
        if self._memory_backend is not None and _c.RAG_MEMORY_STORE_ENABLED:
            # Use the full query_code (not truncated) to avoid the weak-query
            # bug from Ajay's version where only the first 5 lines were used.
            memory_query = request.query_code or request.mutation_type or ""
            if memory_query.strip():
                mem_req = RetrieveRequest(
                    query=memory_query,
                    namespace="memory",
                    top_k=_c.RAG_MEMORY_TOP_K,
                    filters=None,
                    run_id=request.run_id,
                    request_id=request.request_id,
                )
                mem_resp = self._memory_backend.retrieve(mem_req)
                raw_memory = list(mem_resp.blocks)

                # Dedup: discard memory blocks whose gene_id already appears in
                # code blocks (prefer the richer code block over the summary).
                code_gene_ids = {b.document_id for b in code_blocks}
                memory_blocks = [
                    b for b in raw_memory if b.document_id not in code_gene_ids
                ]

        # --- Optional reranking ------------------------------------------- #
        if self._reranker is not None and (code_blocks or text_blocks):
            code_rerank_query = self._build_rerank_query(
                request.query_code, request.mutation_type
            )
            text_rerank_query = " ".join(
                self._build_text_query_parts(
                    template=request.template,
                    query_code=request.query_code,
                    mutation_type=request.mutation_type,
                )
            ) or request.mutation_type or ""

            if code_blocks and code_rerank_query:
                code_blocks = self._rerank_blocks(
                    code_rerank_query, code_blocks, top_k=self._config.top_k
                )
            if text_blocks and text_rerank_query:
                text_blocks = self._rerank_blocks(
                    text_rerank_query, text_blocks, top_k=len(text_blocks)
                )
            reranker_used = True

        # Trim text blocks to configured top_k after reranking
        text_blocks_selected = self._select_text_blocks(
            text_blocks, mutation_type=request.mutation_type
        )

        # --- Format prompt ------------------------------------------------ #
        augmented_prompt, sections_built = self._format_prompt(
            template=request.template,
            code_blocks=code_blocks,
            text_blocks=text_blocks_selected,
            memory_blocks=memory_blocks,
        )

        # --- Collect all blocks used -------------------------------------- #
        all_blocks_used = list(code_blocks) + list(text_blocks_selected) + list(memory_blocks)

        latency_ms = (time.monotonic() - t0) * 1000.0

        # --- Emit observability metric ------------------------------------- #
        self._emit_metric(
            request=request,
            code_blocks=code_blocks,
            text_blocks=text_blocks_selected,
            memory_blocks=memory_blocks,
            reranker_used=reranker_used,
            sections_built=sections_built,
            latency_ms=latency_ms,
        )

        return AugmentResponse(
            augmented_prompt=augmented_prompt,
            blocks_used=all_blocks_used,
            diagnostics={
                "reranker_used": reranker_used,
                "code_blocks_retrieved": len(code_blocks),
                "text_blocks_retrieved": len(text_blocks_selected),
                "memory_blocks_retrieved": len(memory_blocks),
                "sections_built": sections_built,
                "rag_use_code_context": bool(_c.RAG_USE_CODE_CONTEXT),
                "rag_use_text_context": bool(_c.RAG_USE_TEXT_CONTEXT),
                "rag_memory_store_enabled": bool(_c.RAG_MEMORY_STORE_ENABLED),
            },
            latency_ms=latency_ms,
        )

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _build_rerank_query(
        query_code: Optional[str], mutation_type: Optional[str]
    ) -> str:
        if not query_code:
            return mutation_type or ""
        lines = query_code.strip().splitlines()
        if len(lines) > _RERANK_MAX_QUERY_LINES:
            query_code = "\n".join(lines[:_RERANK_MAX_QUERY_LINES])
        return query_code

    @staticmethod
    def _trim_words(text: str, limit: int) -> str:
        words = text.split()
        if len(words) <= limit:
            return " ".join(words)
        return " ".join(words[:limit]).strip()

    def _build_text_query_parts(
        self,
        template: Optional[str] = None,
        query_code: Optional[str] = None,
        mutation_type: Optional[str] = None,
    ) -> list[str]:
        parts: list[str] = []
        if query_code:
            snippet_lines = [
                line.rstrip()
                for line in query_code.strip().splitlines()
                if line.strip()
            ]
            snippet = "\n".join(snippet_lines[:20])[:1600].strip()
            if snippet:
                parts.append(f"Architecture snippet:\n{snippet}")
        if mutation_type:
            parts.append(f"Mutation type: {mutation_type}")
        if template:
            instruction = template.split("```")[0].strip()
            hint = self._trim_words(instruction, 40) if instruction else ""
            if hint:
                parts.append(f"Task hint: {hint}")
            elif mutation_type:
                parts.append(f"CNN mutation objective: {mutation_type}")
        return parts

    def _rerank_blocks(
        self,
        query: str,
        blocks: List[RetrievedBlock],
        top_k: int,
    ) -> List[RetrievedBlock]:
        """Rerank *blocks* using the injected reranker, returning top_k."""
        if self._reranker is None or not blocks:
            return blocks[:top_k]

        try:
            scores = self._reranker.rerank(
                query=query,
                candidates=blocks,
                content_fn=lambda b: b.content,
                top_k=top_k,
            )
            return list(scores)
        except TypeError:
            # Fallback for FakeReranker (no content_fn)
            class _BlockWrapper:
                def __init__(self, block: RetrievedBlock) -> None:
                    self.document = type("_D", (), {"document_id": block.document_id})()
                    self._block = block

            wrappers = [_BlockWrapper(b) for b in blocks]
            reranked_wrappers = self._reranker.rerank(query, wrappers, top_k=top_k)
            return [w._block for w in reranked_wrappers]

    def _select_text_blocks(
        self,
        blocks: List[RetrievedBlock],
        mutation_type: Optional[str] = None,
    ) -> List[RetrievedBlock]:
        """Apply source-type limits and return the final text block list."""
        if not blocks:
            return []

        is_hp_query = mutation_type and "param" in mutation_type.lower()

        selected: List[RetrievedBlock] = []
        n_api = 0
        n_pdf = 0

        for block in blocks:
            diag = block.diagnostics or {}
            doc_type = diag.get("doc_type", block.kind)
            is_api = doc_type in ("api_doc", "pytorch_api")
            is_pdf = doc_type in ("pdf_chunk", "paper")

            if is_hp_query and not is_api:
                continue

            if is_api and n_api < self._config.text_top_k_api:
                selected.append(block)
                n_api += 1
            elif is_pdf and n_pdf < self._config.text_top_k_pdf:
                selected.append(block)
                n_pdf += 1
            elif not is_api and not is_pdf and len(selected) < self._config.text_top_k:
                selected.append(block)

            if len(selected) >= self._config.text_top_k:
                break

        return selected

    @staticmethod
    def _format_prompt(
        template: str,
        code_blocks: List[RetrievedBlock],
        text_blocks: List[RetrievedBlock],
        memory_blocks: Optional[List[RetrievedBlock]] = None,
    ) -> tuple[str, bool]:
        """Build the augmented prompt string from retrieved blocks.

        Section order (prepended before template):
        1. Memory bullets (episodic summaries)  — if any
        2. Text docs (PyTorch API / PDF)         — if any
        3. Code examples (historical mutations)  — if any

        Returns:
            (augmented_prompt, sections_were_built)
        """
        sections: list[str] = []

        if memory_blocks:
            bullets = "\n".join(f"- {b.content}" for b in memory_blocks)
            sections.append(
                "Relevant past attempts from prior runs (for episodic context):\n"
                f"{bullets}"
            )

        if text_blocks:
            text_parts = [
                f"[{b.kind.upper()}] {b.title}\n{b.content}"
                for b in text_blocks
            ]
            text_block_str = "\n\n".join(text_parts)
            sections.append(
                "The following PyTorch documentation and research context may be "
                "relevant to your architectural decisions:\n"
                f"{text_block_str}"
            )

        if code_blocks:
            code_parts = [
                f"# Gene: {b.document_id} | Score: {b.score:.3f}\n{b.content}"
                for b in code_blocks
            ]
            code_block_str = "\n\n".join(code_parts)
            sections.append(
                "### RAG Code Examples\n"
                "The following code blocks are historically successful mutations from this exact codebase. "
                "Notice the `accuracy_delta` and `parameters_delta` metadata to understand their impact. "
                "Do NOT copy them exactly. Extract the architectural motifs (e.g., where they placed SiLU, "
                "how they grouped convolutions, or their choice of optimizer) and apply those lessons to "
                "your current task.\n"
                f"{code_block_str}"
            )

        if not sections:
            return template, False

        augmented = "\n\n".join(sections) + f"\n\n{template.strip()}"
        return augmented, True

    def _emit_metric(
        self,
        request: AugmentRequest,
        code_blocks: List[RetrievedBlock],
        text_blocks: List[RetrievedBlock],
        memory_blocks: List[RetrievedBlock],
        reranker_used: bool,
        sections_built: bool,
        latency_ms: float,
    ) -> None:
        """Emit a rag_context_built metric for downstream analysis."""
        try:
            _c = _get_constants()
            template_hash = hashlib.sha1(
                request.template.encode("utf-8", errors="ignore")
            ).hexdigest()
            query_hash = hashlib.sha1(
                (request.query_code or "").encode("utf-8", errors="ignore")
            ).hexdigest()
            _record_metric(
                "rag_context_built",
                {
                    "run_id": os.getenv("RUN_ID", _c.RUN_ID),
                    "gene_id": request.gene_id,
                    "mutation_type": request.mutation_type,
                    "template_hash": template_hash,
                    "query_hash": query_hash,
                    "rag_use_code_context": bool(_c.RAG_USE_CODE_CONTEXT),
                    "rag_use_text_context": bool(_c.RAG_USE_TEXT_CONTEXT),
                    "reranker_enabled": bool(_c.RAG_RERANKER_ENABLED),
                    "reranker_used": reranker_used,
                    "rag_memory_store_enabled": bool(_c.RAG_MEMORY_STORE_ENABLED),
                    "top_k_code": self._config.top_k,
                    "top_k_text": self._config.text_top_k,
                    "retrieved_code_n": len(code_blocks),
                    "retrieved_text_n": len(text_blocks),
                    "retrieved_memory_n": len(memory_blocks),
                    "template_augmented": sections_built,
                    "latency_ms": latency_ms,
                },
            )
        except Exception:
            pass
