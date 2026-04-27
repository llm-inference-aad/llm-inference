"""RagService — high-level orchestration of the retrieval-augment pipeline.

``RagService`` is the replacement for the bulk of what ``RagRuntime`` does.  It
owns three collaborators:

* a :class:`~src.rag.backend_protocol.BackendProtocol` implementation (default:
  :class:`~src.rag.backends.faiss_backend.FaissBackend`),
* a :class:`~src.rag.reranker.Reranker` instance (can be ``None`` to disable),
* a :class:`~src.rag.prompt_enhancer.PromptEnhancerConfig` for formatting knobs.

All of them are injected at construction time so every dependency is mockable
in unit tests without touching singletons or module-level state.

The two public methods mirror the ``BackendProtocol`` surface plus the higher-
level augment operation:

* :meth:`retrieve` — thin pass-through to the backend (useful for debugging).
* :meth:`augment` — full pipeline: retrieve → optionally rerank → format prompt.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

from .api_types import (
    AugmentRequest,
    AugmentResponse,
    RetrievedBlock,
    RetrieveRequest,
    RetrieveResponse,
)

if TYPE_CHECKING:
    from .backend_protocol import BackendProtocol
    from .bookkeeping import RunLedger
    from .prompt_enhancer import PromptEnhancerConfig
    from .reranker import Reranker


def _get_prompt_enhancer_config_class():
    """Lazy import of PromptEnhancerConfig to avoid triggering faiss at module load."""
    from .prompt_enhancer import PromptEnhancerConfig  # noqa: PLC0415
    return PromptEnhancerConfig


def _get_constants():
    """Lazy import of cfg.constants to avoid loading torch at module import time.

    cfg.constants imports torch at module level.  By deferring the import to
    call time we allow test_service.py to load service.py without triggering
    torch, which would break test_api_types.py::TestNoHeavyImports assertions.
    """
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
# (mirrors the constant in prompt_enhancer.py)
_RERANK_MAX_QUERY_LINES = 20


class RagService:
    """Orchestrates retrieval, optional reranking, and prompt formatting.

    This class is the stable seam between LLMGE and the RAG subsystem.
    Callers (``RagClient``, tests) interact only with this class; they never
    import ``RetrievalService``, ``VectorStoreManager``, or ``EmbeddingService``
    directly.

    Args:
        backend: A ``BackendProtocol``-compliant instance.  If ``None``, a
            :class:`~src.rag.backends.faiss_backend.FaissBackend` is created
            with default configuration.
        reranker: A ``Reranker`` instance.  If ``None``, reranking is skipped
            regardless of the ``RAG_RERANKER_ENABLED`` env var.  Pass
            ``None`` explicitly to disable; omit to use the env-var default
            (``_build_default_reranker()`` is called).
        config: A ``PromptEnhancerConfig`` for formatting knobs (top_k, etc.).
            Defaults to ``PromptEnhancerConfig()`` if not supplied.
        _reranker_sentinel: Internal flag used to distinguish "caller passed
            ``reranker=None`` explicitly to disable" from "caller omitted
            reranker and we should use the env default".  Do not pass this
            from user code.
    """

    _SENTINEL = object()

    def __init__(
        self,
        backend: Optional["BackendProtocol"] = None,
        reranker: Optional["Reranker"] = _SENTINEL,  # type: ignore[assignment]
        config: Optional["PromptEnhancerConfig"] = None,
        ledger: Optional["RunLedger"] = None,
    ) -> None:
        self._backend: "BackendProtocol" = backend or self._build_default_backend()
        _pec_cls = _get_prompt_enhancer_config_class()
        self._config = config or _pec_cls()

        # Reranker handling:
        # - omitted (sentinel)  → use env-var default
        # - None explicitly     → disable reranker
        # - Reranker instance   → use provided instance
        if reranker is RagService._SENTINEL:
            self._reranker: Optional["Reranker"] = self._build_default_reranker()
        else:
            self._reranker = reranker  # type: ignore[assignment]

        # Optional ledger for bookkeeping. When None, no events are emitted.
        self._ledger: Optional["RunLedger"] = ledger

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
        """Pass-through to the backend — useful for direct debugging.

        Args:
            request: A :class:`~src.rag.api_types.RetrieveRequest`.

        Returns:
            A :class:`~src.rag.api_types.RetrieveResponse` from the backend.
        """
        return self._backend.retrieve(request)

    def augment(self, request: AugmentRequest) -> AugmentResponse:
        """Run the full retrieve → rerank → format pipeline.

        Args:
            request: An :class:`~src.rag.api_types.AugmentRequest` describing
                the template, mutation type, and query code.

        Returns:
            An :class:`~src.rag.api_types.AugmentResponse` with the augmented
            prompt, the blocks that were injected, diagnostics, and latency.
        """
        t0 = time.monotonic()
        _c = _get_constants()

        code_blocks: List[RetrievedBlock] = []
        text_blocks: List[RetrievedBlock] = []
        reranker_used = False

        # --- Code namespace retrieval --------------------------------------- #
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

        # --- Text namespace retrieval --------------------------------------- #
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

        # --- Optional reranking -------------------------------------------- #
        if self._reranker is not None and (code_blocks or text_blocks):
            code_rerank_query = self._build_rerank_query(request.query_code, request.mutation_type)
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
            text_blocks,
            mutation_type=request.mutation_type,
        )

        # --- Format prompt ------------------------------------------------- #
        augmented_prompt, sections_built = self._format_prompt(
            template=request.template,
            code_blocks=code_blocks,
            text_blocks=text_blocks_selected,
        )

        # --- Collect all blocks used --------------------------------------- #
        all_blocks_used = list(code_blocks) + list(text_blocks_selected)

        latency_ms = (time.monotonic() - t0) * 1000.0

        # --- Emit observability metric ------------------------------------- #
        self._emit_metric(
            request=request,
            code_blocks=code_blocks,
            text_blocks=text_blocks_selected,
            reranker_used=reranker_used,
            sections_built=sections_built,
            latency_ms=latency_ms,
        )

        # --- Emit bookkeeping ledger event --------------------------------- #
        # This is an "augment event" — eval_outputs / model_response are None.
        # A companion "eval event" will be written by run_improved._log_mutation_result
        # with the same request_id so the two records can be JOINed by reader.
        response = AugmentResponse(
            augmented_prompt=augmented_prompt,
            blocks_used=all_blocks_used,
            diagnostics={
                "reranker_used": reranker_used,
                "code_blocks_retrieved": len(code_blocks),
                "text_blocks_retrieved": len(text_blocks_selected),
                "sections_built": sections_built,
                "rag_use_code_context": bool(_c.RAG_USE_CODE_CONTEXT),
                "rag_use_text_context": bool(_c.RAG_USE_TEXT_CONTEXT),
            },
            latency_ms=latency_ms,
        )
        self._emit_ledger_event(
            request=request,
            response=response,
            sections_built=sections_built,
            latency_ms=latency_ms,
        )
        return response

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _build_rerank_query(query_code: Optional[str], mutation_type: Optional[str]) -> str:
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

        # Build (query, content) pairs for the cross-encoder
        contents = [b.content for b in blocks]
        try:
            scores = self._reranker.rerank(
                query=query,
                candidates=blocks,
                content_fn=lambda b: b.content,
                top_k=top_k,
            )
            # reranker.rerank returns already-sorted items
            return list(scores)
        except TypeError:
            # Fallback: reranker doesn't accept content_fn (e.g. FakeReranker)
            # Build a wrapper so FakeReranker's document_id lookup works.
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

        # Check if this is a hyperparameter-style query
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
    ) -> tuple[str, bool]:
        """Build the augmented prompt string from retrieved blocks.

        Returns:
            (augmented_prompt, sections_were_built)
        """
        sections: list[str] = []

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
                    "top_k_code": self._config.top_k,
                    "top_k_text": self._config.text_top_k,
                    "retrieved_code_n": len(code_blocks),
                    "retrieved_text_n": len(text_blocks),
                    "template_augmented": sections_built,
                    "latency_ms": latency_ms,
                },
            )
        except Exception:
            pass

    def _emit_ledger_event(
        self,
        request: AugmentRequest,
        response: AugmentResponse,
        sections_built: bool,
        latency_ms: float,
    ) -> None:
        """Emit an augment-time bookkeeping event to the injected ledger.

        This is the first of two events for the same mutation attempt.  The
        eval-time event is written by run_improved._log_mutation_result and
        shares the same ``request_id``.

        If ``self._ledger`` is None, nothing is emitted (backward compatible).
        """
        if self._ledger is None:
            return
        try:
            from dataclasses import asdict as _asdict  # noqa: PLC0415
            from .bookkeeping import MutationEvent  # noqa: PLC0415

            # Build a partial MutationEvent with augment-time fields populated.
            raw_prompt = request.template or ""
            event = MutationEvent(
                run_id=request.run_id or os.getenv("RUN_ID", "unknown"),
                generation=-1,  # generation is unknown at augment time; caller fills via eval event
                parent_gene_id=request.gene_id or "unknown",
                child_gene_id=request.gene_id or "unknown",
                mutation_type=request.mutation_type or "unknown",
                backend=self._backend.__class__.__name__,
                request_id=request.request_id or MutationEvent.make_request_id(),
                prompt_id=MutationEvent.make_prompt_id(raw_prompt),
                raw_prompt=raw_prompt,
                augmented_prompt=response.augmented_prompt,
                retrieval_request=_asdict(request),
                retrieval_response={
                    "augmented_prompt": response.augmented_prompt,
                    "blocks_used": len(response.blocks_used),
                    "diagnostics": response.diagnostics,
                    "latency_ms": response.latency_ms,
                },
                model_request=None,
                model_response=None,
                parsed_artifact=None,
                eval_outputs=None,
                latencies={
                    "augment_ms": latency_ms,
                },
                failure_mode=None if sections_built else "retrieval_empty",
            )
            self._ledger.append(event)
        except Exception as exc:
            logger.warning("[RagService] Failed to emit ledger event: %s", exc)
