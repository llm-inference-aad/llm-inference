"""Prompt enhancement helpers that inject RAG context."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from cfg.constants import (
    RAG_MIN_SIMILARITY,
    RAG_RERANKER_ENABLED,
    RAG_TEXT_CANDIDATE_K,
    RAG_TEXT_TOP_K,
    RAG_TEXT_TOP_K_API,
    RAG_TEXT_TOP_K_PDF,
    RAG_USE_CODE_CONTEXT,
    RAG_USE_TEXT_CONTEXT,
    RUN_ID,
)

from .reranker import Reranker
from .retrieval import RetrievedContext, RetrievedMutation, RetrievalService, RetrievalStats
from utils.rag_metrics import record_metric

if TYPE_CHECKING:
    from .backend_protocol import BackendProtocol

# M4: Maximum lines of query code sent to cross-encoder to avoid CPU bottleneck
_RERANK_MAX_QUERY_LINES = 20

# Module-level singleton — lazy-loaded only when reranking is enabled
_reranker: Reranker | None = None


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


@dataclass(frozen=True)
class PromptEnhancerConfig:
    top_k: int = 5
    text_candidate_k: int = RAG_TEXT_CANDIDATE_K
    text_top_k: int = RAG_TEXT_TOP_K
    text_top_k_api: int = RAG_TEXT_TOP_K_API
    text_top_k_pdf: int = RAG_TEXT_TOP_K_PDF
    min_accuracy: float = 0.9
    max_parameters: float | None = None


class PromptEnhancer:
    """Build augmented prompt text that includes retrieval context.

    Can be wired to either the legacy :class:`~src.rag.retrieval.RetrievalService`
    (``retrieval_service`` argument) **or** to any object satisfying
    :class:`~src.rag.backend_protocol.BackendProtocol` (``backend`` argument).
    When *backend* is supplied, :meth:`_retrieve_via_backend` is used for the
    code-context leg of :meth:`build_context_with_stats`.  The text-context leg
    and all prompt-formatting logic are unchanged.

    Backward compatibility: the *retrieval_service* argument still works and is
    the default when *backend* is omitted.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        config: PromptEnhancerConfig | None = None,
        backend: Optional["BackendProtocol"] = None,
    ):
        self.retrieval = retrieval_service
        self.config = config or PromptEnhancerConfig()
        self._backend: Optional["BackendProtocol"] = backend

    def build_context(
        self,
        mutation_type: str | None = None,
        query_code: str | None = None,
    ) -> List[RetrievedMutation]:
        if not RAG_USE_CODE_CONTEXT:
            return []

        mutations: list[RetrievedMutation] = []

        if query_code:
            mutations.extend(
                self.retrieval.retrieve_similar_mutations(
                    query_code, top_k=self.config.top_k, min_similarity=RAG_MIN_SIMILARITY
                )
            )

        if mutation_type and len(mutations) < self.config.top_k:
            extra = self.retrieval.retrieve_by_mutation_type(
                mutation_type, limit=self.config.top_k - len(mutations)
            )
            mutations.extend(extra)

        if len(mutations) < self.config.top_k:
            high_performers = self.retrieval.retrieve_high_performers(
                min_accuracy=self.config.min_accuracy,
                max_parameters=self.config.max_parameters,
                limit=self.config.top_k - len(mutations),
            )
            mutations.extend(high_performers)

        deduped: dict[str, RetrievedMutation] = {}
        for mutation in mutations:
            deduped.setdefault(mutation.gene_id, mutation)
        
        # Sort by score (descending) to ensure best mutations are shown first
        # This helps the LLM focus on the most relevant examples
        result = list(deduped.values())
        result.sort(key=lambda m: m.score, reverse=True)
        return result

    def _retrieve_via_backend(
        self,
        query_code: str,
    ) -> tuple[list[RetrievedMutation], RetrievalStats | None]:
        """Thin wrapper: calls the injected BackendProtocol and converts the
        response back into the (mutations, stats) tuple the rest of this class
        already knows how to handle.

        This is the only place that branches on whether a backend is wired in.
        All other methods remain untouched so that the existing fallback path
        through ``self.retrieval`` continues to work.

        Returns an empty list + None stats if *_backend* is not set.
        """
        if self._backend is None:
            return [], None

        from .api_types import RetrieveRequest
        from .vector_db import VectorStoreManager

        req = RetrieveRequest(
            query=query_code,
            namespace=VectorStoreManager.CODE_NAMESPACE,
            top_k=self.config.top_k,
            filters={"min_similarity": RAG_MIN_SIMILARITY},
        )
        response = self._backend.retrieve(req)

        # Convert RetrievedBlock → RetrievedMutation so the downstream
        # formatting helpers in this class keep working without modification.
        mutations: list[RetrievedMutation] = []
        for block in response.blocks:
            block_diag = block.diagnostics or {}
            mutations.append(
                RetrievedMutation(
                    gene_id=block.document_id,
                    score=block.score,
                    description=block.title,
                    code=block.content,
                    metadata={
                        "gene_id": block.document_id,
                        "description": block.title,
                        "source": block_diag.get("source", "code"),
                        "mutation_type": block_diag.get("mutation_type"),
                        # Preserve any extra diagnostics for downstream logging.
                        **{k: v for k, v in block_diag.items()
                           if k not in ("source", "mutation_type")},
                    },
                )
            )

        # Synthesize a lightweight RetrievalStats from the response diagnostics.
        resp_diag = response.diagnostics or {}
        code_search = resp_diag.get("code_search") or {}
        stats: RetrievalStats | None = None
        if code_search:
            stats = RetrievalStats(
                candidate_k=int(code_search.get("candidate_count", 0)),
                returned_k=int(code_search.get("returned_k", len(mutations))),
                filtered_k=int(code_search.get("filtered_k", len(mutations))),
                min_similarity=RAG_MIN_SIMILARITY,
            )

        return mutations, stats

    def build_context_with_stats(
        self,
        mutation_type: str | None = None,
        query_code: str | None = None,
    ) -> tuple[List[RetrievedMutation], RetrievalStats | None]:
        """Build code context and return similarity-search stats (if used).

        When a :class:`~src.rag.backend_protocol.BackendProtocol` was supplied
        at construction time (``backend=...``), the code-similarity leg is
        dispatched through :meth:`_retrieve_via_backend` so the caller is
        decoupled from ``RetrievalService`` internals.  The fallback (mutation-
        type and high-performer retrieval) still uses ``self.retrieval`` directly
        because those operations have no BackendProtocol equivalent yet.
        """
        if not RAG_USE_CODE_CONTEXT:
            return [], None

        stats: RetrievalStats | None = None
        mutations: list[RetrievedMutation] = []

        if query_code:
            if self._backend is not None:
                # New path: delegate to the injected BackendProtocol.
                similar, stats = self._retrieve_via_backend(query_code)
            else:
                # Legacy path: call RetrievalService directly.
                similar, stats = self.retrieval.retrieve_similar_mutations_with_stats(
                    query_code, top_k=self.config.top_k, min_similarity=RAG_MIN_SIMILARITY
                )
            mutations.extend(similar)

        if mutation_type and len(mutations) < self.config.top_k:
            extra = self.retrieval.retrieve_by_mutation_type(
                mutation_type, limit=self.config.top_k - len(mutations)
            )
            mutations.extend(extra)

        if len(mutations) < self.config.top_k:
            high_performers = self.retrieval.retrieve_high_performers(
                min_accuracy=self.config.min_accuracy,
                max_parameters=self.config.max_parameters,
                limit=self.config.top_k - len(mutations),
            )
            mutations.extend(high_performers)

        deduped: dict[str, RetrievedMutation] = {}
        for mutation in mutations:
            deduped.setdefault(mutation.gene_id, mutation)

        result = list(deduped.values())
        result.sort(key=lambda m: m.score, reverse=True)
        return result, stats

    def build_text_context(
        self,
        query_code: str | None = None,
        mutation_type: str | None = None,
    ) -> List[RetrievedContext]:
        """Retrieve relevant text context (PyTorch docs, research papers).

        Uses the mutation type or query code as a search query against the
        text namespace. Returns domain knowledge to guide LLM reasoning.
        """
        if not RAG_USE_TEXT_CONTEXT:
            return []

        query_parts = self._build_text_query_parts(
            query_code=query_code,
            mutation_type=mutation_type,
        )
        if not query_parts:
            return []

        query = " ".join(query_parts)
        return self.retrieval.retrieve_similar_text(
            query, top_k=self.config.text_top_k, min_similarity=RAG_MIN_SIMILARITY
        )

    @staticmethod
    def _trim_words(text: str, limit: int) -> str:
        words = text.split()
        if len(words) <= limit:
            return " ".join(words)
        return " ".join(words[:limit]).strip()

    def _build_template_hint(self, template: str | None, mutation_type: str | None) -> str:
        if template:
            instruction = template.split("```")[0].strip()
            if instruction:
                return self._trim_words(instruction, 40)
        if mutation_type:
            return f"CNN mutation objective: {mutation_type}"
        return ""

    def _build_code_query_snippet(self, query_code: str | None) -> str:
        if not query_code:
            return ""
        lines = [line.rstrip() for line in query_code.strip().splitlines() if line.strip()]
        if not lines:
            return ""
        return "\n".join(lines[:20])[:1600].strip()

    def _build_text_query_parts(
        self,
        template: str | None = None,
        query_code: str | None = None,
        mutation_type: str | None = None,
    ) -> list[str]:
        query_parts: list[str] = []
        code_snippet = self._build_code_query_snippet(query_code)
        if code_snippet:
            query_parts.append(f"Architecture snippet:\n{code_snippet}")
        if mutation_type:
            query_parts.append(f"Mutation type: {mutation_type}")
        template_hint = self._build_template_hint(template, mutation_type)
        if template_hint:
            query_parts.append(f"Task hint: {template_hint}")
        return query_parts

    @staticmethod
    def _is_hyperparameter_query(mutation_type: str | None, text_query: str) -> bool:
        query_lower = text_query.lower()
        if mutation_type and "param" in mutation_type.lower():
            return True
        keywords = (
            "optimizer",
            "learning rate",
            "schedule",
            "weight decay",
            "hyperparameter",
        )
        return any(keyword in query_lower for keyword in keywords)

    def _select_text_contexts(
        self,
        candidates: List[RetrievedContext],
        mutation_type: str | None,
        text_query: str,
    ) -> tuple[List[RetrievedContext], dict]:
        if not candidates:
            return [], {
                "mode": "empty",
                "selected_api": 0,
                "selected_pdf": 0,
                "selected_other": 0,
            }

        mode = "hyperparameter_api_only" if self._is_hyperparameter_query(mutation_type, text_query) else "hybrid"
        selected: list[RetrievedContext] = []
        selected_ids: set[str] = set()
        selected_api = 0
        selected_pdf = 0
        selected_other = 0

        if mode == "hyperparameter_api_only":
            for ctx in candidates:
                if not self.retrieval.is_api_context(ctx):
                    continue
                selected.append(ctx)
                selected_ids.add(ctx.document_id)
                selected_api += 1
                if len(selected) >= self.config.text_top_k:
                    break
        else:
            for ctx in candidates:
                if self.retrieval.is_api_context(ctx) and selected_api < self.config.text_top_k_api:
                    selected.append(ctx)
                    selected_ids.add(ctx.document_id)
                    selected_api += 1
                elif self.retrieval.is_pdf_context(ctx) and selected_pdf < self.config.text_top_k_pdf:
                    selected.append(ctx)
                    selected_ids.add(ctx.document_id)
                    selected_pdf += 1
                if len(selected) >= self.config.text_top_k:
                    break

            if len(selected) < self.config.text_top_k:
                for ctx in candidates:
                    if ctx.document_id in selected_ids:
                        continue
                    selected.append(ctx)
                    selected_ids.add(ctx.document_id)
                    if self.retrieval.is_api_context(ctx):
                        selected_api += 1
                    elif self.retrieval.is_pdf_context(ctx):
                        selected_pdf += 1
                    else:
                        selected_other += 1
                    if len(selected) >= self.config.text_top_k:
                        break

        if mode == "hyperparameter_api_only":
            selected_other = 0
            selected_pdf = 0
        else:
            selected_other = max(
                0,
                len(selected) - selected_api - selected_pdf,
            )

        return selected, {
            "mode": mode,
            "selected_api": selected_api,
            "selected_pdf": selected_pdf,
            "selected_other": selected_other,
        }

    @staticmethod
    def _candidate_payloads(items: List[object], pre_scores: dict[str, float] | None = None) -> list[dict]:
        payloads: list[dict] = []
        for item in items:
            if isinstance(item, RetrievedContext):
                identifier = item.document_id
                payloads.append(
                    {
                        "document_id": identifier,
                        "source": item.source,
                        "doc_type": item.doc_type,
                        "name": item.metadata.get("name"),
                        "pre_score": None if pre_scores is None else pre_scores.get(identifier),
                        "post_score": float(item.score),
                        "content_preview": (item.content or "")[:160],
                    }
                )
            elif isinstance(item, RetrievedMutation):
                identifier = item.gene_id
                payloads.append(
                    {
                        "document_id": identifier,
                        "source": item.metadata.get("source", "code"),
                        "doc_type": "code_mutation",
                        "name": item.metadata.get("gene_id", identifier),
                        "pre_score": None if pre_scores is None else pre_scores.get(identifier),
                        "post_score": float(item.score),
                        "content_preview": (item.description or item.code or "")[:160],
                    }
                )
        return payloads

    def build_text_context_with_stats(
        self,
        template: str | None = None,
        query_code: str | None = None,
        mutation_type: str | None = None,
    ) -> tuple[List[RetrievedContext], RetrievalStats | None]:
        """Build text context and return similarity-search stats (if used)."""
        if not RAG_USE_TEXT_CONTEXT:
            return [], None

        query_parts = self._build_text_query_parts(
            template=template,
            query_code=query_code,
            mutation_type=mutation_type,
        )
        if not query_parts:
            return [], None

        query = " ".join(query_parts)
        contexts, stats = self.retrieval.retrieve_text_candidates_with_stats(
            query,
            candidate_k=self.config.text_candidate_k,
            min_similarity=RAG_MIN_SIMILARITY,
        )
        selected, _policy = self._select_text_contexts(contexts, mutation_type=mutation_type, text_query=query)
        return selected, stats

    def enhance_template(
        self,
        template: str,
        mutation_type: str | None = None,
        query_code: str | None = None,
        gene_id: str | None = None,
    ) -> tuple[str, List[RetrievedMutation]]:
        mutations, code_stats = self.build_context_with_stats(
            mutation_type=mutation_type, query_code=query_code
        )
        text_candidates, text_stats = self.retrieval.retrieve_text_candidates_with_stats(
            " ".join(
                self._build_text_query_parts(
                    template=template,
                    query_code=query_code,
                    mutation_type=mutation_type,
                )
            ),
            candidate_k=self.config.text_candidate_k,
            min_similarity=RAG_MIN_SIMILARITY,
        ) if RAG_USE_TEXT_CONTEXT else ([], None)
        text_query_parts = self._build_text_query_parts(
            template=template,
            query_code=query_code,
            mutation_type=mutation_type,
        )
        text_query = " ".join(text_query_parts)
        pre_rerank_code_scores = {mutation.gene_id: float(mutation.score) for mutation in mutations}
        pre_rerank_text_scores = {ctx.document_id: float(ctx.score) for ctx in text_candidates}
        code_candidates_pre_rerank = self._candidate_payloads(list(mutations))
        text_candidates_pre_rerank = self._candidate_payloads(list(text_candidates))

        # Rerank both result sets if enabled
        reranker_used = False
        if RAG_RERANKER_ENABLED and (mutations or text_candidates):
            reranker = _get_reranker()
            code_rerank_query = query_code or mutation_type or ""
            text_rerank_query = text_query or mutation_type or ""
            if code_rerank_query:
                lines = code_rerank_query.strip().splitlines()
                if len(lines) > _RERANK_MAX_QUERY_LINES:
                    code_rerank_query = "\n".join(lines[:_RERANK_MAX_QUERY_LINES])
            if mutations:
                mutations = reranker.rerank(
                    code_rerank_query, mutations, content_fn=lambda m: m.code, top_k=self.config.top_k
                )
            if text_candidates:
                text_candidates = reranker.rerank(
                    text_rerank_query, text_candidates, content_fn=lambda c: c.content, top_k=len(text_candidates)
                )
            reranker_used = True
        code_candidates_post_rerank = self._candidate_payloads(list(mutations), pre_scores=pre_rerank_code_scores)
        text_candidates_post_rerank = self._candidate_payloads(
            list(text_candidates),
            pre_scores=pre_rerank_text_scores,
        )
        text_contexts, text_selection = self._select_text_contexts(
            text_candidates,
            mutation_type=mutation_type,
            text_query=text_query,
        )

        sections: list[str] = []

        # Section 1: Domain knowledge from text namespace (PyTorch docs, papers)
        if text_contexts:
            text_block = self.retrieval.format_text_context(text_contexts)
            sections.append(
                "The following PyTorch documentation and research context may be "
                "relevant to your architectural decisions:\n"
                f"{text_block}"
            )

        # Section 2: Historical mutations from code namespace
        if mutations:
            context_block = self.retrieval.format_context(mutations)
            sections.append(
                "### RAG Code Examples\n"
                "The following code blocks are historically successful mutations from this exact codebase. "
                "Notice the `accuracy_delta` and `parameters_delta` metadata to understand their impact. "
                "Do NOT copy them exactly. Extract the architectural motifs (e.g., where they placed SiLU, "
                "how they grouped convolutions, or their choice of optimizer) and apply those lessons to "
                "your current task.\n"
                f"{context_block}"
            )

        augmented_template = template
        if sections:
            augmented_template = "\n\n".join(sections) + f"\n\n{template.strip()}"

        # Emit a structured event so analysis can verify RAG was actually used.
        try:
            template_hash = hashlib.sha1(template.encode("utf-8", errors="ignore")).hexdigest()
            query_hash = hashlib.sha1((query_code or "").encode("utf-8", errors="ignore")).hexdigest()
            record_metric(
                "rag_context_built",
                {
                    "run_id": os.getenv("RUN_ID", RUN_ID),
                    "gene_id": gene_id,
                    "mutation_type": mutation_type,
                    "template_hash": template_hash,
                    "query_hash": query_hash,
                    "rag_use_code_context": bool(RAG_USE_CODE_CONTEXT),
                    "rag_use_text_context": bool(RAG_USE_TEXT_CONTEXT),
                    "reranker_enabled": bool(RAG_RERANKER_ENABLED),
                    "reranker_used": reranker_used,
                    "top_k_code": self.config.top_k,
                    "top_k_text": self.config.text_top_k,
                    "min_similarity": RAG_MIN_SIMILARITY,
                    "retrieved_code_n": len(mutations),
                    "retrieved_text_n": len(text_contexts),
                    "template_augmented": bool(sections),
                    "query_code_present": bool(query_code and query_code.strip()),
                    "text_query_present": bool(text_query.strip()),
                    "text_query_words": len(text_query.split()),
                    "text_query_preview": text_query[:240],
                    "text_selection_mode": text_selection["mode"],
                    "selected_text_api_n": text_selection["selected_api"],
                    "selected_text_pdf_n": text_selection["selected_pdf"],
                    "selected_text_other_n": text_selection["selected_other"],
                    "selected_doc_ids_code": [m.gene_id for m in mutations],
                    "selected_doc_ids_text": [c.document_id for c in text_contexts],
                    "selected_text_doc_types": [c.doc_type for c in text_contexts],
                    "selected_text_sources": [c.source for c in text_contexts],
                    "selected_text_names": [c.metadata.get("name") for c in text_contexts],
                    "context_words_code": sum(len((m.code or "").split()) for m in mutations),
                    "context_words_text": sum(len((c.content or "").split()) for c in text_contexts),
                    "code_candidates_pre_rerank": code_candidates_pre_rerank,
                    "code_candidates_post_rerank": code_candidates_post_rerank,
                    "text_candidates_pre_rerank": text_candidates_pre_rerank,
                    "text_candidates_post_rerank": text_candidates_post_rerank,
                    "code_search": None
                    if code_stats is None
                    else {
                        "candidate_k": code_stats.candidate_k,
                        "returned_k": code_stats.returned_k,
                        "filtered_k": code_stats.filtered_k,
                        "min_similarity": code_stats.min_similarity,
                    },
                    "text_search": None
                    if text_stats is None
                    else {
                        "candidate_k": text_stats.candidate_k,
                        "returned_k": text_stats.returned_k,
                        "filtered_k": text_stats.filtered_k,
                        "min_similarity": text_stats.min_similarity,
                        "selection_mode": text_selection["mode"],
                        "selected_api": text_selection["selected_api"],
                        "selected_pdf": text_selection["selected_pdf"],
                        "selected_other": text_selection["selected_other"],
                    },
                },
            )
        except Exception:
            pass

        if not sections:
            return template, []

        return augmented_template, mutations
