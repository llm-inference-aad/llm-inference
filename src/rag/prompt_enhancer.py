"""Prompt enhancement helpers that inject RAG context."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from cfg.constants import (
    RAG_MIN_SIMILARITY,
    RAG_RERANKER_ENABLED,
    RAG_TEXT_TOP_K,
    RAG_USE_CODE_CONTEXT,
    RAG_USE_TEXT_CONTEXT,
    RUN_ID,
)

from .reranker import Reranker
from .retrieval import RetrievedContext, RetrievedMutation, RetrievalService, RetrievalStats
from utils.rag_metrics import record_metric

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
    text_top_k: int = RAG_TEXT_TOP_K
    min_accuracy: float = 0.9
    max_parameters: float | None = None


class PromptEnhancer:
    """Build augmented prompt text that includes retrieval context."""

    def __init__(self, retrieval_service: RetrievalService, config: PromptEnhancerConfig | None = None):
        self.retrieval = retrieval_service
        self.config = config or PromptEnhancerConfig()

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

    def build_context_with_stats(
        self,
        mutation_type: str | None = None,
        query_code: str | None = None,
    ) -> tuple[List[RetrievedMutation], RetrievalStats | None]:
        """Build code context and return similarity-search stats (if used)."""
        if not RAG_USE_CODE_CONTEXT:
            return [], None

        stats: RetrievalStats | None = None
        mutations: list[RetrievedMutation] = []

        if query_code:
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

        # Build a natural-language query from available signal
        query_parts: list[str] = []
        if mutation_type:
            query_parts.append(f"CNN architecture {mutation_type} mutation")
        if query_code:
            # Use first few lines of code as query context
            code_snippet = "\n".join(query_code.strip().splitlines()[:10])
            query_parts.append(code_snippet)

        if not query_parts:
            return []

        query = " ".join(query_parts)
        return self.retrieval.retrieve_similar_text(
            query, top_k=self.config.text_top_k, min_similarity=RAG_MIN_SIMILARITY
        )

    def build_text_context_with_stats(
        self,
        query_code: str | None = None,
        mutation_type: str | None = None,
    ) -> tuple[List[RetrievedContext], RetrievalStats | None]:
        """Build text context and return similarity-search stats (if used)."""
        if not RAG_USE_TEXT_CONTEXT:
            return [], None

        query_parts: list[str] = []
        if mutation_type:
            query_parts.append(f"CNN architecture {mutation_type} mutation")
        if query_code:
            code_snippet = "\n".join(query_code.strip().splitlines()[:10])
            query_parts.append(code_snippet)
        if not query_parts:
            return [], None

        query = " ".join(query_parts)
        contexts, stats = self.retrieval.retrieve_similar_text_with_stats(
            query, top_k=self.config.text_top_k, min_similarity=RAG_MIN_SIMILARITY
        )
        return contexts, stats

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
        text_contexts, text_stats = self.build_text_context_with_stats(
            query_code=query_code, mutation_type=mutation_type
        )

        # Rerank both result sets if enabled
        reranker_used = False
        if RAG_RERANKER_ENABLED and (mutations or text_contexts):
            reranker = _get_reranker()
            rerank_query = query_code or mutation_type or ""
            # M4: Truncate to avoid cross-encoder CPU bottleneck at population scale
            if rerank_query:
                lines = rerank_query.strip().splitlines()
                if len(lines) > _RERANK_MAX_QUERY_LINES:
                    rerank_query = "\n".join(lines[:_RERANK_MAX_QUERY_LINES])
            if mutations:
                mutations = reranker.rerank(
                    rerank_query, mutations, content_fn=lambda m: m.code, top_k=self.config.top_k
                )
            if text_contexts:
                text_contexts = reranker.rerank(
                    rerank_query, text_contexts, content_fn=lambda c: c.content, top_k=self.config.text_top_k
                )
            reranker_used = True

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
                "Consider the following historically successful mutations. "
                "Emphasize ideas that balance accuracy gains with parameter reductions.\n"
                f"{context_block}"
            )

        if not sections:
            return template, []

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
                    "selected_doc_ids_code": [m.gene_id for m in mutations],
                    "selected_doc_ids_text": [c.document_id for c in text_contexts],
                    "context_words_code": sum(len((m.code or "").split()) for m in mutations),
                    "context_words_text": sum(len((c.content or "").split()) for c in text_contexts),
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
                    },
                },
            )
        except Exception:
            pass

        return augmented_template, mutations
