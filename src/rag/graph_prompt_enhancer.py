"""Prompt enhancement helpers for standalone Graph-RAG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .graph_retrieval import GraphRetrievedMutation, GraphRetrievalService


@dataclass(frozen=True)
class GraphPromptEnhancerConfig:
    # Maximum number of graph-retrieved exemplars injected into the prompt.
    top_k: int = 5
    # Quality guardrail applied during graph retrieval.
    min_accuracy: float = 0.9


class GraphPromptEnhancer:
    """Build augmented prompts using graph-only retrieval results."""

    def __init__(self, retrieval_service: GraphRetrievalService, config: GraphPromptEnhancerConfig | None = None):
        self.retrieval = retrieval_service
        self.config = config or GraphPromptEnhancerConfig()

    def build_context(
        self,
        parent_gene_id: str | None = None,
        query_code: str | None = None,
    ) -> List[GraphRetrievedMutation]:
        """Build ranked context using parent-graph traversal + AST pattern overlap fallback."""
        mutations: list[GraphRetrievedMutation] = []

        if parent_gene_id:
            mutations.extend(
                self.retrieval.retrieve_by_parent_gene(
                    parent_gene_id=parent_gene_id,
                    top_k=self.config.top_k,
                    min_accuracy=self.config.min_accuracy,
                )
            )

        if query_code and len(mutations) < self.config.top_k:
            # If parent-based retrieval returns too few results, backfill from
            # structural overlap with the current code snippet.
            extra = self.retrieval.retrieve_by_pattern_overlap(
                query_code=query_code,
                top_k=self.config.top_k - len(mutations),
                min_accuracy=self.config.min_accuracy,
            )
            mutations.extend(extra)

        # Deduplicate by gene_id and keep only the highest-score instance.
        deduped: dict[str, GraphRetrievedMutation] = {}
        for mutation in mutations:
            existing = deduped.get(mutation.gene_id)
            if existing is None or mutation.score > existing.score:
                deduped[mutation.gene_id] = mutation

        # Stable ranking: strongest signals are shown first to the LLM.
        result = list(deduped.values())
        result.sort(key=lambda mutation: mutation.score, reverse=True)
        return result[: self.config.top_k]

    def enhance_template(
        self,
        template: str,
        parent_gene_id: str | None = None,
        query_code: str | None = None,
    ) -> tuple[str, List[GraphRetrievedMutation]]:
        """Prepend graph-derived mutation context to the original template."""
        mutations = self.build_context(parent_gene_id=parent_gene_id, query_code=query_code)
        if not mutations:
            return template, []

        context_block = self.retrieval.format_context(mutations)
        augmented_template = (
            "Consider the following graph-retrieved successful mutations. "
            "Use the structural patterns as inspiration while preserving correctness.\n"
            f"{context_block}\n\n"
            f"{template.strip()}"
        )
        return augmented_template, mutations
