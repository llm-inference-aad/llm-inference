"""Prompt enhancement helpers that inject RAG context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from cfg.constants import RAG_MIN_SIMILARITY

from .retrieval import RetrievedMutation, RetrievalService


@dataclass(frozen=True)
class PromptEnhancerConfig:
    top_k: int = 5
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

    def enhance_template(
        self,
        template: str,
        mutation_type: str | None = None,
        query_code: str | None = None,
    ) -> tuple[str, List[RetrievedMutation]]:
        mutations = self.build_context(mutation_type=mutation_type, query_code=query_code)
        if not mutations:
            return template, []

        context_block = self.retrieval.format_context(mutations)
        augmented_template = (
            "Consider the following historically successful mutations. "
            "Emphasize ideas that balance accuracy gains with parameter reductions.\n"
            f"{context_block}\n\n"
            f"{template.strip()}"
        )
        return augmented_template, mutations



