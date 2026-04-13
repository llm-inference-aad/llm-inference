"""RetrievalBackend protocol — the contract every RAG backend must satisfy."""

from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple, runtime_checkable

from ..retrieval import RetrievedContext, RetrievedMutation, RetrievalStats


@runtime_checkable
class RetrievalBackend(Protocol):
    """Structural interface for pluggable RAG retrieval backends.

    Implementations must expose these methods so that ``PromptEnhancer``
    and ``RagRuntime`` can consume them without knowing the concrete
    backend (FAISS, PageIndex, Graph RAG, …).
    """

    @property
    def backend_name(self) -> str: ...

    # ── Code retrieval ───────────────────────────────────────────────
    def retrieve_similar_mutations(
        self, query_code: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> List[RetrievedMutation]: ...

    def retrieve_similar_mutations_with_stats(
        self, query_code: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> Tuple[List[RetrievedMutation], RetrievalStats]: ...

    def retrieve_high_performers(
        self,
        min_accuracy: float = 0.9,
        max_parameters: float | None = None,
        limit: int = 5,
    ) -> List[RetrievedMutation]: ...

    def retrieve_by_mutation_type(
        self, mutation_type: str, limit: int = 5
    ) -> List[RetrievedMutation]: ...

    # ── Text retrieval ───────────────────────────────────────────────
    def retrieve_similar_text(
        self, query: str, top_k: int = 3, min_similarity: float = 0.3
    ) -> List[RetrievedContext]: ...

    def retrieve_similar_text_with_stats(
        self, query: str, top_k: int = 3, min_similarity: float = 0.3
    ) -> Tuple[List[RetrievedContext], RetrievalStats]: ...

    # ── Formatting ───────────────────────────────────────────────────
    def format_context(self, mutations: Sequence[RetrievedMutation]) -> str: ...

    def format_text_context(self, contexts: Sequence[RetrievedContext]) -> str: ...

    # ── Mutation logging (runtime feedback loop) ─────────────────────
    def log_mutation_code(self, content: str, metadata: dict) -> str | None: ...
