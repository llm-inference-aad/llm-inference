"""FAISS vector-store retrieval backend."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from ..embeddings import EmbeddingConfig, EmbeddingService
from ..retrieval import RetrievedContext, RetrievedMutation, RetrievalService, RetrievalStats
from ..vector_db import VectorStoreManager


class _DimensionMismatchError(RuntimeError):
    """FAISS index dimensions don't match the configured embedding models."""


class FaissRetrievalBackend:
    """RAG backend backed by FAISS flat-IP indices + dual embedding models."""

    def __init__(
        self,
        rag_data_dir: str | Path,
        code_embed_model: str = "microsoft/codebert-base",
        text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._store = VectorStoreManager(rag_data_dir)
        self._embeddings = EmbeddingService(
            EmbeddingConfig(
                code_model_name=code_embed_model,
                text_model_name=text_embed_model,
            )
        )

        # Verify FAISS index dimensions match the configured embedding models.
        for ns_name, embed_fn, label in [
            (VectorStoreManager.CODE_NAMESPACE, self._embeddings.embed_code, "code"),
            (VectorStoreManager.TEXT_NAMESPACE, self._embeddings.embed_text, "text"),
        ]:
            ns = self._store._namespace(ns_name)
            if ns.index is not None:
                probe = embed_fn("dimension probe")
                if ns.index.d != probe.shape[-1]:
                    raise _DimensionMismatchError(
                        f"FAISS {label} index dimension ({ns.index.d}) != "
                        f"embedding model dimension ({probe.shape[-1]}). "
                        f"Re-run setup_rag.py to re-index with the current models."
                    )

        self._service = RetrievalService(self._store, self._embeddings)

    @property
    def backend_name(self) -> str:
        return "faiss"

    # ── Code retrieval ───────────────────────────────────────────────

    def retrieve_similar_mutations(
        self, query_code: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> List[RetrievedMutation]:
        return self._service.retrieve_similar_mutations(query_code, top_k, min_similarity)

    def retrieve_similar_mutations_with_stats(
        self, query_code: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> Tuple[List[RetrievedMutation], RetrievalStats]:
        return self._service.retrieve_similar_mutations_with_stats(query_code, top_k, min_similarity)

    def retrieve_high_performers(
        self,
        min_accuracy: float = 0.9,
        max_parameters: float | None = None,
        limit: int = 5,
    ) -> List[RetrievedMutation]:
        return self._service.retrieve_high_performers(min_accuracy, max_parameters, limit)

    def retrieve_by_mutation_type(
        self, mutation_type: str, limit: int = 5
    ) -> List[RetrievedMutation]:
        return self._service.retrieve_by_mutation_type(mutation_type, limit)

    # ── Text retrieval ───────────────────────────────────────────────

    def retrieve_similar_text(
        self, query: str, top_k: int = 3, min_similarity: float = 0.3
    ) -> List[RetrievedContext]:
        return self._service.retrieve_similar_text(query, top_k, min_similarity)

    def retrieve_similar_text_with_stats(
        self, query: str, top_k: int = 3, min_similarity: float = 0.3
    ) -> Tuple[List[RetrievedContext], RetrievalStats]:
        return self._service.retrieve_similar_text_with_stats(query, top_k, min_similarity)

    # ── Formatting ───────────────────────────────────────────────────

    def format_context(self, mutations: Sequence[RetrievedMutation]) -> str:
        return self._service.format_context(mutations)

    def format_text_context(self, contexts: Sequence[RetrievedContext]) -> str:
        return self._service.format_text_context(contexts)

    # ── Mutation logging ─────────────────────────────────────────────

    def log_mutation_code(self, content: str, metadata: dict) -> str | None:
        if not content.strip():
            return None
        embeddings = self._embeddings.embed_code(content)
        doc_ids = self._store.add_code_documents([content], embeddings, [metadata])
        return doc_ids[0] if doc_ids else None
