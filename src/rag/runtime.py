"""Runtime helpers for initializing and reusing RAG components."""

from __future__ import annotations

import threading
from typing import List, Sequence

from cfg.constants import (
    RAG_CODE_EMBED_MODEL,
    RAG_DATA_DIR,
    RAG_ENABLED,
    RAG_MAX_PARAMETERS,
    RAG_MIN_ACCURACY,
    RAG_TEXT_EMBED_MODEL,
    RAG_TOP_K,
)

from .embeddings import EmbeddingConfig, EmbeddingService
from .prompt_enhancer import PromptEnhancer, PromptEnhancerConfig
from .retrieval import RetrievedMutation, RetrievalService
from .vector_db import VectorStoreManager


class _DimensionMismatchError(RuntimeError):
    """Internal: raised when FAISS index dims don't match embedding model dims."""


class RagRuntime:
    """Encapsulates the long-lived RAG services."""

    def __init__(self) -> None:
        embedding_config = EmbeddingConfig(
            code_model_name=RAG_CODE_EMBED_MODEL,
            text_model_name=RAG_TEXT_EMBED_MODEL,
        )
        self.store = VectorStoreManager(RAG_DATA_DIR)
        self.embeddings = EmbeddingService(embedding_config)

        # H1: Verify FAISS index dimensions match the configured embedding models.
        # Detects model drift between ingestion (setup_rag.py) and runtime.
        for ns_name, embed_fn, label in [
            (VectorStoreManager.CODE_NAMESPACE, self.embeddings.embed_code, "code"),
            (VectorStoreManager.TEXT_NAMESPACE, self.embeddings.embed_text, "text"),
        ]:
            ns = self.store._namespace(ns_name)
            if ns.index is not None:
                probe = embed_fn("dimension probe")
                if ns.index.d != probe.shape[-1]:
                    raise _DimensionMismatchError(
                        f"FAISS {label} index dimension ({ns.index.d}) != "
                        f"embedding model dimension ({probe.shape[-1]}). "
                        f"Re-run setup_rag.py to re-index with the current models."
                    )

        self.retrieval = RetrievalService(self.store, self.embeddings)
        self.prompt_enhancer = PromptEnhancer(
            self.retrieval,
            PromptEnhancerConfig(
                top_k=RAG_TOP_K,
                min_accuracy=RAG_MIN_ACCURACY,
                max_parameters=RAG_MAX_PARAMETERS,
            ),
        )

    def enhance_template(
        self,
        template: str,
        mutation_type: str | None = None,
        query_code: str | None = None,
        gene_id: str | None = None,
    ) -> tuple[str, Sequence[RetrievedMutation]]:
        return self.prompt_enhancer.enhance_template(
            template=template,
            mutation_type=mutation_type,
            query_code=query_code,
            gene_id=gene_id,
        )

    def log_mutation_code(self, content: str, metadata: dict) -> str | None:
        if not content.strip():
            return None
        embeddings = self.embeddings.embed_code(content)
        document_ids = self.store.add_code_documents([content], embeddings, [metadata])
        return document_ids[0] if document_ids else None

    def collect_context(
        self, mutation_type: str | None = None, query_code: str | None = None
    ) -> Sequence[RetrievedMutation]:
        return self.prompt_enhancer.build_context(mutation_type=mutation_type, query_code=query_code)

    def format_context(self, mutations: Sequence[RetrievedMutation]) -> str:
        return self.retrieval.format_context(mutations)


_runtime_lock = threading.Lock()
_runtime_instance: RagRuntime | None = None


def get_runtime() -> RagRuntime | None:
    """Return the singleton RAG runtime if enabled.

    Soft-disables RAG (returns None) if the FAISS index dimensions don't match
    the configured embedding models, preventing silent retrieval corruption.
    """
    if not RAG_ENABLED:
        return None

    global _runtime_instance
    if _runtime_instance is None:
        with _runtime_lock:
            if _runtime_instance is None:
                try:
                    _runtime_instance = RagRuntime()
                except _DimensionMismatchError as exc:
                    import warnings
                    warnings.warn(
                        f"[RAG] {exc} — RAG disabled for this session.",
                        stacklevel=2,
                    )
                    return None
    return _runtime_instance
