"""Runtime helpers for standalone Graph-RAG retrieval experiments."""

from __future__ import annotations

import threading
from typing import Sequence

from cfg.constants import (
    GRAPH_RAG_ENABLED,
    RAG_CODE_EMBED_MODEL,
    RAG_DATA_DIR,
    RAG_MIN_ACCURACY,
    RAG_TEXT_EMBED_MODEL,
    RAG_TOP_K,
)

from .embeddings import EmbeddingConfig, EmbeddingService
from .graph_prompt_enhancer import GraphPromptEnhancer, GraphPromptEnhancerConfig
from .graph_retrieval import GraphRetrievedMutation, GraphRetrievalService
from .vector_db import VectorStoreManager


class GraphRagRuntime:
    """Encapsulates graph-only RAG services for side-by-side comparisons."""

    def __init__(self) -> None:
        # Embedding models are still used for logging new mutation artifacts into
        # the vector store. Retrieval in graph mode is graph traversal-based.
        embedding_config = EmbeddingConfig(
            code_model_name=RAG_CODE_EMBED_MODEL,
            text_model_name=RAG_TEXT_EMBED_MODEL,
        )
        self.store = VectorStoreManager(RAG_DATA_DIR)
        self.embeddings = EmbeddingService(embedding_config)
        self.retrieval = GraphRetrievalService(self.store)
        # Prompt enhancer applies retrieval policy + prompt injection policy.
        self.prompt_enhancer = GraphPromptEnhancer(
            self.retrieval,
            GraphPromptEnhancerConfig(
                top_k=RAG_TOP_K,
                min_accuracy=RAG_MIN_ACCURACY,
            ),
        )

    def enhance_template(
        self,
        template: str,
        mutation_type: str | None = None,
        query_code: str | None = None,
        parent_gene_id: str | None = None,
    ) -> tuple[str, Sequence[GraphRetrievedMutation]]:
        return self.prompt_enhancer.enhance_template(
            template=template,
            parent_gene_id=parent_gene_id,
            query_code=query_code,
        )

    def collect_context(
        self,
        mutation_type: str | None = None,
        query_code: str | None = None,
        parent_gene_id: str | None = None,
    ) -> Sequence[GraphRetrievedMutation]:
        return self.prompt_enhancer.build_context(parent_gene_id=parent_gene_id, query_code=query_code)

    def log_mutation_code(self, content: str, metadata: dict) -> str | None:
        if not content.strip():
            return None
        # Persist code into vector namespace so future graph builds can ingest it
        # from a single mutation artifact source.
        embeddings = self.embeddings.embed_code(content)
        document_ids = self.store.add_code_documents([content], embeddings, [metadata])
        return document_ids[0] if document_ids else None

    def format_context(self, mutations: Sequence[GraphRetrievedMutation]) -> str:
        return self.retrieval.format_context(mutations)


_graph_runtime_lock = threading.Lock()
_graph_runtime_instance: GraphRagRuntime | None = None


def get_graph_runtime() -> GraphRagRuntime | None:
    """Return the singleton Graph-RAG runtime if enabled."""
    if not GRAPH_RAG_ENABLED:
        return None

    # Singleton keeps model/index initialization costs out of hot paths.
    global _graph_runtime_instance
    if _graph_runtime_instance is None:
        with _graph_runtime_lock:
            if _graph_runtime_instance is None:
                _graph_runtime_instance = GraphRagRuntime()
    return _graph_runtime_instance
