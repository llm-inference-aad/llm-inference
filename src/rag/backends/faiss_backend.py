"""FAISS-backed implementation of BackendProtocol.

This adapter wraps the existing lower-level services
(:class:`~src.rag.embeddings.EmbeddingService`,
:class:`~src.rag.vector_db.VectorStoreManager`,
:class:`~src.rag.retrieval.RetrievalService`) behind the stable
:class:`~src.rag.backend_protocol.BackendProtocol` interface.

Design rules:
- No new retrieval logic: all searching delegates to the existing
  ``RetrievalService`` methods.
- No singleton coupling: the backend holds its own injected service
  references and does **not** touch ``src.rag.runtime._runtime_instance``.
- The ``index()`` method is a thin shim so the protocol surface is satisfied;
  production indexing should use the dedicated ingest pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

from cfg.constants import (
    RAG_DATA_DIR,
    RAG_MIN_SIMILARITY,
    RAG_TEXT_CANDIDATE_K,
)

from ..api_types import RetrievedBlock, RetrieveRequest, RetrieveResponse
from ..embeddings import EmbeddingConfig, EmbeddingService
from ..retrieval import RetrievalService
from ..vector_db import VectorStoreManager


class FaissBackend:
    """BackendProtocol-compliant adapter over the existing FAISS retrieval stack.

    Satisfies :class:`~src.rag.backend_protocol.BackendProtocol` through
    structural sub-typing (PEP 544) — no inheritance required.

    Args:
        store: A :class:`~src.rag.vector_db.VectorStoreManager` instance.
            If *None*, one is constructed from *rag_data_dir*.
        embeddings: An :class:`~src.rag.embeddings.EmbeddingService` instance.
            If *None*, one is constructed with default config.
        rag_data_dir: Path to the RAG data directory (used when *store* is
            not provided).  Defaults to :data:`cfg.constants.RAG_DATA_DIR`.
        min_similarity: Minimum cosine-similarity threshold for candidates.
        text_candidate_k: Candidate pool size for text search (pre-selection).
    """

    def __init__(
        self,
        store: Optional[VectorStoreManager] = None,
        embeddings: Optional[EmbeddingService] = None,
        rag_data_dir: Optional[str] = None,
        min_similarity: float = RAG_MIN_SIMILARITY,
        text_candidate_k: int = RAG_TEXT_CANDIDATE_K,
    ) -> None:
        _data_dir = rag_data_dir or RAG_DATA_DIR
        self._store: VectorStoreManager = store or VectorStoreManager(_data_dir)
        self._embeddings: EmbeddingService = embeddings or EmbeddingService(EmbeddingConfig())
        self._retrieval = RetrievalService(store=self._store, embeddings=self._embeddings)
        self._min_similarity = min_similarity
        self._text_candidate_k = text_candidate_k

    # ---------------------------------------------------------------------- #
    # BackendProtocol surface
    # ---------------------------------------------------------------------- #

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Retrieve relevant blocks for *request*.

        Namespace routing:
        - ``"code"`` → searches code mutations via
          :meth:`~src.rag.retrieval.RetrievalService.retrieve_similar_mutations_with_stats`.
        - ``"text"`` → searches text documents via
          :meth:`~src.rag.retrieval.RetrievalService.retrieve_text_candidates_with_stats`.
        - ``None`` (or any other value) → searches both namespaces and merges
          the results, interleaving by score.

        The ``diagnostics`` field of each :class:`~src.rag.api_types.RetrievedBlock`
        is populated with ``candidate_count``, ``reranked`` (always ``False`` at
        this layer — reranking lives in :class:`~src.rag.prompt_enhancer.PromptEnhancer`),
        and ``source`` (``"code"`` or ``"text"``).
        """
        t0 = time.monotonic()

        namespace = request.namespace
        top_k = request.top_k
        query = request.query
        filters = request.filters or {}

        blocks: List[RetrievedBlock] = []
        request_diagnostics: dict[str, Any] = {"reranker_used": False}

        if namespace in (None, VectorStoreManager.CODE_NAMESPACE, "code"):
            code_blocks, code_diag = self._retrieve_code_blocks(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            blocks.extend(code_blocks)
            request_diagnostics["code_search"] = code_diag

        if namespace in (None, VectorStoreManager.TEXT_NAMESPACE, "text"):
            text_blocks, text_diag = self._retrieve_text_blocks(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            blocks.extend(text_blocks)
            request_diagnostics["text_search"] = text_diag

        # When searching both namespaces, merge and truncate to top_k by score.
        if namespace is None and len(blocks) > top_k:
            blocks = sorted(blocks, key=lambda b: b.score, reverse=True)[:top_k]

        latency_ms = (time.monotonic() - t0) * 1000.0
        return RetrieveResponse(
            blocks=blocks,
            diagnostics=request_diagnostics,
            latency_ms=latency_ms,
        )

    def index(self, document: Any) -> None:  # noqa: ANN401
        """Satisfy the BackendProtocol surface.

        Production indexing should use the dedicated ingest pipeline
        (``src.rag.indexer``, ``scripts/setup_rag.py``).  This shim exists so
        that test fakes and protocol compliance checks can call it without
        errors.
        """
        # Intentional no-op: the ingest pipeline owns index mutations.

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _retrieve_code_blocks(
        self,
        query: str,
        top_k: int,
        filters: dict,
    ) -> tuple[List[RetrievedBlock], dict]:
        """Delegate to RetrievalService and convert to RetrievedBlock list."""
        min_similarity = float(filters.get("min_similarity", self._min_similarity))

        mutations, stats = self._retrieval.retrieve_similar_mutations_with_stats(
            query_code=query,
            top_k=top_k,
            min_similarity=min_similarity,
        )

        diag = {
            "candidate_count": stats.candidate_k,
            "returned_k": stats.returned_k,
            "filtered_k": stats.filtered_k,
        }

        blocks = [
            RetrievedBlock(
                kind="mutation_code",
                document_id=m.gene_id,
                title=m.metadata.get("description", m.gene_id)[:120],
                score=m.score,
                content=m.code,
                diagnostics={
                    "candidate_count": stats.candidate_k,
                    "reranked": False,
                    "source": "code",
                    "gene_id": m.gene_id,
                    "mutation_type": m.metadata.get("mutation_type"),
                },
            )
            for m in mutations
        ]
        return blocks, diag

    def _retrieve_text_blocks(
        self,
        query: str,
        top_k: int,
        filters: dict,
    ) -> tuple[List[RetrievedBlock], dict]:
        """Delegate to RetrievalService text retrieval and convert to RetrievedBlock list."""
        min_similarity = float(filters.get("min_similarity", self._min_similarity))
        candidate_k = int(filters.get("candidate_k", self._text_candidate_k))

        contexts, stats = self._retrieval.retrieve_text_candidates_with_stats(
            query=query,
            candidate_k=candidate_k,
            min_similarity=min_similarity,
        )

        # Truncate to top_k after candidate filtering.
        contexts = contexts[:top_k]

        diag = {
            "candidate_count": stats.candidate_k,
            "returned_k": stats.returned_k,
            "filtered_k": stats.filtered_k,
        }

        blocks = [
            RetrievedBlock(
                kind=ctx.doc_type,
                document_id=ctx.document_id,
                title=ctx.metadata.get("name", ctx.document_id)[:120],
                score=ctx.score,
                content=ctx.content,
                diagnostics={
                    "candidate_count": stats.candidate_k,
                    "reranked": False,
                    "source": ctx.source,
                    "doc_type": ctx.doc_type,
                },
            )
            for ctx in contexts
        ]
        return blocks, diag
