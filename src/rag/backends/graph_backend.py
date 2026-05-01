"""Graph-RAG implementation of BackendProtocol.

This adapter reuses the existing knowledge-graph retrieval stack from
``src.rag.graph_retrieval`` and presents it behind the shared
``BackendProtocol`` interface used by ``RagService``.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from cfg.constants import RAG_DATA_DIR

from src.rag.api_types import RetrievedBlock, RetrieveRequest, RetrieveResponse
from src.rag.graph_retrieval import GraphRetrievalService
from src.rag.vector_db import VectorStoreManager


class GraphBackend:
    """BackendProtocol-compliant graph retrieval adapter.

    Notes:
    - Graph retrieval currently applies to mutation-code context only.
    - ``namespace="text"`` returns an empty response by design.
    """

    def __init__(
        self,
        store: Optional[VectorStoreManager] = None,
        rag_data_dir: Optional[str] = None,
    ) -> None:
        self._store: VectorStoreManager = store or VectorStoreManager(rag_data_dir or RAG_DATA_DIR)
        self._graph = GraphRetrievalService(self._store)

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        t0 = time.monotonic()

        # Graph backend does not serve text corpus queries.
        if request.namespace == "text":
            return RetrieveResponse(
                blocks=[],
                diagnostics={"backend": "graph", "namespace": "text", "supported": False},
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )

        # Default to pattern-overlap retrieval; when caller provides parent gene
        # we prioritize genealogy-aware traversal.
        top_k = max(1, int(request.top_k))
        filters = request.filters or {}
        min_accuracy = float(filters.get("min_accuracy", 0.0))
        parent_gene_id = filters.get("parent_gene_id")

        if isinstance(parent_gene_id, str) and parent_gene_id.strip():
            hits = self._graph.retrieve_by_parent_gene(
                parent_gene_id=parent_gene_id.strip(),
                top_k=top_k,
                min_accuracy=min_accuracy,
            )
            retrieval_mode = "parent_gene"
        else:
            hits = self._graph.retrieve_by_pattern_overlap(
                query_code=request.query,
                top_k=top_k,
                min_accuracy=min_accuracy,
            )
            retrieval_mode = "pattern_overlap"

        blocks: List[RetrievedBlock] = [
            RetrievedBlock(
                kind="mutation_code",
                document_id=h.gene_id,
                title=h.description[:120] if h.description else h.gene_id,
                score=float(h.score),
                content=h.code,
                diagnostics={
                    "source": "graph",
                    "graph_reason": h.metadata.get("graph_reason"),
                    "retrieval_source": h.metadata.get("retrieval_source", "graph"),
                    "mutation_type": h.metadata.get("mutation_type"),
                },
            )
            for h in hits
        ]

        return RetrieveResponse(
            blocks=blocks,
            diagnostics={
                "backend": "graph",
                "retrieval_mode": retrieval_mode,
                "returned_k": len(blocks),
            },
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    def index(self, document: Any) -> None:
        # Index mutations through the shared VectorStoreManager code namespace.
        # The graph is rebuilt lazily from stored docs during retrieval.
        if isinstance(document, str):
            content = document
            metadata: dict = {}
        elif isinstance(document, dict):
            content = str(document.get("content", ""))
            metadata = dict(document.get("metadata") or {})
        else:
            raise TypeError("GraphBackend.index expects str or dict")

        if not content.strip():
            return

        metadata.setdefault("document_id", metadata.get("gene_id"))
        # Graph backend does not own embedding/indexing policy yet; ingestion
        # should normally flow through runtime/service logging paths.
        raise NotImplementedError(
            "GraphBackend.index is not implemented for direct ingestion; "
            "use runtime/service mutation logging pipeline."
        )
