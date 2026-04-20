"""MemoryBackend — episodic memory retrieval channel for RAG.

Stores one-line natural-language summaries of successful mutations in the
``"memory"`` FAISS namespace (384-dim MiniLM-L6 embeddings, same model as the
text namespace).  At prompt-build time ``RagService`` calls this backend after
code/text retrieval; the returned blocks are prepended as a bullet-list digest.

Key design decisions vs. Ajay's original ``MemoryStore``:
- Implements ``BackendProtocol`` so it composes with ``RagService`` without
  coupling to the singleton runtime.
- The retrieval query is the **full mutation description** (or the full
  ``query_code`` string), not just ``mutation_type + first 5 lines of code``.
  The first-5-lines query was identified as a weak-query bug in the eval:
  those lines are almost always boilerplate imports with low discriminating
  power.
- ``min_similarity`` defaults to 0.5 (tighter than Ajay's 0.3) to reduce
  noise from low-relevance summaries.
- Every returned block carries ``diagnostics["source"] = "memory"`` so the
  bookkeeping ledger can measure per-channel contribution.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional

from src.rag.api_types import (
    RetrievedBlock,
    RetrieveRequest,
    RetrieveResponse,
)

if TYPE_CHECKING:
    pass


class MemoryBackend:
    """BackendProtocol-compliant retrieval channel for episodic mutation summaries.

    Args:
        vector_store: A ``VectorStoreManager`` (or fake) that exposes
            ``add_memory_documents`` / ``search_memory``.
        embedding_service: An ``EmbeddingService`` (or fake) that exposes
            ``embed_text(text) -> np.ndarray``.
        min_similarity: Minimum cosine similarity for a result to be included.
            Defaults to 0.5 (overridable via ``RAG_MEMORY_MIN_SIMILARITY``).
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_service: Any,
        min_similarity: float = 0.5,
    ) -> None:
        self._store = vector_store
        self._embeddings = embedding_service
        self._min_similarity = min_similarity

    # ---------------------------------------------------------------------- #
    # BackendProtocol surface
    # ---------------------------------------------------------------------- #

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Search the memory namespace and return matching summaries.

        The query string is used verbatim — it is expected to be the full
        mutation description or the full ``query_code`` string, NOT a truncated
        prefix.

        Returns:
            A :class:`~src.rag.api_types.RetrieveResponse` whose blocks have
            ``diagnostics["source"] == "memory"``.
        """
        t0 = time.monotonic()
        query = request.query
        top_k = request.top_k

        if not query.strip():
            return RetrieveResponse(
                blocks=[],
                diagnostics={"source": "memory", "reason": "empty_query"},
                latency_ms=0.0,
            )

        # Embed the full query string (NOT just the first 5 lines).
        query_embedding = self._embeddings.embed_text(query)
        # embed_text may return a 2-D matrix (N, D) or a 1-D vector for a
        # single string.  Normalise to 1-D for the search call.
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]

        # Retrieve candidates (over-fetch then filter by min_similarity).
        raw_results = self._store.search_memory(query_embedding, top_k=top_k * 2)

        blocks: List[RetrievedBlock] = []
        for result in raw_results:
            score = float(result.score)
            if score < self._min_similarity:
                continue
            doc = result.document
            gene_id = doc.metadata.get("gene_id", doc.document_id)
            blocks.append(
                RetrievedBlock(
                    kind="memory_summary",
                    document_id=gene_id,
                    title=f"Past attempt: {gene_id}",
                    score=score,
                    content=doc.content,
                    diagnostics={"source": "memory", "raw_document_id": doc.document_id},
                )
            )
            if len(blocks) >= top_k:
                break

        latency_ms = (time.monotonic() - t0) * 1000.0
        return RetrieveResponse(
            blocks=blocks,
            diagnostics={"source": "memory", "candidate_count": len(raw_results)},
            latency_ms=latency_ms,
        )

    def index(self, document: Any) -> None:
        """Embed *document* and write it to the memory FAISS namespace.

        Args:
            document: A plain string (the mutation summary) or a dict with a
                ``"text"`` key containing the summary and optional metadata
                keys (``"gene_id"``, ``"mutation_type"``, …).
        """
        if isinstance(document, str):
            text = document
            metadata: dict = {}
        elif isinstance(document, dict):
            text = document.get("text", "")
            metadata = {k: v for k, v in document.items() if k != "text"}
        else:
            raise TypeError(
                f"MemoryBackend.index expects str or dict, got {type(document)!r}"
            )

        if not text.strip():
            return

        embeddings = self._embeddings.embed_text(text)
        # Normalise to 2-D (1, D) for add_memory_documents.
        if embeddings.ndim == 1:
            import numpy as np
            embeddings = np.expand_dims(embeddings, axis=0)

        doc_id = metadata.get("document_id") or str(uuid.uuid4())
        meta_record = {"document_id": doc_id, **metadata}
        self._store.add_memory_documents([text], embeddings, [meta_record])
