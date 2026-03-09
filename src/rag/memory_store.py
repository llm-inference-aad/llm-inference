"""RAG memory store: episodic memory for past interactions (queries, responses, outcomes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .embeddings import EmbeddingService
from .vector_db import VectorStoreManager


@dataclass(frozen=True)
class MemoryEntry:
    """A retrieved memory entry from the store."""

    document_id: str
    score: float
    content: str
    metadata: dict


class MemoryStore:
    """
    Store and retrieve episodic memory: past mutation attempts, outcomes, and context.
    Uses the text embedding model (384-dim) since memory entries are natural language summaries.
    """

    def __init__(self, store: VectorStoreManager, embeddings: EmbeddingService):
        self.store = store
        self.embeddings = embeddings

    def add_entry(
        self,
        summary: str,
        metadata: dict,
    ) -> str | None:
        """
        Add a memory entry. Summary should describe the interaction (e.g., mutation type, outcome, fitness).
        Returns document_id if successful, else None.
        """
        if not summary.strip():
            return None
        embeddings = self.embeddings.embed_text(summary)
        doc_ids = self.store.add_memory_documents([summary], embeddings, [metadata])
        return doc_ids[0] if doc_ids else None

    def search_similar(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.3,
    ) -> List[MemoryEntry]:
        """
        Retrieve memory entries similar to the query (e.g., current mutation type + code context).
        """
        if not query.strip():
            return []
        query_embedding = self.embeddings.embed_text(query)[0]
        results = self.store.search_memory(query_embedding, top_k=top_k * 2)
        filtered = [r for r in results if r.score >= min_similarity][:top_k]
        return [
            MemoryEntry(
                document_id=r.document.document_id,
                score=r.score,
                content=r.document.content,
                metadata=r.document.metadata,
            )
            for r in filtered
        ]
