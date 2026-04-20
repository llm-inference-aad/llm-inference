"""Hand-rolled fakes for the RAG subsystem.

All fakes are pure Python — no torch, no faiss, no sentence-transformers.
They implement the same public surface as their real counterparts so tests can
substitute them without loading GPU-heavy dependencies.

Fake design contracts:
- Deterministic: identical inputs always produce identical outputs.
- Zero-shot: no real model weights are consulted.
- Hermetic: no filesystem I/O unless ``base_dir`` is explicitly passed.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# FakeEmbeddingService
# ---------------------------------------------------------------------------

class FakeEmbeddingService:
    """Deterministic hash-based embedding service.

    Produces float32 vectors whose entries are derived from the SHA-256 digest
    of the input string, spread across the target dimensionality.  The vectors
    are L2-normalised so cosine similarity = dot product (matching real
    sentence-transformer behaviour).

    Dimensions:
    - code embeddings: 768d  (matches microsoft/codebert-base)
    - text embeddings: 384d  (matches all-MiniLM-L6-v2)
    """

    CODE_DIM: int = 768
    TEXT_DIM: int = 384

    @staticmethod
    def _hash_vector(text: str, dim: int) -> np.ndarray:
        """Return a deterministic, L2-normalised float32 vector for *text*."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes
        # Tile the digest to fill *dim* float32 values.
        n_floats_from_digest = len(digest) // 4  # 8 floats from 32 bytes
        seed_floats = np.array(struct.unpack(f"{n_floats_from_digest}f", digest), dtype=np.float32)

        # Repeat/truncate to the required dimension.
        repeats = (dim + n_floats_from_digest - 1) // n_floats_from_digest
        vec = np.tile(seed_floats, repeats)[:dim].copy()

        # Avoid zero-vector edge case (all-zero SHA is impossible but guard anyway).
        if np.allclose(vec, 0.0):
            vec[0] = 1.0

        # L2 normalise.
        norm = np.linalg.norm(vec)
        vec /= norm
        return vec.astype(np.float32)

    def embed_code(self, snippets: str | Sequence[str]) -> np.ndarray:
        """Return (N, 768) float32 embeddings for code snippets."""
        items: list[str] = [snippets] if isinstance(snippets, str) else list(snippets)
        return np.stack([self._hash_vector(s, self.CODE_DIM) for s in items])

    def embed_text(self, documents: str | Sequence[str]) -> np.ndarray:
        """Return (N, 384) float32 embeddings for text documents."""
        items: list[str] = [documents] if isinstance(documents, str) else list(documents)
        return np.stack([self._hash_vector(d, self.TEXT_DIM) for d in items])

    def embed_mixed_batch(
        self, code_snippets: Sequence[str], text_chunks: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.embed_code(list(code_snippets)), self.embed_text(list(text_chunks))


# ---------------------------------------------------------------------------
# FakeVectorStoreManager
# ---------------------------------------------------------------------------

class _InMemoryNamespace:
    """In-memory store for a single namespace (mirrors NamespaceStore surface)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._contents: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[dict] = []
        self._ids: list[str] = []

    def add_documents(
        self,
        contents: Sequence[str],
        embeddings: np.ndarray,
        metadata: Sequence[dict],
    ) -> list[str]:
        import uuid

        new_ids = [str(uuid.uuid4()) for _ in contents]
        self._contents.extend(contents)
        self._embeddings.extend(embeddings)
        self._metadata.extend(metadata)
        self._ids.extend(new_ids)
        return new_ids

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        """Return top-k results sorted by cosine similarity (dot product on normalised vecs)."""
        if not self._embeddings:
            return []

        matrix = np.stack(self._embeddings)  # (N, D)
        scores: np.ndarray = matrix @ query_embedding.astype(np.float32)

        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        # Import lazily to avoid mandatory dependency at module import time.
        from tests.rag.fakes import _FakeRetrievalResult, _FakeStoredDocument

        results = []
        for idx in top_indices:
            doc = _FakeStoredDocument(
                document_id=self._ids[idx],
                content=self._contents[idx],
                metadata=self._metadata[idx],
            )
            results.append(_FakeRetrievalResult(document=doc, score=float(scores[idx])))
        return results


# Lightweight stand-ins for vector_db.StoredDocument / RetrievalResult that
# avoid importing faiss.
class _FakeStoredDocument:
    __slots__ = ("document_id", "content", "metadata")

    def __init__(self, document_id: str, content: str, metadata: dict) -> None:
        self.document_id = document_id
        self.content = content
        self.metadata = metadata


class _FakeRetrievalResult:
    __slots__ = ("document", "score")

    def __init__(self, document: _FakeStoredDocument, score: float) -> None:
        self.document = document
        self.score = score


class FakeVectorStoreManager:
    """In-memory implementation of VectorStoreManager (src/rag/vector_db.py:188-211).

    Implements the public search/add surface used by RetrievalService and
    FaissBackend so tests can run without a real FAISS index on disk.
    Includes the memory namespace added in PR 8.
    """

    CODE_NAMESPACE = "code"
    TEXT_NAMESPACE = "text"
    MEMORY_NAMESPACE = "memory"

    def __init__(self, rag_data_dir: str | None = None) -> None:
        # rag_data_dir accepted for API compatibility but ignored.
        self._namespaces: dict[str, _InMemoryNamespace] = {}

    def _ns(self, name: str) -> _InMemoryNamespace:
        if name not in self._namespaces:
            self._namespaces[name] = _InMemoryNamespace(name)
        return self._namespaces[name]

    # --- add helpers -------------------------------------------------------

    def add_code_documents(
        self,
        contents: Sequence[str],
        embeddings: np.ndarray,
        metadata: Sequence[dict],
    ) -> list[str]:
        return self._ns(self.CODE_NAMESPACE).add_documents(contents, embeddings, metadata)

    def add_text_documents(
        self,
        contents: Sequence[str],
        embeddings: np.ndarray,
        metadata: Sequence[dict],
    ) -> list[str]:
        return self._ns(self.TEXT_NAMESPACE).add_documents(contents, embeddings, metadata)

    def add_memory_documents(
        self,
        contents: Sequence[str],
        embeddings: np.ndarray,
        metadata: Sequence[dict],
    ) -> list[str]:
        """Add episodic memory summaries (uses 384-dim text embeddings)."""
        return self._ns(self.MEMORY_NAMESPACE).add_documents(contents, embeddings, metadata)

    # --- search helpers ----------------------------------------------------

    def search_code(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        return self._ns(self.CODE_NAMESPACE).search(query_embedding, top_k)

    def search_text(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        return self._ns(self.TEXT_NAMESPACE).search(query_embedding, top_k)

    def search_memory(self, query_embedding: np.ndarray, top_k: int = 5) -> list:
        """Search the memory namespace for similar past summaries."""
        return self._ns(self.MEMORY_NAMESPACE).search(query_embedding, top_k)

    # --- introspection helpers (bonus surface) ----------------------------

    def list_documents(self, namespace: str) -> list[_FakeStoredDocument]:
        ns = self._ns(namespace)
        return [
            _FakeStoredDocument(
                document_id=doc_id, content=content, metadata=meta
            )
            for doc_id, content, meta in zip(ns._ids, ns._contents, ns._metadata)
        ]


# ---------------------------------------------------------------------------
# FakeReranker
# ---------------------------------------------------------------------------

class FakeReranker:
    """Returns pre-configured scores from a static dictionary.

    Usage::

        reranker = FakeReranker(scores={"doc_a": 0.9, "doc_b": 0.3})
        results = reranker.rerank(query, candidates)
        # candidates are returned sorted descending by their configured score.

    If a candidate's ``document_id`` is not in the scores dict, it receives a
    default score of 0.5.
    """

    DEFAULT_SCORE = 0.5

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self._scores: dict[str, float] = scores or {}

    def rerank(self, query: str, candidates: list, top_k: int | None = None) -> list:
        """Sort *candidates* by pre-configured score (desc) and return top_k.

        Candidates are expected to be objects with a ``.document.document_id``
        attribute (matching RetrievalResult / _FakeRetrievalResult).  For plain
        dicts with a ``document_id`` key, that value is used directly.
        """
        def _score(candidate: Any) -> float:
            doc_id: str = ""
            if hasattr(candidate, "document") and hasattr(candidate.document, "document_id"):
                doc_id = candidate.document.document_id
            elif isinstance(candidate, dict):
                doc_id = candidate.get("document_id", "")
            return self._scores.get(doc_id, self.DEFAULT_SCORE)

        ranked = sorted(candidates, key=_score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


# ---------------------------------------------------------------------------
# FakeLlmClient
# ---------------------------------------------------------------------------

class FakeLlmClient:
    """Returns canned ``/generate`` responses without hitting a real server.

    Usage::

        client = FakeLlmClient(responses={"mutant0": "def f(): pass"})
        # OR with a default fallback:
        client = FakeLlmClient(default="# stub output")
    """

    DEFAULT_RESPONSE = "# FakeLlmClient default response"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str | None = None,
    ) -> None:
        self._responses: dict[str, str] = responses or {}
        self._default: str = default if default is not None else self.DEFAULT_RESPONSE
        self._calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a canned response for *prompt*.

        The prompt is looked up verbatim first; if not found the default is
        returned.  All calls are recorded in ``self.calls`` for assertion.
        """
        self._calls.append({"prompt": prompt, "kwargs": kwargs})
        return self._responses.get(prompt, self._default)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Read-only list of all recorded generate() calls."""
        return list(self._calls)

    def reset(self) -> None:
        self._calls.clear()
