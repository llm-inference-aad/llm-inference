"""Persistent vector database manager built on FAISS."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - defensive guard
    raise RuntimeError(
        "faiss-cpu must be installed to use the RAG vector database. "
        "Install the optional dependency listed in pyproject.toml."
    ) from exc


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return an L2-normalized float32 matrix for cosine similarity."""
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    faiss.normalize_L2(matrix)
    return matrix


@dataclass(frozen=True)
class StoredDocument:
    """Artifact stored inside the vector database."""

    document_id: str
    content: str
    metadata: dict


@dataclass(frozen=True)
class RetrievalResult:
    """Result element produced by a similarity search."""

    document: StoredDocument
    score: float


class NamespaceStore:
    """Manage a single FAISS index plus metadata for a namespace."""

    def __init__(self, name: str, base_dir: Path):
        self.name = name
        self.base_dir = base_dir
        self.index_path = base_dir / "faiss_index" / f"{name}.index"
        self.metadata_path = base_dir / "metadata" / f"{name}.jsonl"
        self.lock = threading.Lock()
        self.index: faiss.Index | None = None
        self.dimension: int | None = None
        self.documents: list[StoredDocument] = []
        self._load()

    # --------------------------------------------------------------------- #
    # Persistence helpers
    # --------------------------------------------------------------------- #
    def _load(self) -> None:
        _ensure_directory(self.index_path.parent)
        _ensure_directory(self.metadata_path.parent)

        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    self.documents.append(
                        StoredDocument(
                            document_id=payload["document_id"],
                            content=payload["content"],
                            metadata=payload["metadata"],
                        )
                    )

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = self.index.d

    def _persist_metadata(self) -> None:
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            for doc in self.documents:
                handle.write(
                    json.dumps(
                        {
                            "document_id": doc.document_id,
                            "content": doc.content,
                            "metadata": doc.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _persist_index(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def add_documents(
        self, contents: Sequence[str], embeddings: np.ndarray, metadata: Sequence[dict]
    ) -> List[str]:
        if len(contents) != len(metadata):
            raise ValueError("contents and metadata lengths must match.")
        if embeddings.shape[0] != len(contents):
            raise ValueError("Embeddings rows must equal number of contents.")

        normalized_embeddings = _normalize_matrix(np.array(embeddings, dtype=np.float32))
        document_ids: list[str] = []

        with self.lock:
            if self.index is None:
                self.dimension = normalized_embeddings.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)

            if normalized_embeddings.shape[1] != self.index.d:
                raise ValueError(
                    f"Embedding dimension mismatch for namespace '{self.name}'. "
                    f"Expected {self.index.d}, received {normalized_embeddings.shape[1]}"
                )

            existing_ids = {doc.document_id for doc in self.documents}
            for meta in metadata:
                candidate_id = meta.get("document_id")
                if candidate_id and candidate_id in existing_ids:
                    raise ValueError(
                        f"Document id '{candidate_id}' already exists in namespace '{self.name}'."
                    )

            self.index.add(normalized_embeddings)

            for content, meta in zip(contents, metadata):
                document_id = meta.get("document_id") or str(uuid.uuid4())
                self.documents.append(
                    StoredDocument(
                        document_id=document_id,
                        content=content,
                        metadata={**meta, "document_id": document_id},
                    )
                )
                document_ids.append(document_id)

            self._persist_metadata()
            self._persist_index()

        return document_ids

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        if self.index is None or not self.documents:
            return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        normalized_query = _normalize_matrix(np.array(query_embedding, dtype=np.float32))
        distances, indices = self.index.search(normalized_query, top_k)

        results: list[RetrievalResult] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            try:
                document = self.documents[int(idx)]
            except (IndexError, ValueError):
                continue
            results.append(RetrievalResult(document=document, score=float(score)))
        return results


class VectorStoreManager:
    """High-level manager for multiple namespace stores."""

    CODE_NAMESPACE = "code"
    TEXT_NAMESPACE = "text"
    MEMORY_NAMESPACE = "memory"  # Episodic memory: past interactions (uses text embeddings, 384-dim)

    def __init__(self, rag_data_dir: str | Path):
        self.base_dir = Path(rag_data_dir)
        _ensure_directory(self.base_dir)
        self._namespaces: dict[str, NamespaceStore] = {}

    def _namespace(self, name: str) -> NamespaceStore:
        if name not in self._namespaces:
            self._namespaces[name] = NamespaceStore(name=name, base_dir=self.base_dir)
        return self._namespaces[name]

    def add_code_documents(
        self, contents: Sequence[str], embeddings: np.ndarray, metadata: Sequence[dict]
    ) -> List[str]:
        return self._namespace(self.CODE_NAMESPACE).add_documents(contents, embeddings, metadata)

    def add_text_documents(
        self, contents: Sequence[str], embeddings: np.ndarray, metadata: Sequence[dict]
    ) -> List[str]:
        return self._namespace(self.TEXT_NAMESPACE).add_documents(contents, embeddings, metadata)

    def search_code(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        return self._namespace(self.CODE_NAMESPACE).search(query_embedding, top_k)

    def search_text(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        return self._namespace(self.TEXT_NAMESPACE).search(query_embedding, top_k)

    def add_memory_documents(
        self, contents: Sequence[str], embeddings: np.ndarray, metadata: Sequence[dict]
    ) -> List[str]:
        """Add memory entries (uses same 384-dim text embeddings)."""
        return self._namespace(self.MEMORY_NAMESPACE).add_documents(contents, embeddings, metadata)

    def search_memory(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """Search memory namespace for similar past interactions."""
        return self._namespace(self.MEMORY_NAMESPACE).search(query_embedding, top_k)

    def list_documents(self, namespace: str) -> List[StoredDocument]:
        return list(self._namespace(namespace).documents)

