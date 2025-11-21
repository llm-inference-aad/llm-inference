"""Retrieval helpers for RAG-enhanced prompting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .data_ingestion import MutationRecord
from .embeddings import EmbeddingService
from .vector_db import RetrievalResult, StoredDocument, VectorStoreManager


@dataclass(frozen=True)
class RetrievedMutation:
    gene_id: str
    score: float
    description: str
    code: str
    metadata: dict


class RetrievalService:
    """High-level retrieval facade combining the vector DB and embedding service."""

    def __init__(self, store: VectorStoreManager, embeddings: EmbeddingService):
        self.store = store
        self.embeddings = embeddings

    # ------------------------------------------------------------------ #
    # Indexing helpers
    # ------------------------------------------------------------------ #
    def index_mutations(self, records: Sequence[MutationRecord]) -> List[str]:
        contents, metadata = zip(*(record.to_document() for record in records)) if records else ([], [])
        if not contents:
            return []
        embeddings = self.embeddings.embed_code(list(contents))
        return self.store.add_code_documents(list(contents), embeddings, list(metadata))

    def index_text_documents(self, documents: Sequence[dict]) -> List[str]:
        if not documents:
            return []
        contents = [doc["content"] for doc in documents]
        metadata = [doc["metadata"] for doc in documents]
        embeddings = self.embeddings.embed_text(contents)
        return self.store.add_text_documents(contents, embeddings, metadata)

    # ------------------------------------------------------------------ #
    # Retrieval helpers
    # ------------------------------------------------------------------ #
    def retrieve_similar_mutations(self, query_code: str, top_k: int = 5) -> List[RetrievedMutation]:
        if not query_code.strip():
            return []

        query_embedding = self.embeddings.embed_code(query_code)
        results = self.store.search_code(query_embedding, top_k=top_k)
        return [self._to_mutation(result) for result in results]

    def retrieve_high_performers(
        self,
        min_accuracy: float = 0.9,
        max_parameters: float | None = None,
        limit: int = 5,
    ) -> List[RetrievedMutation]:
        documents = self.store.list_documents(VectorStoreManager.CODE_NAMESPACE)
        filtered: list[StoredDocument] = []
        for document in documents:
            fitness = document.metadata.get("fitness") or []
            accuracy = float(fitness[0]) if fitness else 0.0
            parameters = float(fitness[1]) if len(fitness) > 1 else None
            if accuracy < min_accuracy:
                continue
            if max_parameters is not None and parameters is not None and parameters > max_parameters:
                continue
            filtered.append(document)

        filtered.sort(key=lambda doc: doc.metadata.get("fitness", [0])[0], reverse=True)
        return [
            RetrievedMutation(
                gene_id=doc.metadata["gene_id"],
                score=float(doc.metadata.get("fitness", [0])[0]),
                description=doc.metadata.get("description", doc.metadata.get("gene_id")),
                code=doc.content,
                metadata=doc.metadata,
            )
            for doc in filtered[:limit]
        ]

    def retrieve_by_mutation_type(self, mutation_type: str, limit: int = 5) -> List[RetrievedMutation]:
        documents = self.store.list_documents(VectorStoreManager.CODE_NAMESPACE)
        matches = [
            doc for doc in documents if doc.metadata.get("mutation_type", "").lower() == mutation_type.lower()
        ]
        matches.sort(key=lambda doc: doc.metadata.get("fitness", [0])[0], reverse=True)
        return [
            RetrievedMutation(
                gene_id=doc.metadata["gene_id"],
                score=float(doc.metadata.get("fitness", [0])[0]),
                description=doc.metadata.get("description", doc.metadata.get("gene_id")),
                code=doc.content,
                metadata=doc.metadata,
            )
            for doc in matches[:limit]
        ]

    # ------------------------------------------------------------------ #
    # Formatting helpers
    # ------------------------------------------------------------------ #
    def format_context(self, mutations: Sequence[RetrievedMutation]) -> str:
        lines: list[str] = []
        for mutation in mutations:
            fitness = mutation.metadata.get("fitness") or []
            accuracy = f"{fitness[0]:.4f}" if fitness else "unknown"
            params = f"{int(fitness[1])}" if len(fitness) > 1 else "unknown"
            lines.append(
                f"- Gene {mutation.gene_id} (score {mutation.score:.3f}) "
                f"Accuracy {accuracy}, Params {params}\n"
                f"{mutation.description}"
            )
        return "\n".join(lines)

    def _to_mutation(self, result: RetrievalResult) -> RetrievedMutation:
        metadata = {**result.document.metadata}
        metadata.setdefault("description", result.document.content.splitlines()[0])
        return RetrievedMutation(
            gene_id=metadata.get("gene_id", result.document.document_id),
            score=result.score,
            description=metadata.get("description", ""),
            code=result.document.content,
            metadata=metadata,
        )



