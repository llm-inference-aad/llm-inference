"""Embedding utilities for code and natural language contexts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding generation."""

    code_model_name: str = "microsoft/codebert-base"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None


class EmbeddingService:
    """Generate embeddings for both code snippets and textual documents."""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        # Lazy-load models to avoid heavy startup cost for scripts that only need one.
        self._code_model: SentenceTransformer | None = None
        self._text_model: SentenceTransformer | None = None

    def _get_code_model(self) -> SentenceTransformer:
        if self._code_model is None:
            self._code_model = SentenceTransformer(self.config.code_model_name, device=self.config.device)
        return self._code_model

    def _get_text_model(self) -> SentenceTransformer:
        if self._text_model is None:
            self._text_model = SentenceTransformer(self.config.text_model_name, device=self.config.device)
        return self._text_model

    def _encode(self, items: Sequence[str], model: SentenceTransformer) -> np.ndarray:
        embeddings = model.encode(
            list(items),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def embed_code(self, snippets: str | Sequence[str]) -> np.ndarray:
        """Return embeddings for one or more code snippets."""
        items = [snippets] if isinstance(snippets, str) else list(snippets)
        return self._encode(items, self._get_code_model())

    def embed_text(self, documents: str | Sequence[str]) -> np.ndarray:
        """Return embeddings for one or more text documents."""
        items = [documents] if isinstance(documents, str) else list(documents)
        return self._encode(items, self._get_text_model())

    def embed_mixed_batch(
        self, code_snippets: Iterable[str], text_chunks: Iterable[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode code and text batches simultaneously."""
        return self.embed_code(list(code_snippets)), self.embed_text(list(text_chunks))


