"""RAG backend implementations."""

from __future__ import annotations

from .base import RetrievalBackend
from .faiss_backend import FaissRetrievalBackend

__all__ = ["RetrievalBackend", "FaissRetrievalBackend"]
