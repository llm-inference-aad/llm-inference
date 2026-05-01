"""RAG backend implementations.

Exports the production-ready adapters used by ``RagService`` default factory.
``PageIndexBackend`` remains a stub and is intentionally not re-exported.
"""

from .faiss_backend import FaissBackend
from .graph_backend import GraphBackend
from .memory_backend import MemoryBackend

__all__ = ["FaissBackend", "GraphBackend", "MemoryBackend"]
