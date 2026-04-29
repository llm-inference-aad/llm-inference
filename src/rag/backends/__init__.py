"""RAG backend implementations.

Currently ships the FAISS-backed adapter (``FaissBackend``) and the episodic
``MemoryBackend``.  Additional backends (PageIndex, Graph) drop in behind the
same :class:`~src.rag.backend_protocol.BackendProtocol` without interface
churn.  PageIndex/Graph stubs live alongside but are not re-exported here
until they implement non-stub behaviour (callers reach them via deep import
for now).
"""

from .faiss_backend import FaissBackend
from .memory_backend import MemoryBackend

__all__ = ["FaissBackend", "MemoryBackend"]
