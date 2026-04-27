"""RAG backend implementations.

Currently ships the FAISS-backed adapter (``FaissBackend``).  Additional
backends (PageIndex, Graph) will drop in behind the same
:class:`~src.rag.backend_protocol.BackendProtocol` without interface churn.
"""

from .faiss_backend import FaissBackend

__all__ = ["FaissBackend"]
