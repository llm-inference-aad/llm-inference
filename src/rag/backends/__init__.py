"""RAG backend implementations.

Currently ships the FAISS-backed adapter (``FaissBackend``).  Additional
backends (PageIndex, Graph) will drop in behind the same
:class:`~src.rag.backend_protocol.BackendProtocol` without interface churn.
PageIndex/Graph stubs live alongside but are not re-exported here until they
implement non-stub behaviour (callers reach them via deep import for now).
"""

from .faiss_backend import FaissBackend

__all__ = ["FaissBackend"]
