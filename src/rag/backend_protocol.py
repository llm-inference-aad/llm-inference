"""Protocol definition for RAG retrieval backends.

Any class that implements :meth:`retrieve` and :meth:`index` with the correct
signatures satisfies :class:`BackendProtocol` through structural subtyping
(PEP 544).  No registration or inheritance is required.

This module has zero heavy dependencies — it must be importable in any
environment without torch, faiss, or sentence-transformers.

Cross-reference: docs/plans/05_rag_componentization_plan.md Task 1 and Task 2.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .api_types import RetrieveRequest, RetrieveResponse


@runtime_checkable
class BackendProtocol(Protocol):
    """Structural protocol for RAG retrieval backends.

    All backends — ``FaissBackend``, ``PageIndexBackend``, ``GraphBackend``,
    etc. — must satisfy this interface so that :class:`~src.rag.service.RagService`
    can treat them interchangeably.

    The protocol is decorated with :func:`typing.runtime_checkable` so that
    ``isinstance(obj, BackendProtocol)`` works at runtime (it checks that the
    required methods are present, not full signature compatibility).
    """

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Retrieve relevant blocks for *request*.

        Args:
            request: A :class:`~src.rag.api_types.RetrieveRequest` describing
                the query, namespace, top-k, and optional filters.

        Returns:
            A :class:`~src.rag.api_types.RetrieveResponse` containing ordered
            blocks and per-call diagnostics.
        """
        ...  # pragma: no cover

    def index(self, document: Any) -> None:
        """Add *document* to the backend's index.

        Args:
            document: An arbitrary document object.  The exact schema is
                backend-specific; callers should use the backend's own
                ingestion helpers rather than calling this directly in
                production code.  This method exists so that the protocol
                surface is complete and test fakes can implement it.
        """
        ...  # pragma: no cover
