"""RagClient — the public façade for callers outside the RAG subsystem.

``RagClient`` provides the same two-method surface as ``RagService`` but adds a
small dispatch layer that will eventually support HTTP transport (plan Step 9).
In this PR the client dispatches *locally* to an injected ``RagService``
instance.

Design goals:
- ``RagClient`` must NOT import ``src.rag.retrieval`` or ``src.rag.vector_db``
  at any level. All retrieval detail is hidden behind ``RagService``.
- The client catches exceptions, attaches a ``request_id`` if missing, and can
  log / re-raise in a transport-agnostic way. Keeps the call site in
  ``run_improved.py`` clean.
- HTTP transport is deferred — the constructor will later accept a ``base_url``
  parameter, and ``augment`` / ``retrieve`` will POST to it instead of calling
  ``self._service`` directly.
"""

from __future__ import annotations

import uuid
from typing import Optional

from .api_types import AugmentRequest, AugmentResponse, RetrieveRequest, RetrieveResponse
from .service import RagService


class RagClient:
    """Local dispatch client for the RAG pipeline.

    Callers (``run_improved.py``, tests) import and use this class.  They never
    import ``RagService`` directly — ``RagClient`` is the public seam.

    Args:
        service: A ``RagService`` instance to dispatch to.  If ``None``, a
            ``RagService()`` with default configuration is created.
    """

    def __init__(self, service: Optional[RagService] = None) -> None:
        self._service: RagService = service or RagService()

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def augment(self, request: AugmentRequest) -> AugmentResponse:
        """Augment a prompt template with retrieved RAG context.

        Args:
            request: An :class:`~src.rag.api_types.AugmentRequest`.

        Returns:
            An :class:`~src.rag.api_types.AugmentResponse`.
        """
        request = self._ensure_request_id(request)
        return self._service.augment(request)

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Retrieve blocks directly from the backend (debugging / testing).

        Args:
            request: A :class:`~src.rag.api_types.RetrieveRequest`.

        Returns:
            A :class:`~src.rag.api_types.RetrieveResponse`.
        """
        request = self._ensure_request_id(request)
        return self._service.retrieve(request)

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _ensure_request_id(request: AugmentRequest | RetrieveRequest) -> AugmentRequest | RetrieveRequest:
        """Return *request* unchanged if it already has a ``request_id``, else
        attach a freshly generated UUID4.  This ensures every call can be
        correlated in logs without requiring callers to manage IDs.
        """
        if request.request_id is not None:
            return request
        # Frozen dataclasses don't allow in-place mutation — use dataclasses.replace.
        import dataclasses
        return dataclasses.replace(request, request_id=str(uuid.uuid4()))
