"""Placeholder stub for the PageIndex retrieval backend.

Real logic will be ported from feature/rag-pi once that branch stabilises.
This stub satisfies BackendProtocol so protocol-compliance tests can run
across all three backends without a live PageIndex instance.
"""

from __future__ import annotations

from typing import Any

from src.rag.api_types import RetrieveRequest, RetrieveResponse


class PageIndexBackend:
    """Protocol-compliant stub for the PageIndex backend."""

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        raise NotImplementedError("port from feature/rag-pi pending")

    def index(self, document: Any) -> None:
        raise NotImplementedError("port from feature/rag-pi pending")
