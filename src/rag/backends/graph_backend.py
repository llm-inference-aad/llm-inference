"""Placeholder stub for the Graph retrieval backend."""

from __future__ import annotations

from typing import Any

from src.rag.api_types import RetrieveRequest, RetrieveResponse


class GraphBackend:
    """Protocol-compliant stub for the Graph backend."""

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        raise NotImplementedError("graph backend not yet implemented")

    def index(self, document: Any) -> None:
        raise NotImplementedError("graph backend not yet implemented")
