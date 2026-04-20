"""Parametrised protocol-compliance test across all backends.

Each backend class must satisfy BackendProtocol structurally and have
callable retrieve/index. Stubs must raise NotImplementedError (not
AttributeError). FaissBackend is only checked at the class level because
constructing one requires FAISS + an EmbeddingService.

PR 8 addition: MemoryBackend is added to the live-instances list because it
is cheap to construct (FakeVectorStoreManager + FakeEmbeddingService).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rag.api_types import RetrieveRequest  # noqa: E402
from src.rag.backend_protocol import BackendProtocol  # noqa: E402
from src.rag.backends.graph_backend import GraphBackend  # noqa: E402
from src.rag.backends.pageindex_backend import PageIndexBackend  # noqa: E402
from src.rag.backends.memory_backend import MemoryBackend  # noqa: E402
from tests.rag.fakes import FakeEmbeddingService, FakeVectorStoreManager  # noqa: E402


def _sample_request() -> RetrieveRequest:
    return RetrieveRequest(
        query="q",
        namespace="code",
        top_k=3,
        filters={},
        run_id="test-run",
        request_id="req-0",
    )


def _make_memory_backend() -> MemoryBackend:
    """Construct a MemoryBackend with cheap in-memory fakes."""
    return MemoryBackend(
        vector_store=FakeVectorStoreManager(),
        embedding_service=FakeEmbeddingService(),
        min_similarity=0.0,
    )


STUB_BACKENDS = [
    pytest.param(PageIndexBackend, id="PageIndexBackend"),
    pytest.param(GraphBackend, id="GraphBackend"),
]


@pytest.mark.parametrize("backend_cls", STUB_BACKENDS)
class TestStubBackendCompliance:
    def test_isinstance_backend_protocol(self, backend_cls):
        assert isinstance(backend_cls(), BackendProtocol)

    def test_retrieve_callable(self, backend_cls):
        assert callable(getattr(backend_cls(), "retrieve", None))

    def test_index_callable(self, backend_cls):
        assert callable(getattr(backend_cls(), "index", None))

    def test_retrieve_raises_not_implemented(self, backend_cls):
        with pytest.raises(NotImplementedError):
            backend_cls().retrieve(_sample_request())

    def test_index_raises_not_implemented(self, backend_cls):
        with pytest.raises(NotImplementedError):
            backend_cls().index({"text": "doc"})


class TestMemoryBackendProtocolCompliance:
    """Protocol-compliance tests for the live MemoryBackend (PR 8).

    MemoryBackend is fully instantiable with fakes so we can assert it satisfies
    BackendProtocol both at the class level and at runtime.
    """

    def test_isinstance_backend_protocol(self):
        backend = _make_memory_backend()
        assert isinstance(backend, BackendProtocol), (
            "MemoryBackend must satisfy isinstance(..., BackendProtocol)"
        )

    def test_retrieve_callable(self):
        backend = _make_memory_backend()
        assert callable(getattr(backend, "retrieve", None))

    def test_index_callable(self):
        backend = _make_memory_backend()
        assert callable(getattr(backend, "index", None))

    def test_retrieve_returns_retrieve_response(self):
        """retrieve() must return a RetrieveResponse (not raise NotImplementedError)."""
        from src.rag.api_types import RetrieveResponse
        backend = _make_memory_backend()
        resp = backend.retrieve(_sample_request())
        assert isinstance(resp, RetrieveResponse)

    def test_index_does_not_raise(self):
        """index() must not raise NotImplementedError for a valid document."""
        backend = _make_memory_backend()
        backend.index({"text": "some mutation summary | Test Acc: 0.90, Params: 100000"})


class TestFaissBackendClassShape:
    """Class-level protocol check for FaissBackend (no instantiation)."""

    def test_has_retrieve(self):
        pytest.importorskip("faiss", reason="faiss backend requires faiss")
        try:
            from src.rag.backends.faiss_backend import FaissBackend
        except ModuleNotFoundError:
            pytest.skip("FaissBackend not available on this base branch")
        assert callable(getattr(FaissBackend, "retrieve", None))

    def test_has_index(self):
        pytest.importorskip("faiss", reason="faiss backend requires faiss")
        try:
            from src.rag.backends.faiss_backend import FaissBackend
        except ModuleNotFoundError:
            pytest.skip("FaissBackend not available on this base branch")
        assert callable(getattr(FaissBackend, "index", None))
