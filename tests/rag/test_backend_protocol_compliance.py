"""Parametrised protocol-compliance test across all backends.

Each backend class must satisfy BackendProtocol structurally and have
callable retrieve/index. Stubs must raise NotImplementedError (not
AttributeError). FaissBackend is only checked at the class level because
constructing one requires FAISS + an EmbeddingService.
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


def _sample_request() -> RetrieveRequest:
    return RetrieveRequest(
        query="q",
        namespace="code",
        top_k=3,
        filters={},
        run_id="test-run",
        request_id="req-0",
    )


STUB_BACKENDS = [
    pytest.param(PageIndexBackend, id="PageIndexBackend"),
]

# Scaffolding backends are partially-built stubs that DON'T raise — they
# return a structured empty :class:`RetrieveResponse` carrying a diagnostic
# ``reason`` so callers (replays, sanity scripts) get a graceful no-op
# instead of a traceback mid-batch. Workers flip the implementation status
# once they fill in the retrieval body. ``index`` remains a no-op shim.
SCAFFOLDING_BACKENDS = [
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


@pytest.mark.parametrize("backend_cls", SCAFFOLDING_BACKENDS)
class TestScaffoldingBackendCompliance:
    """Compliance for partially-implemented backends.

    They satisfy BackendProtocol structurally but return a diagnostic
    empty response on ``retrieve`` rather than raising. ``index`` is a
    no-op (returns None) — both behaviours match the FaissBackend /
    PageIndexBackend production pattern.
    """

    def test_isinstance_backend_protocol(self, backend_cls):
        assert isinstance(backend_cls(), BackendProtocol)

    def test_retrieve_returns_diagnostic_empty_response(self, backend_cls):
        from src.rag.api_types import RetrieveResponse
        resp = backend_cls().retrieve(_sample_request())
        assert isinstance(resp, RetrieveResponse)
        assert resp.blocks == [], "scaffolding backend must not return blocks"
        diag = resp.diagnostics or {}
        assert diag.get("reason") in {"scaffolding", "no_graph"}, (
            f"expected a diagnostic reason; got {diag}"
        )

    def test_index_is_a_noop(self, backend_cls):
        # Should not raise; should return None (mirrors FaissBackend).
        assert backend_cls().index({"text": "doc"}) is None


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


class TestMemoryBackendCompliance:
    """Live MemoryBackend instance — uses fakes so no torch/faiss needed."""

    def _instance(self):
        from src.rag.backends.memory_backend import MemoryBackend
        from tests.rag.fakes import FakeEmbeddingService, FakeVectorStoreManager

        return MemoryBackend(
            vector_store=FakeVectorStoreManager(),
            embedding_service=FakeEmbeddingService(),
            min_similarity=0.0,
        )

    def test_isinstance_backend_protocol(self):
        assert isinstance(self._instance(), BackendProtocol)

    def test_retrieve_callable(self):
        assert callable(getattr(self._instance(), "retrieve", None))

    def test_index_callable(self):
        assert callable(getattr(self._instance(), "index", None))

    def test_retrieve_returns_response(self):
        from src.rag.api_types import RetrieveResponse
        resp = self._instance().retrieve(_sample_request())
        assert isinstance(resp, RetrieveResponse)

    def test_index_accepts_string_and_dict(self):
        backend = self._instance()
        backend.index("a plain string summary")
        backend.index({"text": "dict summary", "gene_id": "g1"})
