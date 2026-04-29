"""Tests for src/rag/client.py — RagClient.

Tests:
- augment() forwards the request to service.augment() unchanged.
- retrieve() forwards the request to service.retrieve() unchanged.
- RagClient does NOT import src.rag.retrieval or src.rag.vector_db
  (verified via source text inspection, the safest approach).
- A request_id is auto-generated when the caller omits one.

Import strategy: we load client.py directly via importlib.util,
bypassing src/rag/__init__.py which eagerly imports heavy ML deps
(pdfplumber, sentence-transformers) not available in unit-test environments.

RagClient's only direct imports are:
  - src.rag.api_types (pure dataclasses, no heavy deps)
  - src.rag.service   (which we stub out so we never exec service.py)
  - uuid, dataclasses (stdlib)

We inject stub modules for service so the import succeeds without loading
the actual ML pipeline.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pathlib
import sys
import types
from typing import Any, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: worktree root path
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = pathlib.Path(__file__).parent.parent.parent


def _stub_pkg(name: str, path: str) -> types.ModuleType:
    """Create a stub package module and inject into sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_module_direct(rel_path: str, mod_name: str) -> types.ModuleType:
    """Load a module from file path, injecting it into sys.modules."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    full_path = _WORKTREE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Set up stub pkg hierarchy to avoid triggering heavy __init__.py
# ---------------------------------------------------------------------------

_stub_pkg("src", str(_WORKTREE_ROOT / "src"))
_stub_pkg("src.rag", str(_WORKTREE_ROOT / "src" / "rag"))

# Load api_types (pure dataclasses — no heavy deps)
_api_types_mod = _load_module_direct("src/rag/api_types.py", "src.rag.api_types")

# Bind api_types exports
AugmentRequest = _api_types_mod.AugmentRequest
AugmentResponse = _api_types_mod.AugmentResponse
RetrievedBlock = _api_types_mod.RetrievedBlock
RetrieveRequest = _api_types_mod.RetrieveRequest
RetrieveResponse = _api_types_mod.RetrieveResponse


# ---------------------------------------------------------------------------
# Inject a minimal stub for src.rag.service so client.py can import
# ``from .service import RagService`` without triggering the real ML pipeline.
# The tests below use a StubService anyway, not a real RagService.
# ---------------------------------------------------------------------------

class _StubRagService:
    """Minimal stub for RagService used by client.py's type annotations."""
    pass


if "src.rag.service" not in sys.modules:
    _service_stub_mod = types.ModuleType("src.rag.service")
    _service_stub_mod.RagService = _StubRagService  # type: ignore[attr-defined]
    sys.modules["src.rag.service"] = _service_stub_mod

# Now load client.py in isolation
_client_mod = _load_module_direct("src/rag/client.py", "src.rag.client")
RagClient = _client_mod.RagClient


# ---------------------------------------------------------------------------
# Stub service that records calls (used as the real service in tests)
# ---------------------------------------------------------------------------

class StubService:
    """Records every call to augment() and retrieve() for assertion."""

    def __init__(
        self,
        augment_response: AugmentResponse | None = None,
        retrieve_response: RetrieveResponse | None = None,
    ) -> None:
        self.augment_calls: list[AugmentRequest] = []
        self.retrieve_calls: list[RetrieveRequest] = []
        self._augment_resp = augment_response or AugmentResponse(
            augmented_prompt="stub augmented",
            blocks_used=[],
            diagnostics={"stub": True},
            latency_ms=0.1,
        )
        self._retrieve_resp = retrieve_response or RetrieveResponse(
            blocks=[],
            diagnostics={"stub": True},
            latency_ms=0.1,
        )

    def augment(self, request: AugmentRequest) -> AugmentResponse:
        self.augment_calls.append(request)
        return self._augment_resp

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        self.retrieve_calls.append(request)
        return self._retrieve_resp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_augment_request(**overrides) -> AugmentRequest:
    defaults = dict(
        template="Mutate this: {code}",
        mutation_type="Param",
        query_code="class Net: pass",
        gene_id="gene-007",
        run_id="run-001",
        request_id="req-abc",
    )
    defaults.update(overrides)
    return AugmentRequest(**defaults)


def _make_retrieve_request(**overrides) -> RetrieveRequest:
    defaults = dict(
        query="query text",
        namespace="code",
        top_k=3,
        request_id="req-def",
    )
    defaults.update(overrides)
    return RetrieveRequest(**defaults)


# ---------------------------------------------------------------------------
# Tests: RagClient.augment
# ---------------------------------------------------------------------------

class TestRagClientAugment:
    def test_augment_forwards_to_service(self):
        stub = StubService()
        client = RagClient(service=stub)

        req = _make_augment_request()
        resp = client.augment(req)

        assert len(stub.augment_calls) == 1
        forwarded = stub.augment_calls[0]
        assert forwarded.template == req.template
        assert forwarded.mutation_type == req.mutation_type
        assert forwarded.query_code == req.query_code
        assert forwarded.gene_id == req.gene_id

    def test_augment_returns_service_response(self):
        stub = StubService()
        client = RagClient(service=stub)

        req = _make_augment_request()
        resp = client.augment(req)

        assert resp.augmented_prompt == "stub augmented"
        assert resp.diagnostics == {"stub": True}

    def test_augment_generates_request_id_when_missing(self):
        stub = StubService()
        client = RagClient(service=stub)

        # Build a request WITHOUT a request_id
        req = _make_augment_request(request_id=None)
        assert req.request_id is None

        client.augment(req)

        forwarded = stub.augment_calls[0]
        # Client should have injected a UUID4 request_id
        assert forwarded.request_id is not None
        assert len(forwarded.request_id) > 0

    def test_augment_preserves_existing_request_id(self):
        stub = StubService()
        client = RagClient(service=stub)

        req = _make_augment_request(request_id="my-existing-id")
        client.augment(req)

        forwarded = stub.augment_calls[0]
        assert forwarded.request_id == "my-existing-id"

    def test_augment_injects_unique_request_ids_per_call(self):
        """Each call without a request_id gets a distinct UUID."""
        stub = StubService()
        client = RagClient(service=stub)

        req1 = _make_augment_request(request_id=None)
        req2 = _make_augment_request(request_id=None)
        client.augment(req1)
        client.augment(req2)

        id1 = stub.augment_calls[0].request_id
        id2 = stub.augment_calls[1].request_id
        assert id1 != id2


# ---------------------------------------------------------------------------
# Tests: RagClient.retrieve
# ---------------------------------------------------------------------------

class TestRagClientRetrieve:
    def test_retrieve_forwards_to_service(self):
        stub = StubService()
        client = RagClient(service=stub)

        req = _make_retrieve_request()
        resp = client.retrieve(req)

        assert len(stub.retrieve_calls) == 1
        forwarded = stub.retrieve_calls[0]
        assert forwarded.query == req.query
        assert forwarded.namespace == req.namespace
        assert forwarded.top_k == req.top_k

    def test_retrieve_returns_service_response(self):
        stub = StubService()
        client = RagClient(service=stub)

        req = _make_retrieve_request()
        resp = client.retrieve(req)

        assert isinstance(resp, RetrieveResponse)
        assert resp.blocks == []

    def test_retrieve_generates_request_id_when_missing(self):
        stub = StubService()
        client = RagClient(service=stub)

        req = _make_retrieve_request(request_id=None)
        assert req.request_id is None

        client.retrieve(req)

        forwarded = stub.retrieve_calls[0]
        assert forwarded.request_id is not None


# ---------------------------------------------------------------------------
# Isolation test: RagClient must NOT import retrieval or vector_db internals
# ---------------------------------------------------------------------------

class TestRagClientIsolation:
    """Verify that src/rag/client.py has no direct or transitive imports of
    src.rag.retrieval or src.rag.vector_db.

    We check:
    1. The source text of client.py for forbidden import patterns.
    2. The source text of run_improved.py for the same patterns.
    3. The source texts of src/llm_*.py for the same patterns.
    """

    def test_client_file_has_no_direct_retrieval_import(self):
        """client.py must not reference retrieval or vector_db directly."""
        client_src = (_WORKTREE_ROOT / "src" / "rag" / "client.py").read_text()
        assert "from .retrieval" not in client_src, (
            "client.py must not import from .retrieval directly"
        )
        assert "from .vector_db" not in client_src, (
            "client.py must not import from .vector_db directly"
        )
        assert "import retrieval" not in client_src
        assert "import vector_db" not in client_src

    def test_run_improved_has_no_retrieval_imports(self):
        """run_improved.py must not import RAG retrieval internals directly."""
        ri_src = (_WORKTREE_ROOT / "run_improved.py").read_text()
        assert "from src.rag.retrieval" not in ri_src, (
            "run_improved.py must not import src.rag.retrieval directly"
        )
        assert "from src.rag.vector_db" not in ri_src, (
            "run_improved.py must not import src.rag.vector_db directly"
        )

    def test_llm_files_have_no_retrieval_imports(self):
        """src/llm_*.py files must not import retrieval or vector_db internals."""
        llm_files = list((_WORKTREE_ROOT / "src").glob("llm_*.py"))
        for fpath in llm_files:
            src = fpath.read_text()
            assert "from src.rag.retrieval" not in src, (
                f"{fpath.name} imports src.rag.retrieval — violates isolation"
            )
            assert "from src.rag.vector_db" not in src, (
                f"{fpath.name} imports src.rag.vector_db — violates isolation"
            )
