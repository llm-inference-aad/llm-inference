"""Tests for src/rag/service.py — RagService.

Strategy:
- Use FakeBackend (implements BackendProtocol with canned blocks) so no FAISS
  or embedding models are loaded.
- Use a lightweight FakeReranker that records calls.
- Load service.py directly via importlib.util, injecting stubs for its heavy
  transitive dependencies.

Critical isolation rule: Do NOT inject stubs that override modules already
loaded by other tests (especially src.rag.retrieval which test_faiss_backend.py
needs as the real module). We guard each stub injection with `not in sys.modules`
AND we provide PromptEnhancerConfig inline so prompt_enhancer.py (which imports
retrieval) is never triggered.
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
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))
if str(_WORKTREE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT / "src"))


def _stub_pkg(name: str, path: str) -> types.ModuleType:
    """Create a stub package module and inject into sys.modules only if absent."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_module_direct(rel_path: str, mod_name: str) -> types.ModuleType:
    """Load a module from file, injecting into sys.modules if not already present."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    full_path = _WORKTREE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _force_load_module(rel_path: str, mod_name: str) -> types.ModuleType:
    """Force-load a module from disk, replacing any existing entry in sys.modules.

    Used to ensure this test file gets the real module even when a stub was
    installed by a test file that ran earlier (e.g. test_client.py stubs
    src.rag.service before test_service.py runs).
    """
    full_path = _WORKTREE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Define PromptEnhancerConfig inline to avoid loading prompt_enhancer.py
# (which imports .retrieval and would install our stub over the real module)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _InlinePromptEnhancerConfig:
    """Mirror of PromptEnhancerConfig for use in service tests."""
    top_k: int = 5
    text_candidate_k: int = 10
    text_top_k: int = 3
    text_top_k_api: int = 2
    text_top_k_pdf: int = 1
    min_accuracy: float = 0.9
    max_parameters: Optional[float] = None


# ---------------------------------------------------------------------------
# Set up minimal stubs needed by service.py
# ---------------------------------------------------------------------------

_stub_pkg("src", str(_WORKTREE_ROOT / "src"))
_stub_pkg("src.rag", str(_WORKTREE_ROOT / "src" / "rag"))
# Note: do NOT stub src.rag.backends — test_faiss_backend.py needs to import
# FaissBackend from it.  service.py only accesses backends lazily (in _build_default_backend),
# so no stub is needed here.

# Load api_types (pure dataclasses — no heavy deps)
_api_types_mod = _load_module_direct("src/rag/api_types.py", "src.rag.api_types")

# Bind api_types exports
AugmentRequest = _api_types_mod.AugmentRequest
AugmentResponse = _api_types_mod.AugmentResponse
RetrievedBlock = _api_types_mod.RetrievedBlock
RetrieveRequest = _api_types_mod.RetrieveRequest
RetrieveResponse = _api_types_mod.RetrieveResponse

# Stub utils.rag_metrics — guard: only install if not already loaded
if "utils.rag_metrics" not in sys.modules:
    if "utils" not in sys.modules:
        _utils_stub = types.ModuleType("utils")
        _utils_stub.__path__ = [str(_WORKTREE_ROOT / "utils")]  # type: ignore[attr-defined]
        sys.modules["utils"] = _utils_stub
    _metrics_stub = types.ModuleType("utils.rag_metrics")
    _metrics_stub.record_metric = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["utils.rag_metrics"] = _metrics_stub

# Force-load service.py from disk so this test file always gets the real RagService,
# even if a prior test file (e.g. test_client.py) installed a stub into sys.modules.
# service.py uses lazy imports for cfg.constants and prompt_enhancer, so no
# stubs are needed at module load time.
_service_mod = _force_load_module("src/rag/service.py", "src.rag.service")
RagService = _service_mod.RagService


# ---------------------------------------------------------------------------
# Fake backend implementing BackendProtocol with canned blocks
# ---------------------------------------------------------------------------

class FakeBackend:
    """BackendProtocol-compliant fake that returns canned blocks."""

    def __init__(self, blocks: List[RetrievedBlock] | None = None) -> None:
        self._blocks = blocks or []
        self.retrieve_calls: list[RetrieveRequest] = []

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        self.retrieve_calls.append(request)
        ns = request.namespace
        if ns == "code":
            blocks = [b for b in self._blocks
                      if (b.diagnostics or {}).get("source") == "code"]
        elif ns == "text":
            blocks = [b for b in self._blocks
                      if (b.diagnostics or {}).get("source") != "code"]
        else:
            blocks = list(self._blocks)
        return RetrieveResponse(blocks=blocks[:request.top_k], latency_ms=1.0)

    def index(self, document: Any) -> None:
        pass


class FakeRerankerForService:
    """Minimal reranker fake that records calls and returns blocks unchanged."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def rerank(self, query: str, candidates: list, top_k: int | None = None, **kwargs) -> list:
        self.calls.append({"query": query, "n_candidates": len(candidates), "top_k": top_k})
        result = list(candidates)
        if top_k is not None:
            result = result[:top_k]
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_code_block(gene_id: str = "gene-001", score: float = 0.9) -> RetrievedBlock:
    return RetrievedBlock(
        kind="mutation_code",
        document_id=gene_id,
        title=f"Mutation {gene_id}",
        score=score,
        content=f"def forward(self):\n    # code for {gene_id}\n    pass",
        diagnostics={"source": "code", "reranked": False},
    )


def _make_text_block(doc_id: str = "doc-001", score: float = 0.8) -> RetrievedBlock:
    return RetrievedBlock(
        kind="api_doc",
        document_id=doc_id,
        title="torch.nn.Conv2d",
        score=score,
        content="Applies a 2D convolution.",
        diagnostics={"source": "text", "doc_type": "api_doc", "reranked": False},
    )


def _make_augment_request(**overrides) -> AugmentRequest:
    defaults = dict(
        template="Mutate this network:\n{code}",
        mutation_type="Complex",
        query_code="class Net(nn.Module): pass",
        gene_id="gene-test",
        run_id="run-001",
        request_id="req-001",
    )
    defaults.update(overrides)
    return AugmentRequest(**defaults)


# ---------------------------------------------------------------------------
# Patch cfg.constants attributes for hermetic tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_constants(monkeypatch):
    """Patch cfg.constants so tests control RAG feature flags."""
    import cfg.constants as _const
    monkeypatch.setattr(_const, "RAG_ENABLED", True)
    monkeypatch.setattr(_const, "RAG_USE_CODE_CONTEXT", True)
    monkeypatch.setattr(_const, "RAG_USE_TEXT_CONTEXT", False)
    monkeypatch.setattr(_const, "RAG_RERANKER_ENABLED", False)
    monkeypatch.setattr(_const, "RUN_ID", "test-run-001")
    yield


# ---------------------------------------------------------------------------
# Helper: build a RagService with FakeBackend
# ---------------------------------------------------------------------------

def _make_service(blocks=None, reranker=None, *, code_ctx=True, text_ctx=False):
    import cfg.constants as _const
    _const.RAG_USE_CODE_CONTEXT = code_ctx
    _const.RAG_USE_TEXT_CONTEXT = text_ctx
    backend = FakeBackend(blocks=blocks or [])
    # Pass reranker=None explicitly so RagService doesn't try to build one
    return RagService(backend=backend, reranker=reranker if reranker is not None else None)


# ---------------------------------------------------------------------------
# Tests for RagService.augment
# ---------------------------------------------------------------------------

class TestRagServiceAugment:
    """Tests for RagService.augment() with a FakeBackend."""

    def test_augment_returns_augment_response(self):
        service = _make_service()
        req = _make_augment_request()
        resp = service.augment(req)
        assert isinstance(resp, AugmentResponse)

    def test_augmented_prompt_contains_template_when_no_blocks(self):
        """When no blocks are returned, augmented_prompt == original template."""
        service = _make_service(blocks=[])
        req = _make_augment_request(template="My special template")
        resp = service.augment(req)
        assert "My special template" in resp.augmented_prompt

    def test_augmented_prompt_contains_block_content(self):
        """When blocks are returned, augmented_prompt contains their content."""
        code_block = _make_code_block(gene_id="gene-999")
        service = _make_service(blocks=[code_block], code_ctx=True)
        req = _make_augment_request(query_code="class Net: pass")
        resp = service.augment(req)
        assert "gene-999" in resp.augmented_prompt or "code for gene-999" in resp.augmented_prompt

    def test_blocks_used_reflects_backend_return(self):
        """blocks_used must list the blocks that were injected."""
        blocks = [_make_code_block("gene-A"), _make_code_block("gene-B")]
        service = _make_service(blocks=blocks, code_ctx=True)
        req = _make_augment_request()
        resp = service.augment(req)
        block_ids = {b.document_id for b in resp.blocks_used}
        assert "gene-A" in block_ids or "gene-B" in block_ids

    def test_latency_ms_is_populated_and_non_negative(self):
        service = _make_service()
        req = _make_augment_request()
        resp = service.augment(req)
        assert isinstance(resp.latency_ms, float)
        assert resp.latency_ms >= 0.0

    def test_diagnostics_present(self):
        service = _make_service()
        req = _make_augment_request()
        resp = service.augment(req)
        assert resp.diagnostics is not None
        assert "reranker_used" in resp.diagnostics

    def test_reranker_called_when_configured(self):
        """Reranker is called when injected and blocks are available."""
        import cfg.constants as _const
        _const.RAG_USE_CODE_CONTEXT = True

        fake_reranker = FakeRerankerForService()
        backend = FakeBackend(blocks=[_make_code_block("gene-X")])
        service = RagService(backend=backend, reranker=fake_reranker)

        req = _make_augment_request()
        resp = service.augment(req)
        assert len(fake_reranker.calls) >= 1
        assert resp.diagnostics["reranker_used"] is True

    def test_reranker_not_called_when_none(self):
        """When reranker=None, reranker_used should be False."""
        import cfg.constants as _const
        _const.RAG_USE_CODE_CONTEXT = True

        backend = FakeBackend(blocks=[_make_code_block("gene-X")])
        service = RagService(backend=backend, reranker=None)

        req = _make_augment_request()
        resp = service.augment(req)
        assert resp.diagnostics["reranker_used"] is False

    def test_no_blocks_returns_original_template(self):
        """If backend returns no blocks, augmented_prompt == original template."""
        import cfg.constants as _const
        _const.RAG_USE_CODE_CONTEXT = True

        backend = FakeBackend(blocks=[])
        service = RagService(backend=backend, reranker=None)

        req = _make_augment_request(template="Original template text")
        resp = service.augment(req)
        assert resp.augmented_prompt == "Original template text"
        assert resp.blocks_used == []

    def test_empty_query_code_skips_code_retrieval(self):
        """When query_code is empty, no code retrieval request is made."""
        import cfg.constants as _const
        _const.RAG_USE_CODE_CONTEXT = True

        backend = FakeBackend(blocks=[_make_code_block()])
        service = RagService(backend=backend, reranker=None)

        req = _make_augment_request(query_code="")
        resp = service.augment(req)
        code_calls = [c for c in backend.retrieve_calls if c.namespace == "code"]
        assert len(code_calls) == 0

    def test_augmented_prompt_contains_rag_header(self):
        """With code blocks, augmented prompt should include the RAG section header."""
        import cfg.constants as _const
        _const.RAG_USE_CODE_CONTEXT = True

        block = _make_code_block("gene-header-test")
        backend = FakeBackend(blocks=[block])
        service = RagService(backend=backend, reranker=None)

        req = _make_augment_request(query_code="class Net: pass")
        resp = service.augment(req)
        assert "RAG Code Examples" in resp.augmented_prompt


# ---------------------------------------------------------------------------
# Tests for RagService.retrieve (pass-through)
# ---------------------------------------------------------------------------

class TestRagServiceRetrieve:
    def test_retrieve_delegates_to_backend(self):
        code_block = _make_code_block()
        backend = FakeBackend(blocks=[code_block])
        service = RagService(backend=backend, reranker=None)

        req = RetrieveRequest(query="test query", namespace="code", top_k=3)
        resp = service.retrieve(req)

        assert isinstance(resp, RetrieveResponse)
        assert len(backend.retrieve_calls) == 1
        assert backend.retrieve_calls[0] is req

    def test_retrieve_returns_backend_blocks(self):
        block = _make_code_block(gene_id="test-gene", score=0.77)
        backend = FakeBackend(blocks=[block])
        service = RagService(backend=backend, reranker=None)

        req = RetrieveRequest(query="some query", namespace="code", top_k=5)
        resp = service.retrieve(req)

        assert len(resp.blocks) == 1
        assert resp.blocks[0].document_id == "test-gene"
        assert resp.blocks[0].score == 0.77
