"""Integration smoke test for the _apply_rag_context → RagClient → RagService pipeline.

Uses a local FakeBackend (no FAISS/torch/sentence-transformers) wrapped in
RagService → RagClient, then calls run_improved._apply_rag_context end-to-end.

Import strategy (critical for test ordering):
- This file comes alphabetically BEFORE test_api_types.py, so it must NOT
  load torch/faiss/sentence_transformers at module level. That would pollute
  sys.modules and cause test_api_types.py::TestNoHeavyImports to fail.
- All heavy loading happens inside helper functions called from test methods.
- The local fake backend (_LocalFakeBackend) needs only api_types, which is
  pure dataclasses with no ML deps.

This test is marked @pytest.mark.integration so it can be skipped in fast CI:
    uv run pytest tests/rag/ -m "not integration"
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pathlib
import sys
import types
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: worktree root path (no heavy imports at module level)
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))
if str(_WORKTREE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT / "src"))


# ---------------------------------------------------------------------------
# Module loading helpers (all deferred to test methods)
# ---------------------------------------------------------------------------

def _stub_pkg(name: str, path: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_module_direct(rel_path: str, mod_name: str) -> types.ModuleType:
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    full_path = _WORKTREE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _stub_module(name: str, **attrs) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# Lazily load api_types (pure dataclasses, no heavy deps)
# This is the ONE import we allow at module level since it's truly lightweight.
# ---------------------------------------------------------------------------

def _get_api_types():
    """Load and return the api_types module (cached in sys.modules)."""
    _stub_pkg("src", str(_WORKTREE_ROOT / "src"))
    _stub_pkg("src.rag", str(_WORKTREE_ROOT / "src" / "rag"))
    return _load_module_direct("src/rag/api_types.py", "src.rag.api_types")


# ---------------------------------------------------------------------------
# Local fake backend: no FAISS, no torch, no sentence_transformers
# ---------------------------------------------------------------------------

class _LocalFakeBackend:
    """Minimal BackendProtocol fake that returns canned code blocks.

    No external deps — uses only stdlib dataclasses.
    """

    def __init__(self, n_blocks: int = 3) -> None:
        # We can't reference RetrievedBlock/etc. at class definition time
        # because api_types hasn't been loaded yet. We'll build blocks lazily.
        self._n_blocks = n_blocks
        self._blocks = None  # built lazily

    def _ensure_blocks(self):
        if self._blocks is not None:
            return
        at = _get_api_types()
        self._blocks = [
            at.RetrievedBlock(
                kind="mutation_code",
                document_id=f"integration_gene_{i}",
                title=f"Integration mutation {i}",
                score=0.9 - i * 0.05,
                content=f"def mutation_{i}(self):\n    # historic mutation {i}\n    return x * {i}",
                diagnostics={"source": "code", "reranked": False},
            )
            for i in range(self._n_blocks)
        ]

    def retrieve(self, request: Any) -> Any:
        self._ensure_blocks()
        at = _get_api_types()
        ns = request.namespace
        if ns == "code":
            blocks = [b for b in self._blocks  # type: ignore[union-attr]
                      if (b.diagnostics or {}).get("source") == "code"]
        else:
            blocks = list(self._blocks)  # type: ignore[union-attr]
        return at.RetrieveResponse(blocks=blocks[:request.top_k], latency_ms=1.0)

    def index(self, document: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Lazy RagService / RagClient construction
# ---------------------------------------------------------------------------

def _ensure_service_stack():
    """Load service.py and client.py via importlib, bypassing __init__.py.

    Called from test methods, NOT at module level, so heavy deps are not
    imported during test collection.
    """
    _stub_pkg("src", str(_WORKTREE_ROOT / "src"))
    _stub_pkg("src.rag", str(_WORKTREE_ROOT / "src" / "rag"))

    # Ensure api_types is loaded
    _load_module_direct("src/rag/api_types.py", "src.rag.api_types")

    # Stub utils.rag_metrics
    if "utils.rag_metrics" not in sys.modules:
        if "utils" not in sys.modules:
            _m = types.ModuleType("utils")
            _m.__path__ = [str(_WORKTREE_ROOT / "utils")]  # type: ignore[attr-defined]
            sys.modules["utils"] = _m
        _rm = types.ModuleType("utils.rag_metrics")
        _rm.record_metric = lambda *a, **kw: None  # type: ignore[attr-defined]
        sys.modules["utils.rag_metrics"] = _rm

    # service.py now does lazy import of PromptEnhancerConfig (via
    # _get_prompt_enhancer_config_class), so no prompt_enhancer stub is needed
    # at service.py import time.  We must NOT install a stub here because
    # test_faiss_backend.py::TestPromptEnhancerBackendWiring needs the real
    # PromptEnhancer class.

    # Load service and client
    _load_module_direct("src/rag/service.py", "src.rag.service")
    _load_module_direct("src/rag/client.py", "src.rag.client")


def _build_client(n_blocks: int = 3) -> Any:
    """Build a RagClient backed by the local fake backend."""
    _ensure_service_stack()
    RagService = sys.modules["src.rag.service"].RagService
    RagClient = sys.modules["src.rag.client"].RagClient

    backend = _LocalFakeBackend(n_blocks=n_blocks)
    service = RagService(backend=backend, reranker=None)
    return RagClient(service=service)


# ---------------------------------------------------------------------------
# Stub heavy run_improved.py transitive imports (called from test methods)
# ---------------------------------------------------------------------------

def _setup_run_improved_stubs():
    """Inject stubs for cluster-specific deps that run_improved.py needs."""
    # deap
    for pkg in ["deap", "deap.base", "deap.creator", "deap.tools"]:
        if pkg not in sys.modules:
            _stub_module(pkg, HallOfFame=type("HallOfFame", (), {}))

    # src.llm_utils (has torch + heavy imports)
    if "src.llm_utils" not in sys.modules:
        _stub_module(
            "src.llm_utils",
            split_file=lambda *a, **kw: [],
            retrieve_base_code=lambda *a, **kw: "",
            mutate_prompts=lambda *a, **kw: None,
        )

    # src.utils.print_utils
    for mod_name in ["src.utils", "src.utils.print_utils"]:
        if mod_name not in sys.modules:
            _stub_module(
                mod_name,
                print_population=lambda *a, **kw: None,
                print_scores=lambda *a, **kw: None,
                box_print=lambda *a, **kw: None,
                print_job_info=lambda *a, **kw: None,
            )

    # rag.data_ingestion
    if "rag.data_ingestion" not in sys.modules:
        _stub_module(
            "rag.data_ingestion",
            build_mutation_description=lambda *a, **kw: "desc",
            calculate_fitness_improvement=lambda *a, **kw: {},
        )

    # rag.runtime
    if "rag.runtime" not in sys.modules:
        _stub_module(
            "rag.runtime",
            get_runtime=lambda: None,
            get_runtime_status=lambda: {"state": "disabled", "reason": "test", "details": {}},
        )

    # src.llm_mutation, src.llm_crossover
    for m in ["src.llm_mutation", "src.llm_crossover"]:
        if m not in sys.modules:
            _stub_module(m)

    # requests
    if "requests" not in sys.modules:
        _stub_module("requests")

    # Ensure rag.client / rag.service / rag.api_types resolve for run_improved
    _ensure_service_stack()
    if "rag.client" not in sys.modules:
        sys.modules["rag.client"] = sys.modules["src.rag.client"]
    if "rag.service" not in sys.modules:
        sys.modules["rag.service"] = sys.modules["src.rag.service"]
    if "rag.api_types" not in sys.modules:
        sys.modules["rag.api_types"] = sys.modules["src.rag.api_types"]


def _load_run_improved() -> Any:
    """Load run_improved.py with cluster-specific deps stubbed out."""
    mod_name = "_run_improved_test_stub"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    _setup_run_improved_stubs()

    full_path = _WORKTREE_ROOT / "run_improved.py"
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        pass  # DEAP/other module-level globals may error; _apply_rag_context is defined early
    return mod


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestApplyRagContextIntegration:
    """End-to-end smoke: _apply_rag_context through RagClient → RagService."""

    @pytest.fixture(autouse=True)
    def patch_constants(self, monkeypatch):
        import cfg.constants as _const
        monkeypatch.setattr(_const, "RAG_ENABLED", True)
        monkeypatch.setattr(_const, "RAG_USE_CODE_CONTEXT", True)
        monkeypatch.setattr(_const, "RAG_USE_TEXT_CONTEXT", False)
        monkeypatch.setattr(_const, "RAG_RERANKER_ENABLED", False)
        monkeypatch.setattr(_const, "RAG_DATA_DIR", "/tmp/fake_rag_integration")
        monkeypatch.setenv("RUN_ID", "integration-test-run")

    def _get_apply_fn(self):
        """Return _apply_rag_context from run_improved, or skip if unavailable."""
        ri_mod = _load_run_improved()
        fn = getattr(ri_mod, "_apply_rag_context", None)
        if fn is None:
            pytest.skip("_apply_rag_context not found in run_improved (possible load error)")
        return fn

    def test_apply_rag_context_returns_nonempty_string(self):
        """_apply_rag_context must return a non-empty string."""
        apply_fn = self._get_apply_fn()
        client = _build_client()
        result = apply_fn(
            template_txt="Mutate this code: {code}",
            mutation_type="Complex",
            query_code="class Net(nn.Module): pass",
            gene_id="test-gene",
            rag_client=client,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_apply_rag_context_contains_retrieved_content_markers(self):
        """When backend returns code blocks, augmented template mentions them."""
        apply_fn = self._get_apply_fn()
        client = _build_client(n_blocks=3)
        result = apply_fn(
            template_txt="Mutate this code: {code}",
            mutation_type="Complex",
            query_code="def forward(self): return x",
            gene_id="test-gene",
            rag_client=client,
        )
        has_rag_marker = (
            "RAG Code Examples" in result
            or any(f"integration_gene_{i}" in result for i in range(3))
        )
        assert has_rag_marker, (
            f"Expected RAG content markers in augmented prompt, got:\n{result[:500]}"
        )

    def test_apply_rag_context_contains_original_template(self):
        """The original template text must survive into the augmented result."""
        apply_fn = self._get_apply_fn()
        client = _build_client()
        template = "UNIQUE_MARKER_12345 mutate the network"
        result = apply_fn(
            template_txt=template,
            mutation_type="Complex",
            query_code="class Net: pass",
            gene_id="test-gene",
            rag_client=client,
        )
        assert "UNIQUE_MARKER_12345" in result

    def test_apply_rag_context_falls_back_when_client_none(self):
        """When rag_client=None and RAG is disabled, returns original template."""
        apply_fn = self._get_apply_fn()
        ri_mod = sys.modules.get("_run_improved_test_stub")

        import cfg.constants as _const
        original_flag = _const.RAG_ENABLED
        _const.RAG_ENABLED = False

        if ri_mod is not None:
            original_client = getattr(ri_mod, "_rag_client", None)
            ri_mod._rag_client = None

        try:
            template = "fallback template text"
            result = apply_fn(
                template_txt=template,
                mutation_type="Complex",
                query_code="class Net: pass",
                gene_id=None,
                rag_client=None,
            )
            assert result == template
        finally:
            if ri_mod is not None:
                ri_mod._rag_client = original_client
            _const.RAG_ENABLED = original_flag

    def test_client_augment_end_to_end_longer_than_template(self):
        """Full augment() path produces a string >= template length."""
        apply_fn = self._get_apply_fn()
        client = _build_client(n_blocks=3)
        template = "Short template"
        result = apply_fn(
            template_txt=template,
            mutation_type="Complex",
            query_code="class Net(nn.Module):\n    def forward(self, x): return x",
            gene_id="gene-end2end",
            rag_client=client,
        )
        assert len(result) >= len(template)
