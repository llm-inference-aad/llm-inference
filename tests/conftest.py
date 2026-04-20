"""Shared pytest fixtures for the LLMGE test suite.

Fixtures here are available to all tests without explicit import.  RAG-specific
fakes live in ``tests/rag/fakes.py`` so they can be imported directly in tests
that need explicit control.
"""

from __future__ import annotations

import importlib
import sys
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Collection guards
# ---------------------------------------------------------------------------
# test_evo_loop.py imports run_improved at module level which requires SLURM
# env vars, GPU modules, and a full venv with all heavy dependencies.  Skip it
# during collection so `pytest --collect-only` succeeds in CI/unit-test mode.
collect_ignore = ["operators/test_evo_loop.py"]


# ---------------------------------------------------------------------------
# tmp_rag_data — isolated per-test RAG data directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_rag_data(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Yield an empty RAG data directory and monkeypatch cfg.constants.RAG_DATA_DIR.

    Tests that exercise the vector store or indexer should use this fixture to
    avoid writing into the real RAG data directory.
    """
    rag_dir = tmp_path / "rag_data"
    rag_dir.mkdir(parents=True, exist_ok=True)

    # Patch the constant so any code that reads cfg.constants.RAG_DATA_DIR at
    # call time picks up the temporary path.
    try:
        import cfg.constants as _constants
        monkeypatch.setattr(_constants, "RAG_DATA_DIR", str(rag_dir))
    except ImportError:
        # cfg.constants not importable in minimal test environments — skip patch.
        pass

    yield rag_dir


# ---------------------------------------------------------------------------
# reset_rag_runtime — autouse fixture to prevent cross-test state leakage
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_rag_runtime():
    """Reset RAG module-level singletons between tests.

    The RAG subsystem has two module-level singletons that accumulate state
    across tests if not cleared:

    * ``src.rag.runtime._runtime_instance`` — the global ``RagRuntime``.
    * ``src.rag.prompt_enhancer._reranker`` — the module-level Reranker.

    Clearing them here ensures each test starts from a clean slate without
    needing to reload the modules.
    """
    yield  # Let the test run first.

    # Clear singletons after each test.
    runtime_mod = sys.modules.get("src.rag.runtime") or sys.modules.get("rag.runtime")
    if runtime_mod is not None and hasattr(runtime_mod, "_runtime_instance"):
        runtime_mod._runtime_instance = None  # type: ignore[attr-defined]

    enhancer_mod = (
        sys.modules.get("src.rag.prompt_enhancer")
        or sys.modules.get("rag.prompt_enhancer")
    )
    if enhancer_mod is not None and hasattr(enhancer_mod, "_reranker"):
        enhancer_mod._reranker = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# fake_env — hermetic env-var fixture
# ---------------------------------------------------------------------------

# All RAG_* environment variable names that tests may inspect.
_RAG_ENV_VARS = (
    "RAG_ENABLED",
    "RAG_USE_CODE_CONTEXT",
    "RAG_USE_TEXT_CONTEXT",
    "RAG_RERANKER_ENABLED",
    "RAG_DATA_DIR",
    "RAG_TOP_K",
    "RAG_MIN_SIMILARITY",
    "RAG_TEXT_TOP_K",
    "RAG_TEXT_TOP_K_API",
    "RAG_TEXT_TOP_K_PDF",
    "RAG_TEXT_CANDIDATE_K",
    "RAG_MIN_ACCURACY",
    "RAG_MAX_PARAMETERS",
    "RAG_CODE_EMBED_MODEL",
    "RAG_TEXT_EMBED_MODEL",
)


@pytest.fixture()
def fake_env(monkeypatch: pytest.MonkeyPatch):
    """Remove all RAG_* env vars and set known safe defaults.

    Any test that exercises code reading RAG configuration from the environment
    should request this fixture to guarantee hermeticity.
    """
    for var in _RAG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    # Minimal defaults so cfg.constants does not crash when re-evaluated.
    monkeypatch.setenv("RAG_ENABLED", "false")
    monkeypatch.setenv("RAG_USE_CODE_CONTEXT", "false")
    monkeypatch.setenv("RAG_USE_TEXT_CONTEXT", "false")
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "false")

    yield
