"""Tests for src/rag/bookkeeping.py.

Covers:
- Round-trip: write N events, replay, reconstruct exactly N events equal to originals.
- JSON determinism: asdict output is stable.
- Concurrent append: 5 threads × 20 events = 100 lines, no corruption.
- Schema completeness: every event has the 8 required comparison identifiers.
- Malformed line tolerance: skips bad lines, logs a warning.
- Missing ledger tolerance: replay on non-existent path returns empty iterator.
- Emit from RagService: ledger receives an event with retrieval_request populated.

Import strategy: we load bookkeeping.py via importlib.util to bypass
src/rag/__init__.py, which eagerly imports heavy ML deps (sentence-transformers,
faiss, etc.).  The acceptance criterion is that bookkeeping.py itself is
importable without heavy deps, not that the package __init__ is light.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import logging
import pathlib
import sys
import threading
import types
from dataclasses import asdict
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load bookkeeping directly from its file, bypassing __init__.py
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = pathlib.Path(__file__).parent.parent.parent


def _ensure_pkg_stub(name: str, base_path: pathlib.Path) -> types.ModuleType:
    """Create a lightweight stub package in sys.modules if not already present."""
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = [str(base_path)]  # type: ignore[attr-defined]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


def _load_module_direct(rel_path: str, mod_name: str) -> types.ModuleType:
    """Load a module from a file path without triggering the package __init__."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    full_path = _WORKTREE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    assert spec is not None, f"Could not create spec for {full_path}"
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Ensure package stubs exist so relative imports inside bookkeeping work.
_ensure_pkg_stub("src", _WORKTREE_ROOT / "src")
_ensure_pkg_stub("src.rag", _WORKTREE_ROOT / "src" / "rag")

# Load bookkeeping directly (no __init__.py, no torch/faiss).
_bk_mod = _load_module_direct("src/rag/bookkeeping.py", "src.rag.bookkeeping")

# Convenient references used throughout this file.
MutationEvent = _bk_mod.MutationEvent
RunLedger = _bk_mod.RunLedger
replay_ledger = _bk_mod.replay_ledger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    run_id: str = "test-run",
    generation: int = 0,
    parent_gene_id: str = "parent0",
    child_gene_id: str = "child0",
    mutation_type: str = "add_layer",
    backend: str = "faiss",
    request_id: str | None = None,
    prompt_id: str | None = None,
    raw_prompt: str = "Mutate this network.",
    **kwargs: Any,
) -> MutationEvent:
    """Factory for synthetic MutationEvent instances."""
    rid = request_id or MutationEvent.make_request_id()
    pid = prompt_id or MutationEvent.make_prompt_id(raw_prompt)
    return MutationEvent(
        run_id=run_id,
        generation=generation,
        parent_gene_id=parent_gene_id,
        child_gene_id=child_gene_id,
        mutation_type=mutation_type,
        backend=backend,
        request_id=rid,
        prompt_id=pid,
        raw_prompt=raw_prompt,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test: module-level smoke import (no torch, no faiss)
# ---------------------------------------------------------------------------

def test_import_no_heavy_deps():
    """bookkeeping.py must not pull torch or faiss at module import time."""
    # Remove bookkeeping from cache to force a fresh import.
    for key in list(sys.modules.keys()):
        if "bookkeeping" in key:
            del sys.modules[key]

    # Patch torch/faiss so ImportError fires if they are actually imported.
    import unittest.mock as mock

    with mock.patch.dict(sys.modules, {"torch": None, "faiss": None}):
        mod = _load_module_direct("src/rag/bookkeeping.py", "src.rag.bookkeeping_smoke_test")

    assert hasattr(mod, "MutationEvent")
    assert hasattr(mod, "RunLedger")
    assert hasattr(mod, "replay_ledger")


# ---------------------------------------------------------------------------
# Test: round-trip write 10 events + replay
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path):
    """Write 10 synthetic events, replay, assert exactly 10 events with equal values."""
    ledger_path = tmp_path / "rag_ledger.jsonl"
    ledger = RunLedger(run_id="rt-run", ledger_path=ledger_path)

    events = [_make_event(child_gene_id=f"child{i}", generation=i) for i in range(10)]
    for e in events:
        ledger.append(e)

    replayed = list(replay_ledger("rt-run", ledger_path=ledger_path))
    assert len(replayed) == 10

    for original, reconstructed in zip(events, replayed):
        assert asdict(original) == asdict(reconstructed), (
            f"Round-trip mismatch: {asdict(original)} != {asdict(reconstructed)}"
        )


# ---------------------------------------------------------------------------
# Test: JSON determinism
# ---------------------------------------------------------------------------

def test_json_determinism():
    """asdict(event) is stable (key ordering consistent across calls)."""
    event = _make_event(
        raw_prompt="test prompt",
        eval_outputs={"test_acc": 0.95, "total_params": 1234},
        latencies={"augment_ms": 100.0, "llm_ms": 200.0},
    )
    d1 = json.dumps(asdict(event), sort_keys=True)
    d2 = json.dumps(asdict(event), sort_keys=True)
    assert d1 == d2, "asdict output is not deterministic"

    # Also verify all expected keys are present.
    data = asdict(event)
    required_keys = {
        "run_id", "generation", "parent_gene_id", "child_gene_id",
        "mutation_type", "backend", "request_id", "prompt_id",
        "raw_prompt", "augmented_prompt", "retrieval_request",
        "retrieval_response", "model_request", "model_response",
        "parsed_artifact", "eval_outputs", "latencies", "failure_mode",
        "timestamp",
    }
    missing = required_keys - set(data.keys())
    assert not missing, f"Missing keys in asdict output: {missing}"


# ---------------------------------------------------------------------------
# Test: concurrent append (5 threads × 20 events = 100 lines, no corruption)
# ---------------------------------------------------------------------------

def test_concurrent_append(tmp_path):
    """5 threads each writing 20 events produces 100 valid JSONL lines, no corruption."""
    ledger_path = tmp_path / "concurrent_ledger.jsonl"
    ledger = RunLedger(run_id="concurrent-run", ledger_path=ledger_path)

    n_threads = 5
    events_per_thread = 20
    errors: list[Exception] = []

    def writer(thread_idx: int) -> None:
        try:
            for i in range(events_per_thread):
                e = _make_event(
                    child_gene_id=f"t{thread_idx}_e{i}",
                    generation=thread_idx,
                    raw_prompt=f"prompt-thread-{thread_idx}-event-{i}",
                )
                ledger.append(e)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"

    # Read the raw file to check for line corruption.
    raw_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == n_threads * events_per_thread, (
        f"Expected {n_threads * events_per_thread} lines, got {len(raw_lines)}"
    )

    # Every line must parse as valid JSON.
    for lineno, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        assert stripped, f"Empty line at {lineno}"
        try:
            json.loads(stripped)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Line {lineno} is not valid JSON: {exc}\nContent: {stripped!r}")

    # replay_ledger must also return exactly 100 events.
    replayed = list(replay_ledger("concurrent-run", ledger_path=ledger_path))
    assert len(replayed) == n_threads * events_per_thread, (
        f"replay_ledger returned {len(replayed)} events, expected {n_threads * events_per_thread}"
    )


# ---------------------------------------------------------------------------
# Test: schema completeness (8 required comparison identifiers)
# ---------------------------------------------------------------------------

def test_schema_completeness():
    """Every MutationEvent has the 8 required comparison identifiers."""
    required_ids = {
        "run_id",
        "request_id",
        "prompt_id",
        "parent_gene_id",
        "child_gene_id",
        "backend",
        "generation",
        "mutation_type",
    }

    field_names = {f.name for f in dataclasses.fields(MutationEvent)}
    missing = required_ids - field_names
    assert not missing, f"MutationEvent is missing required fields: {missing}"

    # Also verify a constructed event has non-None values for all required ids.
    event = _make_event()
    data = asdict(event)
    for field_name in required_ids:
        assert data[field_name] is not None, (
            f"Required field '{field_name}' is None on constructed event"
        )


# ---------------------------------------------------------------------------
# Test: malformed line tolerance
# ---------------------------------------------------------------------------

def test_malformed_line_tolerance(tmp_path, caplog):
    """replay_ledger skips malformed lines and logs a warning."""
    ledger_path = tmp_path / "malformed_ledger.jsonl"
    ledger = RunLedger(run_id="malformed-run", ledger_path=ledger_path)

    good_events = [_make_event(child_gene_id=f"good{i}") for i in range(5)]
    for e in good_events:
        ledger.append(e)

    # Inject a garbage line into the middle of the file.
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write("THIS IS NOT JSON\n")

    # Add more good events after the bad line.
    for e in [_make_event(child_gene_id=f"good{5 + i}") for i in range(5)]:
        ledger.append(e)

    with caplog.at_level(logging.WARNING, logger="src.rag.bookkeeping"):
        replayed = list(replay_ledger("malformed-run", ledger_path=ledger_path))

    assert len(replayed) == 10, (
        f"Expected 10 valid events (skipping 1 bad line), got {len(replayed)}"
    )

    warning_texts = [r.message for r in caplog.records]
    assert any("malformed" in w.lower() or "skipping" in w.lower() for w in warning_texts), (
        f"Expected a warning about malformed/skipping line, got: {warning_texts}"
    )


# ---------------------------------------------------------------------------
# Test: missing ledger tolerance
# ---------------------------------------------------------------------------

def test_missing_ledger_tolerance(tmp_path):
    """replay_ledger on a non-existent path returns an empty iterator (no crash)."""
    non_existent = tmp_path / "does_not_exist" / "rag_ledger.jsonl"
    result = list(replay_ledger("no-run", ledger_path=non_existent))
    assert result == [], f"Expected empty iterator, got {result}"


# ---------------------------------------------------------------------------
# Test: emit from RagService
# ---------------------------------------------------------------------------

def test_emit_from_rag_service(tmp_path, monkeypatch):
    """RagService.augment emits a ledger event with retrieval_request populated.

    Uses monkeypatch for ALL sys.modules mutations so pytest restores them
    after this test, preventing contamination of later tests (e.g. test_faiss_backend.py
    which needs the real cfg.constants).
    """
    # --- Build stubs -------------------------------------------------------- #

    fake_constants = types.ModuleType("cfg.constants")
    fake_constants.RAG_USE_CODE_CONTEXT = True
    fake_constants.RAG_USE_TEXT_CONTEXT = False
    fake_constants.RAG_RERANKER_ENABLED = False
    fake_constants.RUN_ID = "svc-test-run"

    rag_metrics_mod = types.ModuleType("utils.rag_metrics")
    rag_metrics_mod.record_metric = lambda *a, **kw: None  # type: ignore

    class _FakePEConfig:
        """Minimal PromptEnhancerConfig substitute."""
        def __init__(self) -> None:
            self.top_k = 5
            self.text_candidate_k = 10
            self.text_top_k = 3
            self.text_top_k_api = 2
            self.text_top_k_pdf = 1

    prompt_enhancer_stub = types.ModuleType("src.rag.prompt_enhancer")
    prompt_enhancer_stub.PromptEnhancerConfig = _FakePEConfig  # type: ignore

    # Ensure package stubs via monkeypatch so they're restored after the test.
    if "cfg" not in sys.modules:
        cfg_pkg = types.ModuleType("cfg")
        cfg_pkg.__path__ = [str(_WORKTREE_ROOT / "src" / "cfg")]  # type: ignore
        cfg_pkg.__package__ = "cfg"
        monkeypatch.setitem(sys.modules, "cfg", cfg_pkg)

    monkeypatch.setitem(sys.modules, "cfg.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "utils.rag_metrics", rag_metrics_mod)
    monkeypatch.setitem(sys.modules, "src.rag.prompt_enhancer", prompt_enhancer_stub)

    # Force-reload service.py so it sees the fake cfg.constants from above.
    monkeypatch.delitem(sys.modules, "src.rag.service", raising=False)
    _load_module_direct("src/rag/service.py", "src.rag.service")
    service_mod = sys.modules["src.rag.service"]

    # Patch service module's lazy loaders to our fakes (avoids torch at call time).
    monkeypatch.setattr(service_mod, "_get_constants", lambda: fake_constants)
    monkeypatch.setattr(service_mod, "_get_prompt_enhancer_config_class", lambda: _FakePEConfig)

    # Ensure api_types is loaded (bookkeeping needs it too).
    if "src.rag.api_types" not in sys.modules:
        _load_module_direct("src/rag/api_types.py", "src.rag.api_types")
    if "src.rag.bookkeeping" not in sys.modules:
        _load_module_direct("src/rag/bookkeeping.py", "src.rag.bookkeeping")

    RagService = service_mod.RagService
    _api_mod = sys.modules["src.rag.api_types"]
    AugmentRequest = _api_mod.AugmentRequest
    RetrievedBlock = _api_mod.RetrievedBlock
    RetrieveRequest = _api_mod.RetrieveRequest
    RetrieveResponse = _api_mod.RetrieveResponse

    # --- Build FakeBackend ------------------------------------------------- #
    class FakeBackend:
        def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
            block = RetrievedBlock(
                kind="code",
                document_id="fake-doc-0",
                title="Fake Doc",
                score=0.99,
                content="def fake(): pass",
                diagnostics={"source": "fake"},
            )
            return RetrieveResponse(blocks=[block], diagnostics={}, latency_ms=1.0)

        def index(self, document) -> None:
            pass

    # --- Run test ---------------------------------------------------------- #
    ledger_path = tmp_path / "rag_ledger.jsonl"
    ledger = RunLedger(run_id="svc-test-run", ledger_path=ledger_path)

    service = RagService(
        backend=FakeBackend(),
        reranker=None,
        ledger=ledger,
    )

    req = AugmentRequest(
        template="Mutate this CNN.",
        mutation_type="add_layer",
        query_code="class Net: pass",
        gene_id="gene-test-0",
        request_id="fixed-request-id-001",
    )
    service.augment(req)

    # The ledger file should have exactly one line.
    assert ledger_path.exists(), "Ledger file was not created"
    replayed = list(replay_ledger("svc-test-run", ledger_path=ledger_path))
    assert len(replayed) == 1, f"Expected 1 event, got {len(replayed)}"

    event = replayed[0]
    assert event.retrieval_request is not None, "retrieval_request should be populated"
    assert event.eval_outputs is None, "eval_outputs should be None at augment time"
    assert event.request_id == "fixed-request-id-001", (
        f"request_id mismatch: {event.request_id}"
    )
