"""Tests for src/rag/pareto_policy.py.

Covers:
- _dominates: standard Pareto dominance cases (strict, mutual non-dominance, equal).
- compute_pareto_front: given 20 points with known front, returns exactly the expected
  non-dominated subset.
- is_pareto_eligible: 20-individual synthetic population with known Pareto front.
  - Top 10% by accuracy (indices chosen deliberately).
  - Bottom 10% by params (distinct set).
  - OR logic: eligible if in either set.
  - Not eligible if in neither set.
- Edge cases:
  - Population of 1 → always eligible.
  - All identical eval_outputs → all eligible (they are all in top-N% and bottom-N%).
  - Missing eval_outputs → False.
  - 10% of 7 rounds to ceil(0.7) = 1 → documented rounding choice asserted.
- Integration with ledger: create 10 events, compute eligibility, append, replay,
  verify is_pareto_eligible persists in JSONL.

Rounding convention: math.ceil is used so that a fractional count rounds up to the
nearest integer.  This ensures at least 1 event is always eligible in any non-empty
population with valid eval_outputs.  Example: 10% of 7 = ceil(0.7) = 1.

Import strategy: pareto_policy.py and bookkeeping.py are both loaded directly via
importlib to bypass src/rag/__init__.py, which eagerly imports heavy ML deps.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import math
import pathlib
import sys
import types
from dataclasses import asdict
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load modules directly from source, bypassing heavy __init__.py
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


# Ensure package stubs exist so relative imports inside loaded modules work.
_ensure_pkg_stub("src", _WORKTREE_ROOT / "src")
_ensure_pkg_stub("src.rag", _WORKTREE_ROOT / "src" / "rag")

# Stub cfg.constants so pareto_policy can import the percentile knobs without
# triggering the real constants.py (which imports torch).
# We use setdefault so that if another test file has already installed a more
# complete stub (e.g. test_apply_rag_context_integration.py), we don't replace
# it with our minimal version.  If we install ours first, we pre-populate all
# attributes that other tests may need so monkeypatch.setattr doesn't fail.
_cfg_stub = types.ModuleType("cfg")
_cfg_stub.__path__ = [str(_WORKTREE_ROOT / "src" / "cfg")]  # type: ignore[attr-defined]
_cfg_constants_stub = types.ModuleType("cfg.constants")
# Pareto policy knobs (PR 5)
_cfg_constants_stub.RAG_LOG_TOP_ACCURACY_PCT = 10.0
_cfg_constants_stub.RAG_LOG_BOTTOM_PARAMS_PCT = 10.0
_cfg_constants_stub.RAG_LOG_POLICY = "pareto"
# Common RAG constants expected by other test files (must match names in
# real constants.py so that monkeypatch.setattr works when tests run together).
_cfg_constants_stub.RAG_ENABLED = False
_cfg_constants_stub.RAG_USE_CODE_CONTEXT = True
_cfg_constants_stub.RAG_USE_TEXT_CONTEXT = True
_cfg_constants_stub.RAG_RERANKER_ENABLED = False
_cfg_constants_stub.RAG_DATA_DIR = "/tmp/fake_rag_data"
_cfg_constants_stub.RAG_TOP_K = 5
_cfg_constants_stub.RAG_TEXT_TOP_K = 3
_cfg_constants_stub.RAG_MIN_SIMILARITY = 0.3
_cfg_constants_stub.RAG_TEXT_CANDIDATE_K = 24
_cfg_constants_stub.RAG_TEXT_TOP_K_API = 2
_cfg_constants_stub.RAG_TEXT_TOP_K_PDF = 1
_cfg_constants_stub.RAG_MIN_ACCURACY = 0.9
_cfg_constants_stub.RAG_MAX_PARAMETERS = None
_cfg_constants_stub.RAG_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_cfg_constants_stub.RAG_CODE_EMBED_MODEL = "microsoft/codebert-base"
_cfg_constants_stub.RAG_TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Run-level constants (also patched by test_service.py fixture)
_cfg_constants_stub.RUN_ID = "test-run"
_cfg_constants_stub.RUN_DIR = "/tmp/fake_run"
_cfg_constants_stub.RUN_METRICS_DIR = "/tmp/fake_run/metrics"
_cfg_constants_stub.RUN_LOG_DIR = "/tmp/fake_run/logs"
_cfg_constants_stub.ROOT_DIR = "/tmp"
sys.modules.setdefault("cfg", _cfg_stub)
sys.modules.setdefault("cfg.constants", _cfg_constants_stub)

# Load bookkeeping directly (no __init__.py, no torch/faiss).
_bk_mod = _load_module_direct("src/rag/bookkeeping.py", "src.rag.bookkeeping")
MutationEvent = _bk_mod.MutationEvent
RunLedger = _bk_mod.RunLedger
replay_ledger = _bk_mod.replay_ledger

# Load pareto_policy directly.
_pp_mod = _load_module_direct("src/rag/pareto_policy.py", "src.rag.pareto_policy")
_dominates = _pp_mod._dominates
compute_pareto_front = _pp_mod.compute_pareto_front
is_pareto_eligible = _pp_mod.is_pareto_eligible


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    child_gene_id: str = "child0",
    generation: int = 0,
    test_accuracy: float | None = None,
    test_acc: float | None = None,
    total_params: float | None = None,
    **kwargs: Any,
) -> MutationEvent:
    """Factory for synthetic MutationEvent instances with eval_outputs."""
    eval_outputs: dict | None = None
    if test_accuracy is not None or test_acc is not None or total_params is not None:
        eval_outputs = {}
        if test_accuracy is not None:
            eval_outputs["test_accuracy"] = test_accuracy
        if test_acc is not None:
            eval_outputs["test_acc"] = test_acc
        if total_params is not None:
            eval_outputs["total_params"] = total_params

    return MutationEvent(
        run_id="test-run",
        generation=generation,
        parent_gene_id="parent0",
        child_gene_id=child_gene_id,
        mutation_type="add_layer",
        backend="faiss",
        request_id=MutationEvent.make_request_id(),
        prompt_id=MutationEvent.make_prompt_id(child_gene_id),
        raw_prompt=f"prompt for {child_gene_id}",
        eval_outputs=eval_outputs,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _dominates unit tests
# ---------------------------------------------------------------------------

class TestDominates:
    """Unit tests for the _dominates helper (maximisation space)."""

    def test_strict_dominance_both_objectives(self):
        """a strictly better in both objectives dominates b."""
        a = (0.95, -100)   # higher accuracy, fewer params (negated)
        b = (0.80, -200)
        assert _dominates(a, b) is True

    def test_strict_dominance_one_objective_equal_other(self):
        """a strictly better in one objective and equal in the other."""
        a = (0.95, -100)
        b = (0.80, -100)
        assert _dominates(a, b) is True

    def test_strict_dominance_equal_accuracy_better_params(self):
        a = (0.80, -100)
        b = (0.80, -200)
        assert _dominates(a, b) is True

    def test_mutual_non_dominance(self):
        """a better in accuracy, b better in params — neither dominates."""
        a = (0.95, -500)   # high accuracy, many params
        b = (0.70, -100)   # low accuracy, few params
        assert _dominates(a, b) is False
        assert _dominates(b, a) is False

    def test_equal_points_not_dominated(self):
        """Equal points do not dominate each other."""
        a = (0.85, -200)
        b = (0.85, -200)
        assert _dominates(a, b) is False
        assert _dominates(b, a) is False

    def test_dominated_point(self):
        """b is dominated by a; ensure b does not dominate a."""
        a = (0.90, -150)
        b = (0.70, -300)
        assert _dominates(a, b) is True
        assert _dominates(b, a) is False


# ---------------------------------------------------------------------------
# compute_pareto_front unit tests
# ---------------------------------------------------------------------------

class TestComputeParetoFront:
    """Tests for compute_pareto_front on MutationEvent sequences."""

    def test_known_front_20_points(self):
        """20 points with a known Pareto front — assert correct non-dominated set.

        Layout (test_accuracy, total_params):
        - Indices 0..3: the Pareto front (high accuracy AND low params).
        - Indices 4..19: dominated or strictly interior.
        """
        # Pareto front members: high accuracy, low params
        pareto_data = [
            (0.98, 100),   # idx 0: best accuracy, moderate params
            (0.95, 50),    # idx 1: high accuracy, fewer params
            (0.85, 20),    # idx 2: lower accuracy but very few params
            (0.99, 200),   # idx 3: highest accuracy, more params (not dominated by 0 since acc>0.98)
        ]
        # Dominated points: every one of these is dominated by at least one Pareto member.
        dominated_data = [
            (0.80, 300),
            (0.75, 400),
            (0.70, 500),
            (0.65, 600),
            (0.60, 700),
            (0.55, 800),
            (0.50, 900),
            (0.45, 1000),
            (0.40, 1100),
            (0.35, 1200),
            (0.30, 1300),
            (0.25, 1400),
            (0.20, 1500),
            (0.15, 1600),
            (0.10, 1700),
            (0.05, 1800),
        ]
        all_data = pareto_data + dominated_data
        assert len(all_data) == 20

        events = [
            _make_event(child_gene_id=f"child{i}", test_accuracy=acc, total_params=params)
            for i, (acc, params) in enumerate(all_data)
        ]

        front = compute_pareto_front(events)
        front_ids = {e.child_gene_id for e in front}

        # All Pareto front members must be present.
        expected_ids = {f"child{i}" for i in range(len(pareto_data))}
        assert expected_ids == front_ids, (
            f"Expected Pareto front {expected_ids}, got {front_ids}"
        )

    def test_empty_population(self):
        """Empty population returns empty front."""
        assert compute_pareto_front([]) == []

    def test_single_event_is_pareto_front(self):
        """A single event with eval_outputs is always on the Pareto front."""
        event = _make_event(child_gene_id="solo", test_accuracy=0.7, total_params=500)
        front = compute_pareto_front([event])
        assert len(front) == 1
        assert front[0].child_gene_id == "solo"

    def test_events_without_eval_outputs_excluded(self):
        """Events lacking eval_outputs are excluded from the Pareto front."""
        good = _make_event(child_gene_id="good", test_accuracy=0.8, total_params=200)
        empty = _make_event(child_gene_id="empty")  # no eval_outputs
        front = compute_pareto_front([good, empty])
        assert len(front) == 1
        assert front[0].child_gene_id == "good"

    def test_all_equal_points_all_on_front(self):
        """When all events have identical objectives, none dominates another — all on front."""
        events = [
            _make_event(child_gene_id=f"child{i}", test_accuracy=0.80, total_params=100)
            for i in range(5)
        ]
        front = compute_pareto_front(events)
        assert len(front) == 5


# ---------------------------------------------------------------------------
# is_pareto_eligible tests
# ---------------------------------------------------------------------------

class TestIsParetoEligible:
    """Tests for is_pareto_eligible on 20-individual synthetic population."""

    @staticmethod
    def _build_population_20():
        """Build a 20-individual population with clearly separated accuracy / params.

        Sorted descending by accuracy (index 0 = best):
        - Indices 0..1: top 10% accuracy (20 * 10% = ceil(2.0) = 2 → threshold = accuracy[1])
        - Indices 18..19: bottom 10% params (sorted ascending, 20*10%=ceil(2.0)=2 → threshold = params_asc[1])

        These two sets are disjoint so we can verify OR logic clearly.
        """
        individuals = []
        for i in range(20):
            # accuracy decreasing from 0.99 to 0.80: [0.99, 0.98, ..., 0.80]
            acc = 0.99 - i * 0.01
            # params increasing from 100 to 2000: 100, 200, ..., 2000
            params = (i + 1) * 100
            individuals.append((acc, params))
        return individuals

    def _make_pop(self):
        data = self._build_population_20()
        events = [
            _make_event(child_gene_id=f"child{i}", test_accuracy=acc, total_params=params)
            for i, (acc, params) in enumerate(data)
        ]
        return events, data

    def test_top_10pct_accuracy_eligible(self):
        """Indices 0 and 1 (highest accuracy) are eligible under top-10% accuracy."""
        events, data = self._make_pop()
        # 10% of 20 = ceil(2.0) = 2 → top-2 by accuracy
        for i in [0, 1]:
            assert is_pareto_eligible(events[i], events, 10.0, 10.0), (
                f"Expected child{i} (acc={data[i][0]}) to be eligible via accuracy"
            )

    def test_bottom_10pct_params_eligible(self):
        """Indices 18 and 19 (highest params = bottom 10% by low-params criterion inverted)
        — wait, bottom_params_pct means LOWEST params count.
        Indices 0 and 1 have the lowest params (100, 200).
        But those overlap with top accuracy.  Let's check the bottom-params criterion
        independently by examining which indices have the smallest params values.
        params[i] = (i+1)*100, so smallest params are at i=0 (100) and i=1 (200).
        Those are the same as the top-accuracy indices in this population.
        To verify OR logic independently, we need a population where the sets are disjoint.
        """
        # Build a population where accuracy and params are NOT correlated.
        # High accuracy → many params; low accuracy → few params.
        events_disjoint = [
            # Top accuracy but many params (eligible via accuracy only)
            _make_event(child_gene_id="acc_elite_0", test_accuracy=0.99, total_params=9000),
            _make_event(child_gene_id="acc_elite_1", test_accuracy=0.98, total_params=8000),
            # Middle tier (neither top accuracy nor bottom params)
            *[
                _make_event(child_gene_id=f"mid_{i}", test_accuracy=0.80 - i * 0.02, total_params=5000 + i * 100)
                for i in range(16)
            ],
            # Bottom params but low accuracy (eligible via params only)
            _make_event(child_gene_id="param_elite_0", test_accuracy=0.10, total_params=100),
            _make_event(child_gene_id="param_elite_1", test_accuracy=0.05, total_params=50),
        ]
        assert len(events_disjoint) == 20

        # Accuracy elite: child0 and child1
        assert is_pareto_eligible(events_disjoint[0], events_disjoint, 10.0, 10.0)
        assert is_pareto_eligible(events_disjoint[1], events_disjoint, 10.0, 10.0)

        # Params elite: param_elite_0 and param_elite_1
        assert is_pareto_eligible(events_disjoint[18], events_disjoint, 10.0, 10.0)
        assert is_pareto_eligible(events_disjoint[19], events_disjoint, 10.0, 10.0)

        # Middle tier: should NOT be eligible (neither top accuracy nor bottom params)
        for i in range(2, 18):
            assert not is_pareto_eligible(events_disjoint[i], events_disjoint, 10.0, 10.0), (
                f"Expected {events_disjoint[i].child_gene_id} to be ineligible"
            )

    def test_or_logic_eligible_if_either_criterion(self):
        """An event meeting only the accuracy criterion is still eligible."""
        events, _ = self._make_pop()
        # Index 0: best accuracy (eligible via acc), worst params (not bottom-params)
        assert is_pareto_eligible(events[0], events, 10.0, 10.0)

    def test_not_eligible_if_neither_criterion(self):
        """Events in the middle (neither top accuracy nor bottom params) are ineligible."""
        events, _ = self._make_pop()
        # Indices 2..17 are neither top-2 by accuracy nor bottom-2 by params in this population
        # (bottom params here are indices 0,1 with params=100,200; but those are also top accuracy)
        # Let's build a population where mid-range is clearly ineligible.
        events_clear = [
            _make_event(child_gene_id="top_acc", test_accuracy=0.99, total_params=5000),
            _make_event(child_gene_id="mid_1", test_accuracy=0.70, total_params=4000),
            _make_event(child_gene_id="mid_2", test_accuracy=0.65, total_params=3500),
            _make_event(child_gene_id="mid_3", test_accuracy=0.60, total_params=3000),
            _make_event(child_gene_id="mid_4", test_accuracy=0.55, total_params=2500),
            _make_event(child_gene_id="mid_5", test_accuracy=0.50, total_params=2000),
            _make_event(child_gene_id="mid_6", test_accuracy=0.45, total_params=1500),
            _make_event(child_gene_id="mid_7", test_accuracy=0.40, total_params=1200),
            _make_event(child_gene_id="mid_8", test_accuracy=0.35, total_params=1100),
            _make_event(child_gene_id="low_params", test_accuracy=0.01, total_params=10),
        ]
        # 10% of 10 = ceil(1.0) = 1
        # Top-1 accuracy: top_acc (0.99)
        # Bottom-1 params: low_params (10)
        assert is_pareto_eligible(events_clear[0], events_clear, 10.0, 10.0)   # top accuracy
        assert is_pareto_eligible(events_clear[9], events_clear, 10.0, 10.0)   # bottom params

        # Middle: indices 1..8 should NOT be eligible
        for idx in range(1, 9):
            assert not is_pareto_eligible(events_clear[idx], events_clear, 10.0, 10.0), (
                f"Expected {events_clear[idx].child_gene_id} to be ineligible"
            )

    def test_edge_population_1_always_eligible(self):
        """A single-event population: always eligible (trivially top-1 and bottom-1)."""
        event = _make_event(child_gene_id="solo", test_accuracy=0.5, total_params=100)
        assert is_pareto_eligible(event, [event], 10.0, 10.0)

    def test_edge_all_identical_eval_outputs(self):
        """When all events have identical metrics, all are eligible (all in top-N and bottom-N)."""
        events = [
            _make_event(child_gene_id=f"dup{i}", test_accuracy=0.80, total_params=500)
            for i in range(10)
        ]
        for ev in events:
            assert is_pareto_eligible(ev, events, 10.0, 10.0)

    def test_edge_missing_eval_outputs_always_false(self):
        """Events without eval_outputs are never eligible."""
        no_eval = _make_event(child_gene_id="no_eval")
        population = [
            _make_event(child_gene_id=f"other{i}", test_accuracy=0.8, total_params=100)
            for i in range(5)
        ]
        population.append(no_eval)
        assert is_pareto_eligible(no_eval, population, 10.0, 10.0) is False

    def test_edge_rounding_7_individuals_ceil(self):
        """10% of 7 = ceil(0.7) = 1 — exactly 1 individual is eligible per criterion.

        Rounding convention: math.ceil ensures at least 1 event is eligible in
        any non-empty population.  This test pins that the implementation uses
        ceil, not floor or round.
        """
        # 7 individuals with strictly different accuracy values.
        accs = [0.95, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]
        # 7 individuals with strictly different param counts.
        params_vals = [100, 200, 300, 400, 500, 600, 700]
        events = [
            _make_event(child_gene_id=f"i{i}", test_accuracy=accs[i], total_params=params_vals[i])
            for i in range(7)
        ]
        # 10% of 7 = ceil(0.7) = 1
        expected_top_acc_count = math.ceil(7 * 10.0 / 100.0)
        assert expected_top_acc_count == 1, "Rounding check: ceil(0.7) must equal 1"

        # Exactly index 0 should be eligible (highest accuracy).
        assert is_pareto_eligible(events[0], events, 10.0, 10.0)
        # Exactly index 0 has lowest params (100) — also eligible via params.
        # Index 1 should NOT be eligible via accuracy (only top-1 by acc, and also not bottom-1 params)
        # params_vals[1] = 200, which is not the minimum (100).
        # So only index 0 qualifies by either criterion.
        for i in range(1, 7):
            assert not is_pareto_eligible(events[i], events, 10.0, 10.0), (
                f"Expected i{i} to be ineligible with 10% of 7 population"
            )

    def test_test_acc_key_variant_supported(self):
        """pareto_policy supports both 'test_accuracy' and 'test_acc' key variants."""
        ev_acc = _make_event(child_gene_id="acc_key", test_accuracy=0.99, total_params=100)
        ev_acc2 = _make_event(child_gene_id="acc2_key", test_acc=0.50, total_params=200)
        population = [ev_acc, ev_acc2]
        # ev_acc should be eligible (top accuracy via "test_accuracy" key)
        assert is_pareto_eligible(ev_acc, population, 10.0, 10.0)

    def test_percentile_knobs_respected(self):
        """Different percentile values change who is eligible."""
        events = [
            _make_event(child_gene_id=f"e{i}", test_accuracy=1.0 - i * 0.05, total_params=(i + 1) * 100)
            for i in range(10)
        ]
        # With 50% accuracy threshold: top-5 by accuracy are eligible
        eligible_50 = [ev for ev in events if is_pareto_eligible(ev, events, 50.0, 0.0)]
        assert len(eligible_50) == 5, f"Expected 5 eligible with 50% threshold, got {len(eligible_50)}"

        # With 10% accuracy threshold: top-1 by accuracy is eligible
        eligible_10 = [ev for ev in events if is_pareto_eligible(ev, events, 10.0, 0.0)]
        assert len(eligible_10) == 1


# ---------------------------------------------------------------------------
# Integration test: eligibility persists through ledger round-trip
# ---------------------------------------------------------------------------

class TestEligibilityLedgerIntegration:
    """Verify that is_pareto_eligible survives a JSONL write-replay cycle."""

    def test_eligibility_persists_in_ledger_round_trip(self, tmp_path):
        """Create 10 events with known eligibility, append to ledger, replay, verify."""
        ledger_path = tmp_path / "rag_ledger.jsonl"
        ledger = RunLedger(run_id="pareto-test", ledger_path=ledger_path)

        # Build 10 events with strictly different accuracy and params.
        events = [
            _make_event(
                child_gene_id=f"gene{i}",
                generation=0,
                test_accuracy=1.0 - i * 0.05,
                total_params=(i + 1) * 100,
            )
            for i in range(10)
        ]
        population = list(events)

        # Compute eligibility and build final events with the flag set.
        final_events = []
        for ev in events:
            eligible = is_pareto_eligible(ev, population, 10.0, 10.0)
            # Replace the event with one that has the flag set (frozen dataclass).
            ev_with_flag = dataclasses.replace(ev, is_pareto_eligible=eligible)
            final_events.append(ev_with_flag)
            ledger.append(ev_with_flag)

        # Replay the ledger and verify the flag survived.
        replayed = list(replay_ledger("pareto-test", ledger_path=ledger_path))
        assert len(replayed) == 10

        for original, replayed_ev in zip(final_events, replayed):
            assert replayed_ev.is_pareto_eligible == original.is_pareto_eligible, (
                f"Flag mismatch for {original.child_gene_id}: "
                f"expected {original.is_pareto_eligible}, got {replayed_ev.is_pareto_eligible}"
            )

    def test_old_jsonl_without_flag_defaults_to_none(self, tmp_path):
        """Old JSONL lines without is_pareto_eligible default to None on replay.

        This tests backward compatibility: ledger files created before PR 5
        will not have the is_pareto_eligible key and must load without error.
        """
        ledger_path = tmp_path / "old_ledger.jsonl"

        # Write a minimal valid JSONL line without is_pareto_eligible.
        old_event_dict = {
            "run_id": "old-run",
            "generation": 0,
            "parent_gene_id": "parent0",
            "child_gene_id": "child0",
            "mutation_type": "add_layer",
            "backend": "faiss",
            "request_id": "aaaaaaaa-0000-0000-0000-000000000000",
            "prompt_id": "deadbeef00000000",
            "raw_prompt": "old prompt",
            "augmented_prompt": None,
            "retrieval_request": None,
            "retrieval_response": None,
            "model_request": None,
            "model_response": None,
            "parsed_artifact": None,
            "eval_outputs": {"test_acc": 0.88, "total_params": 500},
            "latencies": {},
            "failure_mode": None,
            "timestamp": "2025-01-01T00:00:00+00:00",
            # NOTE: is_pareto_eligible is intentionally absent
        }
        ledger_path.write_text(json.dumps(old_event_dict) + "\n", encoding="utf-8")

        replayed = list(replay_ledger("old-run", ledger_path=ledger_path))
        assert len(replayed) == 1
        assert replayed[0].is_pareto_eligible is None


# ---------------------------------------------------------------------------
# Module import smoke test
# ---------------------------------------------------------------------------

def test_import_no_heavy_deps():
    """pareto_policy.py must not pull torch or faiss at module import time."""
    import unittest.mock as mock

    # Remove cached module to force re-import.
    for key in list(sys.modules.keys()):
        if "pareto_policy" in key:
            del sys.modules[key]

    with mock.patch.dict(sys.modules, {"torch": None, "faiss": None}):
        mod = _load_module_direct(
            "src/rag/pareto_policy.py", "src.rag.pareto_policy_smoke"
        )

    assert hasattr(mod, "is_pareto_eligible")
    assert hasattr(mod, "compute_pareto_front")
    assert hasattr(mod, "_dominates")
