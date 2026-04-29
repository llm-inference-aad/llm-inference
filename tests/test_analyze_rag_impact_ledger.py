"""Tests for the ledger-aware extension of scripts/analyze_rag_impact.py.

Verifies that:
- When rag_ledger.jsonl is present, load_ledger_stats returns richer metrics.
- When only rag_metrics.jsonl is present, load_ledger_stats returns None
  (caller falls back to legacy load_rag_usage).
- main() integrates both paths without crashing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _import_analyze():
    """Import analyze_rag_impact as a module."""
    spec_name = "analyze_rag_impact"
    if spec_name in sys.modules:
        return sys.modules[spec_name]
    spec = importlib.util.spec_from_file_location(
        spec_name, SCRIPTS_DIR / "analyze_rag_impact.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_event_dict(
    *,
    run_id: str,
    child_gene_id: str,
    test_accuracy: float,
    total_params: float,
    is_pareto_eligible: bool = False,
    failure_mode: str | None = None,
    backend: str = "faiss",
) -> dict:
    """Build a minimal MutationEvent dict for writing to rag_ledger.jsonl."""
    import hashlib
    raw_prompt = f"prompt_{run_id}_{child_gene_id}"
    prompt_id = hashlib.sha256(raw_prompt.encode()).hexdigest()[:16]
    return {
        "run_id": run_id,
        "generation": 1,
        "parent_gene_id": "parent_0",
        "child_gene_id": child_gene_id,
        "mutation_type": "mutate",
        "backend": backend,
        "request_id": f"req-{child_gene_id}",
        "prompt_id": prompt_id,
        "raw_prompt": raw_prompt,
        "augmented_prompt": f"augmented for {child_gene_id}",
        "retrieval_request": {"query": "test query"},
        "retrieval_response": {"blocks": []},
        "model_request": None,
        "model_response": None,
        "parsed_artifact": None,
        "eval_outputs": {
            "test_accuracy": test_accuracy,
            "total_params": total_params,
        },
        "is_pareto_eligible": is_pareto_eligible,
        "latencies": {"retrieve_ms": 10.0},
        "failure_mode": failure_mode,
        "timestamp": "2026-04-20T12:00:00+00:00",
    }


@pytest.fixture()
def run_dir_with_ledger(tmp_path: Path):
    """Create a run directory with a rag_ledger.jsonl and no rag_metrics.jsonl."""
    run_id = "test_run_ledger"
    run_dir = tmp_path / "runs" / run_id
    (run_dir / "metrics").mkdir(parents=True)

    ledger_path = run_dir / "metrics" / "rag_ledger.jsonl"
    events = [
        _make_event_dict(
            run_id=run_id,
            child_gene_id=f"gene_{i}",
            test_accuracy=0.7 + i * 0.05,
            total_params=1_000_000 * (5 - i),
            is_pareto_eligible=(i >= 3),
            backend="faiss",
        )
        for i in range(5)
    ]
    # Add one failed event.
    events.append(
        _make_event_dict(
            run_id=run_id,
            child_gene_id="gene_failed",
            test_accuracy=0.0,
            total_params=0.0,
            failure_mode="syntax_error",
            backend="faiss",
        )
    )
    with ledger_path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    return run_dir, run_id


@pytest.fixture()
def run_dir_with_legacy_only(tmp_path: Path):
    """Create a run directory with rag_metrics.jsonl only (no ledger)."""
    run_id = "test_run_legacy"
    run_dir = tmp_path / "runs" / run_id
    (run_dir / "metrics").mkdir(parents=True)

    rag_metrics_path = run_dir / "metrics" / "rag_metrics.jsonl"
    events = [
        {
            "event_type": "rag_context_built",
            "run_id": run_id,
            "gene_id": f"gene_{i}",
            "retrieved_code_n": 3,
            "retrieved_text_n": 2,
            "context_words_code": 100,
            "context_words_text": 80,
            "selected_text_api_n": 1,
            "selected_text_pdf_n": 1,
            "selected_text_other_n": 0,
        }
        for i in range(3)
    ]
    with rag_metrics_path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    return run_dir, run_id


@pytest.fixture()
def run_dir_with_both(tmp_path: Path, run_dir_with_ledger, run_dir_with_legacy_only):
    """Create a run directory with both ledger and legacy metrics.

    When both are present, the analyzer should prefer the ledger.
    We use the run_dir_with_ledger fixture and add a rag_metrics.jsonl to it.
    """
    run_dir, run_id = run_dir_with_ledger

    # Add legacy rag_metrics.jsonl alongside the ledger.
    rag_metrics_path = run_dir / "metrics" / "rag_metrics.jsonl"
    rag_metrics_path.write_text(
        json.dumps({"event_type": "rag_context_built", "run_id": run_id, "retrieved_code_n": 1}) + "\n"
    )

    return run_dir, run_id


# ---------------------------------------------------------------------------
# Tests: load_ledger_stats
# ---------------------------------------------------------------------------


class TestLoadLedgerStats:
    def test_returns_none_when_no_ledger(self, run_dir_with_legacy_only):
        """Returns None when rag_ledger.jsonl is absent."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_legacy_only

        result = mod.load_ledger_stats(run_dir, run_id)
        assert result is None

    def test_returns_dict_when_ledger_present(self, run_dir_with_ledger):
        """Returns a non-None dict when rag_ledger.jsonl is present."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        result = mod.load_ledger_stats(run_dir, run_id)
        assert result is not None
        assert isinstance(result, dict)

    def test_correct_event_count(self, run_dir_with_ledger):
        """ledger_events_total matches the number of events written."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        result = mod.load_ledger_stats(run_dir, run_id)
        assert result is not None
        # We wrote 5 normal + 1 failed = 6 total.
        assert result["ledger_events_total"] == 6

    def test_eval_event_count(self, run_dir_with_ledger):
        """ledger_eval_events counts events with non-null eval_outputs."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        result = mod.load_ledger_stats(run_dir, run_id)
        # All 6 events have eval_outputs (we set test_accuracy=0.0 for failed).
        assert result["ledger_eval_events"] == 6

    def test_pareto_eligible_count(self, run_dir_with_ledger):
        """ledger_pareto_eligible matches is_pareto_eligible=True events."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        result = mod.load_ledger_stats(run_dir, run_id)
        # Events i=3,4 have is_pareto_eligible=True.
        assert result["ledger_pareto_eligible"] == 2

    def test_backends_detected(self, run_dir_with_ledger):
        """ledger_backends contains the backend names used."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        result = mod.load_ledger_stats(run_dir, run_id)
        assert "faiss" in result["ledger_backends"]

    def test_failure_modes_aggregated(self, run_dir_with_ledger):
        """ledger_failure_modes contains the syntax_error we injected."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        result = mod.load_ledger_stats(run_dir, run_id)
        assert "syntax_error" in result["ledger_failure_modes"]
        assert result["ledger_failure_modes"]["syntax_error"] == 1

    def test_best_accuracy_computed(self, run_dir_with_ledger):
        """ledger_best_test_accuracy is the max test_accuracy across eval events."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        result = mod.load_ledger_stats(run_dir, run_id)
        # Events: 0.70, 0.75, 0.80, 0.85, 0.90, 0.0 (failed)
        assert result["ledger_best_test_accuracy"] == pytest.approx(0.90)

    def test_retrieval_events_counted(self, run_dir_with_ledger):
        """ledger_retrieval_events counts events with non-null retrieval_response."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        result = mod.load_ledger_stats(run_dir, run_id)
        # All 6 events have retrieval_response set.
        assert result["ledger_retrieval_events"] == 6


# ---------------------------------------------------------------------------
# Tests: main() integration
# ---------------------------------------------------------------------------


class TestMainIntegration:
    def test_main_with_ledger_adds_ledger_keys(self, run_dir_with_ledger, tmp_path, monkeypatch):
        """main() report includes ledger_* keys when rag_ledger.jsonl is present."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_ledger

        # Add a minimal results dir so load_results doesn't fail.
        (run_dir / "results").mkdir(exist_ok=True)

        out_path = tmp_path / "report.json"
        monkeypatch.setattr(
            sys, "argv",
            ["analyze_rag_impact", "--run-dir", str(run_dir), "--output", str(out_path)],
        )
        mod.main()

        assert out_path.exists()
        report = json.loads(out_path.read_text())
        assert "ledger_events_total" in report, "Ledger keys missing from report"
        assert "ledger_pareto_eligible" in report

    def test_main_with_legacy_only_no_ledger_keys(self, run_dir_with_legacy_only, tmp_path, monkeypatch):
        """main() report does not include ledger_* keys when only rag_metrics.jsonl exists."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_legacy_only

        (run_dir / "results").mkdir(exist_ok=True)

        out_path = tmp_path / "report_legacy.json"
        monkeypatch.setattr(
            sys, "argv",
            ["analyze_rag_impact", "--run-dir", str(run_dir), "--output", str(out_path)],
        )
        mod.main()

        assert out_path.exists()
        report = json.loads(out_path.read_text())
        # When no ledger is present, load_ledger_stats returns None,
        # so merging {} leaves ledger_* keys absent.
        assert "ledger_events_total" not in report

    def test_main_with_both_prefers_ledger(self, run_dir_with_both, tmp_path, monkeypatch):
        """When both ledger and legacy file exist, ledger keys appear in report."""
        mod = _import_analyze()
        run_dir, run_id = run_dir_with_both

        (run_dir / "results").mkdir(exist_ok=True)

        out_path = tmp_path / "report_both.json"
        monkeypatch.setattr(
            sys, "argv",
            ["analyze_rag_impact", "--run-dir", str(run_dir), "--output", str(out_path)],
        )
        mod.main()

        assert out_path.exists()
        report = json.loads(out_path.read_text())
        assert "ledger_events_total" in report
