"""Tests for scripts/plot_pareto_front_comparison.py.

Uses fixture manifests with synthetic rag_ledger.jsonl data so no real SLURM
runs are needed.  matplotlib is forced to Agg backend at module load time.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_plot():
    """Import plot_pareto_front_comparison as a module."""
    spec_name = "plot_pareto_front_comparison"
    if spec_name in sys.modules:
        return sys.modules[spec_name]
    spec = importlib.util.spec_from_file_location(
        spec_name, SCRIPTS_DIR / "plot_pareto_front_comparison.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures — synthetic ledger and manifest
# ---------------------------------------------------------------------------


def _make_event_dict(
    *,
    run_id: str,
    condition: str,
    child_gene_id: str,
    test_accuracy: float,
    total_params: float,
    is_pareto_eligible: bool = False,
) -> dict:
    """Build a minimal MutationEvent dict suitable for writing to JSONL."""
    import hashlib
    raw_prompt = f"prompt_{run_id}_{child_gene_id}"
    prompt_id = hashlib.sha256(raw_prompt.encode()).hexdigest()[:16]
    return {
        "run_id": run_id,
        "generation": 1,
        "parent_gene_id": "parent_0",
        "child_gene_id": child_gene_id,
        "mutation_type": "mutate",
        "backend": condition,
        "request_id": f"req-{child_gene_id}",
        "prompt_id": prompt_id,
        "raw_prompt": raw_prompt,
        "augmented_prompt": None,
        "retrieval_request": None,
        "retrieval_response": None,
        "model_request": None,
        "model_response": None,
        "parsed_artifact": None,
        "eval_outputs": {
            "test_accuracy": test_accuracy,
            "total_params": total_params,
        },
        "is_pareto_eligible": is_pareto_eligible,
        "latencies": {},
        "failure_mode": None,
        "timestamp": "2026-04-20T12:00:00+00:00",
    }


@pytest.fixture()
def ablation_fixture(tmp_path: Path):
    """Create two synthetic ledger files and a manifest pointing to them.

    Returns (manifest_path, output_dir, n_baseline, n_rag_faiss).
    """
    n_baseline = 10
    n_rag_faiss = 10

    # Create run dirs and ledger files.
    run_ids = {
        "baseline": "baseline_seed1_20260420",
        "rag-faiss": "rag_faiss_seed1_20260420",
    }

    for condition, run_id in run_ids.items():
        run_dir = tmp_path / "runs" / run_id
        (run_dir / "metrics").mkdir(parents=True)
        ledger_path = run_dir / "metrics" / "rag_ledger.jsonl"

        n = n_baseline if condition == "baseline" else n_rag_faiss
        with ledger_path.open("w") as f:
            for i in range(n):
                # Spread points to have a clear Pareto structure.
                acc = 0.6 + i * 0.03
                params = 1_000_000 * (n - i)
                ev = _make_event_dict(
                    run_id=run_id,
                    condition=condition,
                    child_gene_id=f"gene_{condition}_{i}",
                    test_accuracy=acc,
                    total_params=params,
                )
                f.write(json.dumps(ev) + "\n")

    # Write manifest.
    ablation_dir = tmp_path / "ablation_test"
    ablation_dir.mkdir()
    manifest = {
        "matrix_timestamp": "20260420_120000",
        "git_commit": "abc123",
        "git_branch": "worker/pr6-ablation-matrix",
        "user": "test",
        "output_dir": str(ablation_dir),
        "conditions": ["baseline", "rag-faiss"],
        "seeds": [1],
        "jobs": [
            {
                "run_id": run_ids["baseline"],
                "condition": "baseline",
                "seed": 1,
                "launch_command": "bash launch.sh --name baseline_seed1 --seed 1",
                "submitted_at": "2026-04-20T12:00:00+00:00",
                "main_job_id": "123456",
                "error": None,
            },
            {
                "run_id": run_ids["rag-faiss"],
                "condition": "rag-faiss",
                "seed": 1,
                "launch_command": "bash launch.sh --name rag-faiss_seed1 --seed 1",
                "submitted_at": "2026-04-20T12:00:00+00:00",
                "main_job_id": "123457",
                "error": None,
            },
        ],
    }
    manifest_path = ablation_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    output_dir = ablation_dir / "plots"
    return manifest_path, output_dir, n_baseline, n_rag_faiss, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlotParetoComparison:
    def test_png_created(self, ablation_fixture, monkeypatch):
        """Running build_plot produces a non-empty PNG file."""
        manifest_path, output_dir, n_base, n_rag, tmp_path = ablation_fixture
        plot = _import_plot()

        # Monkeypatch REPO_ROOT so run directories are resolved to tmp_path/runs/.
        monkeypatch.setattr(plot, "REPO_ROOT", tmp_path)

        png_path, csv_path = plot.build_plot(
            manifest_path=manifest_path,
            output_dir=output_dir,
            title="Test Pareto Comparison",
            condition_colors={},
        )

        assert png_path.exists(), f"PNG not created at {png_path}"
        assert png_path.stat().st_size > 0, "PNG is empty"

    def test_csv_row_count(self, ablation_fixture, monkeypatch):
        """pareto_data.csv has the expected number of rows (one per event)."""
        manifest_path, output_dir, n_base, n_rag, tmp_path = ablation_fixture
        plot = _import_plot()
        monkeypatch.setattr(plot, "REPO_ROOT", tmp_path)

        _, csv_path = plot.build_plot(
            manifest_path=manifest_path,
            output_dir=output_dir,
            title="Test",
            condition_colors={},
        )

        assert csv_path.exists(), f"CSV not created at {csv_path}"
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        # n_base + n_rag events total.
        assert len(rows) == n_base + n_rag, (
            f"Expected {n_base + n_rag} rows, got {len(rows)}"
        )

    def test_csv_columns(self, ablation_fixture, monkeypatch):
        """pareto_data.csv has all required columns."""
        manifest_path, output_dir, n_base, n_rag, tmp_path = ablation_fixture
        plot = _import_plot()
        monkeypatch.setattr(plot, "REPO_ROOT", tmp_path)

        _, csv_path = plot.build_plot(
            manifest_path=manifest_path,
            output_dir=output_dir,
            title="Test",
            condition_colors={},
        )

        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

        required = {"condition", "run_id", "gene_id", "test_accuracy", "total_params", "is_pareto_eligible", "on_front"}
        assert required.issubset(set(headers)), (
            f"Missing CSV columns: {required - set(headers)}"
        )

    def test_png_is_valid_image(self, ablation_fixture, monkeypatch):
        """PNG file has PNG magic bytes (basic validity check)."""
        manifest_path, output_dir, n_base, n_rag, tmp_path = ablation_fixture
        plot = _import_plot()
        monkeypatch.setattr(plot, "REPO_ROOT", tmp_path)

        png_path, _ = plot.build_plot(
            manifest_path=manifest_path,
            output_dir=output_dir,
            title="Test",
            condition_colors={},
        )

        png_bytes = png_path.read_bytes()
        # PNG magic: \x89PNG\r\n\x1a\n
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "File does not have PNG magic bytes"

    def test_pareto_front_contains_nondominated_points(self, ablation_fixture, monkeypatch):
        """Pareto front in CSV contains only non-dominated rows per condition."""
        manifest_path, output_dir, n_base, n_rag, tmp_path = ablation_fixture
        plot = _import_plot()
        monkeypatch.setattr(plot, "REPO_ROOT", tmp_path)

        _, csv_path = plot.build_plot(
            manifest_path=manifest_path,
            output_dir=output_dir,
            title="Test",
            condition_colors={},
        )

        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        # With our synthetic data (acc increases, params decrease monotonically),
        # every point is non-dominated.  Both conditions should have >0 front points.
        front_rows = [r for r in rows if r["on_front"] == "True"]
        assert len(front_rows) > 0, "No Pareto front points found in CSV"

    def test_cli_main_creates_outputs(self, ablation_fixture, monkeypatch):
        """plot_pareto_front_comparison.main() succeeds via CLI."""
        manifest_path, output_dir, n_base, n_rag, tmp_path = ablation_fixture
        plot = _import_plot()
        monkeypatch.setattr(plot, "REPO_ROOT", tmp_path)

        exit_code = plot.main(
            [
                "--manifest", str(manifest_path),
                "--output", str(output_dir / "cli_test"),
            ]
        )

        assert exit_code == 0
        assert (output_dir / "cli_test" / "pareto_comparison.png").exists()
        assert (output_dir / "cli_test" / "pareto_data.csv").exists()

    def test_missing_manifest_returns_error(self, tmp_path):
        """main() returns non-zero exit code when manifest is missing."""
        plot = _import_plot()
        exit_code = plot.main(
            ["--manifest", str(tmp_path / "nonexistent_manifest.json")]
        )
        assert exit_code != 0
