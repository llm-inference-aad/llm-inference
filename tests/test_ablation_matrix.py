"""Tests for scripts/run_rag_ablation_matrix.py.

All tests use --dry-run; no real sbatch or launch.sh invocations occur.
"""

from __future__ import annotations

import importlib
import json
import sys
from io import StringIO
from pathlib import Path
from typing import List
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers to import the script as a module
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _import_matrix():
    """Import run_rag_ablation_matrix as a module (handles re-import)."""
    spec_name = "run_rag_ablation_matrix"
    if spec_name in sys.modules:
        return sys.modules[spec_name]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        spec_name, SCRIPTS_DIR / "run_rag_ablation_matrix.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDryRun:
    """--dry-run generates correct commands without submitting jobs."""

    def test_default_matrix_job_count(self, tmp_path, capsys):
        """Default 2 conditions x 3 seeds = 6 job records in manifest."""
        mat = _import_matrix()

        with mock.patch.object(
            Path, "exists", return_value=False  # force fresh output dir
        ):
            exit_code = mat.main(
                ["--dry-run", "--output-dir", str(tmp_path / "ablation_test")]
            )

        assert exit_code == 0

        manifest_path = tmp_path / "ablation_test" / "manifest.json"
        assert manifest_path.exists(), "Manifest was not written in dry-run mode"

        manifest = json.loads(manifest_path.read_text())
        jobs = manifest["jobs"]
        assert len(jobs) == 6, f"Expected 6 jobs (2 conds x 3 seeds), got {len(jobs)}"

    def test_correct_conditions_and_seeds(self, tmp_path):
        """Each job record has the expected condition and seed."""
        mat = _import_matrix()

        exit_code = mat.main(
            ["--dry-run", "--output-dir", str(tmp_path / "ablation_test")]
        )
        assert exit_code == 0

        manifest = json.loads((tmp_path / "ablation_test" / "manifest.json").read_text())
        jobs = manifest["jobs"]

        conditions_seen = {j["condition"] for j in jobs}
        seeds_seen = {j["seed"] for j in jobs}

        assert conditions_seen == {"baseline", "rag-faiss"}
        assert seeds_seen == {1, 2, 3}

    def test_launch_command_contains_expected_args(self, tmp_path):
        """Each job's launch_command contains the correct env vars."""
        mat = _import_matrix()

        mat.main(["--dry-run", "--output-dir", str(tmp_path / "ablation_test")])
        manifest = json.loads((tmp_path / "ablation_test" / "manifest.json").read_text())
        jobs = manifest["jobs"]

        for job in jobs:
            cmd = job["launch_command"]
            cond = job["condition"]
            seed = job["seed"]

            # Must contain the seed.
            assert f"EXPERIMENT_SEED={seed}" in cmd, (
                f"Job {cond}/seed{seed}: missing EXPERIMENT_SEED in: {cmd}"
            )

            # Must contain launch.sh invocation.
            assert "launch.sh" in cmd, f"Job {cond}/seed{seed}: no launch.sh in: {cmd}"

            # Baseline must have RAG_ENABLED=false.
            if cond == "baseline":
                assert "RAG_ENABLED=false" in cmd, (
                    f"Baseline job: expected RAG_ENABLED=false in: {cmd}"
                )

            # rag-faiss must have RAG_ENABLED=true.
            if cond == "rag-faiss":
                assert "RAG_ENABLED=true" in cmd, (
                    f"rag-faiss job: expected RAG_ENABLED=true in: {cmd}"
                )

    def test_launch_command_contains_name_arg(self, tmp_path):
        """launch.sh command includes --name <condition>_seed<N>."""
        mat = _import_matrix()

        mat.main(["--dry-run", "--output-dir", str(tmp_path / "ablation_test")])
        manifest = json.loads((tmp_path / "ablation_test" / "manifest.json").read_text())
        jobs = manifest["jobs"]

        for job in jobs:
            cmd = job["launch_command"]
            cond = job["condition"]
            seed = job["seed"]
            assert f"--name {cond}_seed{seed}" in cmd, (
                f"Missing --name {cond}_seed{seed} in: {cmd}"
            )

    def test_single_seed_single_condition(self, tmp_path):
        """--seeds 7 --conditions baseline produces exactly 1 job."""
        mat = _import_matrix()

        exit_code = mat.main(
            [
                "--dry-run",
                "--seeds", "7",
                "--conditions", "baseline",
                "--output-dir", str(tmp_path / "ablation_single"),
            ]
        )
        assert exit_code == 0

        manifest = json.loads(
            (tmp_path / "ablation_single" / "manifest.json").read_text()
        )
        assert len(manifest["jobs"]) == 1
        assert manifest["jobs"][0]["condition"] == "baseline"
        assert manifest["jobs"][0]["seed"] == 7
        assert manifest["jobs"][0]["main_job_id"] is None  # dry run

    def test_dry_run_manifest_schema(self, tmp_path):
        """Manifest has the required top-level keys."""
        mat = _import_matrix()

        mat.main(["--dry-run", "--output-dir", str(tmp_path / "ablation_schema")])
        manifest = json.loads(
            (tmp_path / "ablation_schema" / "manifest.json").read_text()
        )

        required_keys = {
            "matrix_timestamp",
            "git_commit",
            "git_branch",
            "user",
            "output_dir",
            "conditions",
            "seeds",
            "jobs",
        }
        assert required_keys.issubset(set(manifest.keys())), (
            f"Missing keys: {required_keys - set(manifest.keys())}"
        )

    def test_dry_run_job_schema(self, tmp_path):
        """Each job record has the required fields; main_job_id and submitted_at are None."""
        mat = _import_matrix()

        mat.main(["--dry-run", "--output-dir", str(tmp_path / "ablation_job_schema")])
        manifest = json.loads(
            (tmp_path / "ablation_job_schema" / "manifest.json").read_text()
        )
        required_job_keys = {"run_id", "condition", "seed", "launch_command", "submitted_at", "main_job_id"}
        for job in manifest["jobs"]:
            assert required_job_keys.issubset(set(job.keys())), (
                f"Missing job keys: {required_job_keys - set(job.keys())}"
            )
            assert job["submitted_at"] is None
            assert job["main_job_id"] is None

    def test_rerun_appends_suffix(self, tmp_path):
        """Running twice with the same output dir appends _rerun_N suffix."""
        mat = _import_matrix()

        out_dir = str(tmp_path / "ablation_fixed")

        mat.main(["--dry-run", "--output-dir", out_dir])
        assert (tmp_path / "ablation_fixed").exists()

        # Second run: directory already exists, must use _rerun_1.
        mat.main(["--dry-run", "--output-dir", out_dir])
        assert (tmp_path / "ablation_fixed_rerun_1").exists()

        # Third run: _rerun_1 exists, must use _rerun_2.
        mat.main(["--dry-run", "--output-dir", out_dir])
        assert (tmp_path / "ablation_fixed_rerun_2").exists()

    def test_population_and_generation_flags(self, tmp_path):
        """--population-size and --num-generations appear in command strings."""
        mat = _import_matrix()

        mat.main(
            [
                "--dry-run",
                "--seeds", "1",
                "--conditions", "baseline",
                "--population-size", "8",
                "--num-generations", "3",
                "--output-dir", str(tmp_path / "ablation_sizing"),
            ]
        )
        manifest = json.loads(
            (tmp_path / "ablation_sizing" / "manifest.json").read_text()
        )
        cmd = manifest["jobs"][0]["launch_command"]
        assert "POPULATION_SIZE=8" in cmd
        assert "NUM_GENERATIONS=3" in cmd
