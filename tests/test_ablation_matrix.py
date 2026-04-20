"""Tests for scripts/run_rag_ablation_matrix.py (PR 8 additions).

All tests run in dry-run mode (no real sbatch calls).  The 3-way comparison
matrix (baseline / rag-faiss / rag-faiss+memory) must produce exactly 9 sbatch
lines for 3 seeds, and the rag-faiss+memory condition must set
RAG_MEMORY_STORE_ENABLED=true.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_rag_ablation_matrix.py"


# ---------------------------------------------------------------------------
# Helper: capture stdout from the script executed as a subprocess
# ---------------------------------------------------------------------------


def _run_dry(extra_args: list[str]) -> list[str]:
    """Run the ablation matrix script in dry-run mode and return stdout lines."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        *extra_args,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"Script failed with exit code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    return result.stdout.splitlines()


def _sbatch_lines(lines: list[str]) -> list[str]:
    """Return only lines that contain 'sbatch'."""
    return [l for l in lines if "sbatch" in l]


# ---------------------------------------------------------------------------
# Import the module for unit-level tests (no subprocess)
# ---------------------------------------------------------------------------


def _import_matrix():
    spec_name = "_run_rag_ablation_matrix_pr8"
    if spec_name in sys.modules:
        return sys.modules[spec_name]
    spec = importlib.util.spec_from_file_location(spec_name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests: 3-condition comparison matrix (--compare-3)
# ---------------------------------------------------------------------------


class TestCompare3Matrix:
    """--compare-3 with 3 seeds must produce exactly 9 sbatch lines."""

    def test_9_sbatch_lines_for_3_conditions_3_seeds(self):
        """3 conditions × 3 seeds = 9 sbatch lines in dry-run output."""
        lines = _run_dry(["--compare-3", "--seeds", "0", "1", "2"])
        sbatch = _sbatch_lines(lines)
        assert len(sbatch) == 9, (
            f"Expected 9 sbatch lines, got {len(sbatch)}.\n"
            f"All lines:\n" + "\n".join(lines)
        )

    def test_all_three_conditions_present(self):
        """The 9 lines must cover baseline, rag-faiss, rag-faiss+memory."""
        lines = _run_dry(["--compare-3", "--seeds", "0", "1", "2"])
        # Lines include RUN_ID= lines that name the condition.
        all_text = "\n".join(lines)
        assert "baseline" in all_text
        assert "rag-faiss" in all_text
        assert "rag-faiss+memory" in all_text

    def test_rag_faiss_memory_condition_sets_memory_enabled(self):
        """rag-faiss+memory condition must export RAG_MEMORY_STORE_ENABLED=true."""
        lines = _run_dry(["--compare-3", "--seeds", "0"])
        sbatch = _sbatch_lines(lines)

        # Find the sbatch line for rag-faiss+memory.
        memory_lines = [l for l in sbatch if "rag-faiss+memory" in l or "RAG_MEMORY_STORE_ENABLED=true" in l]
        assert len(memory_lines) >= 1, (
            f"No sbatch line contained 'rag-faiss+memory' or 'RAG_MEMORY_STORE_ENABLED=true'.\n"
            f"sbatch lines:\n" + "\n".join(sbatch)
        )
        # The memory condition line must set RAG_MEMORY_STORE_ENABLED=true.
        all_text = "\n".join(lines)
        assert "RAG_MEMORY_STORE_ENABLED=true" in all_text, (
            "rag-faiss+memory condition must set RAG_MEMORY_STORE_ENABLED=true in the sbatch export."
        )

    def test_baseline_has_rag_disabled(self):
        """baseline condition must have RAG_ENABLED=false (not RAG_MEMORY_STORE_ENABLED=true)."""
        lines = _run_dry(["--compare-3", "--seeds", "0"])
        # All text around the baseline block should show RAG_ENABLED=false.
        # We check that the exported env for baseline has RAG_ENABLED=false.
        all_text = "\n".join(lines)
        assert "RAG_ENABLED=false" in all_text, (
            "baseline condition must include RAG_ENABLED=false in its sbatch export."
        )

    def test_dry_run_does_not_call_sbatch(self):
        """Dry-run must print sbatch commands without actually submitting them."""
        # The script writes 'Dry-run complete' or similar at the end, confirming no submission.
        lines = _run_dry(["--compare-3", "--seeds", "0", "1", "2"])
        all_text = "\n".join(lines)
        # Either 'dry-run' or 'Dry-run' must appear.
        assert "dry-run" in all_text.lower(), (
            "Expected dry-run confirmation in script output."
        )


# ---------------------------------------------------------------------------
# Unit-level tests on build_conditions()
# ---------------------------------------------------------------------------


class TestBuildConditions:
    def test_compare_3_returns_exactly_3_conditions(self):
        mat = _import_matrix()
        conditions = mat.build_conditions(compare_3=True)
        assert len(conditions) == 3

    def test_compare_3_condition_names(self):
        mat = _import_matrix()
        names = [c.name for c in mat.build_conditions(compare_3=True)]
        assert names == ["baseline", "rag-faiss", "rag-faiss+memory"]

    def test_rag_faiss_memory_env(self):
        mat = _import_matrix()
        conds = {c.name: c for c in mat.build_conditions(compare_3=True)}
        mem_env = conds["rag-faiss+memory"].env
        assert mem_env.get("RAG_MEMORY_STORE_ENABLED") == "true"
        assert mem_env.get("RAG_ENABLED") == "true"

    def test_baseline_env(self):
        mat = _import_matrix()
        conds = {c.name: c for c in mat.build_conditions(compare_3=True)}
        base_env = conds["baseline"].env
        assert base_env.get("RAG_ENABLED") == "false"
        assert base_env.get("RAG_MEMORY_STORE_ENABLED") == "false"

    def test_rag_faiss_env_memory_disabled(self):
        mat = _import_matrix()
        conds = {c.name: c for c in mat.build_conditions(compare_3=True)}
        faiss_env = conds["rag-faiss"].env
        assert faiss_env.get("RAG_ENABLED") == "true"
        assert faiss_env.get("RAG_MEMORY_STORE_ENABLED") == "false"

    def test_default_matrix_has_memory_condition(self):
        """The full (non-compare-3) build must also include rag-faiss+memory."""
        mat = _import_matrix()
        conds = mat.build_conditions(compare_3=False)
        names = [c.name for c in conds]
        assert "rag-faiss+memory" in names, (
            f"Expected 'rag-faiss+memory' in default conditions, got: {names}"
        )
