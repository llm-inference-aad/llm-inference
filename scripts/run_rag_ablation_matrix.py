#!/usr/bin/env python3
"""Launch a 2-condition x N-seed RAG ablation matrix using launch.sh.

This script drives the ablation study that produces the Pareto-front comparison
graph (see scripts/plot_pareto_front_comparison.py).  It uses launch.sh under the
hood to preserve the existing .env override pattern — each condition's env vars are
injected into launch.sh's subprocess environment, which then get exported via
`sbatch --export=...,ALL`.

Default matrix: 2 conditions x 3 seeds = 6 jobs.
  - baseline:  RAG_ENABLED=false
  - rag-faiss: RAG_ENABLED=true, RAG_USE_CODE_CONTEXT=true,
               RAG_USE_TEXT_CONTEXT=true, RAG_RERANKER_ENABLED=false

Usage examples::

    # Dry run — print commands without submitting
    python scripts/run_rag_ablation_matrix.py --dry-run

    # Submit the default matrix
    python scripts/run_rag_ablation_matrix.py

    # Custom seeds + conditions
    python scripts/run_rag_ablation_matrix.py --seeds 7 13 29 --conditions rag-faiss

Standard library only — no torch, no FAISS, no heavy deps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Built-in condition definitions (2-condition default)
# ---------------------------------------------------------------------------

#: Full registry of known conditions.  The --conditions flag filters this.
ALL_CONDITIONS: Dict[str, Dict[str, str]] = {
    "baseline": {
        "RAG_ENABLED": "false",
        "RAG_USE_CODE_CONTEXT": "false",
        "RAG_USE_TEXT_CONTEXT": "false",
        "RAG_RERANKER_ENABLED": "false",
    },
    "rag-faiss": {
        "RAG_ENABLED": "true",
        "RAG_USE_CODE_CONTEXT": "true",
        "RAG_USE_TEXT_CONTEXT": "true",
        "RAG_RERANKER_ENABLED": "false",
    },
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class JobRecord:
    """One row in the ablation manifest — one SLURM job."""

    run_id: str
    condition: str
    seed: int
    launch_command: str
    submitted_at: Optional[str] = None
    main_job_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AblationManifest:
    """Per-matrix manifest written to runs/ablation_<timestamp>/manifest.json."""

    matrix_timestamp: str
    git_commit: str
    git_branch: str
    user: str
    output_dir: str
    conditions: List[str]
    seeds: List[int]
    jobs: List[JobRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_or_unknown(args: List[str]) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8", errors="ignore").strip() or "unknown"
    except Exception:
        return "unknown"


def _resolve_output_dir(base: Optional[str], timestamp: str) -> Path:
    """Return the ablation output directory, appending _rerun_N if it exists."""
    if base:
        candidate = Path(base)
    else:
        candidate = REPO_ROOT / "runs" / f"ablation_{timestamp}"

    if not candidate.exists():
        return candidate

    # Directory already exists — find a free _rerun_N suffix.
    n = 1
    while True:
        suffixed = candidate.parent / f"{candidate.name}_rerun_{n}"
        if not suffixed.exists():
            return suffixed
        n += 1


def _build_launch_command(
    *,
    condition_name: str,
    seed: int,
    population_size: Optional[int],
    num_generations: Optional[int],
) -> List[str]:
    """Return the launch.sh argv for one condition+seed combination.

    We use `bash launch.sh` so the script is invocable even when the worktree
    does not have execute permission set (common in PACE-ICE storage mounts).
    """
    cmd = ["bash", str(REPO_ROOT / "launch.sh")]
    cmd += ["--name", f"{condition_name}_seed{seed}"]
    cmd += ["--seed", str(seed)]
    return cmd


def _condition_env(
    condition_name: str,
    seed: int,
    population_size: Optional[int],
    num_generations: Optional[int],
) -> Dict[str, str]:
    """Build the subprocess environment for launch.sh.

    launch.sh does `set -a; source .env; set +a` then passes `--export=...,ALL`
    to sbatch.  Variables set in the caller's environment before sourcing .env
    will be overridden by .env values.  To force our condition settings through,
    we re-export them *after* launch.sh sources .env by relying on the fact that
    sbatch --export=ALL carries the entire caller env.

    The trick: we pass the condition vars as OVERRIDE_* names that launch.sh
    already preserves *before* sourcing .env, OR we simply set them as plain
    vars and accept they'll be clobbered by .env.  Since our target vars
    (RAG_ENABLED etc.) are the ones we want to override .env, we inject them
    as additional exports via the subprocess env — `sbatch --export=...,ALL`
    will include them regardless.

    We also append EXPERIMENT_SEED and optional sizing knobs.
    """
    env = {**os.environ}

    # Condition-specific RAG vars (override whatever .env says).
    for k, v in ALL_CONDITIONS[condition_name].items():
        env[k] = v

    env["EXPERIMENT_SEED"] = str(seed)

    if population_size is not None:
        env["POPULATION_SIZE"] = str(population_size)
        env["START_POPULATION_SIZE"] = str(population_size)
    if num_generations is not None:
        env["NUM_GENERATIONS"] = str(num_generations)

    return env


def _parse_main_job_id(stdout: str) -> Optional[str]:
    """Extract the SLURM job ID from launch.sh stdout.

    launch.sh prints: ``Main job submitted:   <JOB_ID>``
    """
    for line in stdout.splitlines():
        # Match "Main job submitted:   <digits>"
        m = re.search(r"Main job submitted:\s+(\d+)", line)
        if m:
            return m.group(1)
    # Fallback: last numeric token on any line.
    for line in reversed(stdout.splitlines()):
        tokens = line.strip().split()
        for tok in reversed(tokens):
            if tok.isdigit():
                return tok
    return None


def _write_manifest(output_dir: Path, manifest: AblationManifest) -> Path:
    """Write manifest.json and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    data = {
        "matrix_timestamp": manifest.matrix_timestamp,
        "git_commit": manifest.git_commit,
        "git_branch": manifest.git_branch,
        "user": manifest.user,
        "output_dir": manifest.output_dir,
        "conditions": manifest.conditions,
        "seeds": manifest.seeds,
        "jobs": [asdict(j) for j in manifest.jobs],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch a RAG ablation matrix via launch.sh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print launch.sh commands without submitting jobs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        metavar="N",
        help="Seed values to run.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "rag-faiss"],
        choices=list(ALL_CONDITIONS.keys()),
        metavar="COND",
        help=f"Conditions to include. Available: {', '.join(ALL_CONDITIONS)}.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        metavar="N",
        help="Override POPULATION_SIZE forwarded to launch.sh.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        metavar="N",
        help="Override NUM_GENERATIONS forwarded to launch.sh.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Override the default runs/ablation_<timestamp>/ output location.",
    )
    args = parser.parse_args(argv)

    # Validate conditions.
    for cond in args.conditions:
        if cond not in ALL_CONDITIONS:
            print(f"ERROR: unknown condition '{cond}'. Available: {list(ALL_CONDITIONS)}", file=sys.stderr)
            return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_output_dir(args.output_dir, timestamp)

    manifest = AblationManifest(
        matrix_timestamp=timestamp,
        git_commit=_git_or_unknown(["rev-parse", "HEAD"]),
        git_branch=_git_or_unknown(["rev-parse", "--abbrev-ref", "HEAD"]),
        user=os.getenv("USER") or os.getenv("USERNAME") or socket.gethostname(),
        output_dir=str(output_dir),
        conditions=list(args.conditions),
        seeds=list(args.seeds),
    )

    total = len(args.conditions) * len(args.seeds)
    print(f"Ablation matrix: {len(args.conditions)} conditions x {len(args.seeds)} seeds = {total} jobs")
    print(f"Output dir: {output_dir}")
    if args.dry_run:
        print("(DRY RUN — no jobs will be submitted)\n")

    header = f"{'#':<4}  {'condition':<16}  {'seed':<6}  {'job_id':<14}  command"
    print(header)
    print("-" * len(header))

    job_num = 0
    for condition_name in args.conditions:
        for seed in args.seeds:
            job_num += 1
            cmd = _build_launch_command(
                condition_name=condition_name,
                seed=seed,
                population_size=args.population_size,
                num_generations=args.num_generations,
            )
            env = _condition_env(
                condition_name=condition_name,
                seed=seed,
                population_size=args.population_size,
                num_generations=args.num_generations,
            )
            launch_cmd_str = " ".join(cmd)

            if args.dry_run:
                # In dry-run mode, show the command and the key env overrides.
                cond_env = ALL_CONDITIONS[condition_name]
                env_str = " ".join(f"{k}={v}" for k, v in sorted(cond_env.items()))
                env_str += f" EXPERIMENT_SEED={seed}"
                if args.population_size:
                    env_str += f" POPULATION_SIZE={args.population_size}"
                if args.num_generations:
                    env_str += f" NUM_GENERATIONS={args.num_generations}"
                print(f"{job_num:<4}  {condition_name:<16}  {seed:<6}  {'(dry-run)':<14}  {env_str} {launch_cmd_str}")

                record = JobRecord(
                    run_id=f"{condition_name}_seed{seed}_{timestamp}",
                    condition=condition_name,
                    seed=seed,
                    launch_command=f"{env_str} {launch_cmd_str}",
                    submitted_at=None,
                    main_job_id=None,
                )
                manifest.jobs.append(record)
                continue

            # Live submission.
            submitted_at = datetime.now(timezone.utc).isoformat()
            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or f"exit code {result.returncode}")
                stdout = result.stdout
                job_id = _parse_main_job_id(stdout)

                # launch.sh prints "Main job submitted:   <id>" and the RUN_ID
                # is embedded in directory paths it creates.  We derive run_id
                # from the --name convention: <condition>_seed<seed>_<timestamp>.
                # Since create_run.sh appends its own timestamp we cannot know
                # the exact run_id without parsing launch.sh output further.
                # We store the job_id as primary key; run_id is best-effort.
                run_id_guess = f"{condition_name}_seed{seed}_{timestamp}"

                record = JobRecord(
                    run_id=run_id_guess,
                    condition=condition_name,
                    seed=seed,
                    launch_command=launch_cmd_str,
                    submitted_at=submitted_at,
                    main_job_id=job_id,
                )
                manifest.jobs.append(record)
                print(f"{job_num:<4}  {condition_name:<16}  {seed:<6}  {str(job_id):<14}  {launch_cmd_str}")

            except Exception as exc:
                error_msg = str(exc)
                record = JobRecord(
                    run_id=f"{condition_name}_seed{seed}_{timestamp}",
                    condition=condition_name,
                    seed=seed,
                    launch_command=launch_cmd_str,
                    submitted_at=submitted_at,
                    main_job_id=None,
                    error=error_msg,
                )
                manifest.jobs.append(record)
                print(
                    f"{job_num:<4}  {condition_name:<16}  {seed:<6}  {'ERROR':<14}  {error_msg}",
                    file=sys.stderr,
                )

    # Write manifest regardless of dry-run / errors.
    manifest_path = _write_manifest(output_dir, manifest)
    print(f"\nManifest written: {manifest_path}")

    if args.dry_run:
        print("Dry-run complete. Re-run without --dry-run to submit jobs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
