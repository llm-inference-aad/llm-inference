#!/usr/bin/env python3
"""
Launch an A/B + ablation matrix for RAG impact evaluation.

By default, this script prints the `sbatch` commands (dry-run). Use `--execute`
to actually submit jobs.

This script is intentionally lightweight and uses only the standard library.
It assumes `run.sh` is the canonical entrypoint (creates/uses runs/<RUN_ID>/...).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Condition:
    name: str
    env: Dict[str, str]


def _run(cmd: List[str], *, cwd: Path) -> str:
    out = subprocess.check_output(cmd, cwd=str(cwd))
    return out.decode("utf-8", errors="ignore").strip()


def _git_or_unknown(args: List[str]) -> str:
    try:
        return _run(["git", *args], cwd=REPO_ROOT) or "unknown"
    except Exception:
        return "unknown"


def _ensure_run_dir(run_id: str, extra_meta: Dict[str, object]) -> Path:
    run_dir = REPO_ROOT / "runs" / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "errors").mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        meta = {
            "run_id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
            "git_branch": _git_or_unknown(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_commit": _git_or_unknown(["rev-parse", "HEAD"]),
            "log_dir": str(run_dir / "logs"),
            "metrics_dir": str(run_dir / "metrics"),
            "errors_dir": str(run_dir / "errors"),
            "status": "initialized",
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    _patch_run_metadata(run_dir, extra_meta)

    # Best-effort "latest" symlink.
    try:
        latest = REPO_ROOT / "runs" / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_id)
    except Exception:
        pass

    return run_dir


def _patch_run_metadata(run_dir: Path, extra: Dict[str, object]) -> None:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(meta, dict):
        return
    meta.update(extra)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _sbatch_export(env: Dict[str, str]) -> str:
    # `sbatch --export=ALL,FOO=bar,BAZ=qux`
    parts = ["ALL"]
    for k in sorted(env):
        parts.append(f"{k}={env[k]}")
    return ",".join(parts)

def _submit(cmd: List[str]) -> str:
    # sbatch typically prints: "Submitted batch job <id>"
    out = subprocess.check_output(cmd, cwd=str(REPO_ROOT))
    return out.decode("utf-8", errors="ignore").strip()


def _parse_sbatch_job_id(s: str) -> str | None:
    parts = s.strip().split()
    if len(parts) >= 4 and parts[-1].isdigit():
        return parts[-1]
    # Fallback: last token numeric
    for tok in reversed(parts):
        if tok.isdigit():
            return tok
    return None


def build_conditions(include_hybrid_rerank: bool) -> List[Condition]:
    conditions: List[Condition] = [
        Condition(
            name="baseline",
            env={
                "RAG_ENABLED": "false",
                "RAG_USE_CODE_CONTEXT": "false",
                "RAG_USE_TEXT_CONTEXT": "false",
                "RAG_RERANKER_ENABLED": "false",
            },
        ),
        Condition(
            name="code_only",
            env={
                "RAG_ENABLED": "true",
                "RAG_USE_CODE_CONTEXT": "true",
                "RAG_USE_TEXT_CONTEXT": "false",
                "RAG_RERANKER_ENABLED": "false",
            },
        ),
        Condition(
            name="text_only",
            env={
                "RAG_ENABLED": "true",
                "RAG_USE_CODE_CONTEXT": "false",
                "RAG_USE_TEXT_CONTEXT": "true",
                "RAG_RERANKER_ENABLED": "false",
            },
        ),
        Condition(
            name="hybrid",
            env={
                "RAG_ENABLED": "true",
                "RAG_USE_CODE_CONTEXT": "true",
                "RAG_USE_TEXT_CONTEXT": "true",
                "RAG_RERANKER_ENABLED": "false",
            },
        ),
    ]
    if include_hybrid_rerank:
        conditions.append(
            Condition(
                name="hybrid_rerank",
                env={
                    "RAG_ENABLED": "true",
                    "RAG_USE_CODE_CONTEXT": "true",
                    "RAG_USE_TEXT_CONTEXT": "true",
                    "RAG_RERANKER_ENABLED": "true",
                },
            )
        )
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="Experiment seeds.")
    parser.add_argument("--num-generations", type=int, default=5, help="Override NUM_GENERATIONS.")
    parser.add_argument("--population-size", type=int, default=16, help="Override POPULATION_SIZE.")
    parser.add_argument("--start-population-size", type=int, default=16, help="Override START_POPULATION_SIZE.")
    parser.add_argument("--threshold", type=float, default=0.90, help="Threshold for time-to-threshold analysis.")
    parser.add_argument(
        "--include-hybrid-rerank",
        action="store_true",
        help="Also run hybrid_rerank condition (RAG_RERANKER_ENABLED=true).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit the sbatch jobs (default is dry-run).",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip submission and only print the planned matrix (useful for review).",
    )
    parser.add_argument(
        "--submit-analysis",
        action="store_true",
        help="When executing, also submit a dependent analysis job per run.",
    )
    args = parser.parse_args()

    conditions = build_conditions(args.include_hybrid_rerank)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    planned = []
    for cond in conditions:
        for seed in args.seeds:
            prefix = f"{cond.name}_seed{seed}"
            run_id = f"{prefix}_{timestamp}"
            run_dir = REPO_ROOT / "runs" / run_id

            extra_meta = {
                "experiment": {
                    "condition": cond.name,
                    "seed": seed,
                    "matrix_timestamp": timestamp,
                    "num_generations": args.num_generations,
                    "population_size": args.population_size,
                    "start_population_size": args.start_population_size,
                    **{k.lower(): v for k, v in cond.env.items()},
                }
            }

            env = {
                "RUN_ID": run_id,
                "NUM_GENERATIONS": str(args.num_generations),
                "POPULATION_SIZE": str(args.population_size),
                "START_POPULATION_SIZE": str(args.start_population_size),
                "EXPERIMENT_SEED": str(seed),
                **cond.env,
            }
            planned.append((cond.name, seed, run_id, env))

    for cond_name, seed, run_id, env in planned:
        export_arg = _sbatch_export(env)
        cmd = ["sbatch", f"--export={export_arg}", "run.sh"]
        printable = " ".join(cmd)
        print(f"[{cond_name} seed={seed}] RUN_ID={run_id}")
        print(f"  {printable}")
        if args.analysis_only:
            continue
        if args.execute:
            extra_meta = {
                "experiment": {
                    "condition": cond_name,
                    "seed": seed,
                    "matrix_timestamp": timestamp,
                    "num_generations": args.num_generations,
                    "population_size": args.population_size,
                    "start_population_size": args.start_population_size,
                    **{k.lower(): v for k, v in env.items() if k.startswith("RAG_")},
                }
            }
            _ensure_run_dir(run_id, extra_meta)
            sbatch_out = _submit(cmd)
            print(f"  {sbatch_out}")
            job_id = _parse_sbatch_job_id(sbatch_out)

            if args.submit_analysis and job_id:
                # Run the stdlib-only analyzer after the main job completes.
                analysis_cmd = [
                    "sbatch",
                    f"--dependency=afterok:{job_id}",
                    f"--export={_sbatch_export({'RUN_ID': run_id})}",
                    "--job-name",
                    f"rag_analyze_{run_id}",
                    "--time",
                    "00:10:00",
                    "--mem",
                    "1G",
                    "--output",
                    str((REPO_ROOT / "runs" / run_id / "logs" / f"slurm-analyze-%j.out")),
                    "--error",
                    str((REPO_ROOT / "runs" / run_id / "errors" / f"slurm-analyze-%j.err")),
                    "--wrap",
                    (
                        f"cd {REPO_ROOT} && "
                        f"python scripts/analyze_rag_impact.py "
                        f"--run-dir runs/{run_id} "
                        f"--threshold {args.threshold}"
                    ),
                ]
                analysis_out = _submit(analysis_cmd)
                print(f"  {analysis_out}")

    if not args.execute:
        print("\nDry-run complete. Re-run with --execute to submit.")


if __name__ == "__main__":
    main()
