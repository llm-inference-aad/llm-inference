#!/usr/bin/env python3
"""
Launch a focused A/B matrix to measure episodic memory-store impact.

Conditions:
- memory_off: RAG enabled, memory store disabled
- memory_on:  RAG enabled, memory store enabled

By default this prints sbatch commands (dry-run). Use --execute to submit.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
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
    return run_dir


def _sbatch_export(env: Dict[str, str]) -> str:
    parts = ["ALL"]
    for key in sorted(env):
        parts.append(f"{key}={env[key]}")
    return ",".join(parts)


def _submit(cmd: List[str]) -> str:
    out = subprocess.check_output(cmd, cwd=str(REPO_ROOT))
    return out.decode("utf-8", errors="ignore").strip()


def _parse_sbatch_job_id(s: str) -> str | None:
    tokens = s.strip().split()
    for tok in reversed(tokens):
        if tok.isdigit():
            return tok
    return None


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def build_conditions(
    *,
    rag_enabled: bool,
    use_code_context: bool,
    use_text_context: bool,
    reranker_enabled: bool,
    memory_top_k: int,
) -> List[Condition]:
    base = {
        "RAG_ENABLED": _bool_str(rag_enabled),
        "RAG_USE_CODE_CONTEXT": _bool_str(use_code_context),
        "RAG_USE_TEXT_CONTEXT": _bool_str(use_text_context),
        "RAG_RERANKER_ENABLED": _bool_str(reranker_enabled),
        "RAG_MEMORY_TOP_K": str(memory_top_k),
    }
    return [
        Condition(
            name="memory_off",
            env={
                **base,
                "RAG_MEMORY_STORE_ENABLED": "false",
            },
        ),
        Condition(
            name="memory_on",
            env={
                **base,
                "RAG_MEMORY_STORE_ENABLED": "true",
            },
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Experiment seeds.")
    parser.add_argument("--num-generations", type=int, default=5, help="Override NUM_GENERATIONS.")
    parser.add_argument("--population-size", type=int, default=16, help="Override POPULATION_SIZE.")
    parser.add_argument("--start-population-size", type=int, default=16, help="Override START_POPULATION_SIZE.")
    parser.add_argument("--threshold", type=float, default=0.90, help="Accuracy threshold for analyzer.")
    parser.add_argument("--memory-top-k", type=int, default=3, help="Value for RAG_MEMORY_TOP_K.")
    parser.add_argument("--rag-enabled", action="store_true", default=True, help="Enable RAG for both arms.")
    parser.add_argument("--no-rag-enabled", action="store_false", dest="rag_enabled")
    parser.add_argument(
        "--use-code-context",
        action="store_true",
        default=True,
        help="Enable code context retrieval in both arms.",
    )
    parser.add_argument("--no-use-code-context", action="store_false", dest="use_code_context")
    parser.add_argument(
        "--use-text-context",
        action="store_true",
        default=True,
        help="Enable text context retrieval in both arms.",
    )
    parser.add_argument("--no-use-text-context", action="store_false", dest="use_text_context")
    parser.add_argument(
        "--reranker-enabled",
        action="store_true",
        default=False,
        help="Enable reranker in both arms.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit sbatch jobs (default is dry-run).",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only print planned matrix (and optional manifest), do not submit.",
    )
    parser.add_argument(
        "--submit-analysis",
        action="store_true",
        help="Submit dependent per-run analysis jobs after each run.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional path to write planned run manifest JSON.",
    )
    args = parser.parse_args()

    conditions = build_conditions(
        rag_enabled=args.rag_enabled,
        use_code_context=args.use_code_context,
        use_text_context=args.use_text_context,
        reranker_enabled=args.reranker_enabled,
        memory_top_k=args.memory_top_k,
    )
    matrix_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    planned: list[dict] = []

    for cond in conditions:
        for seed in args.seeds:
            run_id = f"{cond.name}_seed{seed}_{matrix_id}"
            env = {
                "RUN_ID": run_id,
                "NUM_GENERATIONS": str(args.num_generations),
                "POPULATION_SIZE": str(args.population_size),
                "START_POPULATION_SIZE": str(args.start_population_size),
                "EXPERIMENT_SEED": str(seed),
                **cond.env,
            }
            planned.append(
                {
                    "condition": cond.name,
                    "seed": seed,
                    "run_id": run_id,
                    "env": env,
                }
            )

    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "matrix_id": matrix_id,
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "planned_runs": planned,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote manifest: {manifest_path}")

    for item in planned:
        cond_name = str(item["condition"])
        seed = int(item["seed"])
        run_id = str(item["run_id"])
        env = dict(item["env"])
        export_arg = _sbatch_export(env)
        cmd = ["sbatch", f"--export={export_arg}", "run.sh"]
        print(f"[{cond_name} seed={seed}] RUN_ID={run_id}")
        print(f"  {' '.join(cmd)}")

        if args.analysis_only:
            continue
        if not args.execute:
            continue

        extra_meta = {
            "experiment": {
                "ablation": "memory_store",
                "matrix_id": matrix_id,
                "condition": cond_name,
                "seed": seed,
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
            analysis_cmd = [
                "sbatch",
                f"--dependency=afterok:{job_id}",
                f"--export={_sbatch_export({'RUN_ID': run_id})}",
                "--job-name",
                f"mem_analyze_{run_id}",
                "--time",
                "00:10:00",
                "--mem",
                "1G",
                "--output",
                str((REPO_ROOT / "runs" / run_id / "logs" / "slurm-analyze-%j.out")),
                "--error",
                str((REPO_ROOT / "runs" / run_id / "errors" / "slurm-analyze-%j.err")),
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

