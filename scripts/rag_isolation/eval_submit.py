"""Submit one CIFAR-10 training sbatch per non-fallback gene in a paired run.

Reads ``<run_dir>/results.jsonl`` produced by ``run_paired_eval.py``, copies
each non-fallback ``network.py`` into ``sota/ExquisiteNetV2/models/``, and
submits an sbatch eval job per gene. Results land at
``<run_dir>/results/{eval_gene_id}_results.txt`` once each job completes
(``train.py`` honors ``$RUN_DIR``).

Per spec §5/§7.1/§11. Fallback rows are skipped — they inherit fitness from
their parent during ``collect_fitness.py``.

Usage:
    uv run python scripts/rag_isolation/eval_submit.py \\
        --run-dir experiments/rag_isolation/<run> \\
        [--epochs 8] [--seed 21] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from rag_isolation import eval_common  # type: ignore  # noqa: E402


def _load_results(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Mutation-phase run directory (contains results.jsonl)")
    ap.add_argument("--epochs", type=int, default=8,
                    help="Eval epochs (spec §2 default 8)")
    ap.add_argument("--seed", type=int, default=21,
                    help="Train seed (spec §2 default 21)")
    ap.add_argument("--wall-time", type=str, default="00:30:00",
                    help="sbatch --time for each eval job")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print sbatch lines, do not submit or copy networks")
    ap.add_argument("--limit", type=int, default=None,
                    help="Submit only the first N non-fallback jobs (smoke testing)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        print(f"error: {results_path} not found", file=sys.stderr)
        return 2
    rows = _load_results(results_path)
    if not rows:
        print(f"error: {results_path} is empty", file=sys.stderr)
        return 2

    eval_jobs_path = run_dir / "eval_jobs.jsonl"
    eval_jobs_path.unlink(missing_ok=True)

    sh_dir = run_dir / "eval_scripts"
    slurm_log_dir = run_dir / "logs" / "eval"
    slurm_error_dir = run_dir / "errors" / "eval"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    run_stamp = eval_common.derive_run_stamp(run_dir)
    submitted = 0
    inherited = 0
    skipped_failed = 0
    started_at = time.time()

    for row in rows:
        gene_id = row["gene_id"]
        fallback = bool(row.get("fallback"))
        code_path_rel = row.get("code_path") or ""
        eval_gene_id = eval_common.make_eval_gene_id(run_stamp, gene_id)

        if fallback:
            inherited += 1
            _append_jsonl(eval_jobs_path, {
                "gene_id": gene_id,
                "eval_gene_id": eval_gene_id,
                "case_id": row.get("case_id"),
                "trial": row.get("trial"),
                "arm": row.get("arm"),
                "parent": row.get("parent"),
                "eval_status": "inherited",
                "eval_job_id": None,
                "sbatch_output": None,
                "submitted_at": None,
            })
            continue

        if not code_path_rel:
            skipped_failed += 1
            _append_jsonl(eval_jobs_path, {
                "gene_id": gene_id,
                "eval_gene_id": eval_gene_id,
                "case_id": row.get("case_id"),
                "trial": row.get("trial"),
                "arm": row.get("arm"),
                "parent": row.get("parent"),
                "eval_status": "failed",
                "eval_job_id": None,
                "sbatch_output": "no code_path on row",
                "submitted_at": None,
            })
            continue

        code_path = ROOT_DIR / code_path_rel
        if not code_path.exists():
            skipped_failed += 1
            _append_jsonl(eval_jobs_path, {
                "gene_id": gene_id,
                "eval_gene_id": eval_gene_id,
                "case_id": row.get("case_id"),
                "trial": row.get("trial"),
                "arm": row.get("arm"),
                "parent": row.get("parent"),
                "eval_status": "failed",
                "eval_job_id": None,
                "sbatch_output": f"missing network.py at {code_path}",
                "submitted_at": None,
            })
            continue

        if args.limit is not None and submitted >= args.limit:
            break

        job = eval_common.EvalJob(
            eval_gene_id=eval_gene_id,
            network_src=code_path,
            run_dir=run_dir,
            epochs=args.epochs,
            seed=args.seed,
            wall_time=args.wall_time,
        )

        if args.dry_run:
            sh_path = eval_common.write_bash_script(job, sh_dir, slurm_log_dir, slurm_error_dir)
            print(f"[dry-run] would copy {code_path} → "
                  f"{eval_common.SOTA_MODELS_DIR}/network_{eval_gene_id}.py")
            print(f"[dry-run] would sbatch {sh_path}")
            _append_jsonl(eval_jobs_path, {
                "gene_id": gene_id,
                "eval_gene_id": eval_gene_id,
                "case_id": row.get("case_id"),
                "trial": row.get("trial"),
                "arm": row.get("arm"),
                "parent": row.get("parent"),
                "eval_status": "dry_run",
                "eval_job_id": None,
                "sbatch_script": str(sh_path),
                "submitted_at": None,
            })
            submitted += 1
            continue

        eval_common.copy_network_into_models(code_path, eval_gene_id)
        sh_path = eval_common.write_bash_script(job, sh_dir, slurm_log_dir, slurm_error_dir)
        ok, job_id, raw = eval_common.submit_sbatch(sh_path)
        status = "submitted" if ok else "failed"
        if not ok:
            print(f"warning: sbatch failed for {gene_id}: {raw}", file=sys.stderr)
        else:
            print(f"submitted {gene_id} → job_id={job_id}", flush=True)
        submitted += 1
        _append_jsonl(eval_jobs_path, {
            "gene_id": gene_id,
            "eval_gene_id": eval_gene_id,
            "case_id": row.get("case_id"),
            "trial": row.get("trial"),
            "arm": row.get("arm"),
            "parent": row.get("parent"),
            "eval_status": status,
            "eval_job_id": job_id,
            "sbatch_script": str(sh_path),
            "sbatch_output": raw,
            "submitted_at": time.time(),
        })

    print(
        f"\nDone. submitted={submitted} inherited(fallback)={inherited} "
        f"failed_pre_submit={skipped_failed}"
    )
    print(f"Eval jobs index: {eval_jobs_path}")
    if not args.dry_run:
        print(f"Wall: {time.time() - started_at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
