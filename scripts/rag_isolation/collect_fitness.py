"""Poll for eval results files, merge fitness back into a paired-eval run.

Reads ``<run_dir>/eval_jobs.jsonl`` written by ``eval_submit.py`` and
``<run_dir>/results.jsonl`` written by ``run_paired_eval.py``. Polls every
``--poll-interval`` seconds for ``<run_dir>/results/{eval_gene_id}_results.txt``
to appear. When all submitted jobs are done (or marked failed/timeout),
emits ``results_with_fitness.{jsonl,csv}`` with the fitness columns from
spec §8 populated.

Fallback rows inherit fitness from the cache produced by
``parent_fitness.py``. Failed jobs leave ``fitness_*`` as ``None``.

Per spec §7.2.

Usage:
    uv run python scripts/rag_isolation/collect_fitness.py \\
        --run-dir experiments/rag_isolation/<run> \\
        [--timeout 14400] [--poll-interval 60]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from rag_isolation import eval_common  # type: ignore  # noqa: E402

PARENT_CACHE_FILE = ROOT_DIR / "experiments" / "rag_isolation" / "_parent_fitness_cache.json"

TERMINAL = {"done", "failed", "timeout", "inherited", "dry_run"}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_parent_cache() -> dict:
    if not PARENT_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(PARENT_CACHE_FILE.read_text())
    except Exception:
        return {}


def _resolve_pending(jobs: list[dict], run_dir: Path) -> int:
    """Update jobs in place. Returns the count still pending."""
    pending = 0
    for j in jobs:
        if j.get("eval_status") in TERMINAL:
            continue
        eval_gene_id = j["eval_gene_id"]
        results_file = eval_common.results_path_for(run_dir, eval_gene_id)
        if results_file.exists():
            parsed = eval_common.parse_results_file(results_file)
            if parsed is None:
                j["eval_status"] = "failed"
                j["eval_error"] = "could not parse results file"
                continue
            j["eval_status"] = "done"
            j["fitness_acc"] = parsed["test_acc"]
            j["fitness_params"] = int(parsed["num_params"])
            j["eval_val_acc"] = parsed.get("val_acc")
            j["eval_train_seconds"] = parsed.get("tr_time")
            j["results_file"] = str(results_file.relative_to(ROOT_DIR))
            continue
        # Not done yet — check slurm
        job_id = j.get("eval_job_id")
        if job_id:
            sq = eval_common.squeue_state(job_id)
            if sq is None:
                final = eval_common.sacct_final_state(job_id)
                if final and any(s in final for s in
                                  ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_ME")):
                    j["eval_status"] = "failed"
                    j["eval_error"] = f"slurm: {final}"
                    continue
                if final and "COMPLETED" in final:
                    # Slurm says done but no results file → mark failed.
                    j["eval_status"] = "failed"
                    j["eval_error"] = "completed without results file"
                    continue
        pending += 1
    return pending


def _poll(jobs: list[dict], run_dir: Path, jobs_path: Path,
          timeout: float, poll_interval: float) -> None:
    deadline = time.time() + timeout
    while True:
        pending = _resolve_pending(jobs, run_dir)
        _save_jsonl(jobs_path, jobs)  # checkpoint
        n_done = sum(1 for j in jobs if j.get("eval_status") == "done")
        n_failed = sum(1 for j in jobs if j.get("eval_status") == "failed")
        print(f"[{int(time.time() - (deadline - timeout))}s] "
              f"pending={pending} done={n_done} failed={n_failed}", flush=True)
        if pending == 0:
            return
        if time.time() >= deadline:
            for j in jobs:
                if j.get("eval_status") not in TERMINAL:
                    j["eval_status"] = "timeout"
            _save_jsonl(jobs_path, jobs)
            return
        time.sleep(poll_interval)


def _attach_fitness_to_results(results: list[dict], jobs: list[dict],
                               parent_cache: dict) -> list[dict]:
    """Merge per-gene fitness from jobs (and parent cache for fallbacks) into results."""
    job_by_gene = {j["gene_id"]: j for j in jobs}
    out: list[dict] = []

    for row in results:
        merged = dict(row)
        gene_id = row["gene_id"]
        job = job_by_gene.get(gene_id)
        if job is None:
            merged.update({
                "fitness_acc": None,
                "fitness_params": None,
                "fitness_inherited_from": None,
                "eval_job_id": None,
                "eval_status": "missing",
                "eval_train_seconds": None,
                "eval_val_acc": None,
            })
            out.append(merged)
            continue

        status = job.get("eval_status")
        if status == "done":
            merged.update({
                "fitness_acc": job.get("fitness_acc"),
                "fitness_params": job.get("fitness_params"),
                "fitness_inherited_from": None,
                "eval_job_id": job.get("eval_job_id"),
                "eval_status": "done",
                "eval_train_seconds": job.get("eval_train_seconds"),
                "eval_val_acc": job.get("eval_val_acc"),
            })
        elif status == "inherited":
            parent_rel = row.get("parent")
            entry = parent_cache.get(parent_rel)
            if entry:
                merged.update({
                    "fitness_acc": float(entry["test_acc"]),
                    "fitness_params": int(entry["num_params"]),
                    "fitness_inherited_from": parent_rel,
                    "eval_job_id": None,
                    "eval_status": "inherited",
                    "eval_train_seconds": entry.get("tr_time"),
                    "eval_val_acc": entry.get("val_acc"),
                })
            else:
                merged.update({
                    "fitness_acc": None,
                    "fitness_params": None,
                    "fitness_inherited_from": parent_rel,
                    "eval_job_id": None,
                    "eval_status": "inherited_no_parent_fitness",
                    "eval_train_seconds": None,
                    "eval_val_acc": None,
                })
        else:
            merged.update({
                "fitness_acc": None,
                "fitness_params": None,
                "fitness_inherited_from": None,
                "eval_job_id": job.get("eval_job_id"),
                "eval_status": status,
                "eval_train_seconds": None,
                "eval_val_acc": None,
            })
        out.append(merged)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([_csv_cell(r.get(k)) for k in keys])


def _csv_cell(v):
    if isinstance(v, list):
        return "|".join(str(x) for x in v)
    if v is None:
        return ""
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=4 * 3600,
                    help="Total polling timeout in seconds (default 4h)")
    ap.add_argument("--poll-interval", type=float, default=60.0)
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Keep the SOTA models/network_*.py copies after collection")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    results_path = run_dir / "results.jsonl"
    jobs_path = run_dir / "eval_jobs.jsonl"
    if not results_path.exists():
        print(f"error: {results_path} missing", file=sys.stderr)
        return 2
    if not jobs_path.exists():
        print(f"error: {jobs_path} missing — run eval_submit.py first", file=sys.stderr)
        return 2

    results = _load_jsonl(results_path)
    jobs = _load_jsonl(jobs_path)

    _poll(jobs, run_dir, jobs_path, args.timeout, args.poll_interval)

    parent_cache = _load_parent_cache()
    merged = _attach_fitness_to_results(results, jobs, parent_cache)

    out_jsonl = run_dir / "results_with_fitness.jsonl"
    out_csv = run_dir / "results_with_fitness.csv"
    _save_jsonl(out_jsonl, merged)
    _write_csv(out_csv, merged)

    # Cleanup the run-stamp-prefixed copies in sota/ExquisiteNetV2/models so
    # the dir doesn't grow without bound across runs.
    if not args.no_cleanup:
        for j in jobs:
            if j.get("eval_status") in ("done", "failed", "timeout"):
                eval_common.remove_network_from_models(j["eval_gene_id"])

    n = len(merged)
    n_done = sum(1 for r in merged if r["eval_status"] == "done")
    n_inh = sum(1 for r in merged if r["eval_status"] == "inherited")
    n_failed = sum(1 for r in merged if r["eval_status"] in ("failed", "timeout", "missing", "inherited_no_parent_fitness"))
    print(f"\n{n} rows: done={n_done} inherited={n_inh} failed={n_failed}")
    print(f"JSONL: {out_jsonl}")
    print(f"CSV:   {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
