"""Train every unique parent in a dataset once, cache its fitness.

The paired harness lets a mutation fall back to its parent. To compare
``with_rag`` vs ``no_rag`` fairly when both fall back, we need the parent's
fitness on the same 8-epoch / seed=21 training regime as the children.

This script ensures every parent referenced in ``dataset.json`` has an entry
in ``experiments/rag_isolation/_parent_fitness_cache.json`` — training the
parent if not. The cache is shared across runs so the seed network is only
ever trained once.

Per spec §6 / §7.3.

Usage:
    uv run python scripts/rag_isolation/parent_fitness.py \\
        --dataset scripts/rag_isolation/datasets/smoke.json \\
        [--epochs 8] [--seed 21] [--timeout 14400]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_isolation import eval_common  # type: ignore  # noqa: E402

CACHE_FILE = ROOT_DIR / "experiments" / "rag_isolation" / "_parent_fitness_cache.json"


def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _unique_parents(dataset_path: Path) -> list[str]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    seen = []
    for case in raw.get("cases", []):
        p = case.get("parent")
        if p and p not in seen:
            seen.append(p)
    return seen


def _ensure_cifar10_split() -> None:
    """Run sota/ExquisiteNetV2/split.py if cifar10/ is missing (spec §11)."""
    cifar_dir = eval_common.SOTA_ROOT / "cifar10"
    if cifar_dir.exists() and (cifar_dir / "test").exists():
        return
    raw_dir = eval_common.SOTA_ROOT / "cifar-10-batches-py"
    if not raw_dir.exists():
        print(
            f"warning: {cifar_dir} missing and no raw cifar-10-batches-py to split. "
            "Place cifar-10-batches-py under sota/ExquisiteNetV2/ before running.",
            file=sys.stderr,
        )
        return
    print(f"cifar10/ missing — running split.py from {eval_common.SOTA_ROOT}", flush=True)
    import subprocess
    subprocess.run(
        [sys.executable, "split.py"],
        cwd=str(eval_common.SOTA_ROOT),
        check=True,
    )


def _train_seed_baseline() -> tuple[float, float] | None:
    """Train the canonical seed network synchronously via the production helper.

    Lazily imported because importing ``src.evolution.seed`` pulls in
    ``cfg.constants``, which initializes torch.
    """
    from evolution.seed import train_seed_network_baseline  # type: ignore
    return train_seed_network_baseline()


def _wait_for_results(results_file: Path, job_id: str | None,
                       timeout_s: float, poll_s: float) -> str | None:
    """Block until a results file appears, the job fails, or timeout expires.

    Returns:
        "done" if results file appeared,
        "failed" if sacct reports failure,
        "timeout" otherwise.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if results_file.exists():
            return "done"
        if job_id:
            sq = eval_common.squeue_state(job_id)
            if sq is None:
                # not in queue → check sacct
                final = eval_common.sacct_final_state(job_id)
                if final and any(s in final for s in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL")):
                    return "failed"
                if final and "COMPLETED" in final and not results_file.exists():
                    # Job completed but no results — treat as failed
                    return "failed"
        time.sleep(poll_s)
    return "timeout"


def _train_parent_via_sbatch(parent_path: Path, epochs: int, seed: int,
                              timeout_s: float, poll_s: float) -> dict | None:
    """Submit one sbatch eval for an arbitrary parent and block on its result."""
    parent_hash = _file_hash(parent_path)
    eval_gene_id = f"PARENT_{parent_hash}"
    run_dir = ROOT_DIR / "experiments" / "rag_isolation" / "_parent_fitness_runs" / eval_gene_id
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    slurm_log_dir = run_dir / "logs"
    slurm_error_dir = run_dir / "errors"

    eval_common.copy_network_into_models(parent_path, eval_gene_id)
    job = eval_common.EvalJob(
        eval_gene_id=eval_gene_id,
        network_src=parent_path,
        run_dir=run_dir,
        epochs=epochs,
        seed=seed,
    )
    sh_path = eval_common.write_bash_script(job, run_dir, slurm_log_dir, slurm_error_dir)
    ok, job_id, raw = eval_common.submit_sbatch(sh_path)
    if not ok:
        print(f"sbatch failed for parent {parent_path}: {raw}", file=sys.stderr)
        return None
    print(f"submitted parent {parent_path.name} as job {job_id} — waiting...", flush=True)

    results_file = eval_common.results_path_for(run_dir, eval_gene_id)
    state = _wait_for_results(results_file, job_id, timeout_s, poll_s)
    if state != "done":
        print(f"parent {parent_path} did not finish ({state})", file=sys.stderr)
        return None
    parsed = eval_common.parse_results_file(results_file)
    if not parsed:
        print(f"could not parse {results_file}", file=sys.stderr)
        return None
    return {
        "test_acc": parsed["test_acc"],
        "num_params": parsed["num_params"],
        "val_acc": parsed.get("val_acc"),
        "tr_time": parsed.get("tr_time"),
        "epochs": epochs,
        "seed": seed,
        "eval_job_id": job_id,
        "results_file": str(results_file.relative_to(ROOT_DIR)),
        "trained_at": time.time(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="Dataset JSON — only parents listed here will be trained")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--timeout", type=float, default=4 * 3600,
                    help="Max wait per parent (seconds, default 4h)")
    ap.add_argument("--poll-interval", type=float, default=60.0)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Ignore cache and retrain every parent")
    args = ap.parse_args()

    _ensure_cifar10_split()

    parents = _unique_parents(args.dataset)
    if not parents:
        print(f"no parents in {args.dataset}", file=sys.stderr)
        return 2

    cache = _load_cache()
    print(f"Cache: {CACHE_FILE} ({len(cache)} entries)")
    print(f"Parents from dataset: {len(parents)}")

    for parent_rel in parents:
        cache_key = parent_rel
        if not args.force_retrain and cache_key in cache:
            entry = cache[cache_key]
            print(f"  ✓ cached {parent_rel}: acc={entry.get('test_acc')} "
                  f"params={entry.get('num_params')}")
            continue

        parent_abs = ROOT_DIR / parent_rel
        if not parent_abs.exists():
            print(f"  ✗ parent file missing: {parent_abs}", file=sys.stderr)
            continue

        if parent_rel == "sota/ExquisiteNetV2/network.py":
            # Use the production seed-baseline helper which writes
            # sota/ExquisiteNetV2/results/network_results.txt synchronously.
            os.environ["EPOCHS"] = str(args.epochs)
            print(f"  • training seed network at {args.epochs} epochs...")
            result = _train_seed_baseline()
            if result is None:
                print(f"  ✗ seed training failed", file=sys.stderr)
                continue
            test_acc, num_params = result
            cache[cache_key] = {
                "test_acc": test_acc,
                "num_params": num_params,
                "epochs": args.epochs,
                "seed": args.seed,
                "results_file": "sota/ExquisiteNetV2/results/network_results.txt",
                "trained_at": time.time(),
            }
            _save_cache(cache)
            print(f"  ✓ seed cached: acc={test_acc:.4f} params={int(num_params)}")
            continue

        print(f"  • training parent {parent_rel} via sbatch...")
        entry = _train_parent_via_sbatch(
            parent_abs, args.epochs, args.seed, args.timeout, args.poll_interval,
        )
        if entry is None:
            print(f"  ✗ failed to evaluate {parent_rel}")
            continue
        cache[cache_key] = entry
        _save_cache(cache)
        print(f"  ✓ {parent_rel}: acc={entry['test_acc']:.4f} "
              f"params={int(entry['num_params'])}")

    print(f"\nCache file: {CACHE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
