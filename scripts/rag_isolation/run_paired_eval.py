"""CLI entrypoint for the RAG isolation paired evaluation.

For every (case, trial) in the dataset, run the same mutation twice — once
without RAG context and once with — and write metrics to results.jsonl.

Example:
    uv run python scripts/rag_isolation/run_paired_eval.py \\
        --dataset scripts/rag_isolation/datasets/small_validation.json \\
        --output  experiments/rag_isolation/2026-04-27_v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve repo root before importing core
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _resolve_server_url(explicit: str | None) -> tuple[str, int]:
    """Resolve (hostname, port) from --server-url, env, or hostname.log."""
    if explicit:
        # Accept "host:port" or full http://host:port
        u = explicit.replace("http://", "").replace("https://", "")
        if ":" in u:
            host, port = u.split(":", 1)
            return host.strip().rstrip("/"), int(port.split("/")[0])
        return u.strip(), int(os.environ.get("SERVER_PORT", 8000))
    # Fallback: hostname.log + SERVER_PORT
    hf = Path(os.environ.get("HOSTNAME_LOG_FILE", str(ROOT_DIR / "hostname.log")))
    if hf.exists():
        host = hf.read_text().strip()
        port = int(os.environ.get("SERVER_PORT", 8000))
        return host, port
    raise SystemExit(
        "Could not resolve LLM server URL. Pass --server-url host:port or set "
        "HOSTNAME_LOG_FILE."
    )


def _ping_server(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True,
                    help="Run directory (will be created)")
    ap.add_argument("--server-url", type=str, default=None,
                    help="LLM server URL (host:port). Defaults to hostname.log + SERVER_PORT")
    ap.add_argument("--trials-per-case", type=int, default=None,
                    help="Override dataset trials_per_case")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-cases", type=int, default=None,
                    help="Smoke-test mode: only run the first N cases")
    ap.add_argument("--no-rag-only", action="store_true",
                    help="Run only the no_rag arm (debugging)")
    ap.add_argument("--shuffle-arms", action="store_true", default=True,
                    help="Randomize per-trial arm order to avoid time bias")
    ap.add_argument("--rag-use-code-context", type=str, default=None,
                    choices=["true", "false"])
    ap.add_argument("--rag-use-text-context", type=str, default=None,
                    choices=["true", "false"])
    args = ap.parse_args()

    # CRITICAL: env vars consumed by cfg.constants must be set BEFORE any import
    # from the src/ tree. We always force RAG_ENABLED=true so the runtime can
    # be constructed; the per-arm toggle is handled by us calling (or not
    # calling) enhance_template().
    os.environ["RAG_ENABLED"] = "true"
    if args.rag_use_code_context is not None:
        os.environ["RAG_USE_CODE_CONTEXT"] = args.rag_use_code_context
    if args.rag_use_text_context is not None:
        os.environ["RAG_USE_TEXT_CONTEXT"] = args.rag_use_text_context

    # Resolve server (and let augment_network find it via env)
    host, port = _resolve_server_url(args.server_url)
    if not _ping_server(host, port):
        raise SystemExit(f"LLM server at {host}:{port} is not reachable.")
    os.environ["SERVER_PORT"] = str(port)
    # Write a temp hostname file the production code can consume.
    args.output.mkdir(parents=True, exist_ok=True)
    hostname_log = args.output / "hostname.log"
    hostname_log.write_text(host + "\n")
    os.environ["HOSTNAME_LOG_FILE"] = str(hostname_log)
    # Tell the LLM server we are NOT a slurm job (so it doesn't try to look one up)
    os.environ.setdefault("USE_LOAD_BALANCER", "false")

    # Now safe to import core (which imports from src/)
    from rag_isolation import core  # type: ignore

    cases, cfg = core.load_dataset(args.dataset)
    if args.trials_per_case is not None:
        cfg.trials_per_case = args.trials_per_case
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    arms = ["no_rag"] if args.no_rag_only else ["no_rag", "with_rag"]

    # Build the runtime once. (Heavy: loads CodeBERT + MiniLM into memory.)
    print(f"[{datetime.now().isoformat()}] Loading RagRuntime...", flush=True)
    runtime = core._resolve_runtime() if "with_rag" in arms else None
    print(f"[{datetime.now().isoformat()}] Runtime ready.", flush=True)

    results_path = args.output / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    metadata_path = args.output / "run_metadata.json"

    started_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "started_at": started_at,
        "dataset_path": str(args.dataset),
        "git_commit": _git_commit(),
        "server_url": f"http://{host}:{port}",
        "arms": arms,
        "config": cfg.__dict__,
        "n_cases": len(cases),
        "n_trials_per_case": cfg.trials_per_case,
        "env": {
            "RAG_USE_CODE_CONTEXT": os.environ.get("RAG_USE_CODE_CONTEXT", "true"),
            "RAG_USE_TEXT_CONTEXT": os.environ.get("RAG_USE_TEXT_CONTEXT", "true"),
            "RAG_TOP_K": os.environ.get("RAG_TOP_K", "5"),
            "RAG_MIN_SIMILARITY": os.environ.get("RAG_MIN_SIMILARITY", "0.3"),
            "RAG_MIN_ACCURACY": os.environ.get("RAG_MIN_ACCURACY", "0.9"),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    total = len(cases) * cfg.trials_per_case * len(arms)
    n_done = 0
    t_start = time.perf_counter()
    rng = random.Random(cfg.base_seed)

    for case in cases:
        for trial in range(cfg.trials_per_case):
            order = list(arms)
            if args.shuffle_arms:
                rng.shuffle(order)
            for arm in order:
                n_done += 1
                t_elapsed = time.perf_counter() - t_start
                eta = (t_elapsed / n_done) * (total - n_done) if n_done > 0 else 0
                print(
                    f"[{n_done}/{total}] case={case.case_id} trial={trial} arm={arm} "
                    f"(elapsed={t_elapsed:.0f}s eta={eta:.0f}s)",
                    flush=True,
                )
                result = core.execute_trial(
                    case=case,
                    trial=trial,
                    arm=arm,
                    cfg=cfg,
                    root=ROOT_DIR,
                    out_root=args.output,
                    runtime=runtime,
                )
                core.write_jsonl_row(results_path, result.to_dict())

    metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
    metadata["wall_s_total"] = time.perf_counter() - t_start
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nDone. Results: {results_path}\nMetadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
