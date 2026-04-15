#!/usr/bin/env python3
"""
Analyze a single LLMGE run and summarize whether/ how RAG was used and what it cost.

Design goals:
- Standard library only (no torch/deap/faiss imports).
- Graceful degradation when some artifacts are missing.
- Output a single JSON report under the run's metrics directory by default.

Inputs (best-effort):
- runs/<run_id>/results/*_results.txt                 # fitness (acc, params, val_acc, train_time)
- runs/<run_id>/logs/llm/gene_*.log                   # invalid code attempts (if enabled)
- runs/<run_id>/metrics/latency-*.json                # vLLM token/latency metrics (preferred)
- metrics/data/latency-*.json                         # fallback location (filter by run_id)
- runs/<run_id>/metrics/rag_metrics.jsonl             # RAG context events (rag_context_built, etc.)
- metrics/rag_metrics.jsonl                           # fallback location (unscoped; best-effort)
- runs/<run_id>/run_metadata.json                     # run metadata/config snapshot (if present)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SOTA_MODELS_DIR = REPO_ROOT / "sota" / "ExquisiteNetV2" / "models"


def _safe_json_load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_iso8601(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        # date -Iseconds produces e.g. 2026-02-24T21:58:04-05:00
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return float(values_sorted[0])
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[f])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


@dataclass(frozen=True)
class ResultRecord:
    gene_id: str
    test_acc: float
    params: float
    val_acc: Optional[float]
    train_time_s: Optional[float]
    path: Path
    mtime_s: float


def load_results(run_dir: Path) -> List[ResultRecord]:
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return []

    records: List[ResultRecord] = []
    for path in sorted(results_dir.glob("*_results.txt")):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 2:
            continue
        try:
            test_acc = float(parts[0])
            params = float(parts[1])
            val_acc = float(parts[2]) if len(parts) > 2 and parts[2] else None
            train_time_s = float(parts[3]) if len(parts) > 3 and parts[3] else None
        except Exception:
            continue
        gene_id = path.name.replace("_results.txt", "")
        try:
            mtime_s = path.stat().st_mtime
        except Exception:
            mtime_s = 0.0
        records.append(
            ResultRecord(
                gene_id=gene_id,
                test_acc=test_acc,
                params=params,
                val_acc=val_acc,
                train_time_s=train_time_s,
                path=path,
                mtime_s=mtime_s,
            )
        )

    # Order by completion time proxy: file mtime.
    records.sort(key=lambda r: r.mtime_s)
    return records


def find_first_threshold(records: List[ResultRecord], threshold: float) -> Optional[int]:
    for idx, r in enumerate(records):
        if r.test_acc >= threshold:
            return idx + 1  # 1-based eval index
    return None


def compute_best(records: List[ResultRecord]) -> Dict[str, Any]:
    if not records:
        return {"best_test_acc": None, "best_params_at_best_acc": None, "best_gene_id": None}
    best = max(records, key=lambda r: r.test_acc)
    return {
        "best_test_acc": best.test_acc,
        "best_params_at_best_acc": best.params,
        "best_gene_id": best.gene_id,
    }


def load_invalid_code_stats(run_dir: Path) -> Dict[str, Any]:
    llm_dir = run_dir / "logs" / "llm"
    if not llm_dir.exists():
        return {"invalid_code_gene_rate": None, "avg_invalid_attempts_per_gene": None, "invalid_attempts": {}}
    attempts: Dict[str, int] = {}
    for path in llm_dir.glob("gene_*.log"):
        gene_id = path.stem.replace("gene_", "")
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        attempts[gene_id] = txt.count("INVALID LLM OUTPUT")
    if not attempts:
        return {"invalid_code_gene_rate": 0.0, "avg_invalid_attempts_per_gene": 0.0, "invalid_attempts": {}}
    num_genes = len(attempts)
    invalid_genes = sum(1 for v in attempts.values() if v > 0)
    avg_attempts = sum(attempts.values()) / max(num_genes, 1)
    return {
        "invalid_code_gene_rate": invalid_genes / max(num_genes, 1),
        "avg_invalid_attempts_per_gene": avg_attempts,
        "invalid_attempts": attempts,
    }


def load_fallback_rate(records: List[ResultRecord], run_created_at: Optional[datetime]) -> Dict[str, Any]:
    if not records:
        return {"fallback_rate": None, "fallback_genes": []}
    genes = [r.gene_id for r in records]
    fallback_genes: List[str] = []
    for gid in genes:
        marker = SOTA_MODELS_DIR / f"network_{gid}.py.fallback"
        if not marker.exists():
            continue
        if run_created_at is not None:
            try:
                mtime = datetime.fromtimestamp(marker.stat().st_mtime, tz=run_created_at.tzinfo)
                if mtime < run_created_at:
                    # Likely from a different run; ignore.
                    continue
            except Exception:
                pass
        fallback_genes.append(gid)
    return {
        "fallback_rate": (len(fallback_genes) / max(len(genes), 1)) if genes else None,
        "fallback_genes": fallback_genes,
    }


def _iter_latency_requests(latency_json: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    reqs = latency_json.get("requests")
    if isinstance(reqs, list):
        for r in reqs:
            if isinstance(r, dict):
                yield r


def load_latency_metrics(run_dir: Path, run_id: str) -> Dict[str, Any]:
    metrics_dir = run_dir / "metrics"
    candidates = sorted(metrics_dir.glob("latency-*.json"))
    if not candidates:
        # fallback: global metrics dir
        candidates = sorted((REPO_ROOT / "metrics" / "data").glob("latency-*.json"))

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    latencies: List[float] = []

    matched_files = 0
    matched_requests = 0
    for path in candidates:
        payload = _safe_json_load(path)
        if not isinstance(payload, dict):
            continue
        # If file has a run_id, use it to filter (especially in global metrics/data).
        file_run_id = payload.get("run_id")
        if file_run_id and str(file_run_id) != str(run_id):
            continue
        matched_files += 1
        for req in _iter_latency_requests(payload):
            matched_requests += 1
            try:
                prompt_tokens += int(req.get("prompt_tokens") or 0)
                completion_tokens += int(req.get("completion_tokens") or 0)
                total_tokens += int(req.get("total_tokens") or 0)
            except Exception:
                pass
            try:
                lat = req.get("_latency_sec")
                if lat is not None:
                    latencies.append(float(lat))
            except Exception:
                pass

    return {
        "latency_files": matched_files,
        "latency_requests": matched_requests,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_llm_latency_sec": sum(latencies) if latencies else 0.0,
        "p50_latency_sec": _percentile(latencies, 50) if latencies else None,
        "p95_latency_sec": _percentile(latencies, 95) if latencies else None,
    }


def load_rag_usage(run_dir: Path) -> Dict[str, Any]:
    # Prefer run-scoped metrics if present; fall back to repo-level file.
    candidates = []
    run_scoped = run_dir / "metrics" / "rag_metrics.jsonl"
    if run_scoped.exists():
        candidates.append(run_scoped)
    repo_scoped = REPO_ROOT / "metrics" / "rag_metrics.jsonl"
    if repo_scoped.exists():
        candidates.append(repo_scoped)

    events: List[Dict[str, Any]] = []
    for path in candidates:
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    events.append(obj)
        except Exception:
            continue

        # If we found a run-scoped file, do not mix in the repo-scoped file.
        if path == run_scoped:
            break

    context_events = [e for e in events if e.get("event_type") == "rag_context_built"]
    if not context_events:
        return {
            "rag_events_total": len(events),
            "rag_context_events": 0,
            "rag_nonempty_fraction": None,
            "avg_retrieved_code_n": None,
            "avg_retrieved_text_n": None,
            "avg_retrieved_memory_n": None,
            "avg_context_words_code": None,
            "avg_context_words_text": None,
            "avg_context_words_memory": None,
        }

    code_ns = [int(e.get("retrieved_code_n") or 0) for e in context_events]
    text_ns = [int(e.get("retrieved_text_n") or 0) for e in context_events]
    memory_ns = [int(e.get("retrieved_memory_n") or 0) for e in context_events]
    nonempty = [1 for c, t, m in zip(code_ns, text_ns, memory_ns) if (c + t + m) > 0]
    words_code = [int(e.get("context_words_code") or 0) for e in context_events]
    words_text = [int(e.get("context_words_text") or 0) for e in context_events]
    words_memory = [int(e.get("context_words_memory") or 0) for e in context_events]

    def _avg(xs: List[int]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    return {
        "rag_events_total": len(events),
        "rag_context_events": len(context_events),
        "rag_nonempty_fraction": len(nonempty) / max(len(context_events), 1),
        "avg_retrieved_code_n": _avg(code_ns),
        "avg_retrieved_text_n": _avg(text_ns),
        "avg_retrieved_memory_n": _avg(memory_ns),
        "avg_context_words_code": _avg(words_code),
        "avg_context_words_text": _avg(words_text),
        "avg_context_words_memory": _avg(words_memory),
    }


def write_csv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to a single run directory (runs/<RUN_ID>).")
    parser.add_argument("--threshold", type=float, default=0.90, help="Test accuracy threshold for time-to-threshold.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <run>/metrics/rag_impact_report.json).",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV path to append one summary row (useful for aggregating many runs).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_id = run_dir.name

    metadata = _safe_json_load(run_dir / "run_metadata.json")
    created_at = None
    if isinstance(metadata, dict):
        created_at = _parse_iso8601(str(metadata.get("created_at") or metadata.get("start_time") or ""))
        run_id = str(metadata.get("run_id") or run_id)

    results = load_results(run_dir)
    evals_to_thresh = find_first_threshold(results, threshold=args.threshold)
    best = compute_best(results)

    invalid_stats = load_invalid_code_stats(run_dir)
    fallback_stats = load_fallback_rate(results, run_created_at=created_at)
    latency_stats = load_latency_metrics(run_dir, run_id=run_id)
    rag_usage = load_rag_usage(run_dir)

    wall_time_to_thresh_s = None
    if created_at is not None and evals_to_thresh is not None and results:
        first_hit = results[evals_to_thresh - 1]
        try:
            hit_time = datetime.fromtimestamp(first_hit.mtime_s, tz=created_at.tzinfo)
            wall_time_to_thresh_s = (hit_time - created_at).total_seconds()
        except Exception:
            wall_time_to_thresh_s = None

    report = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "threshold": args.threshold,
        "num_results": len(results),
        "evals_to_reach_threshold": evals_to_thresh,
        "wall_time_to_reach_threshold_sec": wall_time_to_thresh_s,
        **best,
        **fallback_stats,
        **invalid_stats,
        **latency_stats,
        **rag_usage,
        "metadata": metadata if isinstance(metadata, dict) else None,
    }

    out_path = Path(args.output) if args.output else (run_dir / "metrics" / "rag_impact_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.csv_out:
        row = {
            "run_id": run_id,
            "num_results": len(results),
            "evals_to_reach_threshold": evals_to_thresh,
            "best_test_acc": best.get("best_test_acc"),
            "best_params_at_best_acc": best.get("best_params_at_best_acc"),
            "fallback_rate": fallback_stats.get("fallback_rate"),
            "invalid_code_gene_rate": invalid_stats.get("invalid_code_gene_rate"),
            "avg_invalid_attempts_per_gene": invalid_stats.get("avg_invalid_attempts_per_gene"),
            "total_tokens": latency_stats.get("total_tokens"),
            "total_llm_latency_sec": latency_stats.get("total_llm_latency_sec"),
            "rag_nonempty_fraction": rag_usage.get("rag_nonempty_fraction"),
            "avg_retrieved_code_n": rag_usage.get("avg_retrieved_code_n"),
            "avg_retrieved_text_n": rag_usage.get("avg_retrieved_text_n"),
            "avg_retrieved_memory_n": rag_usage.get("avg_retrieved_memory_n"),
        }
        write_csv_row(Path(args.csv_out), row)
        print(f"Appended row to {args.csv_out}")


if __name__ == "__main__":
    main()

