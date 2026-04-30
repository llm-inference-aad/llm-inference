#!/usr/bin/env python3
"""Summarize cache-aware gateway and worker latency metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[idx]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"Skipping unreadable JSON {path}: {exc}")
        return None


def collect_worker_requests(metrics_path: Path) -> list[dict]:
    requests = []
    for path in metrics_path.glob("latency-*.json"):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        for item in data.get("requests", []):
            item = dict(item)
            item["_metrics_file"] = str(path)
            requests.append(item)
    return requests


def collect_gateway_requests(metrics_path: Path) -> list[dict]:
    requests = []
    for path in metrics_path.glob("gateway-*.json"):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        for item in data.get("requests", []):
            item = dict(item)
            item["_metrics_file"] = str(path)
            requests.append(item)
    return requests


def collect_benchmarks(metrics_path: Path) -> list[dict]:
    benchmarks = []
    for path in metrics_path.glob("cache-affinity-benchmark-*.json"):
        data = load_json(path)
        if isinstance(data, dict):
            data["_metrics_file"] = str(path)
            benchmarks.append(data)
    return benchmarks


def summarize_requests(requests: list[dict]) -> dict:
    latencies = [r.get("_latency_sec") for r in requests if isinstance(r.get("_latency_sec"), (int, float))]
    ttfts = [r.get("ttft_sec") for r in requests if isinstance(r.get("ttft_sec"), (int, float))]
    hit_ttfts = [r.get("ttft_sec") for r in requests if r.get("cache_hit") and isinstance(r.get("ttft_sec"), (int, float))]
    miss_ttfts = [r.get("ttft_sec") for r in requests if not r.get("cache_hit") and isinstance(r.get("ttft_sec"), (int, float))]
    hits = sum(1 for r in requests if r.get("cache_hit"))
    total = len(requests)
    return {
        "total_requests": total,
        "cache_hits": hits,
        "cache_misses": total - hits,
        "cache_hit_rate": hits / total if total else None,
        "latency_mean_sec": statistics.mean(latencies) if latencies else None,
        "latency_p50_sec": percentile(latencies, 50),
        "latency_p95_sec": percentile(latencies, 95),
        "ttft_mean_sec": statistics.mean(ttfts) if ttfts else None,
        "hit_ttft_mean_sec": statistics.mean(hit_ttfts) if hit_ttfts else None,
        "miss_ttft_mean_sec": statistics.mean(miss_ttfts) if miss_ttfts else None,
    }


def summarize_gateway(requests: list[dict]) -> dict:
    hits = sum(1 for r in requests if r.get("cache_hit"))
    total = len(requests)
    response_times = [r.get("response_time_sec") for r in requests if isinstance(r.get("response_time_sec"), (int, float))]
    return {
        "total_requests": total,
        "cache_hit_routes": hits,
        "cache_miss_routes": total - hits,
        "cache_hit_rate": hits / total if total else None,
        "gateway_response_mean_sec": statistics.mean(response_times) if response_times else None,
        "gateway_response_p50_sec": percentile(response_times, 50),
        "gateway_response_p95_sec": percentile(response_times, 95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="Run ID under runs/<RUN_ID>/metrics")
    parser.add_argument("--metrics-dir", help="Explicit metrics directory")
    args = parser.parse_args()

    root = repo_root()
    if args.metrics_dir:
        metrics_path = Path(args.metrics_dir)
    elif args.run_id:
        metrics_path = root / "runs" / args.run_id / "metrics"
    else:
        metrics_path = root / "metrics" / "data"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_path}")

    worker_requests = collect_worker_requests(metrics_path)
    gateway_requests = collect_gateway_requests(metrics_path)
    benchmarks = collect_benchmarks(metrics_path)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metrics_dir": str(metrics_path),
        "worker_summary": summarize_requests(worker_requests),
        "gateway_summary": summarize_gateway(gateway_requests),
        "benchmarks": [
            {
                "metrics_file": item.get("_metrics_file"),
                "summary": item.get("summary", {}),
            }
            for item in benchmarks
        ],
    }

    out_file = metrics_path / f"cache-metrics-summary-{int(time.time())}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {out_file}")


if __name__ == "__main__":
    main()
