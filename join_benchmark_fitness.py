#!/usr/bin/env python3
"""Join benchmark outputs with historical fitness artifacts.

This script supports two modes:
1) Direct join: if benchmark request records include `gene_id`, it joins each request
   against fitness parsed from `*_results.txt`.
2) Derived comparison: when there are no gene IDs (typical for RAG decoding sweeps),
   it still produces side-by-side benchmark metrics and historical fitness summaries.

Outputs are written to an output directory:
- benchmark_fitness_summary.json
- benchmark_fitness_direct_join.jsonl (only when direct joins exist)
- benchmark_fitness_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _safe_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    idx = int((len(values_sorted) - 1) * q)
    return values_sorted[idx]


def load_fitness_map(results_dir: Path) -> dict[str, dict[str, float]]:
    fitness: dict[str, dict[str, float]] = {}
    for fp in sorted(results_dir.glob("*_results.txt")):
        gene_id = fp.stem.replace("_results", "")
        try:
            parts = fp.read_text(encoding="utf-8").strip().split(",")
            if len(parts) < 4:
                continue
            test_acc = float(parts[0])
            total_params = float(parts[1])
            val_acc = float(parts[2])
            train_time = float(parts[3])
        except Exception:
            continue
        fitness[gene_id] = {
            "test_accuracy": test_acc,
            "total_parameters": total_params,
            "validation_accuracy": val_acc,
            "train_time_sec": train_time,
        }
    return fitness


def load_requests(requests_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with requests_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_historical_fitness_summary(fitness_map: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows = list(fitness_map.values())
    test_acc = [r["test_accuracy"] for r in rows]
    total_params = [r["total_parameters"] for r in rows]
    train_time = [r["train_time_sec"] for r in rows]

    return {
        "count": len(rows),
        "test_accuracy": {
            "mean": _safe_mean(test_acc),
            "median": _safe_median(test_acc),
            "stdev": _safe_stdev(test_acc),
            "p90": _safe_quantile(test_acc, 0.90),
            "max": max(test_acc) if test_acc else None,
        },
        "total_parameters": {
            "mean": _safe_mean(total_params),
            "median": _safe_median(total_params),
            "stdev": _safe_stdev(total_params),
            "min": min(total_params) if total_params else None,
        },
        "train_time_sec": {
            "mean": _safe_mean(train_time),
            "median": _safe_median(train_time),
            "stdev": _safe_stdev(train_time),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        required=True,
        help="Path to benchmark_summary.json",
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=None,
        help="Path to all_requests.jsonl (defaults to sibling of benchmark summary)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("sota/ExquisiteNetV2/results"),
        help="Directory containing *_results.txt historical fitness files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to benchmark summary parent)",
    )
    args = parser.parse_args()

    benchmark_summary = json.loads(args.benchmark_summary.read_text(encoding="utf-8"))

    requests_path = args.requests
    if requests_path is None:
        requests_path = args.benchmark_summary.parent / "all_requests.jsonl"

    if not requests_path.exists():
        raise FileNotFoundError(f"Requests file not found: {requests_path}")

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Fitness results directory not found: {args.results_dir}")

    out_dir = args.output_dir or args.benchmark_summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fitness_map = load_fitness_map(args.results_dir)
    requests = load_requests(requests_path)

    direct_join_rows: list[dict[str, Any]] = []
    config_joined: dict[str, list[dict[str, Any]]] = {}

    for r in requests:
        gene_id = r.get("gene_id")
        if not gene_id:
            continue
        fit = fitness_map.get(str(gene_id))
        if not fit:
            continue
        cfg = str(r.get("config_name", "unknown"))
        joined = {
            "config_name": cfg,
            "request_id": r.get("request_id"),
            "gene_id": gene_id,
            "response_time_sec": r.get("response_time_sec"),
            "evaluationScore": r.get("evaluationScore"),
            "vllm_num_speculative_tokens": r.get("vllm_num_speculative_tokens"),
            "test_accuracy": fit["test_accuracy"],
            "total_parameters": fit["total_parameters"],
            "validation_accuracy": fit["validation_accuracy"],
            "train_time_sec": fit["train_time_sec"],
        }
        direct_join_rows.append(joined)
        config_joined.setdefault(cfg, []).append(joined)

    historical = build_historical_fitness_summary(fitness_map)

    per_config = benchmark_summary.get("per_config", {})
    derived_configs: dict[str, Any] = {}
    for cfg, metrics in per_config.items():
        joined_rows = config_joined.get(cfg, [])
        if joined_rows:
            acc = [float(x["test_accuracy"]) for x in joined_rows]
            params = [float(x["total_parameters"]) for x in joined_rows]
            derived = {
                "direct_join_count": len(joined_rows),
                "joined_test_accuracy_mean": _safe_mean(acc),
                "joined_test_accuracy_median": _safe_median(acc),
                "joined_total_parameters_mean": _safe_mean(params),
            }
        else:
            derived = {
                "direct_join_count": 0,
                "joined_test_accuracy_mean": None,
                "joined_test_accuracy_median": None,
                "joined_total_parameters_mean": None,
            }

        derived_configs[cfg] = {
            "benchmark_metrics": metrics,
            "fitness_comparison": derived,
        }

    summary = {
        "benchmark_summary_path": str(args.benchmark_summary),
        "requests_path": str(requests_path),
        "results_dir": str(args.results_dir),
        "historical_fitness": historical,
        "total_requests": len(requests),
        "requests_with_gene_id": sum(1 for r in requests if r.get("gene_id")),
        "direct_join_rows": len(direct_join_rows),
        "mode": "direct_join" if direct_join_rows else "derived_only",
        "per_config": derived_configs,
    }

    summary_path = out_dir / "benchmark_fitness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    join_path = out_dir / "benchmark_fitness_direct_join.jsonl"
    if direct_join_rows:
        with join_path.open("w", encoding="utf-8") as f:
            for row in direct_join_rows:
                f.write(json.dumps(row) + "\n")

    report_lines = [
        "# Benchmark + Fitness Comparison",
        "",
        f"- Benchmark summary: `{args.benchmark_summary}`",
        f"- Requests file: `{requests_path}`",
        f"- Fitness source: `{args.results_dir}`",
        f"- Mode: `{summary['mode']}`",
        f"- Requests parsed: `{summary['total_requests']}`",
        f"- Requests with gene_id: `{summary['requests_with_gene_id']}`",
        f"- Direct joins: `{summary['direct_join_rows']}`",
        "",
        "## Historical Fitness Snapshot",
        f"- Count: `{historical['count']}`",
        f"- Test accuracy mean: `{historical['test_accuracy']['mean']}`",
        f"- Test accuracy p90: `{historical['test_accuracy']['p90']}`",
        f"- Parameter median: `{historical['total_parameters']['median']}`",
        f"- Train-time mean (s): `{historical['train_time_sec']['mean']}`",
        "",
        "## Per-Config Comparison",
    ]

    for cfg, data in derived_configs.items():
        bm = data["benchmark_metrics"]
        fc = data["fitness_comparison"]
        report_lines.append(f"- `{cfg}`: latency_mean={bm.get('latency_mean')} score_mean={bm.get('score_mean')} join_count={fc.get('direct_join_count')}")

    report_path = out_dir / "benchmark_fitness_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote summary: {summary_path}")
    if direct_join_rows:
        print(f"Wrote direct joins: {join_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
