#!/usr/bin/env python3
"""Summarize baseline vs SmoothQuant latency metrics across run directories."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def collect_run_rows(runs_dir: Path) -> list[dict]:
    rows: list[dict] = []

    for config_path in sorted(runs_dir.glob("*/experiment_config.json")):
        run_dir = config_path.parent
        if run_dir.is_symlink():
            continue
        config = load_json(config_path)
        metric_files = sorted((run_dir / "metrics").glob("latency-*.json"))

        all_requests = []
        metric_hashes = []
        for metric_file in metric_files:
            metrics = load_json(metric_file)
            metric_hashes.append(metrics.get("run_hash", metric_file.stem.replace("latency-", "")))
            all_requests.extend(metrics.get("requests", []))

        latencies = [float(req["_latency_sec"]) for req in all_requests if "_latency_sec" in req]
        batch_times = [
            float(req["batch_processing_time_sec"])
            for req in all_requests
            if "batch_processing_time_sec" in req
        ]
        eval_scores = [
            float(req["evaluation_score"])
            for req in all_requests
            if req.get("evaluation_score") is not None
        ]

        rows.append(
            {
                "run_id": run_dir.name,
                "label": config.get("label", run_dir.name),
                "smoothquant_enabled": int(config.get("smoothquant_enabled", 0)),
                "smoothquant_alpha": config.get("smoothquant_alpha", ""),
                "batch_size": int(config.get("batch_size", 0) or 0),
                "metric_files": len(metric_files),
                "metric_hashes": ";".join(metric_hashes),
                "request_count": len(latencies),
                "latency_mean_sec": statistics.fmean(latencies) if latencies else 0.0,
                "latency_median_sec": statistics.median(latencies) if latencies else 0.0,
                "latency_p95_sec": percentile(latencies, 95),
                "latency_min_sec": min(latencies) if latencies else 0.0,
                "latency_max_sec": max(latencies) if latencies else 0.0,
                "batch_processing_mean_sec": statistics.fmean(batch_times) if batch_times else 0.0,
                "evaluation_score_mean": statistics.fmean(eval_scores) if eval_scores else 0.0,
            }
        )

    return rows


def grouped_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        if row["request_count"] <= 0:
            continue
        key = (row["smoothquant_enabled"], row["batch_size"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (smoothquant_enabled, batch_size), group_rows in sorted(groups.items()):
        means = [row["latency_mean_sec"] for row in group_rows]
        p95s = [row["latency_p95_sec"] for row in group_rows]
        counts = [row["request_count"] for row in group_rows]
        summary.append(
            {
                "smoothquant_enabled": smoothquant_enabled,
                "batch_size": batch_size,
                "runs_with_requests": len(group_rows),
                "total_requests": sum(counts),
                "mean_of_run_mean_latency_sec": statistics.fmean(means),
                "median_of_run_mean_latency_sec": statistics.median(means),
                "mean_of_run_p95_latency_sec": statistics.fmean(p95s),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("No rows found.")
        return
    widths = {
        col: max(len(col), *(len(f"{row[col]:.4f}") if isinstance(row[col], float) else len(str(row[col])) for row in rows))
        for col in columns
    }
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            cells.append((f"{value:.4f}" if isinstance(value, float) else str(value)).ljust(widths[col]))
        print("  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and SmoothQuant run metrics.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders.")
    parser.add_argument("--out-dir", default="metrics/comparisons", help="Directory for CSV summaries.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    rows = collect_run_rows(runs_dir)
    summary = grouped_summary(rows)

    write_csv(out_dir / "smoothquant_run_summary.csv", rows)
    write_csv(out_dir / "smoothquant_group_summary.csv", summary)

    print("\nPer-run summary:")
    print_table(
        rows,
        [
            "run_id",
            "smoothquant_enabled",
            "batch_size",
            "request_count",
            "latency_mean_sec",
            "latency_p95_sec",
            "evaluation_score_mean",
        ],
    )

    print("\nGrouped summary:")
    print_table(
        summary,
        [
            "smoothquant_enabled",
            "batch_size",
            "runs_with_requests",
            "total_requests",
            "mean_of_run_mean_latency_sec",
            "mean_of_run_p95_latency_sec",
        ],
    )

    print(f"\nWrote {out_dir / 'smoothquant_run_summary.csv'}")
    print(f"Wrote {out_dir / 'smoothquant_group_summary.csv'}")


if __name__ == "__main__":
    main()
