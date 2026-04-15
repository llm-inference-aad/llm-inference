#!/usr/bin/env python3
"""
Summarize memory-store ablation runs and optionally generate comparison plots.

Expected input files:
- runs/*/metrics/rag_impact_report.json
with metadata.experiment.ablation == "memory_store" and
metadata.experiment.condition in {"memory_off", "memory_on"}.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_load(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _condition(report: dict) -> str:
    meta = report.get("metadata")
    if isinstance(meta, dict):
        exp = meta.get("experiment")
        if isinstance(exp, dict):
            cond = exp.get("condition")
            if cond:
                return str(cond)
    return "unknown"


def _ablation(report: dict) -> Optional[str]:
    meta = report.get("metadata")
    if isinstance(meta, dict):
        exp = meta.get("experiment")
        if isinstance(exp, dict):
            value = exp.get("ablation")
            if value is not None:
                return str(value)
    return None


def _matrix_id(report: dict) -> Optional[str]:
    meta = report.get("metadata")
    if isinstance(meta, dict):
        exp = meta.get("experiment")
        if isinstance(exp, dict):
            value = exp.get("matrix_id")
            if value is not None:
                return str(value)
    return None


def _mean_ci(xs: List[float]) -> tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    se = math.sqrt(var) / math.sqrt(len(xs))
    return float(mean), float(1.96 * se)


def _numeric(rs: List[dict], key: str) -> List[float]:
    out: List[float] = []
    for r in rs:
        value = r.get(key)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _write_csv(path: Path, rows: List[dict], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join("" if row.get(h) is None else str(row[h]) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_if_possible(output_dir: Path, grouped: Dict[str, List[dict]]) -> List[str]:
    generated: List[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return generated

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in ("memory_off", "memory_on") if c in grouped]
    if not conditions:
        return generated

    # 1) Per-run best accuracy scatter
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, cond in enumerate(conditions):
        ys = _numeric(grouped[cond], "best_test_acc")
        xs = [idx + 1] * len(ys)
        ax.scatter(xs, ys, label=cond, alpha=0.8)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Memory Store Ablation: Per-run Best Accuracy")
    ax.grid(alpha=0.3, axis="y")
    out1 = output_dir / "best_accuracy_scatter.png"
    fig.tight_layout()
    fig.savefig(out1, dpi=180)
    plt.close(fig)
    generated.append(str(out1))

    # 2) Mean evals-to-threshold (with 95% CI)
    means: List[float] = []
    cis: List[float] = []
    valid_labels: List[str] = []
    for cond in conditions:
        m, ci = _mean_ci(_numeric(grouped[cond], "evals_to_reach_threshold"))
        if m is None:
            continue
        valid_labels.append(cond)
        means.append(m)
        cis.append(ci or 0.0)
    if valid_labels:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(valid_labels, means, yerr=cis, capsize=4)
        ax.set_ylabel("Evals to reach threshold")
        ax.set_title("Memory Store Ablation: Efficiency")
        ax.grid(alpha=0.3, axis="y")
        out2 = output_dir / "evals_to_threshold_bar.png"
        fig.tight_layout()
        fig.savefig(out2, dpi=180)
        plt.close(fig)
        generated.append(str(out2))

    # 3) Mean retrieved-memory count (sanity check that memory arm is active)
    means_mem: List[float] = []
    labels_mem: List[str] = []
    for cond in conditions:
        m, _ = _mean_ci(_numeric(grouped[cond], "avg_retrieved_memory_n"))
        if m is None:
            continue
        labels_mem.append(cond)
        means_mem.append(m)
    if labels_mem:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels_mem, means_mem)
        ax.set_ylabel("Avg retrieved memory entries")
        ax.set_title("Memory Retrieval Activity by Condition")
        ax.grid(alpha=0.3, axis="y")
        out3 = output_dir / "retrieved_memory_bar.png"
        fig.tight_layout()
        fig.savefig(out3, dpi=180)
        plt.close(fig)
        generated.append(str(out3))

    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs", help="Runs directory.")
    parser.add_argument(
        "--glob",
        default="*/metrics/rag_impact_report.json",
        help="Glob under runs-root for per-run report files.",
    )
    parser.add_argument(
        "--matrix-id",
        default=None,
        help="Optional matrix_id filter to compare only one experiment batch.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/memory_ablation",
        help="Output directory for summary artifacts.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )
    args = parser.parse_args()

    runs_root = REPO_ROOT / args.runs_root
    report_paths = sorted(runs_root.glob(args.glob))
    reports: List[dict] = []
    for path in report_paths:
        report = _safe_load(path)
        if not report:
            continue
        if _ablation(report) != "memory_store":
            continue
        if args.matrix_id and _matrix_id(report) != args.matrix_id:
            continue
        reports.append(report)

    grouped: Dict[str, List[dict]] = {}
    for report in reports:
        cond = _condition(report)
        grouped.setdefault(cond, []).append(report)

    rows: List[dict] = []
    summary: Dict[str, Any] = {
        "num_reports": len(reports),
        "conditions": {},
        "matrix_id_filter": args.matrix_id,
    }

    for cond, rs in sorted(grouped.items()):
        best_m, best_ci = _mean_ci(_numeric(rs, "best_test_acc"))
        evals_m, evals_ci = _mean_ci(_numeric(rs, "evals_to_reach_threshold"))
        mem_m, mem_ci = _mean_ci(_numeric(rs, "avg_retrieved_memory_n"))
        row = {
            "condition": cond,
            "n": len(rs),
            "best_test_acc_mean": best_m,
            "best_test_acc_ci95": best_ci,
            "evals_to_threshold_mean": evals_m,
            "evals_to_threshold_ci95": evals_ci,
            "avg_retrieved_memory_n_mean": mem_m,
            "avg_retrieved_memory_n_ci95": mem_ci,
        }
        rows.append(row)
        summary["conditions"][cond] = row

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_headers = [
        "condition",
        "n",
        "best_test_acc_mean",
        "best_test_acc_ci95",
        "evals_to_threshold_mean",
        "evals_to_threshold_ci95",
        "avg_retrieved_memory_n_mean",
        "avg_retrieved_memory_n_ci95",
    ]
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    per_run_csv = out_dir / "per_run.csv"

    _write_csv(summary_csv, rows, csv_headers)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    per_run_rows: List[dict] = []
    for report in reports:
        per_run_rows.append(
            {
                "run_id": report.get("run_id"),
                "condition": _condition(report),
                "best_test_acc": report.get("best_test_acc"),
                "evals_to_reach_threshold": report.get("evals_to_reach_threshold"),
                "avg_retrieved_memory_n": report.get("avg_retrieved_memory_n"),
                "run_dir": report.get("run_dir"),
            }
        )
    _write_csv(
        per_run_csv,
        per_run_rows,
        [
            "run_id",
            "condition",
            "best_test_acc",
            "evals_to_reach_threshold",
            "avg_retrieved_memory_n",
            "run_dir",
        ],
    )

    generated_plots: List[str] = []
    if not args.no_plots:
        generated_plots = _plot_if_possible(out_dir, grouped)

    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {per_run_csv}")
    if generated_plots:
        for plot in generated_plots:
            print(f"Wrote {plot}")
    elif not args.no_plots:
        print("Plot generation skipped (matplotlib unavailable or no data).")


if __name__ == "__main__":
    main()

