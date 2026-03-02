#!/usr/bin/env python3
"""
Summarize many run-level RAG impact reports into a condition-level table.

Reads `runs/*/metrics/rag_impact_report.json` (created by scripts/analyze_rag_impact.py)
and groups by `metadata.experiment.condition` (or `experiment.condition`).

Outputs a CSV plus a compact JSON summary.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_load(path: Path) -> dict | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _get_condition(report: dict) -> str:
    # Prefer report.metadata.experiment.condition
    meta = report.get("metadata")
    if isinstance(meta, dict):
        exp = meta.get("experiment")
        if isinstance(exp, dict):
            cond = exp.get("condition")
            if cond:
                return str(cond)
    # Fallback: report.metadata.condition
    if isinstance(meta, dict) and meta.get("condition"):
        return str(meta["condition"])
    return "unknown"


def _mean_ci(xs: List[float]) -> tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    se = math.sqrt(var) / math.sqrt(len(xs))
    # Normal approx 95% CI half-width. Good enough for quick comparisons.
    return float(mean), float(1.96 * se)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Runs directory (default: runs).",
    )
    parser.add_argument(
        "--glob",
        default="*/metrics/rag_impact_report.json",
        help="Glob under runs-root to find per-run reports.",
    )
    parser.add_argument(
        "--out-csv",
        default="experiments/rag_ablation_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-json",
        default="experiments/rag_ablation_summary.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    runs_root = REPO_ROOT / args.runs_root
    paths = sorted(runs_root.glob(args.glob))
    reports: List[dict] = []
    for p in paths:
        r = _safe_load(p)
        if r:
            reports.append(r)

    by_cond: Dict[str, List[dict]] = {}
    for r in reports:
        by_cond.setdefault(_get_condition(r), []).append(r)

    rows: List[dict] = []
    summary: Dict[str, Any] = {"conditions": {}, "num_reports": len(reports)}

    for cond, rs in sorted(by_cond.items()):
        evals = [r.get("evals_to_reach_threshold") for r in rs]
        evals_f = [float(x) for x in evals if isinstance(x, (int, float))]
        bests_f = [float(r["best_test_acc"]) for r in rs if isinstance(r.get("best_test_acc"), (int, float))]
        tokens_f = [float(r["total_tokens"]) for r in rs if isinstance(r.get("total_tokens"), (int, float))]
        nonempty_f = [
            float(r["rag_nonempty_fraction"])
            for r in rs
            if isinstance(r.get("rag_nonempty_fraction"), (int, float))
        ]

        mean_evals, ci_evals = _mean_ci(evals_f)
        mean_best, ci_best = _mean_ci(bests_f)
        mean_tokens, ci_tokens = _mean_ci(tokens_f)
        mean_nonempty, ci_nonempty = _mean_ci(nonempty_f)

        row = {
            "condition": cond,
            "n": len(rs),
            "evals_to_threshold_mean": mean_evals,
            "evals_to_threshold_ci95": ci_evals,
            "best_test_acc_mean": mean_best,
            "best_test_acc_ci95": ci_best,
            "total_tokens_mean": mean_tokens,
            "total_tokens_ci95": ci_tokens,
            "rag_nonempty_fraction_mean": mean_nonempty,
            "rag_nonempty_fraction_ci95": ci_nonempty,
        }
        rows.append(row)
        summary["conditions"][cond] = row

    out_csv = REPO_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Manual CSV writer to avoid importing csv (still stdlib, but keep it tiny).
    headers = [
        "condition",
        "n",
        "evals_to_threshold_mean",
        "evals_to_threshold_ci95",
        "best_test_acc_mean",
        "best_test_acc_ci95",
        "total_tokens_mean",
        "total_tokens_ci95",
        "rag_nonempty_fraction_mean",
        "rag_nonempty_fraction_ci95",
    ]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join("" if row[h] is None else str(row[h]) for h in headers))
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_json = REPO_ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

