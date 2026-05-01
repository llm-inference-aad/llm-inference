#!/usr/bin/env python3
"""Generate a Pareto-front comparison figure from an ablation matrix manifest.

Reads each run's rag_ledger.jsonl via src.rag.bookkeeping.replay_ledger, computes
per-condition Pareto fronts via src.rag.pareto_policy.compute_pareto_front, and
produces:
  - runs/ablation_<timestamp>/plots/pareto_comparison.png  (300 DPI)
  - runs/ablation_<timestamp>/plots/pareto_data.csv        (one row per event)

Usage::

    python scripts/plot_pareto_front_comparison.py \\
        --manifest runs/ablation_20260420_120000/manifest.json

    # With overrides
    python scripts/plot_pareto_front_comparison.py \\
        --manifest runs/ablation_20260420_120000/manifest.json \\
        --output runs/ablation_20260420_120000/plots/ \\
        --title "RAG vs Baseline Pareto Front" \\
        --condition-colors baseline:#1f77b4 rag-faiss:#ff7f0e
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Use Agg backend before importing pyplot to avoid display errors on headless
# nodes and in test environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — must come after matplotlib.use()

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default palette — enough for 6 conditions.
_DEFAULT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_events_for_run(run_id: str, run_dir: Optional[Path] = None):
    """Stream MutationEvent objects from a run's rag_ledger.jsonl.

    Args:
        run_id: The run identifier.
        run_dir: If provided, look for the ledger relative to this directory.
            Otherwise, fall back to the default ledger path (cfg.constants or
            runs/<run_id>/metrics/rag_ledger.jsonl).

    Returns:
        A list of MutationEvent objects (may be empty).
    """
    from src.rag.bookkeeping import replay_ledger, RunLedger  # noqa: PLC0415

    if run_dir is not None:
        ledger_path = run_dir / "metrics" / "rag_ledger.jsonl"
    else:
        ledger_path = None

    try:
        return list(replay_ledger(run_id, ledger_path=ledger_path))
    except Exception as exc:
        print(f"  WARNING: could not load ledger for run {run_id}: {exc}", file=sys.stderr)
        return []


def _load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pareto computation
# ---------------------------------------------------------------------------


def _get_event_metrics(event) -> Optional[Tuple[float, float]]:
    """Return (test_accuracy, total_params) or None."""
    eo = event.eval_outputs
    if not eo:
        return None
    acc = eo.get("test_accuracy") if eo.get("test_accuracy") is not None else eo.get("test_acc")
    params = eo.get("total_params")
    if acc is None or params is None:
        return None
    try:
        return float(acc), float(params)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _write_csv(
    path: Path,
    rows: List[Dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("condition,run_id,gene_id,test_accuracy,total_params,is_pareto_eligible,on_front\n")
        return
    fieldnames = ["condition", "run_id", "gene_id", "test_accuracy", "total_params", "is_pareto_eligible", "on_front"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _sort_front_for_plot(
    points: Sequence[Tuple[float, float]]
) -> Tuple[List[float], List[float]]:
    """Sort Pareto front points by total_params for a clean step-line plot."""
    sorted_pts = sorted(points, key=lambda p: p[1])  # sort by params ascending
    xs = [p[1] for p in sorted_pts]
    ys = [p[0] for p in sorted_pts]
    return xs, ys


def build_plot(
    *,
    manifest_path: Path,
    output_dir: Path,
    title: str,
    condition_colors: Dict[str, str],
) -> Tuple[Path, Path]:
    """Build and save the Pareto comparison figure.

    Args:
        manifest_path: Path to manifest.json.
        output_dir: Directory to write outputs into.
        title: Figure title.
        condition_colors: Map from condition name to hex color.

    Returns:
        (png_path, csv_path)
    """
    from src.rag.pareto_policy import compute_pareto_front  # noqa: PLC0415

    manifest = _load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])

    # Collect events per condition.
    condition_events: Dict[str, list] = {}
    for job in jobs:
        condition = job.get("condition", "unknown")
        run_id = job.get("run_id", "")

        # Determine run_dir: manifest output_dir/../<run_id> is the standard layout,
        # but also try runs/<run_id> from REPO_ROOT.
        run_dir = REPO_ROOT / "runs" / run_id
        if not run_dir.exists():
            # Try resolving relative to manifest output_dir's parent.
            manifest_out = Path(manifest.get("output_dir", str(manifest_path.parent)))
            candidate = manifest_out.parent / run_id
            if candidate.exists():
                run_dir = candidate

        events = _load_events_for_run(run_id, run_dir=run_dir if run_dir.exists() else None)
        if condition not in condition_events:
            condition_events[condition] = []
        condition_events[condition].extend(events)

    # Prepare figure.
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale("log")
    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)

    csv_rows: List[Dict] = []
    color_cycle = iter(_DEFAULT_COLORS)

    for condition, events in sorted(condition_events.items()):
        color = condition_colors.get(condition) or next(color_cycle, "#333333")

        # Filter to events with valid eval_outputs.
        eval_events = [e for e in events if _get_event_metrics(e) is not None]

        # Compute Pareto front.
        front_events = compute_pareto_front(eval_events)
        front_set = set(id(e) for e in front_events)

        # Scatter: all evaluated events (low alpha).
        scatter_xs = []
        scatter_ys = []
        for event in eval_events:
            metrics = _get_event_metrics(event)
            if metrics is None:
                continue
            acc, params = metrics
            scatter_xs.append(params)
            scatter_ys.append(acc)
            on_front = id(event) in front_set
            csv_rows.append(
                {
                    "condition": condition,
                    "run_id": event.run_id,
                    "gene_id": event.child_gene_id,
                    "test_accuracy": acc,
                    "total_params": params,
                    "is_pareto_eligible": event.is_pareto_eligible,
                    "on_front": on_front,
                }
            )

        ax.scatter(
            scatter_xs,
            scatter_ys,
            color=color,
            alpha=0.3,
            s=40,
            label=None,
        )

        # Pareto front line (high alpha, thicker).
        front_points = []
        for event in front_events:
            metrics = _get_event_metrics(event)
            if metrics is not None:
                front_points.append(metrics)

        label = f"{condition} ({len(eval_events)} pts, {len(front_events)} on front)"
        if front_points:
            xs, ys = _sort_front_for_plot(front_points)
            ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, marker="o", markersize=7, label=label)
        else:
            # No front points — still show legend entry via dummy plot.
            ax.plot([], [], color=color, linewidth=2.5, alpha=0.9, label=label)

    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "pareto_comparison.png"
    csv_path = output_dir / "pareto_data.csv"

    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_csv(csv_path, csv_rows)

    return png_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot Pareto front comparison from an ablation manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        metavar="PATH",
        help="Path to ablation manifest.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Output directory for PNG and CSV. Defaults to <manifest_dir>/plots/.",
    )
    parser.add_argument(
        "--title",
        default="RAG vs Baseline: Pareto Front Comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--condition-colors",
        nargs="+",
        default=[],
        metavar="COND:COLOR",
        help="Custom colors per condition, e.g. baseline:#1f77b4 rag-faiss:#ff7f0e",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = manifest_path.parent / "plots"

    # Parse custom colors.
    condition_colors: Dict[str, str] = {}
    for item in args.condition_colors:
        if ":" in item:
            cond, color = item.split(":", 1)
            condition_colors[cond.strip()] = color.strip()

    png_path, csv_path = build_plot(
        manifest_path=manifest_path,
        output_dir=output_dir,
        title=args.title,
        condition_colors=condition_colors,
    )

    print(f"PNG: {png_path}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
