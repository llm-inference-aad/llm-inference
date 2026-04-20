#!/usr/bin/env python3
"""Generate a Pareto-front comparison figure from an ablation matrix manifest.

Supports any number of conditions; designed for the canonical 3-condition set:
  - ``baseline``          — no RAG
  - ``rag-faiss``         — FAISS code+text retrieval
  - ``rag-faiss+memory``  — FAISS + MemoryBackend (PR 8)

Each condition is plotted with a distinct color and marker shape.  All evaluated
points are shown at low opacity; the Pareto-efficient subset is connected with a
step-line at full opacity.

Output:
  - <output_dir>/pareto_comparison.png  (300 DPI)
  - <output_dir>/pareto_data.csv        (one row per evaluated event)

Usage::

    python scripts/plot_pareto_front_comparison.py \\
        --manifest runs/ablation_20260420_120000/manifest.json

    # Override output directory and figure title
    python scripts/plot_pareto_front_comparison.py \\
        --manifest runs/ablation_20260420_120000/manifest.json \\
        --output runs/ablation_20260420_120000/plots/ \\
        --title "RAG vs Baseline Pareto Front"

    # Custom colors per condition
    python scripts/plot_pareto_front_comparison.py \\
        --manifest runs/ablation_20260420_120000/manifest.json \\
        --condition-colors baseline:#1f77b4 rag-faiss:#ff7f0e rag-faiss+memory:#2ca02c
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

# ---------------------------------------------------------------------------
# Default visual style per condition (up to 6 conditions covered)
# ---------------------------------------------------------------------------

# Each entry: (hex_color, marker_char)
_CONDITION_STYLES: Dict[str, Tuple[str, str]] = {
    "baseline":         ("#1f77b4", "o"),   # blue, circle
    "rag-faiss":        ("#ff7f0e", "s"),   # orange, square
    "rag-faiss+memory": ("#2ca02c", "^"),   # green, triangle-up
    "code_only":        ("#d62728", "D"),   # red, diamond
    "text_only":        ("#9467bd", "v"),   # purple, triangle-down
    "hybrid":           ("#8c564b", "P"),   # brown, plus
    "hybrid_rerank":    ("#e377c2", "X"),   # pink, X
}

_FALLBACK_COLORS = ["#7f7f7f", "#bcbd22", "#17becf"]
_FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _style_for(condition: str, idx: int) -> Tuple[str, str]:
    """Return (color, marker) for *condition*."""
    if condition in _CONDITION_STYLES:
        return _CONDITION_STYLES[condition]
    color = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
    marker = _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)]
    return color, marker


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_events_for_run(run_id: str, run_dir: Optional[Path] = None) -> list:
    """Stream MutationEvent objects from a run's rag_ledger.jsonl.

    Falls back gracefully if bookkeeping is unavailable (returns []).
    """
    try:
        from src.rag.bookkeeping import replay_ledger  # noqa: PLC0415
    except ImportError:
        return []

    if run_dir is not None:
        ledger_path = run_dir / "metrics" / "rag_ledger.jsonl"
    else:
        ledger_path = None

    try:
        return list(replay_ledger(run_id, ledger_path=ledger_path))
    except Exception as exc:
        print(f"  WARNING: could not load ledger for {run_id}: {exc}", file=sys.stderr)
        return []


def _load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pareto computation helpers
# ---------------------------------------------------------------------------


def _get_event_metrics(event) -> Optional[Tuple[float, float]]:
    """Return (test_accuracy, total_params) or None if data is missing."""
    eo = getattr(event, "eval_outputs", None)
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


def _dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Return True if point *a* Pareto-dominates point *b*.

    In our space: higher accuracy is better, lower params is better.
    a dominates b iff a is no worse in both objectives and strictly better in
    at least one.
    """
    return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])


def _compute_pareto_front(
    points: List[Tuple[float, float]],
) -> List[bool]:
    """Return a boolean mask: True = point is on the Pareto front.

    Prefer ``src.rag.pareto_policy.compute_pareto_front`` when available; fall
    back to the inline implementation so the script works without the full chain.
    """
    n = len(points)
    on_front = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(points[j], points[i]):
                on_front[i] = False
                break
    return on_front


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition", "run_id", "gene_id",
        "test_accuracy", "total_params",
        "is_pareto_eligible", "on_front",
    ]
    if not rows:
        path.write_text(",".join(fieldnames) + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plot builder
# ---------------------------------------------------------------------------


def build_plot(
    *,
    manifest_path: Path,
    output_dir: Path,
    title: str,
    condition_colors: Dict[str, str],
) -> Tuple[Path, Path]:
    """Build and save the Pareto comparison figure.

    Supports 1–N conditions.  Each condition gets a distinct color + marker.
    Scatter (all evaluated) at 30 % opacity; Pareto front as a solid step-line
    with high opacity.

    Args:
        manifest_path: Path to manifest.json.
        output_dir: Directory to write PNG and CSV into.
        title: Figure title.
        condition_colors: Optional user-supplied color overrides per condition.

    Returns:
        (png_path, csv_path)
    """
    manifest = _load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])

    # Collect events per condition (aggregate across seeds/runs).
    condition_events: Dict[str, list] = {}
    for job in jobs:
        condition = job.get("condition", "unknown")
        run_id = job.get("run_id", "")
        run_dir = REPO_ROOT / "runs" / run_id
        if not run_dir.exists():
            manifest_out = Path(manifest.get("output_dir", str(manifest_path.parent)))
            candidate = manifest_out.parent / run_id
            if candidate.exists():
                run_dir = candidate

        events = _load_events_for_run(run_id, run_dir=run_dir if run_dir.exists() else None)
        if condition not in condition_events:
            condition_events[condition] = []
        condition_events[condition].extend(events)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xscale("log")
    ax.set_xlabel("Total Parameters (log scale)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(title, fontsize=13)

    csv_rows: List[Dict] = []
    conditions_sorted = sorted(condition_events.keys())

    for cond_idx, condition in enumerate(conditions_sorted):
        events = condition_events[condition]
        color, marker = _style_for(condition, cond_idx)
        if condition in condition_colors:
            color = condition_colors[condition]

        # Extract valid metrics.
        valid: List[Tuple[float, float, object]] = []  # (acc, params, event)
        for event in events:
            m = _get_event_metrics(event)
            if m is not None:
                valid.append((m[0], m[1], event))

        points_2d = [(v[0], v[1]) for v in valid]
        on_front_mask = _compute_pareto_front(points_2d) if points_2d else []

        scatter_xs = [v[1] for v in valid]  # params on x-axis
        scatter_ys = [v[0] for v in valid]  # accuracy on y-axis
        ax.scatter(
            scatter_xs,
            scatter_ys,
            color=color,
            marker=marker,
            alpha=0.25,
            s=40,
        )

        # Pareto front points.
        front_pts: List[Tuple[float, float]] = []
        for i, (acc, params, event) in enumerate(valid):
            is_front = bool(on_front_mask[i]) if i < len(on_front_mask) else False
            ep = getattr(event, "is_pareto_eligible", None)
            csv_rows.append(
                {
                    "condition": condition,
                    "run_id": getattr(event, "run_id", ""),
                    "gene_id": getattr(event, "child_gene_id", ""),
                    "test_accuracy": acc,
                    "total_params": params,
                    "is_pareto_eligible": ep,
                    "on_front": is_front,
                }
            )
            if is_front:
                front_pts.append((acc, params))

        label = f"{condition} ({len(valid)} pts, {len(front_pts)} on front)"
        if front_pts:
            # Sort by params for a clean step-line.
            sorted_front = sorted(front_pts, key=lambda p: p[1])
            fx = [p[1] for p in sorted_front]
            fy = [p[0] for p in sorted_front]
            ax.plot(fx, fy, color=color, marker=marker, linewidth=2.5, markersize=8, alpha=0.9, label=label)
        else:
            ax.plot([], [], color=color, marker=marker, linewidth=2.5, alpha=0.9, label=label)

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
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

    output_dir = Path(args.output).resolve() if args.output else manifest_path.parent / "plots"

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
