#!/usr/bin/env python3
"""Plot Pareto front for a run's results.

Reads files in runs/{run_id}/results/*_results.txt where each file contains
one line with four comma-separated values as described in the repo README:
    test_accuracy, total_parameters, val_accuracy, train_time

This script treats both objectives as minimization objectives. To convert the
test accuracy (which is normally maximized) into a minimization objective we
use (1 - test_accuracy). The two objectives plotted are:
    obj1 = 1 - test_accuracy  (minimize)
    obj2 = total_parameters   (minimize)

Usage:
    python scripts/pareto_front.py --run latest
    python scripts/pareto_front.py --run auto_20251103_210131

Output:
    saves a PNG to runs/{run_id}/pareto_front.png and prints counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_run_dir(run_id: str) -> Path:
    base = Path.cwd() / "runs"
    if run_id == "latest":
        candidate = base / "latest"
        if candidate.exists():
            return candidate
        # fallback: pick newest directory in runs/
        dirs = [p for p in base.iterdir() if p.is_dir()]
        if not dirs:
            raise FileNotFoundError(f"No runs found in {base}")
        latest = max(dirs, key=lambda p: p.stat().st_mtime)
        return latest
    else:
        path = base / run_id
        if not path.exists():
            raise FileNotFoundError(f"Run directory not found: {path}")
        return path


def load_results(results_dir: Path):
    """Load results files from a results directory.

    Returns a list of tuples: (gene_id, test_acc, total_params, val_acc, train_time)
    Ignores files it cannot parse but prints a warning.
    """
    out = []
    for p in sorted(results_dir.glob("*_results.txt")):
        try:
            text = p.read_text().strip()
            if not text:
                print(f"Skipping empty file: {p}")
                continue
            parts = [x.strip() for x in text.split(",")]
            if len(parts) < 4:
                print(f"Skipping malformed file (expected 4 values): {p}")
                continue
            test_acc = float(parts[0])
            total_params = float(parts[1])
            val_acc = float(parts[2])
            train_time = float(parts[3])
            gene_id = p.name.replace("_results.txt", "")
            out.append((gene_id, test_acc, total_params, val_acc, train_time))
        except Exception as e:
            print(f"Warning: failed to parse {p}: {e}")
    return out

def is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """Compute Pareto-efficient (non-dominated) points for minimization.

    points: (N, M) array where lower is better for all objectives.
    returns boolean mask of length N where True indicates Pareto-efficient.
    
    A point is Pareto-efficient if no other point dominates it.
    Point A dominates point B if A is better or equal in all objectives 
    and strictly better in at least one objective.
    """
    if points.size == 0:
        return np.array([], dtype=bool)
    
    n_points = points.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if not is_efficient[i]:
            continue
        # Check if any other point dominates point i
        # A point j dominates i if: points[j] <= points[i] in all dims AND points[j] < points[i] in at least one dim
        better_or_equal = np.all(points <= points[i], axis=1)  # All objectives better or equal
        strictly_better = np.any(points < points[i], axis=1)   # At least one objective strictly better
        dominates = better_or_equal & strictly_better
        dominates[i] = False  # A point doesn't dominate itself
        
        if np.any(dominates):
            is_efficient[i] = False
    
    return is_efficient

def compute_pareto_area(pareto_pts: np.ndarray) -> float:
    """Compute the area under the Pareto front using step-wise integration.
    
    The Pareto front is treated as a step function where we go horizontally
    first (constant y), then vertically down to the next point.
    
    pareto_pts: (N, 2) array of Pareto-efficient points (obj1, obj2).
    returns: area under the step-wise Pareto front curve.
    """
    if pareto_pts.shape[0] == 0:
        return 0.0
    
    # Sort by obj1 (x-axis) in increasing order
    order = np.argsort(pareto_pts[:, 0])
    sorted_pts = pareto_pts[order]
    
    if sorted_pts.shape[0] == 1:
        # Single point - no area under curve in a meaningful sense
        return 0.0
    
    # Calculate area using step function (horizontal-then-vertical)
    # For each segment from point i to point i+1:
    # We go horizontally from x_i to x_{i+1} at height y_i
    # Area of rectangle = (x_{i+1} - x_i) * y_i
    area = 0.0
    for i in range(len(sorted_pts) - 1):
        x_i, y_i = sorted_pts[i]
        x_next, _ = sorted_pts[i + 1]
        # Rectangle width * height (y_i is the height for horizontal step)
        area += (x_next - x_i) * y_i
    
    return area


def plot_pareto(objs: np.ndarray, labels: list[str], out_path: Path) -> float:
    """Scatter plot of objectives and highlight Pareto front.

    objs: (N,2) array where both columns are minimization objectives.
    labels: list of gene ids, length N
    out_path: path to save png
    
    returns: area under the Pareto front
    """
    if objs.size == 0:
        raise ValueError("No objective points to plot")

    # Compute pareto-efficient points
    mask = is_pareto_efficient(objs)
    pareto_pts = objs[mask]

    # Compute area under Pareto front
    area = compute_pareto_area(pareto_pts)

    plt.figure(figsize=(8, 6))
    plt.scatter(objs[:, 0], objs[:, 1], c="C0", alpha=0.6, label="Individuals")
    plt.scatter(pareto_pts[:, 0], pareto_pts[:, 1], c="C3", s=80, label="Pareto front")

    # Connect pareto points with step-wise lines (horizontal then vertical)
    if pareto_pts.shape[0] > 1:
        order = np.argsort(pareto_pts[:, 0])
        sorted_pts = pareto_pts[order]
        # Use step function: 'post' means horizontal line first, then vertical drop
        plt.step(sorted_pts[:, 0], sorted_pts[:, 1], c="C3", linestyle="--", 
                 alpha=0.8, where='post')

    plt.xlabel("Objective 1: 1 - test_accuracy")
    plt.ylabel("Objective 2: total_parameters")
    plt.yscale('log')
    plt.title(f"Pareto Front (minimize both objectives)\nArea under front: {area:.4e}")
    plt.grid(alpha=0.25)
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    return area


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto front from run results")
    parser.add_argument("--run", type=str, default="latest", help="run id under runs/ or 'latest'")
    parser.add_argument("--out", type=str, default=None, help="optional output filename (png)")
    args = parser.parse_args()

    try:
        run_dir = find_run_dir(args.run)
    except Exception as e:
        print(f"Error finding run directory: {e}", file=sys.stderr)
        sys.exit(2)

    results_dir = run_dir / "results"
    if not results_dir.exists():
        print(f"No results directory at {results_dir}", file=sys.stderr)
        sys.exit(2)

    records = load_results(results_dir)
    if not records:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    gene_ids = [r[0] for r in records]
    test_accs = np.array([r[1] for r in records], dtype=float)
    total_params = np.array([r[2] for r in records], dtype=float)

    # Convert to minimization objectives
    obj1 = 1.0 - test_accs
    obj2 = total_params
    objs = np.stack([obj1, obj2], axis=1)

    out_png = Path(args.out) if args.out else (run_dir / "pareto_front.png")
    area = plot_pareto(objs, gene_ids, out_png)

    n = objs.shape[0]
    pareto_mask = is_pareto_efficient(objs)
    print(f"Plotted {n} individuals, pareto front size: {pareto_mask.sum()}")
    print(f"Area under Pareto front: {area:.4e}")
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()
