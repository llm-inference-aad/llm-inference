"""Pareto front visualization for Guided Evolution results.

This script scans a results directory for files named ``*_results.txt`` produced by
``sota/ExquisiteNetV2/train.py`` and plots model test accuracy (maximize) versus
parameter count (minimize). The non-dominated set is highlighted as the Pareto
front.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ModelResult:
    gene_id: str
    test_acc: float
    total_params: float
    val_acc: float
    train_time: float

    @property
    def objectives(self) -> tuple[float, float]:
        """Return objectives ordered as (maximise test_acc, minimise total_params)."""
        return (self.test_acc, self.total_params)


def parse_result_file(path: Path) -> ModelResult | None:
    """Attempt to parse a single ``*_results.txt`` file."""
    contents = path.read_text().strip()
    if not contents:
        return None

    parts = [seg.strip() for seg in contents.split(",") if seg.strip()]
    if len(parts) < 4:
        raise ValueError(
            f"Expected 4 comma-separated values in {path}, found {len(parts)}: {contents!r}"
        )

    test_acc, total_params, val_acc, train_time = map(float, parts[:4])
    gene_id = path.stem.replace("_results", "")
    return ModelResult(gene_id, test_acc, total_params, val_acc, train_time)


def load_results(results_dir: Path) -> List[ModelResult]:
    files = sorted(results_dir.glob("*_results.txt"))
    records: List[ModelResult] = []
    for file_path in files:
        try:
            record = parse_result_file(file_path)
        except ValueError as exc:
            print(f"Skipping {file_path}: {exc}")
            continue
        if record is not None:
            records.append(record)
    return records


def dominates(a: ModelResult, b: ModelResult) -> bool:
    """Return True if ``a`` dominates ``b`` for (maximize acc, minimize params)."""
    a_test, a_params = a.objectives
    b_test, b_params = b.objectives
    return (a_test >= b_test and a_params <= b_params) and (
        a_test > b_test or a_params < b_params
    )


def pareto_front(results: Sequence[ModelResult]) -> List[ModelResult]:
    front: List[ModelResult] = []
    for candidate in results:
        dominated = any(dominates(other, candidate) for other in results)
        if dominated:
            continue
        front.append(candidate)
    # Sort by parameter count ascending for plotting
    front.sort(key=lambda r: (r.total_params, -r.test_acc))
    return front


def plot_front(
    results: Sequence[ModelResult],
    front: Sequence[ModelResult],
    output: Path,
    show: bool = False,
) -> None:
    if not results:
        print("No valid results to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    params = [math.log10(r.total_params) if r.total_params > 0 else float("nan") for r in results]
    accuracies = [r.test_acc for r in results]
    ax.scatter(params, accuracies, label="All models", alpha=0.4, color="#6baed6")

    if front:
        front_params = [math.log10(r.total_params) for r in front]
        front_acc = [r.test_acc for r in front]
        ax.scatter(front_params, front_acc, label="Pareto front", color="#d95f02", zorder=5)
        ax.plot(front_params, front_acc, color="#d95f02", linewidth=1.5, alpha=0.8)
        for record in front:
            ax.annotate(
                record.gene_id[-6:],
                (math.log10(record.total_params), record.test_acc),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("log10(Total Parameters)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Guided Evolution Pareto Front")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Pareto front saved to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Pareto front for Guided Evolution results.")
    parser.add_argument(
        "--results-dir",
        default="sota/ExquisiteNetV2/results",
        type=Path,
        help="Directory containing *_results.txt files (default: sota/ExquisiteNetV2/results).",
    )
    parser.add_argument(
        "--output",
        default="pareto_front.png",
        type=Path,
        help="Filename for the generated plot (default: pareto_front.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory {results_dir} not found. Run evaluations or pass --results-dir."
        )

    records = load_results(results_dir)
    if not records:
        print(f"No result files found in {results_dir}. Nothing to plot.")
        return

    front = pareto_front(records)
    plot_front(records, front, args.output, show=args.show)


if __name__ == "__main__":
    main()
