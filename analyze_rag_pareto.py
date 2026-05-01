#!/usr/bin/env python3
"""Compute Pareto fronts from a benchmark summary file.

Default behavior:
- Reads runs/.../benchmark_summary.json
- Uses per-config aggregates
- Computes a Pareto front for:
  1) minimize latency_mean, maximize score_mean
  2) minimize latency_mean, maximize spec_tokens_mean (optional proxy for speculation burden)

The goal is to help compare RAG-only, RAG+speculative, RAG+constrained, and
RAG+both configurations in a presentation-friendly way.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Point:
    name: str
    metrics: Dict[str, float]


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def dominates(a: Point, b: Point, objectives: Sequence[Tuple[str, str]]) -> bool:
    """Return True if a dominates b for the given objectives.

    objectives is a list of (metric_name, direction) where direction is one of
    'min' or 'max'.
    """
    better_or_equal = True
    strictly_better = False
    for metric, direction in objectives:
        av = a.metrics.get(metric)
        bv = b.metrics.get(metric)
        if av is None or bv is None:
            return False
        if direction == "min":
            if av > bv:
                better_or_equal = False
            if av < bv:
                strictly_better = True
        elif direction == "max":
            if av < bv:
                better_or_equal = False
            if av > bv:
                strictly_better = True
        else:
            raise ValueError(f"Invalid direction: {direction}")
    return better_or_equal and strictly_better


def pareto_front(points: Sequence[Point], objectives: Sequence[Tuple[str, str]]) -> List[Point]:
    front: List[Point] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if dominates(q, p, objectives):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def summarize(summary: Dict[str, Any]) -> List[Point]:
    points: List[Point] = []
    for name, stats in summary.get("per_config", {}).items():
        if not isinstance(stats, dict):
            continue
        metrics = {
            "latency_mean": float(stats.get("latency_mean", 0.0)),
            "score_mean": float(stats.get("score_mean", 0.0)),
            "spec_tokens_mean": float(stats.get("spec_tokens_mean", 0.0)),
        }
        points.append(Point(name=name, metrics=metrics))
    return points


def fmt_point(p: Point) -> str:
    return (
        f"{p.name}: latency_mean={p.metrics.get('latency_mean', 0):.3f}, "
        f"score_mean={p.metrics.get('score_mean', 0):.3f}, "
        f"spec_tokens_mean={p.metrics.get('spec_tokens_mean', 0):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("runs/vllm_300request/metrics/benchmark_summary.json"),
        help="Path to benchmark_summary.json",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    summary = load_summary(args.summary)
    points = summarize(summary)
    if not points:
        raise SystemExit(f"No per_config data found in {args.summary}")

    front_score = pareto_front(points, [("latency_mean", "min"), ("score_mean", "max")])
    front_spec = pareto_front(points, [("latency_mean", "min"), ("spec_tokens_mean", "min")])

    print(f"Source: {args.summary}")
    print("\nPareto front: minimize latency_mean, maximize score_mean")
    for p in sorted(front_score, key=lambda x: (x.metrics["latency_mean"], -x.metrics["score_mean"])):
        print("  -", fmt_point(p))

    print("\nPareto front: minimize latency_mean, minimize spec_tokens_mean")
    for p in sorted(front_spec, key=lambda x: (x.metrics["latency_mean"], x.metrics["spec_tokens_mean"])):
        print("  -", fmt_point(p))

    result = {
        "source": str(args.summary),
        "front_latency_score": [p.name for p in front_score],
        "front_latency_spec_tokens": [p.name for p in front_spec],
        "points": {p.name: p.metrics for p in points},
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nWrote: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
