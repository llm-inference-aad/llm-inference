#!/usr/bin/env python3
"""Compare decoding runs across baseline and adaptive vLLM outputs.

Defaults:
- baseline: runs/server-only/metrics/smoke_tests
- candidate: runs/vllm_adaptive/metrics (or pass via CLI)

Compares per-config JSON files by filename stem and prints a compact table for:
- response_time_sec
- evaluationScore
- speculative_accepted
- vllm_num_speculative_tokens
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Optional

FIELDS = [
    "response_time_sec",
    "evaluationScore",
    "speculative_accepted",
    "vllm_num_speculative_tokens",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt(value: Any, digits: int = 3) -> str:
    num = numeric(value)
    if num is None:
        return "-"
    return f"{num:.{digits}f}"


def pct_change(base: Any, cand: Any) -> str:
    b = numeric(base)
    c = numeric(cand)
    if b in (None, 0) or c is None:
        return "-"
    return f"{((c - b) / b) * 100:+.1f}%"


def collect(dir_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(dir_path.glob("config_*.json")):
        try:
            out[f.stem] = load_json(f)
        except Exception as e:
            out[f.stem] = {"error": str(e)}
    return out


def avg(field: str, rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    vals = [numeric(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def print_table(baseline: Dict[str, Dict[str, Any]], candidate: Dict[str, Dict[str, Any]]) -> None:
    keys = sorted(set(baseline) & set(candidate))
    print("\nPer-config comparison")
    print("=" * 110)
    header = f"{'config':<28} {'metric':<28} {'baseline':>12} {'candidate':>12} {'delta':>12}"
    print(header)
    print("-" * len(header))
    for key in keys:
        b = baseline[key]
        c = candidate[key]
        for field in FIELDS:
            print(f"{key:<28} {field:<28} {fmt(b.get(field)):>12} {fmt(c.get(field)):>12} {pct_change(b.get(field), c.get(field)):>12}")
        print("-" * len(header))


def print_summary(baseline: Dict[str, Dict[str, Any]], candidate: Dict[str, Dict[str, Any]]) -> None:
    print("\nSummary")
    print("=" * 110)
    for field in FIELDS:
        b = avg(field, baseline.values())
        c = avg(field, candidate.values())
        if b is None or c is None:
            delta = "-"
        elif b == 0:
            delta = "-"
        else:
            delta = f"{((c - b) / b) * 100:+.1f}%"
        print(f"{field:<28} baseline={fmt(b):>10}  candidate={fmt(c):>10}  delta={delta:>8}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("runs/server-only/metrics/smoke_tests"))
    parser.add_argument("--candidate", type=Path, default=Path("runs/vllm_adaptive/metrics"))
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write a JSON summary")
    args = parser.parse_args()

    if not args.baseline.exists():
        raise SystemExit(f"Baseline directory not found: {args.baseline}")
    if not args.candidate.exists():
        raise SystemExit(f"Candidate directory not found: {args.candidate}")

    baseline = collect(args.baseline)
    candidate = collect(args.candidate)

    if not baseline:
        raise SystemExit(f"No config_*.json files in baseline directory: {args.baseline}")
    if not candidate:
        raise SystemExit(f"No config_*.json files in candidate directory: {args.candidate}")

    print(f"Baseline : {args.baseline}")
    print(f"Candidate: {args.candidate}")

    print_table(baseline, candidate)
    print_summary(baseline, candidate)

    summary = {
        "baseline_dir": str(args.baseline),
        "candidate_dir": str(args.candidate),
        "configs_compared": sorted(set(baseline) & set(candidate)),
        "baseline_means": {field: avg(field, baseline.values()) for field in FIELDS},
        "candidate_means": {field: avg(field, candidate.values()) for field in FIELDS},
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
