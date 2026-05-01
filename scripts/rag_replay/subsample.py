"""Stratified unique-parent subsampler for past_genes.csv.

Picks N rows from `past_genes.csv` such that:
- Every row's `orig_parent_path` is unique (one eval per parent network).
- Restricted to `orig_eligible_for_rag=True`.
- Fallback ratio matches the eligible-pool population (29.2% by default).
- Deterministic for a given seed.

Usage:
    python scripts/rag_replay/subsample.py --n 30 --seed 21 \\
        --in scripts/rag_replay/datasets/past_genes.csv \\
        --out scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def _is_true(v) -> bool:
    return str(v).lower() == "true"


def subsample(rows: list[dict], n: int, seed: int) -> list[dict]:
    eligible = [r for r in rows if _is_true(r["orig_eligible_for_rag"])]

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        by_parent[r["orig_parent_path"]].append(r)

    # Population fallback ratio in the eligible pool, applied to N
    pop_fb_rate = sum(_is_true(r["orig_was_fallback"]) for r in eligible) / len(eligible)
    n_fb = round(n * pop_fb_rate)
    n_nonfb = n - n_fb

    fb_capable = sorted(p for p, gs in by_parent.items()
                        if any(_is_true(g["orig_was_fallback"]) for g in gs))
    nonfb_capable = sorted(p for p, gs in by_parent.items()
                           if any(not _is_true(g["orig_was_fallback"]) for g in gs))

    if n_fb > len(fb_capable):
        raise SystemExit(f"need {n_fb} FB parents but pool only has {len(fb_capable)}")

    rng = random.Random(seed)
    chosen_fb = set(rng.sample(fb_capable, n_fb))

    # Step 2: non-FB parents, drawn from non-FB-capable minus parents already chosen
    nonfb_pool = [p for p in nonfb_capable if p not in chosen_fb]
    if n_nonfb > len(nonfb_pool):
        raise SystemExit(f"need {n_nonfb} non-FB parents but available pool has {len(nonfb_pool)}")
    chosen_nonfb = set(rng.sample(nonfb_pool, n_nonfb))

    out: list[dict] = []
    for parent in chosen_fb:
        candidates = sorted(
            (g for g in by_parent[parent] if _is_true(g["orig_was_fallback"])),
            key=lambda g: g["orig_gene_id"],
        )
        out.append(candidates[0])
    for parent in chosen_nonfb:
        candidates = sorted(
            (g for g in by_parent[parent] if not _is_true(g["orig_was_fallback"])),
            key=lambda g: g["orig_gene_id"],
        )
        out.append(candidates[0])

    out.sort(key=lambda r: r["orig_gene_id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=Path("scripts/rag_replay/datasets/past_genes.csv"))
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    with args.in_path.open() as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    sample = subsample(rows, args.n, args.seed)

    with args.out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sample:
            w.writerow(r)

    n_fb = sum(_is_true(r["orig_was_fallback"]) for r in sample)
    n_unique = len(set(r["orig_parent_path"] for r in sample))
    print(f"wrote {args.out_path}")
    print(f"  rows           : {len(sample)}")
    print(f"  unique parents : {n_unique}/{len(sample)}")
    print(f"  fallback rows  : {n_fb} ({n_fb/len(sample):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
