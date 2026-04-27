"""Analyze paired RAG isolation results: paired tests + per-template breakdown.

Usage:
    uv run python scripts/rag_isolation/analyze.py <run_dir>

Reads <run_dir>/results.jsonl, writes:
  - <run_dir>/summary.csv  (one row per arm × metric)
  - <run_dir>/paired.csv   (one row per case × trial with both arms side-by-side)
  - <run_dir>/report.md    (human-readable summary with paired tests)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pivot_pairs(rows: list[dict]) -> list[dict]:
    """Group results by (case_id, trial); emit one paired row per group."""
    groups: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        groups[(r["case_id"], r["trial"])][r["arm"]] = r
    pairs: list[dict] = []
    for (case_id, trial), arms in groups.items():
        if "no_rag" not in arms or "with_rag" not in arms:
            continue
        a, b = arms["no_rag"], arms["with_rag"]
        row = {
            "case_id": case_id,
            "trial": trial,
            "template": a["template"],
            "mutation_type": a["mutation_type"],
            "augment_idx_no_rag": a["augment_idx"],
            "augment_idx_with_rag": b["augment_idx"],
            "augment_idx_match": a["augment_idx"] == b["augment_idx"],
        }
        for k in (
            "syntax_valid_first_try", "module_valid_first_try", "n_attempts",
            "fallback", "prompt_chars", "response_chars", "llm_latency_s",
            "rag_block_chars", "retrieved_n_code", "retrieved_n_text",
            "parent_changed",
        ):
            row[f"{k}_no_rag"] = a.get(k)
            row[f"{k}_with_rag"] = b.get(k)
        pairs.append(row)
    return pairs


# --- statistical tests (no scipy dependency required for the basics) ---


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Exact two-sided McNemar's test on counts b (only A wins), c (only B wins).

    Returns (statistic_b_minus_c, p_value). Uses binomial sum.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    k = min(b, c)
    # Two-sided p = 2 * P(X <= k) under Binomial(n, 0.5), capped at 1.
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2 * p)
    return float(b - c), p


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test using normal approximation.

    Returns (W_statistic, approx_two_sided_p).
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0
    abs_d = [abs(d) for d in nz]
    order = sorted(range(n), key=lambda i: abs_d[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[order[j + 1]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz) if d < 0)
    W = min(w_plus, w_minus)
    mean = n * (n + 1) / 4
    std = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std == 0:
        return float(W), 1.0
    z = (W - mean) / std
    # two-sided p from normal approx
    p = math.erfc(abs(z) / math.sqrt(2))
    return float(W), float(p)


def median_iqr(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    s = sorted(xs)
    n = len(s)
    med = statistics.median(s)
    q1 = s[max(0, int(0.25 * (n - 1)))]
    q3 = s[min(n - 1, int(0.75 * (n - 1)))]
    return med, q1, q3


def summarize_arm(rows: list[dict], arm: str) -> dict[str, Any]:
    arm_rows = [r for r in rows if r["arm"] == arm]
    if not arm_rows:
        return {}
    n = len(arm_rows)
    out = {"arm": arm, "n": n}
    for metric in ("syntax_valid_first_try", "module_valid_first_try", "fallback", "parent_changed"):
        out[f"{metric}_rate"] = sum(1 for r in arm_rows if r.get(metric)) / n
    for metric in ("n_attempts", "prompt_chars", "response_chars", "llm_latency_s",
                   "rag_block_chars", "retrieved_n_code", "retrieved_n_text"):
        vals = [float(r.get(metric, 0) or 0) for r in arm_rows]
        med, q1, q3 = median_iqr(vals)
        out[f"{metric}_mean"] = sum(vals) / n
        out[f"{metric}_median"] = med
        out[f"{metric}_q1"] = q1
        out[f"{metric}_q3"] = q3
    return out


def paired_tests(pairs: list[dict]) -> list[dict]:
    """Run paired tests on the binary + continuous metrics."""
    tests = []
    n_pairs = len(pairs)
    if n_pairs == 0:
        return tests

    # Binary outcomes: McNemar
    binary_metrics = [
        ("syntax_valid_first_try", "Did NOT have syntax error on first try"),
        ("module_valid_first_try", "Module instantiated on first try"),
        ("fallback", "Fell back to parent code"),
        ("parent_changed", "Generated code differs from parent"),
    ]
    for key, label in binary_metrics:
        a_only = sum(1 for p in pairs if p[f"{key}_no_rag"] and not p[f"{key}_with_rag"])
        b_only = sum(1 for p in pairs if not p[f"{key}_no_rag"] and p[f"{key}_with_rag"])
        both = sum(1 for p in pairs if p[f"{key}_no_rag"] and p[f"{key}_with_rag"])
        neither = sum(1 for p in pairs if not p[f"{key}_no_rag"] and not p[f"{key}_with_rag"])
        stat, p = mcnemar_exact(a_only, b_only)
        tests.append({
            "metric": key,
            "label": label,
            "test": "McNemar (exact)",
            "n_pairs": n_pairs,
            "no_rag_only": a_only,
            "with_rag_only": b_only,
            "both": both,
            "neither": neither,
            "statistic": stat,
            "p_value": p,
        })

    # Continuous: Wilcoxon signed-rank
    cont_metrics = [
        "n_attempts", "prompt_chars", "response_chars",
        "llm_latency_s", "rag_block_chars",
    ]
    for key in cont_metrics:
        diffs = [float(p[f"{key}_with_rag"] or 0) - float(p[f"{key}_no_rag"] or 0) for p in pairs]
        med, q1, q3 = median_iqr(diffs)
        W, pv = wilcoxon_signed_rank(diffs)
        tests.append({
            "metric": key,
            "label": f"Δ {key} (with_rag - no_rag)",
            "test": "Wilcoxon signed-rank (normal approx)",
            "n_pairs": n_pairs,
            "median_delta": med,
            "iqr_low": q1,
            "iqr_high": q3,
            "statistic": W,
            "p_value": pv,
        })
    return tests


def per_template_breakdown(pairs: list[dict]) -> list[dict]:
    """For each template, summarize pair-level deltas."""
    by_tpl: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_tpl[p["template"]].append(p)
    out = []
    for tpl, ps in by_tpl.items():
        n = len(ps)
        row = {"template": tpl, "n_pairs": n}
        for key in ("syntax_valid_first_try", "fallback"):
            row[f"{key}_no_rag_rate"] = sum(1 for p in ps if p[f"{key}_no_rag"]) / n
            row[f"{key}_with_rag_rate"] = sum(1 for p in ps if p[f"{key}_with_rag"]) / n
        for key in ("n_attempts", "llm_latency_s", "response_chars"):
            diffs = [float(p[f"{key}_with_rag"] or 0) - float(p[f"{key}_no_rag"] or 0) for p in ps]
            row[f"{key}_median_delta"] = median_iqr(diffs)[0]
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    path.write_text("\n".join(lines))


def render_report(
    pairs: list[dict],
    arm_summaries: list[dict],
    tests: list[dict],
    by_tpl: list[dict],
    metadata: dict,
) -> str:
    lines = []
    lines.append(f"# RAG Isolation Report\n")
    lines.append(f"**Run started:** {metadata.get('started_at', 'n/a')}  ")
    lines.append(f"**Run ended:** {metadata.get('ended_at', 'n/a')}  ")
    lines.append(f"**Wall time:** {metadata.get('wall_s_total', 0):.1f} s  ")
    lines.append(f"**Server:** {metadata.get('server_url', 'n/a')}  ")
    lines.append(f"**Git commit:** `{metadata.get('git_commit', 'n/a')[:12]}`  ")
    lines.append(f"**Cases × trials × arms = pairs:** {metadata.get('n_cases', '?')} × "
                 f"{metadata.get('n_trials_per_case', '?')} × 2 = **{len(pairs)} pairs**\n")

    lines.append("## Per-arm marginal summary\n")
    if arm_summaries:
        keys = list(arm_summaries[0].keys())
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join("---" for _ in keys) + "|")
        for s in arm_summaries:
            lines.append("| " + " | ".join(_fmt(s.get(k)) for k in keys) + " |")
        lines.append("")

    lines.append("## Paired tests (no_rag vs with_rag)\n")
    lines.append("Negative deltas mean the with_rag arm produced *less* of the metric. "
                 "Read p-values cautiously: small N → low power. Look at effect sizes.\n")
    lines.append("| Metric | Test | n | Effect | p-value |")
    lines.append("|---|---|---|---|---|")
    for t in tests:
        if "median_delta" in t:
            effect = (f"median Δ = {t['median_delta']:.3g} "
                      f"(IQR {t['iqr_low']:.3g}–{t['iqr_high']:.3g})")
        else:
            effect = (f"only_no_rag={t['no_rag_only']}  only_with_rag={t['with_rag_only']}  "
                      f"both={t['both']}  neither={t['neither']}")
        lines.append(f"| `{t['metric']}` | {t['test']} | {t['n_pairs']} | {effect} | {t['p_value']:.4f} |")
    lines.append("")

    lines.append("## Per-template breakdown\n")
    if by_tpl:
        keys = list(by_tpl[0].keys())
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "|".join("---" for _ in keys) + "|")
        for r in by_tpl:
            lines.append("| " + " | ".join(_fmt(r.get(k)) for k in keys) + " |")
        lines.append("")

    # Quick "interesting case" callout: pairs where one arm worked and other didn't
    flips_syntax = [p for p in pairs
                    if p["syntax_valid_first_try_no_rag"] != p["syntax_valid_first_try_with_rag"]]
    if flips_syntax:
        lines.append("## Pairs where syntax-valid-first-try flipped\n")
        lines.append("These are the most informative cases for qualitative review.\n")
        for p in flips_syntax[:20]:
            lines.append(f"- `{p['case_id']}` trial {p['trial']}: "
                         f"no_rag={p['syntax_valid_first_try_no_rag']} "
                         f"with_rag={p['syntax_valid_first_try_with_rag']}")
        lines.append("")

    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    rows = _load_jsonl(args.run_dir / "results.jsonl")
    if not rows:
        print(f"No rows in {args.run_dir / 'results.jsonl'}", file=sys.stderr)
        return 1

    metadata = json.loads((args.run_dir / "run_metadata.json").read_text())

    pairs = _pivot_pairs(rows)
    arm_summaries = [s for s in (summarize_arm(rows, "no_rag"), summarize_arm(rows, "with_rag")) if s]
    tests = paired_tests(pairs)
    by_tpl = per_template_breakdown(pairs)

    write_csv(args.run_dir / "paired.csv", pairs)
    write_csv(args.run_dir / "summary.csv", arm_summaries)
    write_csv(args.run_dir / "tests.csv", tests)

    report = render_report(pairs, arm_summaries, tests, by_tpl, metadata)
    (args.run_dir / "report.md").write_text(report)

    print(f"Report: {args.run_dir / 'report.md'}")
    print(f"Pairs:  {args.run_dir / 'paired.csv'}")
    print(f"Tests:  {args.run_dir / 'tests.csv'}")
    print(f"Arm summary: {args.run_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
