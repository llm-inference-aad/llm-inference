"""Compute paired statistics from a replay run and emit `report.md` + `paired.csv`.

Reads `journal.jsonl` (filled in by 03_replay.py + its polling phase), joins the
`no_rag` and `with_rag` arms by `orig_gene_id`, and surfaces:

- Goodput: P(syntax_valid_first_try) per arm + delta (McNemar exact).
- Fallback rescue: of source genes that originally fell back, what fraction did
  the with_rag arm recover into a valid module on first try?
- Preservation: of source genes that originally did NOT fall back, what
  fraction did the with_rag arm keep valid?
- Accuracy delta: paired median + IQR + Wilcoxon over rows where both arms
  successfully trained.
- Cost delta: prompt size, LLM wall time, training time.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import wilcoxon, binomtest
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _read_journal(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # If we have multiple entries per (orig_gene_id, arm) — keep the last (post-poll wins).
    keyed: dict[tuple[str, str], dict] = {}
    for r in rows:
        keyed[(r["orig_gene_id"], r["arm"])] = r
    return list(keyed.values())


def _pair(rows: list[dict]) -> list[dict]:
    by_gid: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_gid[r["orig_gene_id"]][r["arm"]] = r
    out: list[dict] = []
    for orig_gid, arms in by_gid.items():
        if "no_rag" not in arms or "with_rag" not in arms:
            continue
        n = arms["no_rag"]; w = arms["with_rag"]
        out.append({
            "orig_gene_id": orig_gid,
            "orig_run_id": n.get("orig_run_id"),
            "orig_was_fallback": n.get("orig_was_fallback"),
            "orig_test_acc": n.get("orig_test_acc"),
            "orig_params": n.get("orig_params"),
            "orig_train_time_s": n.get("orig_train_time_s"),
            "norag_gene_id": n.get("new_gene_id"),
            "norag_n_attempts": n.get("n_attempts"),
            "norag_was_fallback": n.get("was_fallback"),
            "norag_syntax_valid_first": n.get("syntax_valid_first_try"),
            "norag_test_acc": n.get("test_acc"),
            "norag_params": n.get("params"),
            "norag_train_time_s": n.get("train_time_s"),
            "norag_llm_wall_s": n.get("llm_wall_s"),
            "norag_prompt_chars": n.get("prompt_chars"),
            "norag_status": n.get("status"),
            "rag_gene_id": w.get("new_gene_id"),
            "rag_n_attempts": w.get("n_attempts"),
            "rag_was_fallback": w.get("was_fallback"),
            "rag_syntax_valid_first": w.get("syntax_valid_first_try"),
            "rag_retrieved_n_code": w.get("retrieved_n_code"),
            "rag_retrieved_n_text": w.get("retrieved_n_text"),
            "rag_block_chars": w.get("rag_block_chars"),
            "rag_test_acc": w.get("test_acc"),
            "rag_params": w.get("params"),
            "rag_train_time_s": w.get("train_time_s"),
            "rag_llm_wall_s": w.get("llm_wall_s"),
            "rag_prompt_chars": w.get("prompt_chars"),
            "rag_status": w.get("status"),
        })
    return out


def _frac(values: list[bool]) -> tuple[float, int]:
    if not values:
        return 0.0, 0
    n = len(values)
    return sum(1 for v in values if v) / n, n


def _mcnemar_2x2(pairs: list[tuple[bool, bool]]) -> dict:
    """McNemar exact test on pairs of (no_rag_success, with_rag_success).

    Returns counts plus statistic + p when scipy/statsmodels available.
    """
    a = sum(1 for n, w in pairs if n and w)
    b = sum(1 for n, w in pairs if n and not w)
    c = sum(1 for n, w in pairs if not n and w)
    d = sum(1 for n, w in pairs if not n and not w)
    out = {"both_success": a, "only_no_rag": b, "only_with_rag": c, "both_fail": d, "n": len(pairs)}
    if _HAS_SCIPY and (b + c) > 0:
        # McNemar exact: binomial test on the smaller off-diagonal count
        # against b+c trials with p=0.5 (two-sided).
        k = min(b, c); n = b + c
        res = binomtest(k, n, 0.5, alternative="two-sided")
        out["statistic"] = float(min(b, c))
        out["p_value"] = float(res.pvalue)
    return out


def _wilcoxon_paired(pairs: list[tuple[float, float]]) -> dict:
    """Paired Wilcoxon signed-rank on (no_rag_value, with_rag_value)."""
    deltas = [w - n for n, w in pairs if n is not None and w is not None]
    if not deltas:
        return {"n": 0}
    out = {
        "n": len(deltas),
        "median_delta": statistics.median(deltas),
        "mean_delta": statistics.mean(deltas),
        "iqr_delta_low": statistics.quantiles(deltas, n=4)[0] if len(deltas) >= 4 else min(deltas),
        "iqr_delta_high": statistics.quantiles(deltas, n=4)[2] if len(deltas) >= 4 else max(deltas),
    }
    if _HAS_SCIPY and len(deltas) >= 5 and any(d != 0 for d in deltas):
        try:
            res = wilcoxon([w for n, w in pairs if n is not None and w is not None],
                           [n for n, w in pairs if n is not None and w is not None])
            out["statistic"] = float(res.statistic)
            out["p_value"] = float(res.pvalue)
        except Exception:
            pass
    return out


def _bool(v) -> bool:
    return bool(v) if not isinstance(v, str) else v.lower() == "true"


def _f(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def report(replay_dir: Path) -> int:
    journal = replay_dir / "journal.jsonl"
    if not journal.exists():
        raise SystemExit(f"No journal.jsonl found at {journal}")

    rows = _read_journal(journal)
    paired = _pair(rows)
    if not paired:
        raise SystemExit("No paired rows found — both arms must have completed.")

    paired_csv = replay_dir / "paired.csv"
    fieldnames = list(paired[0].keys())
    with paired_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in paired:
            writer.writerow(p)
    print(f"wrote {paired_csv}")

    gp_pairs = [
        (_bool(p.get("norag_syntax_valid_first")), _bool(p.get("rag_syntax_valid_first")))
        for p in paired
    ]
    gp_no_rag, n = _frac([n for n, _ in gp_pairs])
    gp_with_rag, _ = _frac([w for _, w in gp_pairs])
    goodput_test = _mcnemar_2x2(gp_pairs)

    fb_pairs = [
        (_bool(p.get("norag_was_fallback")), _bool(p.get("rag_was_fallback")))
        for p in paired
    ]
    fb_no_rag, _ = _frac([n for n, _ in fb_pairs])
    fb_with_rag, _ = _frac([w for _, w in fb_pairs])
    fb_test = _mcnemar_2x2([(not n, not w) for n, w in fb_pairs])

    rescue_pairs = [
        _bool(p.get("rag_syntax_valid_first")) for p in paired if _bool(p.get("orig_was_fallback"))
    ]
    preserve_pairs = [
        _bool(p.get("rag_syntax_valid_first")) for p in paired if not _bool(p.get("orig_was_fallback"))
    ]
    rescue, n_rescue = _frac(rescue_pairs)
    preserve, n_preserve = _frac(preserve_pairs)

    acc_pairs = [
        (_f(p.get("norag_test_acc")), _f(p.get("rag_test_acc"))) for p in paired
    ]
    acc_pairs = [(n, w) for n, w in acc_pairs if n is not None and w is not None]
    acc_test = _wilcoxon_paired(acc_pairs)

    train_pairs = [(_f(p.get("norag_train_time_s")), _f(p.get("rag_train_time_s")))
                   for p in paired]
    train_pairs = [(n, w) for n, w in train_pairs if n is not None and w is not None]
    train_test = _wilcoxon_paired(train_pairs)

    prompt_pairs = [(_f(p.get("norag_prompt_chars")), _f(p.get("rag_prompt_chars")))
                    for p in paired]
    prompt_pairs = [(n, w) for n, w in prompt_pairs if n is not None and w is not None]
    prompt_test = _wilcoxon_paired(prompt_pairs)

    llm_pairs = [(_f(p.get("norag_llm_wall_s")), _f(p.get("rag_llm_wall_s")))
                 for p in paired]
    llm_pairs = [(n, w) for n, w in llm_pairs if n is not None and w is not None]
    llm_test = _wilcoxon_paired(llm_pairs)

    md = []
    md.append("# RAG Replay — Paired Comparison Report\n")
    md.append(f"\n**Replay dir:** `{replay_dir}`")
    md.append(f"**Paired rows:** {len(paired)}\n")

    md.append("## Headline goodput\n")
    md.append("| Arm | P(valid first try) | Was-fallback rate |")
    md.append("|---|---|---|")
    md.append(f"| no_rag (regenerated) | {gp_no_rag:.3f} ({n}) | {fb_no_rag:.3f} |")
    md.append(f"| with_rag | {gp_with_rag:.3f} ({n}) | {fb_with_rag:.3f} |")
    md.append(f"| **delta** | **{gp_with_rag - gp_no_rag:+.3f}** | **{fb_with_rag - fb_no_rag:+.3f}** |\n")

    md.append("### McNemar paired test on `syntax_valid_first_try`")
    md.append(f"```\n{json.dumps(goodput_test, indent=2)}\n```\n")
    md.append("### McNemar paired test on `was_fallback` (success = NOT fallback)")
    md.append(f"```\n{json.dumps(fb_test, indent=2)}\n```\n")

    md.append("## Fallback sub-cohort\n")
    md.append(f"- **Recovery rate** (orig was fallback → with_rag valid first try): **{rescue:.3f}** (n={n_rescue})")
    md.append(f"- **Preservation rate** (orig was NOT fallback → with_rag valid first try): **{preserve:.3f}** (n={n_preserve})\n")

    md.append("## Accuracy delta\n")
    md.append("Paired Wilcoxon on `rag_test_acc - norag_test_acc` over rows where both arms trained:\n")
    md.append(f"```\n{json.dumps(acc_test, indent=2)}\n```\n")

    md.append("## Cost / size deltas\n")
    md.append("Prompt chars (sanity — RAG prompts must be larger):")
    md.append(f"```\n{json.dumps(prompt_test, indent=2)}\n```")
    md.append("LLM wall seconds:")
    md.append(f"```\n{json.dumps(llm_test, indent=2)}\n```")
    md.append("Train seconds:")
    md.append(f"```\n{json.dumps(train_test, indent=2)}\n```\n")

    md.append("## Sanity assertions\n")
    bad_rag_zero = sum(1 for p in paired if (p.get("rag_block_chars") or 0) <= 0)
    bad_norag_nonzero = sum(1 for p in paired
                             if (p.get("rag_block_chars") or 0) > 0
                             and p.get("rag_status") != "with_rag")  # always true; sanity skipped
    md.append(f"- with_rag rows where `rag_block_chars <= 0`: **{bad_rag_zero}** (should be 0 or close)")
    md.append(f"- scipy available: {_HAS_SCIPY}\n")

    if not _HAS_SCIPY:
        md.append("\n> Install scipy for p-values: `uv add scipy`\n")

    report_path = replay_dir / "report.md"
    report_path.write_text("\n".join(md))
    print(f"wrote {report_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("replay_dir", type=Path)
    args = ap.parse_args()
    return report(args.replay_dir)


if __name__ == "__main__":
    raise SystemExit(main())
