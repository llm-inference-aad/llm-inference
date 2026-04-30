"""Aggregate historical genes from RAG-OFF runs into a single CSV for replay.

Walks `runs/<RUN_ID>/` directories, keeps genes that came from runs whose
`run_metadata.json` recorded `RAG_ENABLED in {"false", None}`, joins each gene's
on-disk fitness with its ancestry-derived mutation type, extracts the original
`[PROMPT TO LLM]` block from the per-gene LLM log, and emits one CSV row per
gene.

Output:
    scripts/rag_replay/datasets/past_genes.csv
    scripts/rag_replay/datasets/prompts/<orig_gene_id>.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class GeneRow:
    orig_gene_id: str
    orig_run_id: str
    orig_rag_enabled: str
    orig_mutation_op: str
    orig_eligible_for_rag: bool
    orig_parent_id: str
    orig_parent_path: str
    orig_was_fallback: bool
    orig_test_acc: float | None
    orig_params: float | None
    orig_val_acc: float | None
    orig_train_time_s: float | None
    orig_prompt_path: str
    orig_prompt_chars: int


PROMPT_HEADER_RE = re.compile(r"PROMPT TO LLM")
SECTION_DIV = "=" * 80
INNER_DIV = "-" * 80


def extract_first_prompt(log_text: str) -> str | None:
    """Pull the first `[PROMPT TO LLM]` block from a per-gene LLM log."""
    sections = log_text.split(SECTION_DIV)
    for sec in sections:
        if not PROMPT_HEADER_RE.search(sec):
            continue
        parts = sec.split(INNER_DIV, 1)
        if len(parts) != 2:
            continue
        body = parts[1].strip()
        if body:
            return body
    return None


def latest_checkpoint(run_dir: Path) -> Path | None:
    cks = sorted(
        run_dir.glob("checkpoints/checkpoint_gen_*.pkl"),
        key=lambda p: int(re.search(r"gen_(\d+)", p.name).group(1)),
    )
    return cks[-1] if cks else None


def load_ancestry(run_dir: Path) -> dict:
    ck = latest_checkpoint(run_dir)
    if ck is None:
        return {}
    try:
        with ck.open("rb") as f:
            d = pickle.load(f)
    except Exception as exc:
        print(f"[warn] failed to load {ck}: {exc}", file=sys.stderr)
        return {}
    return d.get("GLOBAL_DATA_ANCESTRY", {}) or {}


def load_results(run_dir: Path) -> dict[str, tuple]:
    """gene_id -> (test_acc, params, val_acc, train_time_s) from results/*.txt."""
    out: dict[str, tuple] = {}
    for path in (run_dir / "results").glob("*_results.txt"):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 2:
            continue
        try:
            test_acc = float(parts[0])
            params = float(parts[1])
            val_acc = float(parts[2]) if len(parts) > 2 and parts[2] else None
            train_time = float(parts[3]) if len(parts) > 3 and parts[3] else None
        except Exception:
            continue
        gid = path.name.replace("_results.txt", "")
        out[gid] = (test_acc, params, val_acc, train_time)
    return out


def is_rag_off_run(metadata: dict) -> bool:
    """Return True if the run was conducted without RAG.

    We treat both an explicit "false" string and missing/None values as off,
    since several pre-RAG runs simply lack the field.
    """
    exp = metadata.get("experiment", {}) or {}
    val = exp.get("RAG_ENABLED")
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() in {"false", "0", "no", "off", ""}
    return not bool(val)


def parse_parent_id(genes: list[str]) -> str:
    """Pull the immediate parent gene id from an ancestry GENES list.

    `genes` may be a list of plain ids (from mutation lineage) or strings of the
    form `P:<parent>-C:<child>` from crossover entries. We take the immediate
    predecessor and unwrap the P:/-C: form if present.
    """
    if not genes or len(genes) < 2:
        return "network"  # only the seed sentinel — parent is the root network
    raw = genes[-2]
    if not isinstance(raw, str):
        return "network"
    m = re.search(r"P:([^-]+)", raw)
    if m:
        return m.group(1)
    return raw


def resolve_parent_path(parent_id: str, sota_models: Path, sota_root: Path) -> Path | None:
    if parent_id in {"network", "SEED"}:
        return sota_root / "network.py"
    candidate = sota_models / f"network_{parent_id}.py"
    if candidate.exists():
        return candidate
    return sota_root / "network.py"


def aggregate(runs_root: Path, sota_root: Path, out_dir: Path) -> int:
    sota_models = sota_root / "models"
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[GeneRow] = []
    skip_counts = {"no_log": 0, "no_results": 0, "no_prompt": 0, "rag_on": 0}

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "run_metadata.json"
        if not meta_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            continue
        if not is_rag_off_run(metadata):
            skip_counts["rag_on"] += 1
            continue

        run_id = metadata.get("run_id", run_dir.name)
        results = load_results(run_dir)
        ancestry = load_ancestry(run_dir)
        log_dir = run_dir / "logs" / "llm"

        for gid, (test_acc, params, val_acc, train_time) in results.items():
            log_path = log_dir / f"gene_{gid}.log"
            if not log_path.exists():
                skip_counts["no_log"] += 1
                continue
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                skip_counts["no_log"] += 1
                continue
            prompt = extract_first_prompt(log_text)
            if not prompt:
                skip_counts["no_prompt"] += 1
                continue

            anc = ancestry.get(gid, {})
            mutate_types = anc.get("MUTATE_TYPE") or []
            mutation_op = mutate_types[-1] if mutate_types else "UNKNOWN"
            parent_id = parse_parent_id(anc.get("GENES") or [])
            parent_path = resolve_parent_path(parent_id, sota_models, sota_root)

            fallback_marker = sota_models / f"network_{gid}.py.fallback"
            was_fallback = fallback_marker.exists()

            prompt_path = prompts_dir / f"{gid}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

            rows.append(
                GeneRow(
                    orig_gene_id=gid,
                    orig_run_id=run_id,
                    orig_rag_enabled="false",
                    orig_mutation_op=mutation_op,
                    orig_eligible_for_rag=mutation_op == "TEMPLATE_BASED",
                    orig_parent_id=parent_id,
                    orig_parent_path=str(parent_path.relative_to(ROOT_DIR)) if parent_path else "",
                    orig_was_fallback=was_fallback,
                    orig_test_acc=test_acc,
                    orig_params=params,
                    orig_val_acc=val_acc,
                    orig_train_time_s=train_time,
                    orig_prompt_path=str(prompt_path.relative_to(ROOT_DIR)),
                    orig_prompt_chars=len(prompt),
                )
            )

    csv_path = out_dir / "past_genes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(GeneRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)

    op_counts: dict[str, int] = {}
    for r in rows:
        op_counts[r.orig_mutation_op] = op_counts.get(r.orig_mutation_op, 0) + 1
    eligible_n = sum(1 for r in rows if r.orig_eligible_for_rag)
    fallback_n = sum(1 for r in rows if r.orig_was_fallback)

    print(f"wrote {csv_path} with {len(rows)} rows")
    print(f"  by mutation_op: {op_counts}")
    print(f"  eligible_for_rag (TEMPLATE_BASED): {eligible_n}")
    print(f"  fallback marker present: {fallback_n}")
    print(f"  skipped: {skip_counts}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=ROOT_DIR / "runs")
    ap.add_argument("--sota-root", type=Path, default=ROOT_DIR / "sota" / "ExquisiteNetV2")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="Output directory for past_genes.csv and prompts/",
    )
    args = ap.parse_args()
    return aggregate(args.runs_dir, args.sota_root, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
