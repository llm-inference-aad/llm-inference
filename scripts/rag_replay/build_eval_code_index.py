"""Build a leak-free FAISS code namespace for the RAG replay eval.

Why this exists
---------------
The replay subset (``scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv``)
samples genes from two source runs. If those genes' code (or AST-equivalent
clones from other runs) end up in the ``code`` namespace at retrieval time,
the with_rag arm sees its own answer and the experiment is invalid.

This script builds a fresh ``rag_data_eval/`` next to the production
``rag_data/`` with two guarantees:

  1. The two source runs of the eval subset are excluded entirely
     (``--excluded-run`` flag, default = both subset runs).
  2. Any gene whose AST-normalized code hash matches one of the 30 target
     networks is dropped, even from "kept" runs (catches reformatted clones).

Outputs
-------
- ``rag_data_eval/faiss_index/code.index``  — the FAISS index
- ``rag_data_eval/metadata/code.jsonl``     — per-document metadata
- ``rag_data_eval/holdout_dropped.jsonl``   — audit trail of every gene
  dropped by the holdout, with reason and source run
- prints the index sha and the dropped/kept counts at the end so the
  integration owner can pin the sha in the run-instructions doc.

Usage
-----
::

    uv run python scripts/rag_replay/build_eval_code_index.py \\
        --csv scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv \\
        --runs-dir runs/ \\
        --models-dir sota/ExquisiteNetV2/ \\
        --output rag_data_eval/

The text namespace is left to the production ``rag_data/`` corpus (the PDF
and pytorch.json sources do not reference any eval gene by ID, so they
cannot leak).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _target_gene_hashes(csv_path: Path, models_root: Path) -> tuple[set[str], list[dict]]:
    """Compute AST-normalized hashes for every target gene's network file."""
    from rag.data_ingestion import ast_normalized_hash  # type: ignore

    hashes: set[str] = set()
    audit: list[dict] = []
    rows = list(csv.DictReader(csv_path.open()))
    for row in rows:
        gid = row["orig_gene_id"]
        nf = models_root / "models" / f"network_{gid}.py"
        if not nf.exists():
            audit.append({"gene_id": gid, "status": "target_network_missing", "path": str(nf)})
            continue
        h = ast_normalized_hash(nf.read_text(encoding="utf-8"))
        hashes.add(h)
        audit.append({"gene_id": gid, "ast_hash": h})
    return hashes, audit


def _target_run_ids(csv_path: Path) -> set[str]:
    """Source-run IDs the eval subset was drawn from."""
    rows = list(csv.DictReader(csv_path.open()))
    return {row["orig_run_id"] for row in rows if row.get("orig_run_id")}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=ROOT_DIR
        / "scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv",
        help="Eval subset CSV (drives the holdout target-hash set + run-id set).",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT_DIR / "runs",
        help="Root containing per-run dirs with checkpoints/.",
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT_DIR / "sota" / "ExquisiteNetV2",
        help="Root containing models/network_<gid>.py.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "rag_data_eval",
        help="Output directory for the eval-time index.",
    )
    ap.add_argument(
        "--excluded-run",
        action="append",
        default=None,
        help="Run name to exclude from the corpus. Repeat. "
             "Default: union of run IDs present in the eval CSV.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Cap on checkpoint files scanned (debug).",
    )
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metadata").mkdir(parents=True, exist_ok=True)

    # Lazy heavy imports
    from rag.data_ingestion import extract_mutations_from_checkpoints  # type: ignore
    from rag.embeddings import EmbeddingService  # type: ignore
    from rag.vector_db import VectorStoreManager  # type: ignore

    target_hashes, target_audit = _target_gene_hashes(args.csv, args.models_dir)
    auto_excluded_runs = _target_run_ids(args.csv)
    excluded_runs = set(args.excluded_run) if args.excluded_run else auto_excluded_runs

    print(f"[build] eval CSV: {args.csv}")
    print(f"[build] target gene count: {len(target_audit)}, "
          f"target hashes: {len(target_hashes)}")
    print(f"[build] excluded runs ({len(excluded_runs)}): "
          f"{sorted(excluded_runs)}")

    records = extract_mutations_from_checkpoints(
        runs_dir=str(args.runs_dir),
        models_dir=str(args.models_dir),
        limit=args.limit,
        excluded_runs=excluded_runs,
        excluded_code_hashes=target_hashes,
    )

    if not records:
        print("[build] WARNING: no mutations extracted — nothing to index", flush=True)

    # Embed + store in the FAISS code namespace.
    embeddings = EmbeddingService()
    contents: list[str] = []
    metadata: list[dict] = []
    for r in records:
        content, meta = r.to_document()
        contents.append(content)
        metadata.append(meta)

    if contents:
        store = VectorStoreManager(rag_data_dir=str(args.output))
        code_embs = embeddings.embed_code(contents)
        ids = store.add_code_documents(contents, code_embs, metadata)
        print(f"[build] indexed {len(ids)} mutations into {args.output}/faiss_index/code.index")

    # Self-check: walk the metadata jsonl, confirm no target_hash slipped in.
    metadata_jsonl = args.output / "metadata" / "code.jsonl"
    if metadata_jsonl.exists():
        leaks = []
        for line in metadata_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            ah = (entry.get("metadata") or {}).get("code_ast_hash")
            if ah in target_hashes:
                leaks.append(entry["metadata"].get("gene_id", "?"))
        if leaks:
            print(f"[build] FAIL: {len(leaks)} target gene(s) leaked into the index: "
                  f"{leaks[:5]}", file=sys.stderr)
            return 2
        print(f"[build] self-check PASS: no target genes in code.jsonl")

    # Audit trail.
    audit_path = args.output / "holdout_dropped.jsonl"
    with audit_path.open("w") as f:
        for entry in target_audit:
            f.write(json.dumps({"target": entry}) + "\n")
        f.write(json.dumps({
            "summary": {
                "excluded_runs": sorted(excluded_runs),
                "target_hashes": len(target_hashes),
                "indexed_count": len(records),
            }
        }) + "\n")
    print(f"[build] audit trail: {audit_path}")

    # Print the index sha so it can be pinned in the run-instructions doc.
    code_index = args.output / "faiss_index" / "code.index"
    if code_index.exists():
        print(f"[build] code.index sha256: {_file_sha256(code_index)}")
    if metadata_jsonl.exists():
        print(f"[build] code.jsonl sha256: {_file_sha256(metadata_jsonl)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
