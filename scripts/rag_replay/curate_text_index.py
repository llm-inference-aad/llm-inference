"""Curate the FAISS text namespace down to a whitelist of PDF stems.

Why: the production text index ingests every PDF in ``rag_corpus/`` plus
``pytorch.json`` API docs. For the RAG-replay eval we want the FAISS text
namespace and the PageIndex tree set to cover the **same** documents — anything
in FAISS but not in PageIndex (or vice versa) biases the cross-backend
comparison.

This script reads the existing ``<rag_data>/metadata/text.jsonl``, keeps
only chunks whose source PDF stem is in the curated whitelist, re-embeds
them, and writes a fresh ``text.index`` + ``text.jsonl`` into the target
data dir. Fast: no pdfplumber pass, just metadata filter + re-embed.

By default it operates in-place on ``rag_data/`` (the production text
index). Pass ``--input`` to read from one dir and ``--output`` to write
into another (e.g. seed ``rag_data_eval/`` from ``rag_data/`` while
preserving the production index).

Usage::

    uv run python scripts/rag_replay/curate_text_index.py \\
        --input rag_data/ --output rag_data_eval/

The whitelist is hard-coded to the ten PDFs the PageIndex trees cover —
see PR description / runbook §7.8 for provenance. Override with
``--keep-stem`` (repeatable) for ablations.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Whitelist mirrors the PageIndex tree set on feature/rag-pipeline-ben.
# Compare against the source filenames in metadata.source.
DEFAULT_KEEP_STEMS: tuple[str, ...] = (
    "4388-Article Text-28466-1-10-20230612",
    "CIFAR 10 Dataset_ Everything You Need To Know - AskPython",
    "Cifar-10_Classification_using_Deep_Convolutional_Neural_Network",
    "Dropout Regularization in Deep Learning - GeeksforGeeks",
    "LLM_Guided_Evolution___The_Automation_of_Models_Advancing_Models",
    "The Emerging Science of Machine Learning Benchmarks _ SIAM",
    "activation-functions",
    "depthwise-separable-convultions",
    "efficientNet",
    "learning-rate-schedules",
)


def _entry_keep(entry: dict, keep_stems: set[str]) -> bool:
    """True iff the entry's source PDF stem is in *keep_stems*.

    The metadata schema written by ``data_ingestion.process_pdfs`` has
    ``metadata.source`` = absolute path to the PDF and ``metadata.source_type``
    = "pdf". Non-PDF entries (api docs from ``pytorch.json``) are dropped
    unconditionally — they're not in the PageIndex tree set so keeping them
    breaks coverage symmetry.
    """
    meta = entry.get("metadata") or {}
    if meta.get("source_type") != "pdf":
        return False
    source = meta.get("source") or ""
    stem = Path(source).stem
    return stem in keep_stems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=ROOT_DIR / "rag_data",
                    help="Source rag_data dir to read text.jsonl from.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Target rag_data dir to write the curated text "
                         "index into. Defaults to --input (in-place).")
    ap.add_argument("--keep-stem", action="append", default=None,
                    help="PDF stem to retain (repeatable). Defaults to the "
                         "ten-PDF PageIndex-aligned set.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be kept/dropped, do not write.")
    args = ap.parse_args()

    output_dir = args.output or args.input
    keep_stems = set(args.keep_stem) if args.keep_stem else set(DEFAULT_KEEP_STEMS)

    src_jsonl = args.input / "metadata" / "text.jsonl"
    if not src_jsonl.exists():
        print(f"FAIL: {src_jsonl} does not exist.", file=sys.stderr)
        return 1

    kept: list[dict] = []
    dropped_pdf: dict[str, int] = {}
    dropped_other = 0
    for line in src_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if _entry_keep(entry, keep_stems):
            kept.append(entry)
        else:
            meta = entry.get("metadata") or {}
            if meta.get("source_type") == "pdf":
                stem = Path(meta.get("source", "")).stem
                dropped_pdf[stem] = dropped_pdf.get(stem, 0) + 1
            else:
                dropped_other += 1

    print(f"[curate] input:  {src_jsonl}")
    print(f"[curate] output: {output_dir}")
    print(f"[curate] kept:   {len(kept)} chunks across {len({Path(e['metadata']['source']).stem for e in kept})} PDFs")
    print(f"[curate] dropped: {dropped_other} non-PDF + "
          f"{sum(dropped_pdf.values())} PDF chunks across "
          f"{len(dropped_pdf)} excluded PDFs")
    for stem in sorted(keep_stems):
        present = sum(1 for e in kept if Path(e["metadata"]["source"]).stem == stem)
        marker = "✓" if present > 0 else "✗ MISSING"
        print(f"  {marker} {stem}: {present} chunks")

    if args.dry_run:
        print("[curate] --dry-run, exiting without writing")
        return 0

    if not kept:
        print("FAIL: no chunks survived the whitelist; refusing to write empty index",
              file=sys.stderr)
        return 2

    # Wipe the target's existing text namespace before re-embedding to avoid
    # mixed-vintage state.
    out_metadata = output_dir / "metadata" / "text.jsonl"
    out_index = output_dir / "faiss_index" / "text.index"
    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    if out_metadata.exists():
        out_metadata.unlink()
    if out_index.exists():
        out_index.unlink()

    # Re-embed via the production EmbeddingService and re-add to a freshly
    # constructed NamespaceStore. NamespaceStore takes care of FAISS write +
    # jsonl persistence on add_documents.
    from rag.embeddings import EmbeddingService  # type: ignore
    from rag.vector_db import VectorStoreManager  # type: ignore

    print(f"[curate] re-embedding {len(kept)} chunks…", flush=True)
    embeddings = EmbeddingService()
    contents = [e["content"] for e in kept]
    metadata = [e["metadata"]    for e in kept]
    text_embs = embeddings.embed_text(contents)

    store = VectorStoreManager(rag_data_dir=str(output_dir))
    store.add_text_documents(contents, text_embs, metadata)

    # Sha report so the runbook can pin the artifact.
    import hashlib
    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    print(f"[curate] text.index sha256: {_sha(out_index)}")
    print(f"[curate] text.jsonl sha256: {_sha(out_metadata)}")
    print("[curate] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
