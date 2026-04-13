#!/usr/bin/env python3
"""
Offline tree builder: constructs PageIndex tree structures for corpus PDFs.

Run once to pre-build trees, then use them for PageIndex retrieval.
Results are cached in rag_data/pageindex_trees/.

Usage:
    python scripts/build_pageindex_trees.py
    python scripts/build_pageindex_trees.py --model local_server
    python scripts/build_pageindex_trees.py --corpus-dir rag_corpus --output-dir rag_data/pageindex_trees
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pageindex.pageindex.page_index import page_index_main
from src.pageindex.pageindex.utils import ConfigLoader, structure_to_list

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO_ROOT / "rag_corpus"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "rag_data" / "pageindex_trees"


def build_trees(corpus_dir: Path, output_dir: Path, model: str, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {corpus_dir}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {corpus_dir}")

    opt = ConfigLoader().load({
        "model": model,
        "if_add_node_summary": "yes",
        "if_add_node_text": "yes",
        "if_add_node_id": "yes",
    })

    for pdf_path in pdfs:
        stem = pdf_path.stem
        out_path = output_dir / f"{stem}_structure.json"

        if out_path.exists() and not force:
            print(f"  [skip] {stem} — tree already exists at {out_path}")
            continue

        if out_path.exists() and force:
            print(f"  [force] Removing existing tree: {out_path}")
            out_path.unlink()

        print(f"  Building tree for: {pdf_path.name} ...")
        try:
            result = page_index_main(str(pdf_path), opt)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Print summary
            structure = result.get("structure", [])
            nodes = structure_to_list(structure) if structure else []
            print(f"    -> {out_path.name}: {len(nodes)} nodes")
        except Exception as exc:
            print(f"    [ERROR] {pdf_path.name}: {exc}")

    # Final summary
    built = list(output_dir.glob("*_structure.json"))
    print(f"\nDone. {len(built)} tree(s) in {output_dir}")
    for p in sorted(built):
        data = json.loads(p.read_text(encoding="utf-8"))
        structure = data.get("structure", [])
        nodes = structure_to_list(structure) if structure else []
        print(f"  {p.name}: {len(nodes)} nodes, doc_name={data.get('doc_name', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PageIndex trees for corpus PDFs")
    parser.add_argument(
        "--model", type=str, default="local_server",
        help="Model for tree building (default: local_server)",
    )
    parser.add_argument(
        "--corpus-dir", type=str, default=str(DEFAULT_CORPUS_DIR),
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save tree JSON files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing trees and rebuild from scratch",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging for pageindex LLM calls",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=level,
    )

    build_trees(
        corpus_dir=Path(args.corpus_dir),
        output_dir=Path(args.output_dir),
        model=args.model,
        force=args.force,
    )


if __name__ == "__main__":
    main()
