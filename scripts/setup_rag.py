#!/usr/bin/env python3
"""Bootstrap the local RAG vector database."""

from __future__ import annotations

import argparse
from pathlib import Path

from cfg.constants import RAG_DATA_DIR, ROOT_DIR, SOTA_ROOT
from rag.data_ingestion import extract_mutations_from_checkpoints, process_pdfs
from rag.embeddings import EmbeddingService
from rag.retrieval import RetrievalService
from rag.vector_db import VectorStoreManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the local RAG corpus.")
    parser.add_argument(
        "--runs-dir",
        default=str(Path(ROOT_DIR) / "runs"),
        help="Directory containing past evolution runs.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(Path(ROOT_DIR) / "rag_corpus"),
        help="Directory containing PDF research papers.",
    )
    parser.add_argument(
        "--rag-dir",
        default=RAG_DATA_DIR,
        help="Directory where the vector database should be stored.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of checkpoints to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag_dir = args.rag_dir
    print(f"[RAG] Initializing vector database at {rag_dir}")

    store = VectorStoreManager(rag_dir)
    embeddings = EmbeddingService()
    retrieval = RetrievalService(store, embeddings)

    existing_ids = {doc.document_id for doc in store.list_documents(VectorStoreManager.CODE_NAMESPACE)}
    mutation_records = extract_mutations_from_checkpoints(
        runs_dir=args.runs_dir,
        models_dir=SOTA_ROOT,
        limit=args.limit,
    )
    new_records = [record for record in mutation_records if record.gene_id not in existing_ids]
    if new_records:
        retrieval.index_mutations(new_records)
        print(f"[RAG] Indexed {len(new_records)} new mutation records.")
    else:
        print("[RAG] No new mutation records found.")

    pdf_documents = process_pdfs(args.pdf_dir)
    if pdf_documents:
        retrieval.index_text_documents(pdf_documents)
        print(f"[RAG] Indexed {len(pdf_documents)} PDF text chunks.")
    else:
        print("[RAG] No PDF documents found.")

    print("[RAG] Setup complete.")


if __name__ == "__main__":
    main()



