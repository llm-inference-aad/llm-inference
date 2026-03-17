#!/usr/bin/env python3
"""Bootstrap the local RAG vector database."""

from __future__ import annotations

import argparse
from pathlib import Path

from cfg.constants import (
    RAG_CODE_EMBED_MODEL,
    RAG_DATA_DIR,
    RAG_MIN_ACCURACY,
    RAG_TEXT_EMBED_MODEL,
    ROOT_DIR,
    SOTA_ROOT,
)
from rag.data_ingestion import (
    extract_mutations_from_checkpoints,
    ingest_pytorch_docs,
    process_pdfs,
)
from rag.embeddings import EmbeddingConfig, EmbeddingService
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
        "--pytorch-json",
        default=str(Path(ROOT_DIR) / "rag_corpus" / "pytorch.json"),
        help="Path to pytorch.json API documentation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of checkpoints to process.",
    )
    parser.add_argument(
        "--rebuild-text",
        action="store_true",
        help="Clear the text namespace before re-indexing PDFs and PyTorch docs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag_dir = args.rag_dir
    print(f"[RAG] Initializing vector database at {rag_dir}")
    if args.rebuild_text:
        text_index = Path(rag_dir) / "faiss_index" / "text.index"
        text_metadata = Path(rag_dir) / "metadata" / "text.jsonl"
        for path in (text_index, text_metadata):
            if path.exists():
                path.unlink()
                print(f"[RAG] Removed {path} for text namespace rebuild.")

    store = VectorStoreManager(rag_dir)
    embeddings = EmbeddingService(EmbeddingConfig(
        code_model_name=RAG_CODE_EMBED_MODEL,
        text_model_name=RAG_TEXT_EMBED_MODEL,
    ))
    retrieval = RetrievalService(store, embeddings)

    existing_ids = {doc.document_id for doc in store.list_documents(VectorStoreManager.CODE_NAMESPACE)}
    mutation_records = extract_mutations_from_checkpoints(
        runs_dir=args.runs_dir,
        models_dir=SOTA_ROOT,
        limit=args.limit,
        min_accuracy=RAG_MIN_ACCURACY,
    )
    new_records = [record for record in mutation_records if record.gene_id not in existing_ids]
    if new_records:
        retrieval.index_mutations(new_records)
        print(f"[RAG] Indexed {len(new_records)} new mutation records.")
    else:
        print("[RAG] No new mutation records found.")

    # --- Text namespace: PDFs + PyTorch docs ---
    existing_text_ids = {doc.document_id for doc in store.list_documents(VectorStoreManager.TEXT_NAMESPACE)}

    pdf_documents = process_pdfs(args.pdf_dir)
    if pdf_documents:
        new_pdfs = [d for d in pdf_documents if d["metadata"].get("document_id") not in existing_text_ids]
        if new_pdfs:
            retrieval.index_text_documents(new_pdfs)
            print(f"[RAG] Indexed {len(new_pdfs)} new PDF text chunks.")
        else:
            print("[RAG] PDF chunks already indexed, skipping.")
    else:
        print("[RAG] No PDF documents found.")

    pytorch_docs = ingest_pytorch_docs(args.pytorch_json)
    if pytorch_docs:
        new_docs = [d for d in pytorch_docs if d["metadata"].get("document_id") not in existing_text_ids]
        if new_docs:
            retrieval.index_text_documents(new_docs)
            print(f"[RAG] Indexed {len(new_docs)} PyTorch API doc chunks.")
        else:
            print("[RAG] PyTorch docs already indexed, skipping.")
    else:
        print(f"[RAG] No PyTorch docs found at {args.pytorch_json}")

    print("[RAG] Setup complete.")


if __name__ == "__main__":
    main()
