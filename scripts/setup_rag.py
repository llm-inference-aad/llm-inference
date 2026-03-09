#!/usr/bin/env python3
"""Bootstrap the local RAG vector database."""

from __future__ import annotations

import argparse
from pathlib import Path

from cfg.constants import (
    PAGEINDEX_API_KEY,
    PAGEINDEX_DATA_DIR,
    PAGEINDEX_MODEL,
    PAGEINDEX_POLL_INTERVAL_SEC,
    PAGEINDEX_QUERY_THINKING,
    PAGEINDEX_TREE_TIMEOUT_SEC,
    RAG_BACKEND,
    RAG_CODE_EMBED_MODEL,
    RAG_DATA_DIR,
    RAG_FAIL_OPEN,
    RAG_TEXT_EMBED_MODEL,
    ROOT_DIR,
    SOTA_ROOT,
)
from rag.data_ingestion import extract_mutations_from_checkpoints, process_pdfs
from rag.embeddings import EmbeddingConfig, EmbeddingService
from rag.retrieval import FaissRetrievalBackend, RetrievalService
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
        "--backend",
        default=RAG_BACKEND,
        choices=["faiss", "pageindex"],
        help="RAG backend to initialize.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of checkpoints to process.",
    )
    return parser.parse_args()


def _build_retrieval(backend_name: str, rag_dir: str) -> tuple[RetrievalService, VectorStoreManager | None]:
    if backend_name == "pageindex":
        try:
            from rag.pageindex_backend import PageIndexRetrievalBackend

            backend = PageIndexRetrievalBackend(
                data_dir=rag_dir or PAGEINDEX_DATA_DIR,
                model_name=PAGEINDEX_MODEL,
                api_key=PAGEINDEX_API_KEY,
                tree_timeout_sec=PAGEINDEX_TREE_TIMEOUT_SEC,
                poll_interval_sec=PAGEINDEX_POLL_INTERVAL_SEC,
                query_thinking=PAGEINDEX_QUERY_THINKING,
            )
            if backend.client is None:
                raise RuntimeError("PAGEINDEX_API_KEY is missing; cannot generate PageIndex trees.")
            return RetrievalService(backend), None
        except Exception as exc:
            if not RAG_FAIL_OPEN:
                raise RuntimeError("Failed to initialize PageIndex backend.") from exc
            print(f"[RAG] PageIndex backend unavailable ({exc}). Falling back to FAISS backend.")

    embedding_config = EmbeddingConfig(
        code_model_name=RAG_CODE_EMBED_MODEL,
        text_model_name=RAG_TEXT_EMBED_MODEL,
    )
    store = VectorStoreManager(rag_dir)
    embeddings = EmbeddingService(embedding_config)
    backend = FaissRetrievalBackend(store, embeddings)
    return RetrievalService(backend), store


def main() -> None:
    args = parse_args()
    rag_dir = args.rag_dir
    print(f"[RAG] Initializing {args.backend} backend at {rag_dir}")

    retrieval, store = _build_retrieval(args.backend, rag_dir)
    print(f"[RAG] Active backend: {retrieval.backend_name}")

    existing_ids: set[str] = set()
    if store is not None:
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

    pdf_paths = sorted(str(path) for path in Path(args.pdf_dir).glob("*.pdf"))
    if retrieval.backend_name == "pageindex" and hasattr(retrieval.backend, "index_pdf_documents"):
        if pdf_paths:
            indexed_docs = retrieval.backend.index_pdf_documents(pdf_paths)
            print(f"[RAG] Generated/updated {indexed_docs} PageIndex trees from {len(pdf_paths)} PDFs.")
        else:
            print("[RAG] No PDF documents found for PageIndex tree generation.")
    else:
        pdf_documents = process_pdfs(args.pdf_dir)
        if pdf_documents:
            retrieval.index_text_documents(pdf_documents)
            print(f"[RAG] Indexed {len(pdf_documents)} PDF text chunks.")
        else:
            print("[RAG] No PDF documents found.")

    print("[RAG] Setup complete.")


if __name__ == "__main__":
    main()



