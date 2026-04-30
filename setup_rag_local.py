#!/usr/bin/env python3
"""Bootstrap the local RAG vector database with sys.path setup."""

import sys
import importlib.util
import types
from pathlib import Path

# Load rag modules directly from src/rag to avoid importing src/rag/__init__.py
project_root = Path(__file__).parent
src_rag = project_root / "src" / "rag"
# Ensure 'rag' package exists in sys.modules so dataclasses and relative
# module lookups inside rag modules resolve correctly.
if "rag" not in sys.modules:
    rag_pkg = types.ModuleType("rag")
    rag_pkg.__path__ = [str(src_rag)]
    sys.modules["rag"] = rag_pkg

# Ensure 'cfg' package exists for cfg.constants
cfg_path = project_root / "src" / "cfg"
if "cfg" not in sys.modules:
    cfg_pkg = types.ModuleType("cfg")
    cfg_pkg.__path__ = [str(cfg_path)]
    sys.modules["cfg"] = cfg_pkg


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    # Ensure the module is visible in sys.modules under its name
    sys.modules[name] = module
    return module

# Load cfg.constants
cfg_mod = _load_module_from_path("cfg.constants", project_root / "src" / "cfg" / "constants.py")
RAG_DATA_DIR = getattr(cfg_mod, "RAG_DATA_DIR")
ROOT_DIR = getattr(cfg_mod, "ROOT_DIR")
SOTA_ROOT = getattr(cfg_mod, "SOTA_ROOT")

# Load individual rag modules
data_ingestion_mod = _load_module_from_path("data_ingestion_local", src_rag / "data_ingestion.py")
embeddings_mod = _load_module_from_path("embeddings_local", src_rag / "embeddings.py")
retrieval_mod = _load_module_from_path("retrieval_local", src_rag / "retrieval.py")
vector_db_mod = _load_module_from_path("vector_db_local", src_rag / "vector_db.py")

# Export expected symbols
extract_mutations_from_checkpoints = getattr(data_ingestion_mod, "extract_mutations_from_checkpoints")
process_pdfs = getattr(data_ingestion_mod, "process_pdfs")
EmbeddingService = getattr(embeddings_mod, "EmbeddingService")
RetrievalService = getattr(retrieval_mod, "RetrievalService")
VectorStoreManager = getattr(vector_db_mod, "VectorStoreManager")

import argparse


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
    print(f"[RAG] Found {len(existing_ids)} existing mutation records")
    
    print(f"[RAG] Extracting mutations from {args.runs_dir}...")
    mutation_records = extract_mutations_from_checkpoints(
        runs_dir=args.runs_dir,
        models_dir=SOTA_ROOT,
        limit=args.limit,
    )
    print(f"[RAG] Found {len(mutation_records)} total mutation records")
    
    new_records = [record for record in mutation_records if record.gene_id not in existing_ids]
    if new_records:
        print(f"[RAG] Indexing {len(new_records)} new mutation records...")
        retrieval.index_mutations(new_records)
        print(f"[RAG] Successfully indexed {len(new_records)} new mutation records.")
    else:
        print("[RAG] No new mutation records to index.")

    print(f"[RAG] Processing PDFs from {args.pdf_dir}...")
    pdfs_dir = Path(args.pdf_dir)
    if pdfs_dir.exists():
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        print(f"[RAG] Found {len(pdf_files)} PDF files")
        if pdf_files:
            documents = process_pdfs(str(pdfs_dir))
            print(f"[RAG] Processed {len(documents)} document chunks from PDFs")
            if documents:
                retrieval.index_text_documents(documents)
                print(f"[RAG] Successfully indexed PDF documents.")
        else:
            print("[RAG] No PDF files found in corpus directory.")
    else:
        print(f"[RAG] PDF directory not found: {pdfs_dir}")

    print("[RAG] Vector database initialization complete!")


if __name__ == "__main__":
    main()
