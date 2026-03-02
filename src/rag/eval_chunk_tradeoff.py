"""
Evaluate precision/recall tradeoffs for different chunk sizes in the PyTorch RAG corpus.

This script rebuilds (or reuses) a text index per chunk size and reports:
- Precision@K
- Recall@K
- MRR@K
- Average context length (words)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rag.data_ingestion import ingest_pytorch_docs
from src.rag.embeddings import EmbeddingService
from src.rag.retrieval import RetrievalService
from src.rag.vector_db import VectorStoreManager


DEFAULT_GOLDEN_PATH = Path(__file__).with_name("golden_queries.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAG_DATA_DIR = os.environ.get("RAG_DATA_DIR", str(REPO_ROOT / "rag_data"))


def load_golden_dataset(path: Path, tier: str, limit: int | None = None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tiers = payload.get("tiers") or {}
    if tier not in tiers:
        available = ", ".join(sorted(tiers)) or "none"
        raise ValueError(f"Unknown tier '{tier}'. Available: {available}")
    items = list(tiers[tier])
    if limit is not None:
        items = items[: max(0, limit)]
    return items


def _match_expected(name: str, expected: Sequence[str]) -> List[str]:
    name_lower = name.lower()
    hits = [exp for exp in expected if exp.lower() in name_lower]
    return hits


def _prepare_store(
    json_path: Path,
    rag_data_dir: Path,
    chunk_words: int,
    chunk_threshold_words: int,
    reindex: bool,
) -> tuple[VectorStoreManager, EmbeddingService]:
    store = VectorStoreManager(rag_data_dir)
    embeddings = EmbeddingService()

    existing_docs = store.list_documents(VectorStoreManager.TEXT_NAMESPACE)
    if existing_docs and not reindex:
        return store, embeddings

    if existing_docs and reindex:
        suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        rag_data_dir = rag_data_dir.with_name(f"{rag_data_dir.name}_{suffix}")
        store = VectorStoreManager(rag_data_dir)

    documents = ingest_pytorch_docs(
        str(json_path),
        chunk_words=chunk_words,
        chunk_threshold_words=chunk_threshold_words,
    )
    retrieval = RetrievalService(store, embeddings)
    retrieval.index_text_documents(documents)
    return store, embeddings


def evaluate_store(
    store: VectorStoreManager,
    embeddings: EmbeddingService,
    top_k: int,
    dataset: Iterable[dict],
) -> dict:
    total = 0
    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    context_words_sum = 0

    for item in dataset:
        query = item["query"]
        expected = item["expected"]
        total += 1

        q_emb = embeddings.embed_text(query)[0]
        results = store.search_text(q_emb, top_k=top_k)

        matched = set()
        rank = None
        for i, res in enumerate(results):
            name = res.document.metadata.get("name", "")
            hits = _match_expected(name, expected)
            if hits:
                matched.update(hits)
                if rank is None:
                    rank = i + 1

        precision_sum += (len(matched) / max(top_k, 1))
        recall_sum += (len(matched) / max(len(expected), 1))
        if rank is not None:
            mrr_sum += 1.0 / rank

        context_words_sum += sum(len(res.document.content.split()) for res in results)

    return {
        "queries": total,
        "precision_at_k": precision_sum / max(total, 1),
        "recall_at_k": recall_sum / max(total, 1),
        "mrr_at_k": mrr_sum / max(total, 1),
        "avg_context_words": context_words_sum / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk-words",
        type=int,
        nargs="+",
        default=[200, 400, 800],
        help="Chunk sizes (words) to evaluate.",
    )
    parser.add_argument(
        "--chunk-threshold-words",
        type=int,
        default=500,
        help="Chunk only if document exceeds this many words.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K documents to retrieve.")
    parser.add_argument(
        "--rag-data-root",
        type=str,
        default="rag_data_eval",
        help="Root directory for evaluation indices.",
    )
    parser.add_argument(
        "--rag-data-dir",
        type=str,
        default=DEFAULT_RAG_DATA_DIR,
        help="Existing RAG data directory to reuse (ignored if you reindex into rag-data-root).",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="rag_corpus/pytorch.json",
        help="Path to the PyTorch JSON corpus.",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="small",
        choices=("small", "medium"),
        help="Golden query tier to evaluate.",
    )
    parser.add_argument(
        "--golden-path",
        type=str,
        default=str(DEFAULT_GOLDEN_PATH),
        help="Path to golden_queries.json.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of queries.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindexing by writing into a new timestamped directory.",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {json_path}")

    rag_root = Path(args.rag_data_root)
    rag_root.mkdir(parents=True, exist_ok=True)

    print("Chunk Size Tradeoff Evaluation")
    print("-" * 60)

    dataset = load_golden_dataset(Path(args.golden_path), args.tier, limit=args.limit)

    for chunk_words in args.chunk_words:
        rag_dir = rag_root / f"chunk_{chunk_words}"
        store, embeddings = _prepare_store(
            json_path=json_path,
            rag_data_dir=rag_dir,
            chunk_words=chunk_words,
            chunk_threshold_words=args.chunk_threshold_words,
            reindex=args.reindex,
        )
        metrics = evaluate_store(
            store=store,
            embeddings=embeddings,
            top_k=args.top_k,
            dataset=dataset,
        )
        print(
            f"chunk_words={chunk_words} | "
            f"P@{args.top_k}={metrics['precision_at_k']:.3f} | "
            f"R@{args.top_k}={metrics['recall_at_k']:.3f} | "
            f"MRR@{args.top_k}={metrics['mrr_at_k']:.3f} | "
            f"avg_ctx_words={metrics['avg_context_words']:.1f}"
        )


if __name__ == "__main__":
    main()
