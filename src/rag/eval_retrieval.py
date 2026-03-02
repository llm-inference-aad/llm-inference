
"""
RAG Retrieval Evaluation Benchmark (text namespace).

Measures Recall@K and MRR@K for a golden query set against the text namespace.
Golden queries live in `src/rag/golden_queries.json` and support multiple tiers.
"""
import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.cfg.constants import RAG_DATA_DIR, RAG_TOP_K
from src.rag.vector_db import VectorStoreManager
from src.rag.embeddings import EmbeddingService

DEFAULT_GOLDEN_PATH = Path(__file__).with_name("golden_queries.json")


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


def evaluate_retrieval(top_k: int = 5, *, dataset: list[dict]) -> None:
    print(f"Initializing RAG Components (Top-K={top_k})...")
    try:
        store = VectorStoreManager(RAG_DATA_DIR)
        embeddings = EmbeddingService()
    except Exception as e:
        print(f"Error initializing: {e}")
        return
    
    print(f"Running Evaluation on {len(dataset)} queries...")
    print("-" * 60)
    
    hits = 0
    total = len(dataset)
    mrr_sum = 0.0

    for item in dataset:
        query = item["query"]
        expected_list = item["expected"]
        expected = expected_list[0] if expected_list else ""
        start_time = time.time()
        q_emb = embeddings.embed_text(query)[0]
        results = store.search_text(q_emb, top_k=top_k)
        duration = time.time() - start_time
        
        found = False
        rank = 0
        
        print(f"Query: '{query}'")
        print(f"  Expect: '{expected}'")
        
        for i, res in enumerate(results):
            doc_name = res.document.metadata.get("name", "")
            # Check if expected substring is in the retrieved doc name
            if expected and expected.lower() in (doc_name or "").lower():
                found = True
                rank = i + 1
                break
        
        if found:
            print(f"  ✅ Found at Rank {rank} (Score: {results[rank-1].score:.4f})")
            hits += 1
            mrr_sum += 1.0 / rank
        else:
            print(f"  ❌ Not Found in Top-{top_k}")
            print(f"     TopResult: {results[0].document.metadata.get('name')} ({results[0].score:.4f})")
        
        print("-" * 60)

    recall = hits / total
    mrr = mrr_sum / total
    
    print(f"\nEvaluation Results:")
    print(f"  Recall@{top_k}: {recall:.2%} ({hits}/{total})")
    print(f"  MRR@{top_k}:    {mrr:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-K documents to retrieve")
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
    args = parser.parse_args()
    
    dataset = load_golden_dataset(Path(args.golden_path), args.tier, limit=args.limit)
    evaluate_retrieval(top_k=args.k, dataset=dataset)
