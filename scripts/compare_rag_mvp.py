#!/usr/bin/env python3
"""
MVP comparison: Vector RAG vs PageIndex RAG on the text namespace.

Runs both retrieval methods on the same set of queries and compares
latency, chunk count, context size, and (optionally) LLM-judged relevance.

Usage:
    python scripts/compare_rag_mvp.py
    python scripts/compare_rag_mvp.py --eval-relevance
    python scripts/compare_rag_mvp.py --top-k 5 --model gpt-4o-2024-11-20
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.vector_db import VectorStoreManager
from src.rag.embeddings import EmbeddingService, EmbeddingConfig
from src.rag.retrieval import RetrievalService
from src.rag.pageindex_retriever import PageIndexRetriever
from src.utils.rag_metrics import record_metric
from src.cfg.constants import RAG_DATA_DIR, RAG_MIN_SIMILARITY, RAG_TEXT_TOP_K

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREES_DIR = Path(RAG_DATA_DIR) / "pageindex_trees"

# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "id": "q1",
        "query": "CNN architecture parameter reduction techniques for CIFAR-10",
        "type": "Pure NL, param-focused",
    },
    {
        "id": "q2",
        "query": "CNN architecture Expert_Params mutation depthwise separable convolution",
        "type": "Mutation-type style",
    },
    {
        "id": "q3",
        "query": "data augmentation strategies for CIFAR-10 classification",
        "type": "Training-focused",
    },
    {
        "id": "q4",
        "query": "CNN architecture Complex mutation residual connections batch normalization",
        "type": "Complex mutation",
    },
    {
        "id": "q5",
        "query": (
            "CNN architecture EoT mutation\n"
            "class ExquisiteNetV2(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.features = nn.Sequential(\n"
            "            nn.Conv2d(3, 32, 3, padding=1),\n"
            "            nn.BatchNorm2d(32),\n"
            "            nn.ReLU(),\n"
            "            nn.Conv2d(32, 64, 3, padding=1),\n"
            "            nn.BatchNorm2d(64),\n"
            "            nn.ReLU(),\n"
            "        )"
        ),
        "type": "Mixed code+text",
    },
    {
        "id": "q6",
        "query": "retrieval augmented generation techniques for knowledge-intensive tasks",
        "type": "RAG survey content",
    },
]

# ---------------------------------------------------------------------------
# LLM-as-judge relevance scoring
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = """Rate the relevance of the following retrieved passage to the query on a scale of 1-5.

1 = Completely irrelevant
2 = Marginally relevant
3 = Somewhat relevant
4 = Highly relevant
5 = Perfectly relevant

Query: {query}

Retrieved passage:
{passage}

Reply with ONLY a JSON object: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""


def judge_relevance(query: str, passage: str, model: str) -> int:
    """Use an LLM to score passage relevance on 1-5 scale."""
    from src.pageindex.pageindex.utils import ChatGPT_API, extract_json

    # Truncate passage to avoid token blow-up
    words = passage.split()
    if len(words) > 500:
        passage = " ".join(words[:500]) + "..."

    prompt = RELEVANCE_PROMPT.format(query=query, passage=passage)
    response = ChatGPT_API(model=model, prompt=prompt)
    if response == "Error":
        return 0
    parsed = extract_json(response)
    score = parsed.get("score", 0)
    try:
        return max(1, min(5, int(score)))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_vector_rag(
    retrieval_svc: RetrievalService,
    query: str,
    top_k: int,
    min_similarity: float,
) -> dict:
    t0 = time.time()
    contexts, stats = retrieval_svc.retrieve_similar_text_with_stats(
        query=query, top_k=top_k, min_similarity=min_similarity,
    )
    latency_ms = (time.time() - t0) * 1000

    chunks = []
    for ctx in contexts:
        chunks.append({
            "document_id": ctx.document_id,
            "score": round(ctx.score, 4),
            "source": ctx.source,
            "content": ctx.content,
            "doc_type": ctx.doc_type,
        })

    total_words = sum(len(c["content"].split()) for c in chunks)
    return {
        "method": "vector",
        "latency_ms": round(latency_ms, 1),
        "retrieved_chunks": len(chunks),
        "context_word_count": total_words,
        "chunks": chunks,
        "retrieval_stats": asdict(stats),
    }


def run_pageindex_rag(
    retriever: PageIndexRetriever,
    query: str,
    top_k: int,
) -> dict:
    t0 = time.time()
    results, metadata = retriever.retrieve(query, top_k=top_k)
    latency_ms = (time.time() - t0) * 1000

    chunks = []
    thinking_parts = []
    for r in results:
        chunks.append({
            "document_id": r.document_id,
            "node_id": r.node_id,
            "title": r.title,
            "summary": r.summary,
            "source": r.source,
            "content": r.content,
        })
        if r.thinking and r.thinking not in thinking_parts:
            thinking_parts.append(r.thinking)

    total_words = sum(len(c["content"].split()) for c in chunks)
    return {
        "method": "pageindex",
        "latency_ms": round(latency_ms, 1),
        "retrieved_chunks": len(chunks),
        "context_word_count": total_words,
        "chunks": chunks,
        "pageindex_thinking": " | ".join(thinking_parts),
        "pageindex_metadata": metadata,
    }


def add_relevance_scores(result: dict, query: str, model: str) -> None:
    scores = []
    for chunk in result["chunks"]:
        score = judge_relevance(query, chunk["content"], model)
        scores.append(score)
    result["relevance_scores"] = scores
    result["avg_relevance"] = round(sum(scores) / len(scores), 2) if scores else 0.0


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_query_table(query_info: dict, vector_result: dict, pi_result: dict) -> None:
    qid = query_info["id"]
    qtext = query_info["query"][:80] + ("..." if len(query_info["query"]) > 80 else "")
    print(f"\nQuery {qid}: \"{qtext}\"")
    print(f"  Type: {query_info['type']}")

    header = f"{'Method':<12} {'Latency':>10} {'Chunks':>8} {'Words':>8}"
    has_rel = "avg_relevance" in vector_result
    if has_rel:
        header += f" {'Relevance':>11}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for res in [vector_result, pi_result]:
        name = res["method"].capitalize()
        if name == "Pageindex":
            name = "PageIndex"
        line = f"  {name:<12} {res['latency_ms']:>8.0f}ms {res['retrieved_chunks']:>8} {res['context_word_count']:>8}"
        if has_rel:
            line += f" {res.get('avg_relevance', 0):>8.1f}/5"
        print(line)


def print_aggregate(all_vector: list[dict], all_pi: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)

    def avg(results: list[dict], key: str) -> float:
        vals = [r[key] for r in results if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    header = f"{'Metric':<25} {'Vector RAG':>14} {'PageIndex RAG':>14}"
    print(header)
    print("-" * len(header))

    v_lat = avg(all_vector, "latency_ms")
    p_lat = avg(all_pi, "latency_ms")
    print(f"{'Avg Latency (ms)':<25} {v_lat:>14.0f} {p_lat:>14.0f}")

    v_chunks = avg(all_vector, "retrieved_chunks")
    p_chunks = avg(all_pi, "retrieved_chunks")
    print(f"{'Avg Chunks':<25} {v_chunks:>14.1f} {p_chunks:>14.1f}")

    v_words = avg(all_vector, "context_word_count")
    p_words = avg(all_pi, "context_word_count")
    print(f"{'Avg Context Words':<25} {v_words:>14.0f} {p_words:>14.0f}")

    if "avg_relevance" in all_vector[0]:
        v_rel = avg(all_vector, "avg_relevance")
        p_rel = avg(all_pi, "avg_relevance")
        print(f"{'Avg Relevance':<25} {v_rel:>12.1f}/5 {p_rel:>12.1f}/5")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Vector RAG vs PageIndex RAG")
    parser.add_argument(
        "--top-k", type=int, default=RAG_TEXT_TOP_K,
        help=f"Number of chunks to retrieve (default: {RAG_TEXT_TOP_K})",
    )
    parser.add_argument(
        "--min-similarity", type=float, default=RAG_MIN_SIMILARITY,
        help=f"Min similarity for vector RAG (default: {RAG_MIN_SIMILARITY})",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-2024-11-20",
        help="Model for PageIndex tree search and relevance eval",
    )
    parser.add_argument(
        "--trees-dir", type=str, default=str(DEFAULT_TREES_DIR),
        help="Directory with PageIndex tree JSON files",
    )
    parser.add_argument(
        "--eval-relevance", action="store_true",
        help="Run LLM-as-judge relevance scoring (adds latency & cost)",
    )
    args = parser.parse_args()

    # --- Init Vector RAG ---
    print("Initializing Vector RAG...")
    store = VectorStoreManager(RAG_DATA_DIR)
    embeddings = EmbeddingService(EmbeddingConfig())
    retrieval_svc = RetrievalService(store, embeddings)

    # --- Init PageIndex RAG ---
    print("Initializing PageIndex RAG...")
    pi_retriever = PageIndexRetriever(
        trees_dir=args.trees_dir,
        model=args.model,
    )
    print(f"  Loaded {len(pi_retriever._trees)} tree(s)")

    # --- Run comparison ---
    all_vector_results: list[dict] = []
    all_pi_results: list[dict] = []

    print(f"\nRunning {len(QUERIES)} queries (top_k={args.top_k})...")
    if args.eval_relevance:
        print("  (with LLM relevance evaluation)")

    for q in QUERIES:
        query_text = q["query"]

        # Vector RAG
        v_result = run_vector_rag(retrieval_svc, query_text, args.top_k, args.min_similarity)
        v_result["query_id"] = q["id"]

        # PageIndex RAG
        pi_result = run_pageindex_rag(pi_retriever, query_text, args.top_k)
        pi_result["query_id"] = q["id"]

        # Optional relevance eval
        if args.eval_relevance:
            add_relevance_scores(v_result, query_text, args.model)
            add_relevance_scores(pi_result, query_text, args.model)

        # Print per-query table
        print_query_table(q, v_result, pi_result)

        # Log metrics (strip full chunk content to keep metrics lean)
        for res in [v_result, pi_result]:
            metric_payload = {k: v for k, v in res.items() if k != "chunks"}
            record_metric("rag_comparison", metric_payload)

        all_vector_results.append(v_result)
        all_pi_results.append(pi_result)

    # --- Aggregate ---
    print_aggregate(all_vector_results, all_pi_results)
    print(f"\nMetrics logged to rag_metrics.jsonl")


if __name__ == "__main__":
    main()
