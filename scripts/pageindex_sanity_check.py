#!/usr/bin/env python3
"""Sanity check for the PageIndex backend.

Two parts:

1. **Tree structure check** — for each ``*_structure.json`` under the trees
   directory, print node count, max depth, total text words, and the title
   of the largest node.  This is a static check (no LLM, no server).

2. **Live retrieval check** — instantiate ``PageIndexBackend`` against the
   trees directory and run a small set of natural-language queries through
   the live local LLM server.  Print the top-k blocks per query.

Usage::

    export RAG_DATA_DIR=$(pwd)/rag_data
    export HOSTNAME_LOG_FILE=$(pwd)/hostname-runtime.log
    export SERVER_PORT=8000
    .venv/bin/python scripts/pageindex_sanity_check.py
    .venv/bin/python scripts/pageindex_sanity_check.py --static-only

The script is intended to be run by hand for verification; it is not part
of the test suite.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.api_types import RetrieveRequest  # noqa: E402
from src.rag.backends.pageindex_backend import PageIndexBackend  # noqa: E402


# --------------------------------------------------------------------------- #
# Static tree structure check
# --------------------------------------------------------------------------- #

def _walk(node, depth=0):
    """Yield (node_dict, depth) pairs for every node in *structure*."""
    if isinstance(node, list):
        for item in node:
            yield from _walk(item, depth)
        return
    if not isinstance(node, dict):
        return
    yield node, depth
    children = node.get("nodes")
    if children:
        yield from _walk(children, depth + 1)


def _summarise_tree(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    structure = data.get("structure", []) or []
    items = list(_walk(structure))
    if not items:
        return {
            "path": path.name,
            "doc_name": data.get("doc_name"),
            "node_count": 0,
        }

    node_count = len(items)
    max_depth = max(d for _, d in items)
    total_words = sum(len((n.get("text") or "").split()) for n, _ in items)
    largest = max(items, key=lambda nd: len((nd[0].get("text") or "")))
    return {
        "path": path.name,
        "doc_name": data.get("doc_name"),
        "node_count": node_count,
        "max_depth": max_depth,
        "total_text_words": total_words,
        "largest_node_title": largest[0].get("title"),
        "largest_node_words": len((largest[0].get("text") or "").split()),
        "summary_present_for_root": bool(structure[0].get("summary")) if structure else False,
    }


def static_check(trees_dir: Path) -> list[dict]:
    print(f"\n=== STATIC TREE CHECK: {trees_dir} ===\n")
    summaries = []
    for path in sorted(trees_dir.glob("*_structure.json")):
        summary = _summarise_tree(path)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print()
    if not summaries:
        print(f"No *_structure.json files found in {trees_dir}.")
    return summaries


# --------------------------------------------------------------------------- #
# Live retrieval check
# --------------------------------------------------------------------------- #

DEFAULT_QUERIES = [
    "What CNN architectures perform well on CIFAR-10?",
    "How does retrieval-augmented generation improve large language models?",
    "What training tricks (data augmentation, regularization) help with image classification?",
]


def _build_fake_llm(trees_dir: Path):
    """Return a deterministic fake LLM callable that selects the highest-
    information-density leaf node from each tree by simple keyword overlap
    against the query. Used by ``--fake-llm`` to exercise the backend
    end-to-end on real on-disk trees without depending on the live server."""
    import json as _json

    def _walk(struct, out=None):
        if out is None:
            out = []
        if isinstance(struct, list):
            for item in struct:
                _walk(item, out)
            return out
        if isinstance(struct, dict):
            out.append(struct)
            children = struct.get("nodes")
            if children:
                _walk(children, out)
        return out

    cache: dict[str, list[dict]] = {}
    for path in sorted(trees_dir.glob("*_structure.json")):
        data = _json.loads(path.read_text(encoding="utf-8"))
        cache[data.get("doc_name", path.stem)] = _walk(data.get("structure", []))

    def _score(node, query_terms):
        text = (
            (node.get("title") or "") + " " +
            (node.get("summary") or "") + " " +
            (node.get("text") or "")
        ).lower()
        return sum(1 for q in query_terms if q in text)

    def fake_llm(model: str, prompt: str) -> str:
        # The prompt embeds the query right after "Query:".
        try:
            after = prompt.split("Query:", 1)[1]
            query = after.split("\n", 1)[0].strip()
        except Exception:
            query = ""
        # The doc_name appears in `The document "<name>"`
        doc_name = ""
        try:
            doc_name = prompt.split('The document "', 1)[1].split('"', 1)[0]
        except Exception:
            pass
        nodes = cache.get(doc_name, [])
        terms = [t for t in query.lower().replace("?", "").split() if len(t) > 3]
        if not nodes or not terms:
            return _json.dumps({"thinking": "fake_llm", "selected_nodes": []})
        scored = [(n, _score(n, terms)) for n in nodes]
        scored.sort(key=lambda p: (p[1], not bool(p[0].get("nodes"))), reverse=True)
        picked = [n for n, s in scored[:3] if s > 0]
        return _json.dumps({
            "thinking": f"fake_llm picked top-{len(picked)} nodes by keyword overlap",
            "selected_nodes": [
                {"node_id": n["node_id"], "relevance": min(5, 3 + i)}
                for i, n in enumerate(picked)
            ],
        })

    return fake_llm


def live_check(trees_dir: Path, queries: list[str], top_k: int = 3,
               use_fake_llm: bool = False) -> None:
    print(f"\n=== {'FAKE-LLM' if use_fake_llm else 'LIVE'} RETRIEVAL CHECK: {trees_dir} ===\n")
    if use_fake_llm:
        backend = PageIndexBackend(
            trees_dir=str(trees_dir),
            llm_call=_build_fake_llm(trees_dir),
        )
    else:
        backend = PageIndexBackend(trees_dir=str(trees_dir))

    for q in queries:
        print(f"--- query: {q!r} ---")
        resp = backend.retrieve(RetrieveRequest(query=q, top_k=top_k))
        diag = resp.diagnostics or {}
        print(f"  trees_searched={diag.get('trees_searched')} "
              f"latency_ms={resp.latency_ms:.0f}")
        if not resp.blocks:
            print("  (no blocks returned)")
            print()
            continue
        for i, b in enumerate(resp.blocks, 1):
            bd = b.diagnostics or {}
            preview = (b.content or "").strip().replace("\n", " ")[:200]
            print(
                f"  [{i}] doc={bd.get('doc_name')!r} "
                f"node_id={bd.get('node_id')} "
                f"title={b.title!r} "
                f"score={b.score} "
                f"is_leaf={bd.get('is_leaf')}"
            )
            print(f"      preview: {preview!r}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trees-dir",
        type=str,
        default=os.path.join(
            os.environ.get("RAG_DATA_DIR", str(REPO_ROOT / "rag_data")),
            "pageindex_trees",
        ),
        help="Directory containing *_structure.json files.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Top-k for live retrieval check.",
    )
    parser.add_argument(
        "--static-only", action="store_true",
        help="Skip the live LLM retrieval check.",
    )
    parser.add_argument(
        "--fake-llm", action="store_true",
        help="Use a deterministic keyword-based fake LLM instead of the "
             "real local server. Useful for verifying the backend pipeline "
             "without depending on the model's prompt-following.",
    )
    parser.add_argument(
        "--queries", type=str, nargs="+", default=DEFAULT_QUERIES,
        help="Queries for the live retrieval check.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    trees_dir = Path(args.trees_dir)
    static_check(trees_dir)

    if args.static_only:
        return

    if not trees_dir.exists() or not list(trees_dir.glob("*_structure.json")):
        print(f"\n(skipping live check — no trees in {trees_dir})")
        return

    live_check(trees_dir, args.queries, top_k=args.top_k, use_fake_llm=args.fake_llm)


if __name__ == "__main__":
    main()
