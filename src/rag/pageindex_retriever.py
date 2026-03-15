"""PageIndex tree-search retrieval adapter.

Implements LLM-driven tree search over pre-built PageIndex document structures.
For each query, sends the compact tree (without full text) to an LLM which
selects relevant nodes, then extracts full text from those nodes.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from src.pageindex.pageindex.utils import (
    ChatGPT_API,
    extract_json,
    remove_fields,
    structure_to_list,
)

TREE_SEARCH_PROMPT = """You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {compact_tree_json}

Reply in the following JSON format:
{{
  "thinking": <your reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}"""


@dataclass(frozen=True)
class PageIndexResult:
    """A single retrieved node from PageIndex tree search."""

    document_id: str   # "{pdf_stem}:{node_id}"
    node_id: str
    content: str       # full text of the node
    title: str
    summary: str
    source: str        # pdf filename
    thinking: str      # LLM's reasoning for selecting this node


@dataclass
class _LoadedTree:
    """Internal container for a loaded PageIndex tree."""

    stem: str
    doc_name: str
    structure: list | dict
    node_map: dict = field(default_factory=dict)  # node_id -> node dict


class PageIndexRetriever:
    """Retrieves relevant document sections via LLM tree search over PageIndex structures."""

    def __init__(self, trees_dir: str, model: str = "gpt-4o-2024-11-20"):
        self.trees_dir = Path(trees_dir)
        self.model = model
        self._trees: list[_LoadedTree] = []
        self._load_trees()

    def _load_trees(self) -> None:
        for path in sorted(self.trees_dir.glob("*_structure.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            stem = path.stem.replace("_structure", "")
            structure = data.get("structure", [])

            # Build node_id -> node lookup from flat list
            nodes = structure_to_list(structure) if structure else []
            node_map = {}
            for node in nodes:
                nid = node.get("node_id")
                if nid is not None:
                    node_map[str(nid)] = node

            tree = _LoadedTree(
                stem=stem,
                doc_name=data.get("doc_name", stem),
                structure=structure,
                node_map=node_map,
            )
            self._trees.append(tree)

        if not self._trees:
            raise FileNotFoundError(
                f"No PageIndex tree files found in {self.trees_dir}. "
                "Run scripts/build_pageindex_trees.py first."
            )

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> Tuple[List[PageIndexResult], dict]:
        """Search all loaded trees for nodes relevant to *query*.

        Returns:
            (results, metadata) where metadata includes latency_ms and token_estimate.
        """
        t0 = time.time()
        all_results: list[PageIndexResult] = []

        for tree in self._trees:
            results = self._search_tree(query, tree)
            all_results.extend(results)

        # Trim to top_k (results are already ordered per-document)
        all_results = all_results[:top_k]

        latency_ms = (time.time() - t0) * 1000
        metadata = {
            "latency_ms": round(latency_ms, 1),
            "trees_searched": len(self._trees),
            "total_results": len(all_results),
        }
        return all_results, metadata

    def _search_tree(self, query: str, tree: _LoadedTree) -> list[PageIndexResult]:
        """Send compact tree to LLM, parse selected node_ids, extract full text."""
        compact_tree = remove_fields(
            copy.deepcopy(tree.structure), fields=["text"]
        )
        compact_tree_json = json.dumps(compact_tree, ensure_ascii=False)

        prompt = TREE_SEARCH_PROMPT.format(
            query=query,
            compact_tree_json=compact_tree_json,
        )

        response = ChatGPT_API(model=self.model, prompt=prompt)
        if response == "Error":
            return []

        parsed = extract_json(response)
        thinking = parsed.get("thinking", "")
        node_ids = parsed.get("node_list", [])

        results: list[PageIndexResult] = []
        for nid in node_ids:
            nid_str = str(nid).zfill(4) if str(nid).isdigit() else str(nid)
            node = tree.node_map.get(nid_str)
            if node is None:
                continue
            results.append(PageIndexResult(
                document_id=f"{tree.stem}:{nid_str}",
                node_id=nid_str,
                content=node.get("text", ""),
                title=node.get("title", ""),
                summary=node.get("summary", ""),
                source=tree.doc_name,
                thinking=thinking,
            ))

        return results
