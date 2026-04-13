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

TREE_SEARCH_PROMPT = """You are a domain expert evaluating document sections for relevance to a query.
The document "{doc_name}" is organized as a tree. Each node has a node_id, title, and summary.

Domain context: This system supports CNN evolution research for CIFAR-10 image classification.
The documents cover CNN architectures, training techniques, and retrieval-augmented generation.

Query: {query}

Document tree structure:
{compact_tree_json}

Instructions:
- Select nodes whose content is likely to answer or inform the query.
- Prefer leaf nodes (nodes without children) over parent nodes when both cover the same content.
  Parent nodes aggregate child text, so selecting a parent when a specific child suffices adds noise.
- For each selected node, assign a relevance score from 1 to 5:
  1 = Marginally related  2 = Somewhat related  3 = Relevant  4 = Highly relevant  5 = Perfectly relevant
- Only include nodes with relevance >= 3.

Reply in JSON:
{{
  "thinking": "<your reasoning>",
  "selected_nodes": [
    {{"node_id": "<id>", "relevance": <int 1-5>}},
    ...
  ]
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
    relevance_score: float = 0.0  # 1-5 from LLM, 0 = unscored (legacy)
    is_leaf: bool = True          # whether node has no children


@dataclass
class _LoadedTree:
    """Internal container for a loaded PageIndex tree."""

    stem: str
    doc_name: str
    structure: list | dict
    node_map: dict = field(default_factory=dict)  # node_id -> node dict


class PageIndexRetriever:
    """Retrieves relevant document sections via LLM tree search over PageIndex structures."""

    def __init__(self, trees_dir: str, model: str = "local_server"):
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

        # Cross-document ranking: highest relevance first, leaf nodes break ties
        all_results.sort(
            key=lambda r: (r.relevance_score, r.is_leaf), reverse=True
        )
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
            doc_name=tree.doc_name,
            compact_tree_json=compact_tree_json,
        )

        response = ChatGPT_API(model=self.model, prompt=prompt)
        if response == "Error":
            return []

        parsed = extract_json(response)
        thinking = parsed.get("thinking", "")

        # Parse new format (selected_nodes with relevance) or legacy (node_list)
        selected_nodes = parsed.get("selected_nodes", [])
        if selected_nodes:
            node_entries = [
                (str(sn.get("node_id", "")), float(sn.get("relevance", 0)))
                for sn in selected_nodes
            ]
        else:
            # Backward compat: flat node_list with no relevance scores
            node_ids = parsed.get("node_list", [])
            node_entries = [(str(nid), 0.0) for nid in node_ids]

        results: list[PageIndexResult] = []
        for raw_nid, relevance in node_entries:
            nid_str = raw_nid.zfill(4) if raw_nid.isdigit() else raw_nid
            node = tree.node_map.get(nid_str)
            if node is None:
                continue
            # Filter by minimum relevance threshold when scores are available
            if relevance > 0 and relevance < 3:
                continue
            is_leaf = not bool(node.get("nodes"))
            results.append(PageIndexResult(
                document_id=f"{tree.stem}:{nid_str}",
                node_id=nid_str,
                content=node.get("text", ""),
                title=node.get("title", ""),
                summary=node.get("summary", ""),
                source=tree.doc_name,
                thinking=thinking,
                relevance_score=relevance,
                is_leaf=is_leaf,
            ))

        return results
