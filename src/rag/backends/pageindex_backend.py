"""PageIndex retrieval backend.

LLM-driven tree search over pre-built PageIndex document structures
(VectifyAI/PageIndex source, vendored under :mod:`src.pageindex`, with all
LLM calls routed to the local FastAPI ``/generate`` endpoint).

Trees are produced offline by ``scripts/build_pageindex_trees.py`` and
persisted as ``<doc_stem>_structure.json`` under
``${RAG_DATA_DIR}/pageindex_trees/``.  At retrieve time the backend sends
each tree (with text fields stripped, summaries kept) plus the query to the
LLM, parses the selected ``node_id``s, and returns the corresponding nodes'
full text as :class:`~src.rag.api_types.RetrievedBlock`s.

Design rules (mirrors :class:`~src.rag.backends.faiss_backend.FaissBackend`):

- Satisfies :class:`~src.rag.backend_protocol.BackendProtocol` structurally;
  no inheritance.
- No singleton coupling: the trees directory and LLM callable are injected
  at construction time.
- :meth:`index` is an intentional no-op shim — production indexing is owned
  by the offline tree builder.
- Trees are lazy-loaded on the first :meth:`retrieve` call so a missing
  trees directory does not break import or instance construction; it only
  surfaces (with a clear diagnostic) when the caller actually asks for
  retrieval.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from ..api_types import RetrievedBlock, RetrieveRequest, RetrieveResponse


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


@dataclass
class _LoadedTree:
    """Internal container for a loaded PageIndex tree."""

    stem: str
    doc_name: str
    structure: list
    node_map: dict = field(default_factory=dict)  # node_id -> node dict


def _default_llm_call(model: str, prompt: str) -> str:
    """Default LLM client: vendored ``ChatGPT_API`` (routes to local server).

    Imported lazily so test fakes never trigger the import of
    ``src.pageindex.pageindex.utils`` (and its tiktoken/PyMuPDF deps).
    """
    from src.pageindex.pageindex.utils import ChatGPT_API
    return ChatGPT_API(model=model, prompt=prompt)


class PageIndexBackend:
    """BackendProtocol-compliant adapter over PageIndex tree search.

    Args:
        trees_dir: Directory containing ``*_structure.json`` files produced
            by :mod:`scripts.build_pageindex_trees`.  Defaults to
            ``${RAG_DATA_DIR}/pageindex_trees/``.
        model: Model name forwarded to the underlying LLM client.  Defaults
            to ``"local_server"`` to match the vendored client.
        llm_call: Optional ``(model, prompt) -> str`` callable.  When not
            provided the backend uses
            :func:`src.pageindex.pageindex.utils.ChatGPT_API`.  Tests inject
            a fake here.
        min_relevance: Minimum LLM-assigned relevance (1-5) for a node to
            survive into the response.  Default 3 matches the prompt.
    """

    def __init__(
        self,
        trees_dir: Optional[str] = None,
        model: str = "local_server",
        llm_call: Optional[Callable[[str, str], str]] = None,
        min_relevance: float = 3.0,
    ) -> None:
        if trees_dir is None:
            # Lazy import — cfg.constants pulls in torch, and the protocol
            # tests assert that backend imports stay heavy-deps-free.
            from cfg.constants import RAG_DATA_DIR
            trees_dir = os.path.join(RAG_DATA_DIR, "pageindex_trees")
        self._trees_dir: Path = Path(trees_dir)
        self._model = model
        self._llm_call = llm_call or _default_llm_call
        self._min_relevance = float(min_relevance)
        self._trees: Optional[List[_LoadedTree]] = None  # lazy-loaded

    # ---------------------------------------------------------------------- #
    # BackendProtocol surface
    # ---------------------------------------------------------------------- #

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Run LLM tree search across all loaded PageIndex trees.

        Returns blocks ordered by ``relevance_score`` (descending), with
        ties broken in favour of leaf nodes.  Each block's ``diagnostics``
        dict contains ``source="pageindex"``, ``relevance``, ``is_leaf``,
        ``thinking`` (the LLM's reasoning), and ``doc_name``.
        """
        t0 = time.monotonic()
        query = request.query
        top_k = request.top_k

        if not query.strip():
            return RetrieveResponse(
                blocks=[],
                diagnostics={"source": "pageindex", "reason": "empty_query"},
                latency_ms=0.0,
            )

        try:
            trees = self._ensure_trees_loaded()
        except FileNotFoundError as exc:
            return RetrieveResponse(
                blocks=[],
                diagnostics={
                    "source": "pageindex",
                    "reason": "no_trees",
                    "trees_dir": str(self._trees_dir),
                    "error": str(exc),
                },
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )

        all_blocks: List[RetrievedBlock] = []
        per_tree_diag: List[dict] = []
        for tree in trees:
            blocks, diag = self._search_tree(query, tree)
            all_blocks.extend(blocks)
            per_tree_diag.append(diag)

        # Cross-document ranking: relevance desc, leaf nodes break ties.
        all_blocks.sort(
            key=lambda b: (
                b.score,
                bool((b.diagnostics or {}).get("is_leaf", True)),
            ),
            reverse=True,
        )
        all_blocks = all_blocks[:top_k]

        latency_ms = (time.monotonic() - t0) * 1000.0
        return RetrieveResponse(
            blocks=all_blocks,
            diagnostics={
                "source": "pageindex",
                "trees_searched": len(trees),
                "per_tree": per_tree_diag,
            },
            latency_ms=latency_ms,
        )

    def index(self, document: Any) -> None:  # noqa: ANN401
        """Satisfy the BackendProtocol surface.

        Production indexing is the offline tree-builder
        (``scripts/build_pageindex_trees.py``).  This shim exists so test
        fakes and protocol-compliance checks can call it without errors.
        """
        # Intentional no-op: the offline tree builder owns index mutations.

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _ensure_trees_loaded(self) -> List[_LoadedTree]:
        if self._trees is not None:
            return self._trees
        self._trees = self._load_trees(self._trees_dir)
        return self._trees

    @staticmethod
    def _load_trees(trees_dir: Path) -> List[_LoadedTree]:
        """Load all ``*_structure.json`` files from *trees_dir*."""
        from src.pageindex.pageindex.utils import structure_to_list

        if not trees_dir.exists():
            raise FileNotFoundError(
                f"PageIndex trees directory does not exist: {trees_dir}. "
                "Run scripts/build_pageindex_trees.py first."
            )

        loaded: List[_LoadedTree] = []
        for path in sorted(trees_dir.glob("*_structure.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            stem = path.stem.replace("_structure", "")
            structure = data.get("structure", [])
            nodes = structure_to_list(structure) if structure else []

            node_map = {}
            for node in nodes:
                nid = node.get("node_id")
                if nid is not None:
                    node_map[str(nid)] = node

            loaded.append(
                _LoadedTree(
                    stem=stem,
                    doc_name=data.get("doc_name", stem),
                    structure=structure,
                    node_map=node_map,
                )
            )

        if not loaded:
            raise FileNotFoundError(
                f"No PageIndex tree files (*_structure.json) found in {trees_dir}. "
                "Run scripts/build_pageindex_trees.py first."
            )
        return loaded

    def _search_tree(
        self, query: str, tree: _LoadedTree
    ) -> tuple[List[RetrievedBlock], dict]:
        """Send the compact tree to the LLM, parse selected nodes, build blocks."""
        from src.pageindex.pageindex.utils import extract_json, remove_fields

        compact_tree = remove_fields(copy.deepcopy(tree.structure), fields=["text"])
        compact_tree_json = json.dumps(compact_tree, ensure_ascii=False)

        prompt = TREE_SEARCH_PROMPT.format(
            query=query,
            doc_name=tree.doc_name,
            compact_tree_json=compact_tree_json,
        )

        try:
            response = self._llm_call(self._model, prompt)
        except Exception as exc:  # pragma: no cover — defensive
            return [], {
                "doc_name": tree.doc_name,
                "error": f"llm_call_failed: {exc}",
                "selected_count": 0,
            }

        if response is None or response == "Error":
            return [], {
                "doc_name": tree.doc_name,
                "error": "llm_error_response",
                "selected_count": 0,
            }

        parsed = extract_json(response)
        # The model sometimes drops the wrapper and returns just the
        # selected_nodes list (e.g. ``[{"node_id": "0001", "relevance": 5}]``).
        # Coerce both shapes to a (thinking, selected_nodes) pair.
        if isinstance(parsed, list):
            thinking = ""
            selected_nodes = parsed
        elif isinstance(parsed, dict):
            thinking = str(parsed.get("thinking", "") or "")
            selected_nodes = parsed.get("selected_nodes") or []
            if not isinstance(selected_nodes, list):
                selected_nodes = []
        else:
            thinking = ""
            selected_nodes = []

        blocks: List[RetrievedBlock] = []
        kept = 0
        for sn in selected_nodes:
            raw_nid = str(sn.get("node_id", ""))
            relevance = float(sn.get("relevance", 0) or 0)
            nid_str = raw_nid.zfill(4) if raw_nid.isdigit() else raw_nid
            node = tree.node_map.get(nid_str)
            if node is None:
                continue
            if relevance > 0 and relevance < self._min_relevance:
                continue
            is_leaf = not bool(node.get("nodes"))
            blocks.append(
                RetrievedBlock(
                    kind="pageindex_node",
                    document_id=f"{tree.stem}:{nid_str}",
                    title=str(node.get("title", ""))[:120],
                    score=relevance,
                    content=str(node.get("text", "")),
                    diagnostics={
                        "source": "pageindex",
                        "doc_name": tree.doc_name,
                        "node_id": nid_str,
                        "is_leaf": is_leaf,
                        "summary": node.get("summary", ""),
                        "thinking": thinking,
                        "relevance": relevance,
                    },
                )
            )
            kept += 1

        diag = {
            "doc_name": tree.doc_name,
            "selected_count": len(selected_nodes),
            "kept_count": kept,
        }
        return blocks, diag
