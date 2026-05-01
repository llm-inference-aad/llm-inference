"""Graph retrieval backend — scaffolding for the Graph team to fill in.

Status: **scaffolding**, not a working backend. ``retrieve()`` returns an
empty :class:`~src.rag.api_types.RetrieveResponse` whose diagnostics
explain why; ``index()`` is a no-op shim. The shape and DI surface mirror
``FaissBackend`` and ``PageIndexBackend`` (the latter on
``feature/rag-pipeline-ben``) so a worker implementing the real graph
search can replace the body of ``_search_graph`` without touching call
sites.

What "Graph" means here
-----------------------
Documents (PDFs, prior mutation networks, API references) are pre-processed
offline into a node-edge graph rooted at a small set of entry concepts
("CIFAR-10", "depthwise separable convolution", "BatchNorm", …). Edges
encode topical similarity, citation, and code-symbol references. At
retrieve time, the backend walks the graph from query-anchored entry nodes
outward, ranks visited nodes, and returns the top-K as
:class:`~src.rag.api_types.RetrievedBlock`s.

The graph is built by an offline tool (TBD by the Graph worker, a sibling
of ``scripts/build_pageindex_trees.py``) and persisted as JSON / pickle
under ``${RAG_DATA_DIR}/graph/``.

Worker checklist (to flip this from scaffolding to implementation)
------------------------------------------------------------------
1. Decide the graph storage format (JSON adjacency list vs networkx pickle).
2. Implement ``_load_graph`` to read it, populating ``self._graph``.
3. Implement ``_search_graph(query, graph)``: rank nodes, return
   ``(list[RetrievedBlock], diagnostics)``. Embed-and-cosine over node
   summaries is the simplest first cut; richer beam-search-from-anchors is
   a follow-up.
4. Add the offline build tool at ``scripts/build_graph.py``.
5. Replace ``ImplementationStatus.SCAFFOLDING`` with ``IMPLEMENTED`` so the
   protocol-compliance test in ``tests/rag/test_backend_protocol_compliance.py``
   stops marking the backend as a stub.
6. Remove ``"graph"`` from ``_STUB_BACKENDS`` in
   ``scripts/rag_replay/02_rag_service.py`` once retrieval works
   end-to-end against a real graph file.

Until that's done, calling ``retrieve()`` returns a structured "not
implemented" response rather than raising — so a misconfigured replay run
fails with a useful diagnostic block instead of a stack trace mid-batch.
The fail-fast at ``02_rag_service.py`` validation still blocks
``RAG_BACKEND=graph`` at the harness boundary; this scaffolding is for
direct in-process callers (tests, sanity scripts) and for the eventual
implementation to slot into.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from ..api_types import RetrievedBlock, RetrieveRequest, RetrieveResponse


class ImplementationStatus(Enum):
    """Tracks where this backend is in its rollout."""

    SCAFFOLDING = "scaffolding"   # Stub, returns diagnostic empty response
    PARTIAL = "partial"           # Some queries work, some return empty
    IMPLEMENTED = "implemented"   # Worker has filled in retrieval


@dataclass
class _LoadedGraph:
    """Container for an in-memory graph.

    The fields below are placeholders — the implementing worker should
    replace them (or extend this dataclass) with whatever native graph
    representation they pick (e.g. ``networkx.DiGraph``, an adjacency dict,
    or a ``node_id → node_metadata`` map plus an ``edges`` list).
    """

    name: str
    nodes: dict = field(default_factory=dict)  # node_id -> {summary, source, …}
    edges: list = field(default_factory=list)  # (src, dst, weight, kind)


def _default_llm_call(model: str, prompt: str) -> str:  # pragma: no cover
    """Default LLM client — mirrors the PageIndex backend's default.

    Imported lazily so the protocol-compliance test (which forbids
    transitive heavy-dep imports from this module's import-time path)
    keeps passing while this file remains a stub.
    """
    from src.pageindex.pageindex.utils import ChatGPT_API  # type: ignore
    return ChatGPT_API(model=model, prompt=prompt)


class GraphBackend:
    """BackendProtocol-compliant scaffolding over a (yet-to-be-built) graph.

    Args:
        graph_dir: Directory containing the graph artifact(s). Defaults to
            ``${RAG_DATA_DIR}/graph/``. May not exist yet — that's fine,
            ``retrieve()`` returns an empty diagnostic response in that
            case rather than raising.
        model: Model name forwarded to the LLM client when graph search
            uses an LLM (mirrors PageIndexBackend).
        llm_call: Optional ``(model, prompt) -> str`` callable. When not
            provided the backend uses the same vendored client as
            PageIndex. Tests inject a fake here.
        min_score: Minimum node score for inclusion. Reasonable default
            once retrieval is implemented; held as instance state for now.
    """

    STATUS = ImplementationStatus.SCAFFOLDING

    def __init__(
        self,
        graph_dir: Optional[str] = None,
        model: str = "local_server",
        llm_call: Optional[Callable[[str, str], str]] = None,
        min_score: float = 0.3,
    ) -> None:
        if graph_dir is None:
            from cfg.constants import RAG_DATA_DIR  # type: ignore
            graph_dir = os.path.join(RAG_DATA_DIR, "graph")
        self._graph_dir: Path = Path(graph_dir)
        self._model = model
        self._llm_call = llm_call or _default_llm_call
        self._min_score = float(min_score)
        self._graph: Optional[_LoadedGraph] = None  # lazy-loaded

    # ------------------------------------------------------------------ #
    # BackendProtocol surface — these match PageIndexBackend exactly so a
    # caller swapping `RAG_BACKEND=pageindex → graph` doesn't see any
    # interface change.
    # ------------------------------------------------------------------ #

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Run graph-search for *query*.

        Until a worker fills in :meth:`_search_graph`, this returns an
        empty :class:`RetrieveResponse` whose diagnostics tell the caller
        which step is missing. That's intentional: structured diagnostics
        > exception traceback in the middle of a 30-row replay batch.
        """
        t0 = time.monotonic()
        query = request.query

        if not query.strip():
            return RetrieveResponse(
                blocks=[],
                diagnostics={"source": "graph", "reason": "empty_query"},
                latency_ms=0.0,
            )

        if self.STATUS == ImplementationStatus.SCAFFOLDING:
            return RetrieveResponse(
                blocks=[],
                diagnostics={
                    "source": "graph",
                    "reason": "scaffolding",
                    "message": (
                        "GraphBackend is scaffolding only — _search_graph and "
                        "_load_graph are not implemented. Worker checklist is "
                        "in the module docstring."
                    ),
                    "graph_dir": str(self._graph_dir),
                    "graph_dir_exists": self._graph_dir.exists(),
                },
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )

        try:
            graph = self._ensure_graph_loaded()
        except FileNotFoundError as exc:
            return RetrieveResponse(
                blocks=[],
                diagnostics={
                    "source": "graph",
                    "reason": "no_graph",
                    "graph_dir": str(self._graph_dir),
                    "error": str(exc),
                },
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )

        blocks, diag = self._search_graph(query, graph)
        # Filter by min_score, keep top_k.
        blocks = [b for b in blocks if b.score >= self._min_score]
        blocks.sort(key=lambda b: b.score, reverse=True)
        blocks = blocks[: request.top_k]

        return RetrieveResponse(
            blocks=blocks,
            diagnostics={
                "source": "graph",
                "graph_name": graph.name,
                "candidate_count": len(blocks),
                **(diag or {}),
            },
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    def index(self, document: Any) -> None:
        """Indexing is owned by the offline graph builder, not this backend.

        Kept as a no-op (matches FaissBackend / PageIndexBackend) so the
        BackendProtocol surface is satisfied. The worker implementing the
        real backend may keep it as a no-op or wire it to a live builder
        if online indexing becomes a requirement.
        """
        return None

    # ------------------------------------------------------------------ #
    # Internals — these are what the Graph worker fills in.
    # ------------------------------------------------------------------ #

    def _ensure_graph_loaded(self) -> _LoadedGraph:  # pragma: no cover
        """Load (or return cached) graph. Worker fills in.

        Should raise FileNotFoundError with a clear message if the graph
        artifact is missing — ``retrieve()`` catches that to produce a
        structured "no_graph" response.
        """
        if self._graph is not None:
            return self._graph
        if not self._graph_dir.exists():
            raise FileNotFoundError(
                f"graph directory {self._graph_dir} does not exist; build it "
                f"with scripts/build_graph.py (TBD)"
            )
        # TODO(graph-worker): read the graph file(s) and populate _LoadedGraph.
        # Minimum viable shape:
        #   self._graph = _LoadedGraph(
        #       name="cifar10_corpus",
        #       nodes={node_id: {"summary": str, "source": str, ...}, ...},
        #       edges=[(src, dst, weight, kind), ...],
        #   )
        raise NotImplementedError(
            "GraphBackend._ensure_graph_loaded: not implemented. "
            "See module docstring §'Worker checklist'."
        )

    def _search_graph(
        self, query: str, graph: _LoadedGraph,
    ) -> Tuple[List[RetrievedBlock], dict]:  # pragma: no cover
        """Rank nodes for *query* over *graph*. Worker fills in.

        Returns ``(blocks, diagnostics)``. ``blocks`` should be a list of
        :class:`RetrievedBlock` with ``kind="graph_node"``,
        ``document_id`` set to the node id, ``title`` to a short label,
        ``score`` to the rank score, and ``content`` to the node summary
        or surrounding context.
        """
        raise NotImplementedError(
            "GraphBackend._search_graph: not implemented. "
            "See module docstring §'Worker checklist' (item 3)."
        )
