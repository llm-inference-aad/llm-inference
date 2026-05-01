"""In-process RAG augmentation service for the replay harness.

Wraps the production RAG augment surface behind a single ``augment_via_rag(...)``
function. Shape is deliberately HTTP-ready: the request payload + response
dataclass mirror what a ``POST /augment`` endpoint would expose, so swapping
the call site to ``requests.post(...)`` is a 5-line change.

Backend selection
-----------------

The ``RAG_BACKEND`` env var picks which retrieval backend services the call:

- ``"faiss"`` (default) — the legacy ``RagRuntime.enhance_template`` path
  (CodeBERT + MiniLM + FAISS).
- ``"pageindex"`` — the LLM-driven tree-search backend
  (``src.rag.backends.pageindex_backend.PageIndexBackend``) wrapped in a
  ``RagService``. Trees must be pre-built under
  ``${RAG_DATA_DIR}/pageindex_trees/`` via
  ``scripts/build_pageindex_trees.py``.
- ``"graph"`` — still a stub; fails fast at warmup with a pointed message.

The runtime is constructed lazily on first call and cached as a module-level
singleton: building either live backend is expensive (FAISS loads ~1 GB of
indices + embedding models, ~30 s; PageIndex loads tree JSONs cheaply but
each retrieve fires an LLM tree-search, ~10–80 s).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
# src/ on the path gives us `from rag.runtime import RagRuntime` etc.; the
# repo root is needed too because the vendored PageIndex source imports
# itself as `src.pageindex.pageindex.utils` (no `src.` shortcut alias).
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Backend registry — names that may be passed via `RAG_BACKEND`.
# `faiss` (default) routes through `RagRuntime`. `pageindex` is wrapped in a
# `RagService(backend=PageIndexBackend())`. `graph` is still a stub on this
# branch and fails fast so the team running it knows their blocker is the
# backend port, not the harness.
_KNOWN_BACKENDS = {"faiss", "pageindex", "graph"}
_STUB_BACKENDS = {"graph"}


@dataclass
class AugmentRequest:
    template: str
    mutation_type: str | None = None
    query_code: str | None = None
    gene_id: str | None = None


@dataclass
class AugmentResponse:
    augmented_template: str
    retrieved_n_code: int
    retrieved_n_text: int
    rag_block_chars: int


_runtime = None
_runtime_backend: str | None = None


def selected_backend() -> str:
    """Return the RAG_BACKEND env value, defaulting to `faiss`."""
    return (os.environ.get("RAG_BACKEND") or "faiss").strip().lower()


def _build_pageindex_runtime():
    """Build a ``RagService`` whose backend is ``PageIndexBackend``.

    Reranker is disabled — PageIndex blocks score on a 1-5 relevance scale, not
    cosine similarity, and the cross-encoder reranker is calibrated for FAISS-
    style scores. Trees are loaded lazily on the first ``retrieve`` call.
    """
    from rag.backends.pageindex_backend import PageIndexBackend  # type: ignore
    from rag.service import RagService  # type: ignore

    return RagService(backend=PageIndexBackend(), reranker=None)


def _get_runtime():
    """Construct and cache the backend-specific runtime.

    Bypasses ``rag.runtime.get_runtime()``'s ``RAG_ENABLED`` gate: the replay
    always needs a runtime regardless of the env toggle (the toggle is consumed
    by production paths, not by us).

    ``RAG_BACKEND`` is validated here:
      - ``faiss`` (default) → ``RagRuntime`` (CodeBERT + MiniLM + FAISS).
      - ``pageindex`` → ``RagService(backend=PageIndexBackend())``.
      - ``graph`` → still a stub; raise ``SystemExit`` with a pointed message.
      - anything else → unknown backend.
    """
    global _runtime, _runtime_backend
    if _runtime is not None:
        return _runtime

    backend = selected_backend()
    if backend not in _KNOWN_BACKENDS:
        raise SystemExit(
            f"RAG_BACKEND={backend!r} is not in {sorted(_KNOWN_BACKENDS)}. "
            f"Set RAG_BACKEND=faiss, pageindex, or graph."
        )
    if backend in _STUB_BACKENDS:
        raise SystemExit(
            f"RAG_BACKEND={backend!r} is a stub on feature/rag-pipeline-surya "
            f"(src/rag/backends/{backend}_backend.py raises NotImplementedError). "
            f"Port the backend (and wire it into RagRuntime) before running the replay."
        )

    if backend == "faiss":
        from rag.runtime import RagRuntime  # type: ignore
        _runtime = RagRuntime()
    elif backend == "pageindex":
        _runtime = _build_pageindex_runtime()
    else:  # pragma: no cover — guarded above
        raise SystemExit(f"Unhandled RAG_BACKEND={backend!r}")

    _runtime_backend = backend
    return _runtime


def _augment_via_faiss(runtime, req: AugmentRequest) -> AugmentResponse:
    augmented, mutations = runtime.enhance_template(
        template=req.template,
        mutation_type=req.mutation_type,
        query_code=req.query_code,
        gene_id=req.gene_id,
    )
    text_n = sum(1 for m in mutations if getattr(m, "kind", None) == "text")
    code_n = len(mutations) - text_n
    return AugmentResponse(
        augmented_template=augmented,
        retrieved_n_code=code_n,
        retrieved_n_text=text_n,
        rag_block_chars=max(0, len(augmented) - len(req.template)),
    )


def _augment_via_pageindex(service, req: AugmentRequest) -> AugmentResponse:
    """Run the PageIndex backend through ``RagService.augment``.

    PageIndex doesn't differentiate code vs text namespaces — it returns
    document tree nodes regardless. We classify the returned blocks by
    ``kind`` so the journal still shows separate code/text counters: any block
    of ``kind == "pageindex_node"`` is treated as a text-source contribution
    (the trees in ``rag_corpus/`` are PDFs of papers, not code snippets).
    """
    from rag.api_types import AugmentRequest as ApiAugmentRequest  # type: ignore

    api_req = ApiAugmentRequest(
        template=req.template,
        mutation_type=req.mutation_type or "",
        query_code=req.query_code or "",
        gene_id=req.gene_id,
    )
    resp = service.augment(api_req)
    text_n = sum(1 for b in resp.blocks_used if b.kind == "pageindex_node")
    code_n = len(resp.blocks_used) - text_n
    return AugmentResponse(
        augmented_template=resp.augmented_prompt,
        retrieved_n_code=code_n,
        retrieved_n_text=text_n,
        rag_block_chars=max(0, len(resp.augmented_prompt) - len(req.template)),
    )


def augment_via_rag(req: AugmentRequest) -> AugmentResponse:
    runtime = _get_runtime()
    if _runtime_backend == "pageindex":
        return _augment_via_pageindex(runtime, req)
    return _augment_via_faiss(runtime, req)


def warmup() -> None:
    _get_runtime()


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Smoke-test the augment function.")
    ap.add_argument("--template", type=Path, required=True, help="Path to a prompt file.")
    ap.add_argument("--mutation-type", default=None)
    ap.add_argument("--query-code-file", type=Path, default=None)
    args = ap.parse_args()

    template = args.template.read_text()
    query_code = args.query_code_file.read_text() if args.query_code_file else None
    print(f"[smoke] warming up runtime...", flush=True)
    warmup()
    print(f"[smoke] augmenting...", flush=True)
    resp = augment_via_rag(AugmentRequest(
        template=template, mutation_type=args.mutation_type,
        query_code=query_code, gene_id="smoke",
    ))
    print(json.dumps({
        "raw_chars": len(template),
        "aug_chars": len(resp.augmented_template),
        "retrieved_n_code": resp.retrieved_n_code,
        "retrieved_n_text": resp.retrieved_n_text,
        "rag_block_chars": resp.rag_block_chars,
    }, indent=2))
