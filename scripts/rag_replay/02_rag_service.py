"""In-process RAG augmentation service for the replay harness.

Wraps `src.rag.runtime.RagRuntime.enhance_template` behind a single
`augment_via_rag(...)` function. Shape is deliberately HTTP-ready: the request
payload + response dataclass mirror what a `POST /augment` endpoint would
expose, so swapping the call site to `requests.post(...)` is a 5-line change.

The runtime is constructed lazily on first call and cached as a module-level
singleton — this is intentional because building it loads CodeBERT (768-dim) +
MiniLM (384-dim) + the FAISS indices into memory, which takes ~30s.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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


def _get_runtime():
    """Construct and cache `RagRuntime` directly, bypassing the env-gated singleton.

    `rag.runtime.get_runtime()` returns `None` when `RAG_ENABLED=false`, but we
    always need the runtime here regardless of the env toggle (the toggle is
    consumed by production, not by us). We instantiate the class directly.
    """
    global _runtime
    if _runtime is None:
        from rag.runtime import RagRuntime  # type: ignore
        _runtime = RagRuntime()
    return _runtime


def augment_via_rag(req: AugmentRequest) -> AugmentResponse:
    runtime = _get_runtime()
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
