"""In-process RAG augmentation service for the replay harness.

Uses the same `RagClient -> RagService` seam as production. This means replay
respects `RAG_BACKEND` (e.g. `faiss`, `graph`) instead of hardwiring FAISS via
legacy `RagRuntime`.
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
    """Construct and cache `RagClient` for replay augmentation."""
    global _runtime
    if _runtime is None:
        from rag.client import RagClient  # type: ignore
        _runtime = RagClient()
    return _runtime


def augment_via_rag(req: AugmentRequest) -> AugmentResponse:
    client = _get_runtime()
    from rag.api_types import AugmentRequest as ApiAugmentRequest  # type: ignore

    response = client.augment(
        ApiAugmentRequest(
            template=req.template,
            mutation_type=req.mutation_type or "",
            query_code=req.query_code or "",
            gene_id=req.gene_id,
        )
    )
    text_n = sum(1 for b in response.blocks_used if b.kind != "mutation_code")
    code_n = sum(1 for b in response.blocks_used if b.kind == "mutation_code")
    return AugmentResponse(
        augmented_template=response.augmented_prompt,
        retrieved_n_code=code_n,
        retrieved_n_text=text_n,
        rag_block_chars=max(0, len(response.augmented_prompt) - len(req.template)),
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
