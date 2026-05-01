# PageIndex Backend Implementation Plan

> Branch: `feature/rag-pipeline-surya-pi`
> Started: 2026-04-30
> Goal: Replace the `PageIndexBackend` stub with a working implementation that
> satisfies the post-componentization `BackendProtocol`, vendoring PageIndex
> source from VectifyAI/PageIndex (with all LLM calls routed to the local
> FastAPI server, not OpenAI).

## 1. Context

After PR-merging the `feature/rag-pipeline-surya` componentization, the RAG
backend abstraction is:

- `src/rag/backend_protocol.py` — `BackendProtocol` (PEP-544, structural):
  `retrieve(RetrieveRequest) -> RetrieveResponse` and `index(Any) -> None`.
- `src/rag/api_types.py` — frozen request/response dataclasses.
- `src/rag/service.py` — orchestrates retrieve → optional rerank → format.
- `src/rag/backends/{faiss,memory,pageindex,graph}_backend.py`. PageIndex and
  Graph are protocol-satisfying stubs that raise `NotImplementedError`.

A working PageIndex existed on `feature/rag-pi` (pre-componentization). It
contained:

- `src/pageindex/` — VectifyAI source, **patched** so `ChatGPT_API*` calls
  POST to the local FastAPI `/generate` endpoint (`_get_server_url()` and
  `_post_to_server()` helpers added in `utils.py`, mirroring
  `submit_local_server` in `src/llm_utils.py`).
- `src/rag/pageindex_retriever.py` — old `PageIndexRetriever` adapter.
  Pre-componentization. Returns `(List[PageIndexResult], dict)`.
- `scripts/build_pageindex_trees.py` — offline tree builder.
- `tests/pageindex/test_section_splitting.py` — unit test.

These are the upstream VectifyAI sources with the minimum changes required
to route to the local LLM. They satisfy the user's "VectifyAI source + local
LLM" constraint and do not need to be re-derived from upstream.

## 2. What this PR delivers

1. Vendor `src/pageindex/` (VectifyAI source + local-LLM patches) onto the
   current branch.
2. Port `scripts/build_pageindex_trees.py` for offline tree building.
3. **Replace** the `PageIndexBackend` stub with an implementation that
   satisfies `BackendProtocol`:
   - `retrieve(RetrieveRequest)` — loads pre-built trees, runs the tree
     search prompt against each tree, returns ordered `RetrievedBlock`s with
     `diagnostics["source"] = "pageindex"`.
   - `index(Any)` — no-op shim (matches `FaissBackend.index()` policy:
     production indexing is owned by the offline build script).
4. Update protocol-compliance test (`test_backend_protocol_compliance.py`)
   to reflect the new status.
5. Add a focused unit test that exercises `PageIndexBackend.retrieve()`
   against a fake LLM and a synthetic in-memory tree (no real model calls).
6. Sanity verification artefacts (`docs/features/rag_pageindex_backend.md`):
   tree counts/depth on the real corpus and a sample retrieval.

## 3. Non-goals

- Not adding PageIndex to any A/B harness, ablation matrix, or default
  composition. Selection remains "inject at construction time."
- Not modifying `RagService` or `RagClient`.
- Not modifying the FAISS or Memory backends.
- Not touching the corpus contents.
- Not committing unless the user asks.

## 4. Step-by-step

| # | Step | Output / verification |
|---|------|----------------------|
| 1 | Read prior code on `feature/rag-pi` (done). | — |
| 2 | Write this plan + checkpoint MD. | This file. |
| 3 | Vendor `src/pageindex/` from `feature/rag-pi` (full subtree). | `src/pageindex/pageindex/{__init__,page_index,page_index_md,utils,config.yaml}`. |
| 4 | Port `scripts/build_pageindex_trees.py`. | `scripts/build_pageindex_trees.py` exists, `--help` runs. |
| 5 | Implement `PageIndexBackend`. | New `src/rag/backends/pageindex_backend.py`. |
| 6 | Update `tests/rag/test_backend_protocol_compliance.py` so PageIndex is no longer treated as a stub. | `pytest tests/rag/test_backend_protocol_compliance.py` green. |
| 7 | Add unit test `tests/rag/test_pageindex_backend.py` (fake LLM, synthetic tree). | Green. |
| 8 | Build trees on `rag_corpus/*.pdf` against the local LLM server. | `${RAG_DATA_DIR}/pageindex_trees/*_structure.json`. |
| 9 | Sanity-check tree structure: node counts, depths, content lengths. | Numbers documented in checkpoint MD. |
| 10 | Sanity-check retrieval: run `PageIndexBackend.retrieve()` on one query, inspect the returned blocks. | Sample logged in checkpoint MD. |
| 11 | Final state checkpoint. | `docs/features/rag_pageindex_backend.md`. |

## 5. Open questions / risks

- **Local LLM server availability** for steps 8–10. If not running, tree
  build will fail at `_post_to_server`. Plan: check `HOSTNAME_LOG_FILE`
  before building; if absent, document this and run only the synthetic
  sanity tests (steps 6–7). The user can run the real build later by
  invoking `scripts/build_pageindex_trees.py` themselves.
- **Tree dedup fix** from the previous memory note (page-level sibling
  dedup `add_node_text_deduped`) was applied to `feature/rag-pi`'s
  `utils.py`. Vendoring from that branch carries the fix forward, so we
  don't need to re-derive it.

## 6. Files added / modified

**Added** (vendored from `feature/rag-pi` or new):
- `src/pageindex/` (vendored, full subtree minus tutorials/cookbook noise)
- `scripts/build_pageindex_trees.py`
- `tests/rag/test_pageindex_backend.py`
- `docs/features/rag_pageindex_backend.md` (final checkpoint)
- `docs/plans/06_pageindex_backend_plan.md` (this file)

**Modified**:
- `src/rag/backends/pageindex_backend.py` — stub → real implementation.
- `tests/rag/test_backend_protocol_compliance.py` — drop the "raises
  NotImplementedError" expectation for PageIndex.
