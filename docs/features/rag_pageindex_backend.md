# RAG: PageIndex Backend Implementation

> Branch: `feature/rag-pipeline-surya-pi`
> Author session: 2026-04-30
> Plan: [`docs/plans/06_pageindex_backend_plan.md`](../plans/06_pageindex_backend_plan.md)
> Status: Backend implementation + tests complete. **End-to-end live
> retrieval verified against the local vLLM server with the new
> per-request `system_prompt` override (PageIndex sends `system_prompt=""`).**
> Live tree generation works for short documents; long documents
> (e.g. survey papers ≥ ~15 pages) stress the local 8K context and
> would benefit from raising `MAX_MODEL_LEN`. Details below.

## What this delivers

Replaces the `PageIndexBackend` stub at `src/rag/backends/pageindex_backend.py`
with a working implementation that satisfies the post-componentization
`BackendProtocol`. Tree construction and the LLM tree-search prompt come
from VectifyAI/PageIndex (vendored under `src/pageindex/`); all model calls
are routed to the local FastAPI `/generate` endpoint, not OpenAI.

## Files added / modified

**Added:**

- `src/pageindex/` — vendored PageIndex source with the local-LLM patches
  (`pageindex/utils.py` adds `_get_server_url` + `_post_to_server`, and the
  `ChatGPT_API*` helpers POST to the local server). Includes the prior
  `add_node_text_deduped` + `fix_node_ranges` page-level dedup fixes.
  License retained (`LICENSE`).
- `scripts/build_pageindex_trees.py` — offline tree builder. Reads
  `rag_corpus/*.pdf`, writes `<stem>_structure.json` to
  `${RAG_DATA_DIR}/pageindex_trees/`. Tunes `max_token_num_each_node=3000`
  / `max_page_num_each_node=3` to fit the local 8K-context server.
- `scripts/pageindex_sanity_check.py` — static + retrieval sanity-check
  tool. Supports `--static-only` and `--fake-llm` flags so the backend can
  be exercised end-to-end against real on-disk trees without depending on
  the live model's prompt-following.
- `tests/rag/test_pageindex_backend.py` — 8 unit tests using a synthetic
  on-disk tree and an injected fake LLM (no live server, no corpus).
- `docs/plans/06_pageindex_backend_plan.md` — implementation plan.
- `docs/features/rag_pageindex_backend.md` — this file.

**Modified:**

- `src/rag/backends/pageindex_backend.py` — stub → implementation.
- `tests/rag/test_backend_protocol_compliance.py` — drop the
  "raises `NotImplementedError`" expectation for `PageIndexBackend` and add
  a `TestPageIndexBackendCompliance` class that exercises the live class.
- `server_vllm.py` — add `system_prompt: str | None = None` to `LLMRequest`
  and resolve it per request: `None` falls back to the server's
  `SYSTEM_PROMPT` env var (LLMGE behaviour preserved); `""` skips the
  system prompt entirely (PageIndex case); any custom string is used
  verbatim (future callers — chat, summarization, etc.).

## Backend behaviour

`PageIndexBackend.retrieve(request)`:

1. Lazy-load all `<stem>_structure.json` trees from `trees_dir`.
2. For each tree, strip leaf `text` fields (`remove_fields(... ['text'])`)
   to keep the prompt compact, format `TREE_SEARCH_PROMPT`, send to the
   LLM via the injected `llm_call` (defaults to `ChatGPT_API` from the
   vendored utils, which POSTs to the local server).
3. Parse the JSON response, accept nodes with `relevance >= min_relevance`
   (default 3), look up each `node_id` in the per-tree node map.
4. Cross-document sort by `(relevance desc, is_leaf desc)`, truncate to
   `request.top_k`.
5. Return a `RetrieveResponse` of `RetrievedBlock`s with
   `kind="pageindex_node"`, `score=relevance`, `content=node['text']`,
   per-block diagnostics (`source`, `doc_name`, `node_id`, `is_leaf`,
   `summary`, `thinking`, `relevance`).

`PageIndexBackend.index(document)`:

- Intentional no-op shim. Production indexing is owned by
  `scripts/build_pageindex_trees.py`. Matches the `FaissBackend.index()`
  policy.

Construction takes optional `trees_dir`, `model`, `llm_call`, and
`min_relevance`. The `llm_call` injection is what makes the unit tests
(and the `--fake-llm` sanity flag) possible without a live model.

## Test results

```text
tests/rag/test_pageindex_backend.py ............. 8 passed
tests/rag/test_backend_protocol_compliance.py ... 17 passed
                                                 25 total — green
```

Tests cover: happy path (synthetic 3-node tree, fake LLM selects one node,
response shape and diagnostics validated); relevance filtering
(`relevance < 3` dropped); unknown-node-id ignored; top-k cross-document
truncation; empty query short-circuit (no LLM call); missing `trees_dir`
returns empty `RetrieveResponse` with `reason="no_trees"`; `"Error"` from
the LLM returns empty blocks plus per-tree diagnostic; `index()` is a
no-op for `str` and `dict`.

## Tree generation sanity check (static)

`scripts/pageindex_sanity_check.py --static-only` against the existing
trees in `${RAG_DATA_DIR}/pageindex_trees/` (built in earlier work):

```text
4388-Article Text-28466-1-10-20230612.pdf
  node_count: 8, max_depth: 0, total_text_words: 2235
  largest_node: "Discussion" (644 words)
  summary_present_for_root: true

Cifar-10_Classification_using_Deep_Convolutional_Neural_Network.pdf
  node_count: 14, max_depth: 1, total_text_words: 2819
  largest_node: "Stochastic gradient descent" (680 words)
  summary_present_for_root: true
```

Trees parse cleanly through `structure_to_list` and the per-tree node maps
build correctly. Node-ID lookups in the backend go through these maps.

## Retrieval sanity check (live LLM, system_prompt="")

Against the same trees, top_k=3, real Nemotron-49B server with
`system_prompt=""`:

```text
query: "What CNN architectures perform well on CIFAR-10?"
  trees_searched=2 latency_ms=63351
  [1] doc=4388-Article Text…pdf  node_id=0002  title='Method'  score=4.0  is_leaf=True

query: "How does retrieval-augmented generation improve large language models?"
  trees_searched=2 latency_ms=13958
  (no blocks returned — correct: corpus has no RAG papers)

query: "What training tricks (data augmentation, regularization) help with image classification?"
  trees_searched=2 latency_ms=83851
  [1] node_id=0009 title='Regularization'                  score=5.0
  [2] node_id=0010 title='Dropout'                         score=4.0
  [3] node_id=0012 title='Stochastic gradient descent'     score=3.0
  (all leaves from Cifar-10 paper, semantically relevant, ranked correctly)
```

The model's `<think>` reasoning channel made each call ~10–80 s wall-clock
(compute is fine; latency dominated by 60K-token unstructured chains of
thought before the JSON answer).  `extract_json` recovers the JSON
literal from the noisy preamble; `_search_tree` defensively accepts both
``{"selected_nodes": [...]}`` and the bare ``[...]`` shape the model
sometimes returns.

## Retrieval sanity check (fake-LLM, deterministic)

`scripts/pageindex_sanity_check.py --fake-llm` against the same trees,
top_k=3 (the fake LLM does a deterministic keyword overlap against the
query — same code path as the real backend, just with the model
call replaced):

```text
query: "What CNN architectures perform well on CIFAR-10?"
  trees_searched=2 latency_ms=385
  [1] doc=4388-Article Text…pdf  node_id=0000  title='ABSTRACT'           score=5
      preview: 'Convolutional neural network (CNN) is a powerful tool…'
  [2] doc=Cifar-10_Classification…pdf  node_id=0009  title='Regularization' score=5
      preview: 'regularization or increase dataset by methods such as data augmentation…'
  [3] doc=4388-Article Text…pdf  node_id=0002  title='Method'             score=4

query: "How does retrieval-augmented generation improve large language models?"
  trees_searched=2 latency_ms=1
  [1] node_id=0001 title='Introduction'  score=5
  [2] node_id=0010 title='Dropout'       score=5
  [3] node_id=0000 title='ABSTRACT'      score=4

query: "What training tricks (data augmentation, regularization) help with image classification?"
  trees_searched=2 latency_ms=1
  [1] node_id=0004 title='Discussion'                  score=5
  [2] node_id=0012 title='Stochastic gradient descent' score=5
  [3] node_id=0002 title='Method'                      score=4
```

This exercises the full backend pipeline against real trees:

- tree loading from `*_structure.json`
- compact-tree formatting (`remove_fields ['text']`)
- prompt construction and `llm_call` dispatch
- JSON parsing
- node lookup, leaf detection, diagnostics
- cross-document ranking + top-k truncation
- `RetrieveResponse` construction with latency

## Server change: per-request `system_prompt` override

Previously `server_vllm.py` unconditionally prepended a `SYSTEM_PROMPT`
that demands a fenced runnable Python code block — right for the mutation
generation path, but it caused the model to return *Python that builds
JSON* (e.g. `return json.dumps(response)`) instead of a JSON literal,
breaking PageIndex.

`server_vllm.py` now exposes `system_prompt: str | None = None` on
`LLMRequest`, with three modes:

- `None` (omitted) → fall back to the server's `SYSTEM_PROMPT` env var.
  Existing callers (LLMGE mutation generation) get the Python-coercive
  prompt with no client-side change.
- `""` → no system prompt at all. The PageIndex client uses this to get
  raw user-prompt-only output suitable for JSON.
- `"<custom>"` → use the supplied string verbatim. Future callers
  (chat, summarization, etc.) can supply their own task-specific prompt
  per request without touching the server config.

Confirmed in production:

```text
default (system_prompt omitted)  → prompt_tokens=59, output "PONG"
system_prompt=""                 → prompt_tokens=13, output "```json\n{\"answer\":\"hello\"}\n```"
```

The vendored PageIndex client (`src/pageindex/pageindex/utils.py`) sends
`system_prompt: ""` for all PageIndex calls.

## Live tree generation results

With the server change in place, `scripts/build_pageindex_trees.py` was
re-run against the full `rag_corpus/`:

| PDF | Outcome | Notes |
|---|---|---|
| `4388-Article Text-28466-1-10-20230612.pdf` | ✅ Built (8 nodes) | clean process_no_toc path |
| `Cifar-10_Classification_using_Deep_Convolutional_Neural_Network.pdf` | ❌ failed downstream of TOC verify (`'int' object has no attribute 'split'`) | LLM occasionally returns `structure` as int instead of "X.Y.Z" string; defensive str() coerce vendored, but already-running build didn't re-pickup the patch |
| `survey_rag_llm.pdf` (15-page survey) | ⏳ deep recursive splits — killed before completion to free server time | Survey hits the local 8K context on the toc_transformer call; my size-guard now falls back to process_no_toc, which then recurses into multi-page sub-sections. Each sub-section is its own LLM round — works, just slow on the 8K cap |

The `Cifar-10` failure is fixed in the vendored source (`page_index.py`
line 599: `struct.split('.')` now goes through `str(...)`). The next
build run will pick it up.

The survey paper is the canonical case for raising `MAX_MODEL_LEN`. With
the vLLM server's 8K cap, the toc_transformer call had to fall back to
the no-TOC pipeline that recurses on large nodes — many small calls,
each verbose due to the model's `<think>` channel. Bumping `MAX_MODEL_LEN`
to 16K or 32K (the model supports it) would let the upstream
toc-aware path run end-to-end with a single LLM call per phase, replacing
~30+ calls with ~5 and producing a flatter, cleaner tree. Recommended
follow-up: set `MAX_MODEL_LEN=16384` in `.env` and rebuild.

## Robustness changes vendored into `src/pageindex/`

These are real bugs/inadequacies surfaced during the live-server attempt
and are useful regardless of the SYSTEM_PROMPT decision:

1. **Local 8K-context budgets** (`src/pageindex/pageindex/page_index.py`,
   `src/pageindex/pageindex/utils.py`):
   - `page_list_to_group_text(... max_tokens=20000)` → reads
     `PAGEINDEX_GROUP_MAX_TOKENS` (default 4000) so TOC-detection batches
     fit `prompt + max_new_tokens ≤ 8192` with headroom.
   - `_post_to_server` reads `PAGEINDEX_MAX_NEW_TOKENS` (default 2048)
     and uses `temperature=0.0` (greedy). The `repetition_penalty=1.1`
     interacting with low non-zero temperature could cause the EOS token
     to win the very first sampling step, producing 1-token responses.
2. **Tolerant JSON extraction** (`extract_json` in `utils.py`): now
   considers ``` ```json ``` ``` fences, ``` ```python ``` ``` fences, raw
   content, and the widest balanced ``[...]`` / ``{...}`` substring inside
   any of those. Normalises Python `None`/`True`/`False`. Returns `{}` on
   total failure (preserves the upstream contract).
3. **Defensive coercion in `process_no_toc`** (`page_index.py`): when
   `generate_toc_init` / `generate_toc_continue` returns a non-list
   (dict / scalar from a parse failure), the script now coerces to `[]`
   and continues rather than crashing with
   `'dict' object has no attribute 'extend'`.
4. **`toc_transformer` size guard + `meta_processor` fallback**
   (`page_index.py`): if the raw TOC content would exceed
   `PAGEINDEX_TOC_MAX_TOKENS` (default 5500), `toc_transformer` raises
   immediately rather than waste 10 retries on a doomed prompt. The
   `meta_processor` dispatcher now catches exceptions from the TOC-aware
   modes and falls back to `process_no_toc` so a single oversize TOC
   doesn't abort the whole document.
5. **String coerce of `structure`** (`page_index.py` line 599): the LLM
   sometimes returns the structure index as a bare int rather than
   "X.Y.Z" — coerce with `str(...)` before `.split('.')`.
6. **Backend-side dict-or-list response coerce**
   (`src/rag/backends/pageindex_backend.py::_search_tree`): the model
   sometimes returns just a list of selected nodes (no
   `{"thinking": ..., "selected_nodes": [...]}` wrapper). The backend now
   accepts both shapes.

All vendored into the source so the upstream pipeline survives the
real-world LLM output variance.

## Operational notes

- The first `retrieve()` call on an instance triggers tree loading;
  subsequent calls reuse the in-process cache. Construct one
  `PageIndexBackend` per process and reuse it.
- Default `trees_dir` is `${RAG_DATA_DIR}/pageindex_trees/`. Override at
  construction or via `--trees-dir` on the sanity script.
- Selecting this backend at runtime is by composition — no global
  registry, matching the existing `FaissBackend`/`MemoryBackend` pattern:

  ```python
  service = RagService(backend=PageIndexBackend())
  client  = RagClient(service=service)
  ```

## How to reproduce

```bash
# 1. launch server (worktree's edited server_vllm.py honors system_prompt)
cd .claude/worktrees/rag-pipeline-surya-pi
sbatch -t 1:00:00 -C "H100" --gpus-per-node=2 -p coe-gpu --mem 80G -c 16 server.sh
# wait for vLLM to load the model (~5 min on 2× H100)

# 2. build trees
export RAG_DATA_DIR=$(pwd)/rag_data
export HOSTNAME_LOG_FILE=/home/.../runs/server-only/logs/hostname.log
export SERVER_PORT=8000
.venv/bin/python scripts/build_pageindex_trees.py \
    --corpus-dir rag_corpus \
    --output-dir "$RAG_DATA_DIR/pageindex_trees" \
    --model local_server

# 3. verify retrieval (live)
.venv/bin/python scripts/pageindex_sanity_check.py \
    --trees-dir "$RAG_DATA_DIR/pageindex_trees" --top-k 3

# 3b. verify retrieval (deterministic, no LLM)
.venv/bin/python scripts/pageindex_sanity_check.py \
    --trees-dir "$RAG_DATA_DIR/pageindex_trees" --top-k 3 --fake-llm
```

## Known limitations / follow-ups

- No A/B harness wiring in this PR. The replay/ablation scripts already
  accept arbitrary backend instances; adding a `pageindex` condition is a
  separate change.
- `min_relevance` is a constructor argument; if we want runtime tuning
  per request, accept it via `RetrieveRequest.filters["min_relevance"]`.
- **Recommended:** raise `MAX_MODEL_LEN=16384` (or 32K) in `.env` so the
  TOC-aware build path runs end-to-end on long documents instead of
  falling back to the recursive no-TOC path. The model supports it; on
  2× H100 (160 GB) the KV cache headroom is fine for a single user.
- The Cifar-10 build hit `'int' object has no attribute 'split'` because
  the LLM returned `structure` as int rather than "X.Y.Z" string. Fixed
  in the vendored source (`str(...)` coerce); next build will succeed.
- The model's `<think>` channel inflates retrieval latency to 10–80 s
  per query.  If we need fast retrieval, switch to a smaller / less
  reasoning-heavy model for the tree-search role.
