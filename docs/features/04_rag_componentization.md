# RAG Componentization (PRs 0-8)

> Last updated: 2026-04-29
> Status: Merged into `feature/rag-pipeline-surya` at `e12224bdd`
> Plan source: [`docs/plans/05_rag_componentization_plan.md`](../plans/05_rag_componentization_plan.md)

## What this delivers

Refactors `src/rag/` from a monolith embedded in `run_improved.py` into a
separately callable component with a stable request/response surface, a
pluggable backend protocol, two new backend stubs (PageIndex, Graph), a
working episodic memory backend, and the bookkeeping/observability/Pareto
infrastructure needed to A/B-test backends end-to-end.

The work was split into nine parallel PRs (0-8) that were developed on
isolated worker branches, code-reviewed, then merged into a single
`integration/rag-componentization` branch and finally folded into
`feature/rag-pipeline-surya`. PR 8 was reimplemented on top of the merged
componentization (rather than its original off-branch version) to avoid a
six-way structural conflict with PRs 3 and 6.

## Architecture before / after

### Before

```
run_improved.py
  └── _apply_rag_context()
       └── RagRuntime (singleton)
            ├── PromptEnhancer
            ├── RetrievalService
            ├── VectorStoreManager (FAISS)
            └── EmbeddingService
```

`RagRuntime` was a coupled singleton: retrieval, formatting, ranking, and
observability all lived inside the same object, and `run_improved.py` reached
through it to mutate state.

### After

```
run_improved.py
  └── RagClient            (src/rag/client.py)
       └── RagService      (src/rag/service.py)
            ├── BackendProtocol  (src/rag/backend_protocol.py)
            │    ├── FaissBackend       (src/rag/backends/faiss_backend.py)
            │    ├── PageIndexBackend   (src/rag/backends/pageindex_backend.py — stub)
            │    ├── GraphBackend       (src/rag/backends/graph_backend.py     — stub)
            │    └── MemoryBackend      (src/rag/backends/memory_backend.py)
            ├── Reranker         (optional, env-gated)
            ├── PromptEnhancerConfig
            └── RunLedger        (src/rag/bookkeeping.py — optional)
```

Retrieval is now expressed as a single `BackendProtocol` (PEP 544 structural)
that any backend can satisfy. `RagService` orchestrates retrieve → optional
rerank → format. `RagClient` is the public façade (HTTP transport later
drops in here without changing call sites).

`MutationEvent` records flow through `RunLedger` with two events per gene
(augment-time + eval-time), JOINable by `request_id`.

## PR-by-PR contribution map

The merge-commit subjects on `feature/rag-pipeline-surya` follow the form
`merge: PR N — <title>` and resolve to:

### PR 0 — Out-of-band teardown + pytest infrastructure
- **Branch:** `worker/pr0-teardown-testinfra` → merge `d9537e4c4`
- **Why first:** all subsequent PRs need a working test loop and a `run.sh`
  that does not leak SLURM server jobs on cancellation.
- Re-architects `run.sh` so `cleanup_server` runs as a TRAP under
  `EXIT/INT/TERM` (commits `90bc6de64`, `4d93d8ecb`, `e409c89a7`,
  `ea019534a`). The pre-PR-0 version blocked in `uv run python …` and the
  trap fired only after Python returned, which meant `scancel` left the
  nested vLLM server orphaned. PR 0 backgrounds Python and `wait`s,
  forwarding `SIGTERM`/`SIGINT` to the child so cleanup is deterministic.
- Persists `SERVER_JOB_ID` to `${HOSTNAME_LOG_FILE%.log}_server_job.txt` so
  the trap can scancel it even if the main shell env is lost.
- Adds `tests/conftest.py` with `reset_rag_runtime` (autouse) and `fake_env`
  fixtures, and `tests/test_run_teardown.sh` for the end-to-end teardown
  contract.
- Drops `vllm` / `bitsandbytes` deps from `pyproject.toml` (commit
  `3dfa6acf5`) so the unit-test env doesn't need GPU wheels.

### PR 1 — API types + backend protocol
- **Branch:** `worker/pr1-api-types` → merge `6fba214fa`
- Frozen `@dataclass` types in [`src/rag/api_types.py`](../../src/rag/api_types.py):
  `RetrieveRequest`, `RetrievedBlock`, `RetrieveResponse`, `AugmentRequest`,
  `AugmentResponse`, plus `from_dict` JSON-roundtrip helpers.
- [`src/rag/backend_protocol.py`](../../src/rag/backend_protocol.py): a
  `runtime_checkable Protocol` with two methods (`retrieve`, `index`). Zero
  heavy dependencies — `tests/rag/test_api_types.py::TestNoHeavyImports`
  asserts that importing this module does not pull in torch / faiss /
  sentence_transformers.
- 41 unit tests in `tests/rag/test_api_types.py`.

### PR 2 — FAISS backend adapter
- **Branch:** `worker/pr2-faiss-backend` → merge `9beb1daa8`
- [`src/rag/backends/faiss_backend.py`](../../src/rag/backends/faiss_backend.py):
  wraps the existing `EmbeddingService` + `VectorStoreManager` +
  `RetrievalService` behind `BackendProtocol`. No new retrieval logic —
  every search delegates to the existing `RetrievalService`.
- No singleton coupling: holds injected service refs, never touches
  `runtime._runtime_instance`.
- 23 unit tests in `tests/rag/test_faiss_backend.py`.

### PR 3 — RagClient + RagService seam
- **Branch:** `worker/pr3-rag-client-service` → merge `bc990a7ee`
- The stable boundary between LLMGE and RAG.
- [`src/rag/service.py`](../../src/rag/service.py) `RagService`: full
  retrieve → rerank → format pipeline with all collaborators
  (`backend`, `reranker`, `config`, `ledger`) injected at construction.
  No singletons.
- [`src/rag/client.py`](../../src/rag/client.py) `RagClient`: thin façade
  that forwards `augment` / `retrieve` to `RagService` and ensures every
  request carries a `request_id` for cross-event JOIN.
- 14 unit tests in `tests/rag/test_service.py` + 12 in `test_client.py`.

### PR 4 — Bookkeeping ledger
- **Branch:** `worker/pr4-bookkeeping` → merge `d7752d578` (includes review
  auto-fixes `dceeb47d1`, `247fd1a5b`)
- [`src/rag/bookkeeping.py`](../../src/rag/bookkeeping.py): jsonl-based
  `RunLedger` writing one `MutationEvent` row per call, two events per
  gene (augment-time fields populated by `RagService.augment`, eval-time
  fields populated by `run_improved._log_mutation_result`).
- Both events share the same `request_id` so a downstream replayer can
  JOIN them via `replay_ledger`.
- Review fixes: `dceeb47d1` persists the augment-time `request_id` in a
  module-level dict (rather than `GLOBAL_DATA`, which doesn't exist yet at
  augment time); `247fd1a5b` pre-assigns the `request_id` on the request
  before the call so the caller sees the same value the ledger records.
- 18 unit tests in `tests/rag/test_bookkeeping.py`.

### PR 5 — Pareto-aware mutation logging policy
- **Branch:** `worker/pr5-pareto-policy` → merge `c6da51abf`
- [`src/rag/pareto_policy.py`](../../src/rag/pareto_policy.py):
  per-generation percentile windows decide which events get
  `is_pareto_eligible=True`. Replaces the prior global `RAG_MIN_ACCURACY` /
  `RAG_MAX_PARAMETERS` cutoff that systematically excluded high-accuracy +
  high-param mutations from the logged corpus.
- Adds `_generation_population: dict[int, list[MutationEvent]]` cache in
  `run_improved.py`. Every event is always appended to the ledger
  regardless of the flag — only `is_pareto_eligible` changes.
- 19 unit tests in `tests/rag/test_pareto_policy.py`.

### PR 6 — Ablation matrix runner + Pareto comparison report
- **Branch:** `worker/pr6-ablation-matrix` → merge `2a4adcdbe`
- [`scripts/run_rag_ablation_matrix.py`](../../scripts/run_rag_ablation_matrix.py):
  sbatch-based driver that fans 3 conditions × N seeds out as 3·N jobs
  (`baseline_norag`, `rag_faiss`, `rag_faiss+reranker`).
- [`scripts/plot_pareto_front_comparison.py`](../../scripts/plot_pareto_front_comparison.py):
  reads the resulting `runs/<RUN_ID>/results/*.txt` and emits a paired
  Pareto-front overlay plot.
- 22 unit tests in `tests/test_ablation_matrix.py` and 14 in
  `tests/test_plot_pareto_comparison.py`.

### PR 7 — PageIndex + Graph backend stubs
- **Branch:** `worker/pr7-backend-stubs` → merge `8c25d97c6`
- Two `BackendProtocol`-compliant stubs that raise `NotImplementedError`
  (not `AttributeError`) when called:
  [`src/rag/backends/pageindex_backend.py`](../../src/rag/backends/pageindex_backend.py),
  [`src/rag/backends/graph_backend.py`](../../src/rag/backends/graph_backend.py).
- Lets the ablation matrix driver and integration tests reference the
  protocol surface without owning a real implementation yet.
- Also lands the lazy `__getattr__` in
  [`src/rag/__init__.py`](../../src/rag/__init__.py) (commit `b6a001312`)
  so submodule imports defer torch/faiss until actually needed.
- 17 parametrized tests in `tests/rag/test_backend_protocol_compliance.py`.

### PR 8 — MemoryBackend (re-derived on top of PR 3)
- **Original branch:** `worker/pr8-memory-backend` was developed off
  `feature/rag-pipeline-surya` *before* PR 3 merged, so its FAISS service
  layer collided with PR 3's `RagService` and PR 6's ablation script. The
  six-way structural conflict was resolved by **reimplementing** PR 8 as
  commit `b0c336037` on top of integration HEAD.
- What ships:
  - [`src/rag/backends/memory_backend.py`](../../src/rag/backends/memory_backend.py):
    `BackendProtocol`-compliant adapter that stores one-line
    natural-language summaries of past mutations in a separate `"memory"`
    FAISS namespace (384-dim MiniLM-L6, distinct from the 768-dim
    CodeBERT code namespace).
  - `vector_db.py` gains `MEMORY_NAMESPACE` + `add_memory_documents` +
    `search_memory`.
  - `cfg/constants.py` gains `RAG_MEMORY_STORE_ENABLED` (default false),
    `RAG_MEMORY_TOP_K=3`, `RAG_MEMORY_MIN_SIMILARITY=0.5`.
  - `RagService` accepts an optional `memory_backend` parameter and merges
    memory blocks into the augmented prompt with a dedup-by-`gene_id` rule
    (the richer code block always wins over a one-line memory bullet).
  - `run_improved.py` adds:
    - `_build_memory_backend()` — per-run factory rooted at
      `runs/<RUN_ID>/rag_memory/` (per-run scope, deterministic empty
      start per experiment, no cross-run contamination).
    - `_enqueue_memory_summary()` called from `_log_mutation_result` for
      **both successes and failures** (failures tagged `success=false`
      become negative examples).
    - `_flush_memory_buffer()` called immediately after `save_checkpoint`
      at end of each generation, mirroring the checkpoint write rhythm.
- Bug-fix vs. the original prototype: the "weak-query" bug (embedding
  only the first 5 lines of code, which is almost always boilerplate
  imports) is fixed — the full `query_code` is used.
- Stricter `min_similarity=0.5` default (vs. the original 0.3) cuts
  noise from low-relevance summaries.
- 13 unit tests in `tests/rag/test_memory_backend.py`.
- **Deferred from original PR 8** (re-target as a follow-up after this
  merge lands): `rag-faiss+memory` ablation condition, the 9-job
  `--compare-3` flag in the ablation runner, and matching test assertions.
  These collide with PR 6's already shipped ablation surface.

## Other commits worth flagging in review

These commits land on `feature/rag-pipeline-surya` but are not part of any
numbered PR:

- `645dc433c rag: add replay harness for paired RAG evaluation on historical
  genes` — `scripts/rag_replay/{01_aggregate,02_rag_service,03_replay,04_compare}.py`,
  the harness that re-runs 156 historical genes through both the no-RAG and
  with-RAG arms for the paired analysis described in the stand-up plan
  ([`~/.claude/plans/docs-rag-isolation-00-summary-md-ignore-linked-neumann.md`](https://example.invalid/)).
- `d40302d9f fix(rag-replay): absolute RUN_DIR, 8h driver budget, CUDA-busy
  retry` — operational fixes for the replay harness.
- `04805e342 docs(rag): add eval dataset spec + componentization plan for
  backend A/B testing` — adds the full `docs/rag_replay/` documentation set
  alongside this file:
  - [`docs/rag_replay/00_overview.md`](../rag_replay/00_overview.md) —
    motivation, deliverables, success criteria for the paired evaluation.
  - [`docs/rag_replay/01_aggregate.md`](../rag_replay/01_aggregate.md) —
    `01_aggregate.py` design: how 156 source genes are pulled from the
    historical RAG-OFF runs into `past_genes.csv` and prompt files.
  - [`docs/rag_replay/02_replay_loop.md`](../rag_replay/02_replay_loop.md) —
    `03_replay.py` design: per-arm sbatch submission, journal-based resume,
    and the in-process `RagRuntime.enhance_template` call that produces the
    with-RAG arm.
  - [`docs/rag_replay/03_metrics.md`](../rag_replay/03_metrics.md) —
    `04_compare.py` design: paired CSV joining no-RAG / with-RAG arms with
    syntax-validity, fallback, accuracy, params, train-time, and prompt
    sizes; goodput / accuracy / cost delta calculations and the stratified
    fallback sub-cohort.
  - [`docs/rag_replay/05_eval_dataset_spec.md`](../rag_replay/05_eval_dataset_spec.md) —
    schema for `past_genes.csv` and the prompt files that feed the harness.
- `c7c04b665 fix(tests): produce non-NaN deterministic vectors in
  FakeEmbeddingService` — fix for an upstream NaN edge-case from the all-zero
  SHA256 codepath.
- `20f272202 test(rag): prefer src.rag.data_ingestion to dodge stub
  pollution` — orders `from src.rag.data_ingestion ...` ahead of
  `from rag.data_ingestion ...` so `TestFitnessInSummary` doesn't pick up
  the lambda stub installed by `test_apply_rag_context_integration.py`.

## Test status

`uv run pytest tests/rag/` from the repo root: **151 / 154 pass**.

The 3 failures are pre-existing test-pollution flakes, not regressions
caused by this branch:

| Test                                                             | When isolated | Why it "fails" in suite           |
| ---------------------------------------------------------------- | ------------- | --------------------------------- |
| `TestNoHeavyImports::test_torch_not_imported`                    | pass (41/41)  | earlier suite tests import torch  |
| `TestNoHeavyImports::test_faiss_not_imported`                    | pass (41/41)  | earlier suite tests import faiss  |
| `TestNoHeavyImports::test_sentence_transformers_not_imported`    | pass (41/41)  | earlier suite tests import st     |

The protocol guarantee (api_types itself does not pull heavy deps) is real;
the in-suite assertion is only meaningful with `pytest --forked` or
`pytest -p no:cacheprovider` — both are out of scope for this PR.

## Operational changes worth highlighting in review

- **Memory backend is opt-in.** Set `RAG_MEMORY_STORE_ENABLED=true` to turn
  it on; default behaviour is unchanged.
- **Per-run scope for memory.** `runs/<RUN_ID>/rag_memory/` — each
  experiment starts with an empty memory namespace. Deliberate choice (per
  user decision in conversation) so the experiment is reproducible
  without cross-run contamination. Cumulative-across-runs is a v2 ask.
- **Ledger writes are best-effort.** `RagService._emit_ledger_event`
  swallows exceptions and logs a warning; a ledger write failure can
  never tank an evolution run.
- **Memory writes are batched + best-effort.** `_flush_memory_buffer`
  swallows per-record errors and clears the buffer either way, so a
  transient embed failure does not pile up across generations.
- **`run.sh` SIGTERM behaviour changed (PR 0).** Cleanup now runs from a
  TRAP, not a post-Python block. Anyone scancel-ing a job will see the
  server job get scancelled within ~2 seconds rather than after Python's
  process tree finally exits.

## Migration guide for downstream callers

Before:

```python
from rag.runtime import get_runtime
runtime = get_runtime()
augmented = runtime.enhance_template(template, mutation_type, query_code)
```

After:

```python
from rag.client import RagClient
from rag.api_types import AugmentRequest

client = RagClient()  # uses default RagService with FaissBackend
resp = client.augment(AugmentRequest(
    template=template,
    mutation_type=mutation_type,
    query_code=query_code,
    gene_id=gene_id,
    run_id=os.getenv("RUN_ID"),
))
augmented = resp.augmented_prompt
```

The legacy `RagRuntime.enhance_template` path still works (HTTP-shape-
preserving wrapper in `augment_via_rag()` for the replay harness), but new
code should use `RagClient`.

## What's next (not in this PR)

- **PR 8 follow-up:** rebase the original `worker/pr8-memory-backend`
  ablation-matrix changes onto post-merge HEAD as a separate small PR
  (adds `rag_faiss+memory` condition, `--compare-3` flag, 9-job
  assertions).
- **PageIndex / Graph backends:** the stubs exist; real implementations
  are tracked in `docs/plans/05_rag_componentization_plan.md` Phase 2.
- **HTTP transport for `RagClient`:** deferred per the componentization
  plan. The seam is in place — `RagClient.__init__` will gain a `base_url`
  parameter when needed.
