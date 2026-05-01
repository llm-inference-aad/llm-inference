# RAG Debugging and Validation

> **Last updated:** March 17, 2026
> **Purpose:** Operational guide for validating whether RAG is active, architecture-grounded, and worth keeping in LLMGE prompts.

---

## Goal

The current validation loop is not just "did retrieval return something?" The bar is higher:

- FixedPrompts retrieval must be grounded in the current parent architecture.
- Text retrieval must prefer PyTorch/API context and only use PDFs selectively.
- Injected context must be explainable from run-scoped metrics.
- The final context must look useful to a human reviewer, not like noise.

---

## Core Signals

RAG writes structured events to `${RUN_METRICS_DIR}/rag_metrics.jsonl`.

The events that matter most:

- `rag_runtime_status`
  - Confirms whether RAG initialized and why it may have failed.
- `rag_query_code_loaded`
  - Confirms FixedPrompts successfully loaded parent model code for retrieval.
- `rag_query_code_load_failed`
  - Confirms the query-grounding path failed and RAG fell back to prompt prose.
- `rag_generation_skipped`
  - RAG did not run for this generation.
- `rag_generation_failed`
  - RAG ran but crashed before prompt augmentation completed.
- `rag_context_built`
  - Main retrieval event. Includes query preview, selected doc IDs, source mix, and pre/post-rerank candidates.
- `rag_generation_context`
  - Final selected counts after prompt augmentation.

---

## Smoke Loop

Use short runs to verify retrieval behavior before any medium-scale A/B experiment.

### 1. Query-Grounding Probe

```bash
RAG_USE_CODE_CONTEXT=false \
RAG_USE_TEXT_CONTEXT=true \
RAG_RERANKER_ENABLED=true \
NUM_GENERATIONS=1 \
POPULATION_SIZE=4 \
START_POPULATION_SIZE=4 \
./launch.sh --name rag_text_queryfix --rag --seed 42
```

What to look for:

- `query_code_present=true` in `rag_context_built`
- `text_query_preview` contains architecture terms from the parent model
- selected docs are mainly PyTorch/API chunks

### 2. Source-Policy Probe

```bash
RAG_USE_CODE_CONTEXT=false \
RAG_USE_TEXT_CONTEXT=true \
RAG_RERANKER_ENABLED=true \
NUM_GENERATIONS=1 \
POPULATION_SIZE=4 \
START_POPULATION_SIZE=4 \
./launch.sh --name rag_text_sourcepolicy --rag --seed 42
```

What to look for:

- `Param` mutations select only API docs
- architecture prompts select mostly API docs and at most one PDF
- no OCR/table garbage in the top selected chunks

### 3. Final Small Validation

Text-only:

```bash
RAG_USE_CODE_CONTEXT=false \
RAG_USE_TEXT_CONTEXT=true \
RAG_RERANKER_ENABLED=true \
NUM_GENERATIONS=2 \
POPULATION_SIZE=8 \
START_POPULATION_SIZE=8 \
./launch.sh --name rag_text_finalprobe --rag --seed 42
```

Hybrid:

```bash
RAG_USE_CODE_CONTEXT=true \
RAG_USE_TEXT_CONTEXT=true \
RAG_RERANKER_ENABLED=true \
NUM_GENERATIONS=2 \
POPULATION_SIZE=8 \
START_POPULATION_SIZE=8 \
./launch.sh --name rag_hybrid_finalprobe --rag --seed 42
```

---

## Reindex and Retrieval Eval

Rebuild the text namespace after changing PDF filtering or PyTorch chunking:

```bash
uv run python scripts/setup_rag.py --pytorch-json rag_corpus/pytorch.json --rebuild-text
```

Validate retrieval quality:

```bash
uv run python src/rag/eval_retrieval.py --tier medium --k 5 --limit 200
```

Current reference baseline:

- `Recall@5 = 85.0%`
- `MRR@5 = 0.6996`

If section chunking or corpus cleanup materially regresses this benchmark, treat that as a blocking issue before running more LLMGE probes.

---

## Inspecting a Run

### Analyzer Summary

```bash
./.venv/bin/python3 scripts/analyze_rag_impact.py --run-dir runs/<RUN_ID>
```

Useful fields from `rag_impact_report.json`:

- `rag_context_events`
- `rag_nonempty_fraction`
- `avg_retrieved_text_n`
- `avg_selected_text_api_n`
- `avg_selected_text_pdf_n`
- `param_events_with_pdf_fraction`
- `avg_text_top_pre_rerank_score`
- `avg_text_top_post_rerank_score`
- `rag_observed_issue`

### Candidate-Level Inspection

```bash
./.venv/bin/python3 scripts/inspect_rag_context.py --run-dir runs/<RUN_ID>
```

Filter to a gene:

```bash
./.venv/bin/python3 scripts/inspect_rag_context.py \
  --run-dir runs/<RUN_ID> \
  --gene-id <GENE_ID>
```

This prints:

- `query_code_present`
- `text_query_preview`
- `text_selection_mode`
- selected text doc IDs
- text candidates before rerank
- text candidates after rerank
- code candidates before rerank
- code candidates after rerank

---

## Interpreting Scores

The relevance values shown in prompt context are **post-rerank cross-encoder scores**.

That means:

- a very low displayed score usually means the reranker also thinks the match is weak
- it is not the same as raw FAISS cosine similarity
- candidate inspection is the right way to distinguish "retrieved by vector search" from "survived reranking and source policy"

---

## Worthy Retrieval Criteria

Do not move back to broader LLMGE evaluation until all of these are true:

- FixedPrompts retrieval consistently shows `query_code_present=true`
- top injected docs are architecture-grounded rather than generic prompt matches
- no obvious OCR/table-heavy PDF chunk appears in top-3
- `Param` prompts retrieve optimizer/scheduler APIs instead of random PDFs
- selected context is something a human reviewer would keep in the prompt
- every selected chunk is explainable from `rag_metrics.jsonl` and `scripts/inspect_rag_context.py`

---

## Failure Modes

### RAG disabled by config

Symptoms:

- `rag_runtime_status` shows `disabled`
- no `rag_context_built` events

Fix:

- verify `RAG_ENABLED=true`
- verify `RAG_USE_TEXT_CONTEXT` or `RAG_USE_CODE_CONTEXT` matches the intended probe

### FixedPrompts still using prompt prose

Symptoms:

- `query_code_present=false`
- `rag_query_code_load_failed` events
- `text_query_preview` looks like template prose instead of architecture code

Fix:

- verify the parent model path exists for the selected gene
- inspect `run_improved.py:generate_template()`

### PDFs still dominate

Symptoms:

- `avg_selected_text_pdf_n` is high
- `param_events_with_pdf_fraction > 0`
- selected docs are surveys, OCR fragments, or tables

Fix:

- lower `RAG_TEXT_TOP_K_PDF`
- inspect `RAG_TEXT_CANDIDATE_K`
- rebuild text index after tightening PDF filtering

### Reranker is selecting weak matches

Symptoms:

- low `avg_text_top_post_rerank_score`
- selected results still look irrelevant

Fix:

- inspect candidate previews with `scripts/inspect_rag_context.py`
- validate query quality first
- then revisit source policy and corpus quality
