# RAG Replay Harness — Notebook Update

**Date:** 2026-04-27
**Branch:** `feature/rag-pipeline-surya`
**Window:** Stand-up sprint preceding finals

## Executive summary

For finals we needed numbers showing whether RAG-augmented prompts produce
better mutations than the baseline RAG-OFF prompts that LLMGE has been running
all semester. The existing tooling answers either condition-level questions
(whole runs, via `scripts/run_rag_ablation_matrix.py`) or synthetic per-prompt
questions on hand-curated cases (the paired isolation harness on
`feature/rag-testing`). Neither replays our actual historical genes through
RAG.

This update introduces a **replay harness** at `scripts/rag_replay/` that:

1. Aggregates every gene from past RAG-OFF runs into one CSV.
2. For each row, regenerates the same mutation twice — once without RAG, once
   with RAG injected via `RagRuntime.enhance_template` (called as an
   in-process function with HTTP-ready shape, so it can be promoted to a
   FastAPI service in a 5-line refactor).
3. Submits a SLURM `train.py` job per arm so we get test_acc + param count for
   both regenerated children.
4. Joins the two arms into `paired.csv` and emits paired-statistics report.md.

## What ships

### Code

| File | Lines | Purpose |
|---|---|---|
| `scripts/rag_replay/01_aggregate.py` | 230 | Walk `runs/`, build `past_genes.csv` + `prompts/<gid>.txt`. |
| `scripts/rag_replay/02_rag_service.py` | 100 | `augment_via_rag()` wrapper around `RagRuntime.enhance_template`. |
| `scripts/rag_replay/03_replay.py` | 380 | For-loop driver: regenerate × 2 arms, submit SLURM, journal, poll. |
| `scripts/rag_replay/04_compare.py` | 250 | Paired stats — McNemar (binary) + Wilcoxon (continuous) + report.md. |

### Docs

| File | Purpose |
|---|---|
| `docs/rag_replay/00_overview.md` | Entry doc — diagram, decisions, file map, quickstart. |
| `docs/rag_replay/01_aggregate.md` | Aggregator design + Mermaid + edge cases. |
| `docs/rag_replay/02_replay_loop.md` | Sequence diagram + retry semantics + augment-idx inference. |
| `docs/rag_replay/03_metrics.md` | Authoritative metric definitions. |

## Source data inventory

After running the aggregator (`.venv/bin/python scripts/rag_replay/01_aggregate.py`):

```
156 RAG-OFF genes total
  - by mutation_op: TEMPLATE_BASED=130, CrossOver=9, CREATED=17
  - eligible_for_rag (TEMPLATE_BASED): 130
  - fallback marker present: 39
```

Per-run breakdown (genes / TEMPLATE_BASED / fallback):

| run_id | total | eligible | fallback |
|---|---|---|---|
| `nemotron_rag_text_20260311_022245` | 86 | 74 | 18 |
| `nemotron_baseline_20260311_022419` | 70 | 56 | 21 |

> The `nemotron_rag_text_20260311_022245` run is named aspirationally —
> `run_metadata.json` records `RAG_ENABLED=false` and there is no
> `metrics/rag_metrics.jsonl` written, so it is genuinely RAG-OFF and was
> correctly classified.

## Decisions resolved with the user

| # | Decision |
|---|---|
| Q1 | Source = all genes from runs that ran without RAG. Fallback genes kept and analyzed as a sub-cohort (rescue rate). |
| Q2 | Regenerate **both** arms per source gene — controls LLM stochasticity (~140 SLURM jobs). |
| Q3 | RAG via in-process Python call. Function shape is HTTP-ready so a FastAPI promotion is trivial later. |

## Headline metrics the harness will produce

(Populated after the full run lands.)

```
goodput_no_rag    = mean(norag_syntax_valid_first)
goodput_with_rag  = mean(rag_syntax_valid_first)
delta_goodput     = goodput_with_rag - goodput_no_rag

recovery_rate     = mean(rag_syntax_valid_first | orig_was_fallback)
preservation_rate = mean(rag_syntax_valid_first | not orig_was_fallback)

median_delta_test_acc, IQR, Wilcoxon p
median_delta_prompt_chars  (sanity — should be positive)
median_delta_llm_wall_s    (cost)
median_delta_train_time_s  (incidental)
```

Plus two McNemar exact tests (paired binary): `syntax_valid_first` and
`was_fallback`.

## Run plan

```bash
# 1. Server up
sbatch server.sh

# 2. Smoke (3 genes, 6 sbatch jobs, ~5 epochs each — verifies plumbing)
.venv/bin/python scripts/rag_replay/03_replay.py \
  --output experiments/rag_replay/smoke_$(date +%Y%m%d_%H%M) \
  --max-rows 3 --epochs 5 --eligible-only

.venv/bin/python scripts/rag_replay/04_compare.py experiments/rag_replay/smoke_<...>

# 3. Full replay (130 source genes × 2 arms = 260 SLURM jobs at 24 epochs each)
.venv/bin/python scripts/rag_replay/03_replay.py \
  --output experiments/rag_replay/full_$(date +%Y%m%d_%H%M) \
  --eligible-only --epochs 24 --poll-timeout-hours 12

.venv/bin/python scripts/rag_replay/04_compare.py experiments/rag_replay/full_<...>
```

Expected wall time: dominated by SLURM queue depth. ~31 GPU-hours of training
is parallelized across however many nodes are available.

## Open work (post-stand-up)

- **HTTP promotion.** Wrap `augment_via_rag()` in `@app.post("/augment")` if
  the deck wants a literal microservice in the architecture diagram.
  Touchpoint: `scripts/rag_replay/02_rag_service.py:50`.
- **Per-template stratification.** Current aggregator only stores
  `mutation_op = TEMPLATE_BASED` rather than the specific template. Recovering
  template hints from prompt prefix matching against `templates/FixedPrompts/`
  would let us say *"RAG helped Param more than Significant"* — useful but
  cosmetic for finals.
- **Index leakage check.** The current FAISS index was rebuilt from the same
  baseline runs the replay sources from. RAG could "retrieve the gene we're
  asking it to regenerate". Mitigation: rebuild the index excluding
  `nemotron_baseline_20260311_022419` and compare; or accept and disclose.
