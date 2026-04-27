# RAG Isolation Harness — Handoff

**Date:** 2026-04-27
**Owner:** the RAG-isolation harness in `scripts/rag_isolation/`

## What this is

A standalone tool to answer questions of the form *"does RAG change the
behavior of this specific mutation?"* — at the per-mutation level — by
running each (parent, template, augment_idx) case twice with everything
held constant except whether the prompt receives RAG augmentation.

This complements (does **not** replace) the existing
`scripts/run_rag_ablation_matrix.py`, which compares whole evolutionary
runs at the condition level.

## Files shipped

```
docs/rag_isolation/
├── 01_llmge_pipeline.md      # context map: how LLMGE actually executes
├── 02_rag_subsystem.md       # context map: how RAG works
├── 03_design.md              # design of this harness
└── 04_handoff.md             # this file

scripts/rag_isolation/
├── README.md                 # how to run
├── core.py                   # per-trial execution
├── run_paired_eval.py        # CLI entrypoint
├── analyze.py                # paired stats + report
├── run_harness.sbatch        # slurm wrapper
└── datasets/
    ├── smoke.json            # 2 cases × 2 trials (smoke test)
    └── small_validation.json # 5 cases × 3 trials (initial validation)

experiments/rag_isolation/    # outputs land here
└── <run_dir>/
    ├── results.jsonl
    ├── paired.csv
    ├── summary.csv
    ├── tests.csv
    ├── report.md
    ├── run_metadata.json
    └── cases/<case_id>/<trial>/<arm>/
        ├── prompt_template.txt
        ├── prompt_template.raw.txt
        ├── network.py             (or .py.fallback)
        ├── logs/llm/gene_*.log
        └── logs/validation_errors.csv
```

## Quickstart

```bash
# 1. Make sure the LLM server is running
sbatch server.sh
# wait until hostname.log is populated and the port is listening

# 2. Smoke-test the harness end-to-end
uv run python scripts/rag_isolation/run_paired_eval.py \
  --dataset scripts/rag_isolation/datasets/smoke.json \
  --output  experiments/rag_isolation/smoke_$(date +%Y%m%d_%H%M)

# 3. Read the report
cat experiments/rag_isolation/smoke_*/report.md

# 4. For real numbers, use the larger dataset
sbatch scripts/rag_isolation/run_harness.sbatch
```

## Bugfixes shipped along the way

The harness work surfaced two pre-existing bugs in `src/rag/` that were
preventing **any** import of the RAG runtime — so production
`run_improved.py` would have crashed too the first time it tried to use
RAG. Both are fixed in this branch as part of Phase 3:

1. **`src/rag/prompt_enhancer.py`** — `from .reranker import Reranker` at
   module top-level crashed because `reranker.py` doesn't exist anywhere
   in the repo. Made the import lazy inside `_get_reranker()` so RAG
   imports work; reranker only fails when explicitly enabled (which is
   off by default).
2. **`src/rag/retrieval.py`** — `format_context` did
   `int(fitness[1])` which raises `OverflowError` when a stored mutation
   has a non-finite fitness (e.g., `INVALID_FITNESS_MAX = (inf, -inf)`).
   Added `_safe_float_str` / `_safe_int_str` helpers that print
   `"unknown"` for non-finite values.

Neither change alters behavior on the happy path; they just make
existing edge cases survivable.

## Limitations / out-of-scope

- **No `--full-eval` yet.** The design supports dispatching `train.py`
  via slurm to get test-accuracy deltas per pair, but that increases
  runtime by orders of magnitude. The cheap metrics (syntax-validity,
  retries, latency, RAG block size) already test the core claim and run
  in minutes.
- **No EoT** — EoT prompt construction depends on a live `TOP_N_GENES`
  population. It's reachable from this harness with a richer case spec,
  but ship-1 only covers FixedPrompts.
- **No crossover** — production LLMGE crossover doesn't use RAG, so
  there's nothing to A/B.
- **Reranker untested** — the `Reranker` class is referenced but not
  implemented in this branch.

## How to extend

| Want to | Do |
|---|---|
| Add a new mutation template | Add a case to a dataset JSON; the template path is repo-relative |
| Test on a different parent network | Same: parent path is repo-relative |
| Add a new metric | Add a field to `core.TrialResult`, populate it inside `core.execute_trial`, and add it to the test list in `analyze.py:paired_tests` |
| Compare different RAG configs (top_k, min_similarity, …) | Set the corresponding env vars before invoking — `core.py` reads them at runtime via `cfg.constants` |
| Run with `--full-eval` to get accuracy deltas | Implement `eval_runner.py` per the design; submit one slurm job per (case, trial, arm) using `cfg.constants.PYTHON_BASH_SCRIPT_TEMPLATE` |

## Validation evidence

See `experiments/rag_isolation/<smoke_run>/report.md` for the smoke-test
output. The smoke run is a plumbing check, not a statistical claim — N=4
pairs, two templates. To produce defensible numbers, run the
`small_validation.json` dataset (5 cases × 3 trials = 15 pairs) and
ideally upsize to ≥30 pairs per template.
