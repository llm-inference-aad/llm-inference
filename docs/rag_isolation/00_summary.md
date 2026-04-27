# RAG Isolation Testing — Summary

**One-liner:** A standalone harness that runs the *same* mutation twice — once
without RAG and once with RAG — so we can attribute differences in syntax errors,
retries, and latency to RAG itself rather than to evolutionary noise.

## Why this exists

LLMGE already has condition-level RAG ablation tooling (`run_rag_ablation_matrix.py`,
`summarize_rag_ablation.py`) which compares whole evolutionary runs. That's
useful for top-line claims like *"hybrid RAG reduced evals-to-threshold by 24%
on average"*, but it cannot answer *"did RAG actually reduce LLM syntax
errors at the prompt level?"* — because every run sees a different population
trajectory, different parents, different mutation selections.

This harness fills that gap by **pairing**: every (parent, template,
augment_idx) "case" runs in both arms, with everything else held constant.
The only thing that varies between arms is whether the prompt was passed
through `RagRuntime.enhance_template()`.

## What "the same mutation" means

For each (case_id, trial_idx) the two arms share:

- Same parent network file
- Same `augment_idx` (which split section is being templated in) — pinned
  via `np.random.seed(derived_seed)` immediately before each call
- Same template + ConstantRules.txt
- Same temperature, top_p, max_new_tokens
- Same LLM server (same model, same checkpoint)

The only difference: in the `with_rag` arm, the template is augmented
with retrieved code mutations + text doc snippets via the production
`RagRuntime.enhance_template()` call before being passed to the LLM.

## Architecture

```
scripts/rag_isolation/
├── core.py                 # per-trial execution
├── run_paired_eval.py      # CLI entrypoint
├── analyze.py              # paired stats + report
├── run_harness.sbatch      # slurm wrapper
└── datasets/
    ├── smoke.json          # 2 cases × 2 trials (plumbing test)
    └── small_validation.json  # 5 cases × 3 trials (initial real run)
```

The harness **reuses the production `augment_network` function** in
`src/llm_mutation.py`, which means the captured metrics reflect what
actually happens in production: the real validation loop, real retry
logic, real fallback behavior.

## What gets measured

### Cheap (always recorded — no GPU eval)

| Metric | Source | What it tells us |
|---|---|---|
| `syntax_valid_first_try` | `_validate_python_snippet` first attempt | Did the LLM produce parseable Python on the first shot? |
| `module_valid_first_try` | `validate_module_source` first attempt | Did the module instantiate (catches NameError, missing imports)? |
| `n_attempts` | retry loop in `augment_network` | Effort needed to converge |
| `fallback` | `.fallback` marker | All retries failed; reverted to parent code |
| `error_types_per_attempt` | `validation_errors.csv` | Distribution of failure modes |
| `prompt_chars`, `response_chars` | per-gene LLM log | Token bloat from RAG vs response verbosity |
| `llm_latency_s` | wall time around the call | Latency cost |
| `retrieved_n_code`, `retrieved_n_text` | RAG runtime | How much RAG actually surfaced |
| `rag_block_chars` | template size delta | Size of the RAG insertion |
| `parent_changed` | hash compare | Sanity check that the LLM produced something different |

### Expensive (designed-in, not yet wired)

`test_accuracy`, `param_count`, `train_time_s` — would dispatch `train.py`
via slurm. The harness can be extended via the `--full-eval` flag pattern
described in the design doc; ship-1 ships without it because the cheap
metrics already test the core hypothesis and run in minutes.

## Statistical methodology

Paired by `(case_id, trial_idx)`:

| Metric type | Test |
|---|---|
| Paired binary (`syntax_valid_first_try`, `fallback`, …) | **McNemar's exact** |
| Paired continuous (`n_attempts`, `llm_latency_s`, …) | **Wilcoxon signed-rank** |

Effect sizes (median Δ + IQR) reported alongside p-values. With small
N, do not over-interpret p-values.

The analyze.py script also produces:

- Per-arm marginal summary (means, medians, IQRs)
- Per-template breakdown (so we can say *"RAG helps Param but hurts
  Significant"* if true)
- Callouts for paired flips (cases where one arm worked and the other
  didn't) — the most informative for qualitative review

## Output artifacts

```
experiments/rag_isolation/<run_id>/
├── results.jsonl       # one row per (case, trial, arm)
├── paired.csv          # one row per pair, both arms side-by-side
├── summary.csv         # per-arm marginal stats
├── tests.csv           # paired test results
├── report.md           # human-readable summary
├── run_metadata.json   # run config, env, git commit
└── cases/<case>/<trial>/<arm>/
    ├── prompt_template.txt        # what was sent to the LLM
    ├── prompt_template.raw.txt    # the un-augmented template (for diffing)
    ├── network.py                  # the generated module
    ├── network.py.fallback        # marker if augment_network gave up
    └── logs/llm/gene_*.log         # per-attempt LLM IO
```

## Bugs found and fixed along the way

While wiring up the harness I discovered two pre-existing bugs in
`src/rag/` that were preventing **any** import of the RAG runtime —
production `run_improved.py` would have hit them too the first time it
tried to use RAG. Both fixed in this branch:

1. **`src/rag/prompt_enhancer.py`** — top-level `from .reranker import Reranker`
   crashed because `reranker.py` doesn't exist. Made the import lazy
   inside `_get_reranker()`.
2. **`src/rag/retrieval.py`** — `format_context` did `int(fitness[1])`
   which raises `OverflowError` when a stored mutation has non-finite
   fitness (e.g., `INVALID_FITNESS_MAX`). Added defensive
   `_safe_float_str` / `_safe_int_str` helpers.

Neither change alters the happy path; they just make existing edge
cases survivable.

## How to run

```bash
# 1. LLM server must be running (writes to hostname.log)
sbatch server.sh

# 2. Smoke test first (8 LLM calls, ~5 min)
uv run python scripts/rag_isolation/run_paired_eval.py \
  --dataset scripts/rag_isolation/datasets/smoke.json \
  --output  experiments/rag_isolation/smoke_$(date +%Y%m%d_%H%M)

# 3. Real validation (60 LLM calls, ~30 min)
sbatch scripts/rag_isolation/run_harness.sbatch

# 4. Inspect
cat experiments/rag_isolation/<run>/report.md
```

## Limitations / out of scope

- **No `--full-eval`** — accuracy/param-count deltas would require
  dispatching `train.py` per pair. Designed for, not implemented.
- **No EoT / Crossover** — EoT depends on a live population; LLMGE
  crossover doesn't use RAG. Both reachable from this harness with a
  richer case spec, but out of scope for ship-1.
- **No Reranker** — referenced in code but unimplemented (`Reranker`
  class missing).
- **LLM-side stochasticity** — at temperature > 0 the LLM is
  non-deterministic. We use multiple trials per case rather than asking
  the server for a seed (the local server doesn't accept one).

## Where to read more

- `docs/rag_isolation/01_llmge_pipeline.md` — full context map of LLMGE
- `docs/rag_isolation/02_rag_subsystem.md` — full context map of RAG
- `docs/rag_isolation/03_design.md` — the design rationale and statistical model
- `docs/rag_isolation/04_handoff.md` — handoff notes (file inventory, extension recipes)
- `scripts/rag_isolation/README.md` — operator-facing usage guide
