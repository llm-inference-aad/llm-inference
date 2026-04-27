# RAG Isolation Test Pipeline

A standalone harness for measuring RAG's effect on LLMGE mutation quality
**at the per-mutation level** — paired across `no_rag` and `with_rag` arms
so claims about RAG are not confounded by evolutionary trajectory.

See [`docs/rag_isolation/03_design.md`](../../docs/rag_isolation/03_design.md) for the
full design rationale and statistical model.

## Files

```
scripts/rag_isolation/
├── core.py                      # per-trial execution; reuses production augment_network
├── run_paired_eval.py           # CLI entrypoint
├── analyze.py                   # paired McNemar + Wilcoxon tests, per-template breakdown
├── datasets/
│   └── small_validation.json    # 5 cases × 3 trials = 30 pairs (smoke test)
└── README.md                    # this file
```

## Prerequisites

1. **A running LLM server** that exposes `/generate` (the same endpoint
   `submit_local_server` in `src/llm_utils.py` uses). Launch one via:

   ```bash
   sbatch server.sh
   # wait until logs show "Server ready", then note the hostname
   ```

   Or reuse a running server's `hostname.log` + `SERVER_PORT` env.

2. **A populated FAISS index** at `rag_data/`. If empty, build with:

   ```bash
   uv run python scripts/setup_rag.py --runs-dir runs/ --pdf-dir rag_corpus/
   ```

3. **`.env` loaded** so `LLM_INFERENCE_ROOT_DIR` and `VENV_PATH` are set.

## Run

```bash
uv run python scripts/rag_isolation/run_paired_eval.py \
  --dataset scripts/rag_isolation/datasets/small_validation.json \
  --output  experiments/rag_isolation/$(date +%Y-%m-%d_%H%M) \
  --server-url <hostname>:8000
```

The harness does **not** dispatch slurm jobs for the LLM step — it makes
HTTP requests against the server directly, in-process. A run of the small
validation set takes roughly 60 calls × 3–5 sec/call = **~5 minutes**.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--max-cases 1` | Smoke test with 1 case before committing to the full run |
| `--trials-per-case 1` | Same idea — fewer trials |
| `--no-rag-only` | Run only the no_rag arm (e.g., to debug RAG-side issues) |
| `--rag-use-code-context false` | Disable code retrieval, keep text retrieval |
| `--rag-use-text-context false` | Disable text retrieval, keep code retrieval |

## Analyze

```bash
uv run python scripts/rag_isolation/analyze.py experiments/rag_isolation/<run_dir>
```

Writes:
- `report.md` — human-readable summary with paired test results
- `paired.csv` — one row per (case, trial) with both arms side-by-side
- `summary.csv` — per-arm marginal stats
- `tests.csv` — paired test outputs (statistic, p-value, effect size)

## What "the same mutation" means

For each (case_id, trial_idx) pair, both arms see byte-identical inputs
**except** for the optional RAG-augmented context block prepended by
`RagRuntime.enhance_template()`. Specifically pinned across arms:

- Same parent code file
- Same `augment_idx` (which split section is templated in) — pinned via
  `np.random.seed(derived_seed)` immediately before each `augment_network` call
- Same template + ConstantRules.txt
- Same temperature, top_p, max_new_tokens
- Same LLM server

## What we measure

**Cheap (always recorded):**
- `syntax_valid_first_try` — did `_validate_python_snippet` pass on attempt 1?
- `module_valid_first_try` — did `validate_module_source` pass on attempt 1? (Catches NameError/ImportError inside `__init__`)
- `n_attempts` — how many retries before convergence (1 = first try)
- `fallback` — did all retries fail and `augment_network` reverted to parent?
- `error_types_per_attempt` — distribution of failure modes (`SyntaxError`, `NameError`, …)
- `prompt_chars`, `response_chars`, `llm_latency_s` — cost of the call
- `retrieved_n_code`, `retrieved_n_text`, `rag_block_chars` — how much RAG actually contributed
- `parent_changed` — sanity check that the LLM produced something different from the parent

**Expensive (not yet wired):**
- `test_accuracy`, `param_count`, `train_time_s` — would dispatch `train.py` via slurm. The design supports it but Phase 1 ships without.

## Statistical methodology

| Metric type | Test | Why |
|---|---|---|
| Paired binary (`syntax_valid_first_try`, `fallback`, …) | **McNemar's exact** | Standard for paired binary outcomes, no large-N assumption |
| Paired continuous/ordinal (`n_attempts`, `llm_latency_s`, …) | **Wilcoxon signed-rank** | Paired, robust to skew |

Effect size: report median delta + IQR alongside p-values. With small N,
do not over-interpret p-values.

## Limitations / out of scope

- **Crossover** — LLMGE crossover doesn't use RAG, so no harness needed.
- **EoT** — supported in principle but the EoT template depends on
  `TOP_N_GENES` from a current population; deferred until we want to A/B
  that flow specifically.
- **Reranker** — referenced in code but not implemented (`Reranker` class missing).
- **LLM-side determinism** — temperature > 0 makes the LLM stochastic; we
  use multiple trials per case rather than asking the server for a seed.
