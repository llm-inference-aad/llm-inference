# RAG Isolation Test Pipeline — Design

**Date:** 2026-04-27
**Status:** Design draft — to be implemented as `scripts/rag_isolation/`.

## Goal (one sentence)

For each (parent_network, mutation_template, augment_idx) case, run the **same** mutation N times in two arms — `no_rag` and `with_rag` — keeping every other variable identical, and emit per-trial metrics that can be aggregated into statistically defensible claims about RAG's effect on mutation quality.

This is **paired**: the unit of analysis is the case, not the run. Both arms see the same prompt up to the RAG context block.

## Why this is needed

The existing `scripts/run_rag_ablation_matrix.py` launches independent multi-generation runs per condition and `summarize_rag_ablation.py` aggregates run-level outcomes (best accuracy, evals to threshold). Those are **run-level** comparisons — useful, but noisy and confounded by the evolutionary trajectory itself (population diversity, lucky elites, etc.). They cannot answer questions like *"does RAG reduce LLM syntax errors at the prompt level?"* which requires holding the prompt fixed.

Per `docs/rag_isolation/02_rag_subsystem.md` §6: *"To implement per-gene paired A/B comparison, you would need a new harness."* This document is that harness's design.

## Conceptual model

```
case = (parent_path, template_path, augment_idx)
       fixed across both arms; defines "the same mutation"

trial = (case, trial_idx, llm_seed)
        varies the LLM stochasticity to give us statistical power

arm   = no_rag | with_rag
        varies ONLY whether the prompt is augmented with RAG context
        before being sent to the LLM
```

For each `(case, trial)` we record one row per arm. Pairing is by `(case_id, trial_idx)`.

## What "the same mutation" means

The two arms see prompts that are byte-identical **except** for the optional RAG context block prepended/inserted by `RagRuntime.enhance_template()`. Specifically, both arms:

- Load the same parent code from `parent_path`
- Use the same `augment_idx` (which split section of parent code is templated in)
- Use the same template (e.g., `templates/FixedPrompts/concise/Param.txt + ConstantRules.txt`)
- Use the same `temperature`, `top_p`, and `max_new_tokens` for the LLM call
- Use the same `numpy.random` seed (so any internal sampling within `augment_network` is reproducible)
- Hit the same LLM server (same model, same checkpoint)

The only difference: in `with_rag`, the template is first passed through `RagRuntime.enhance_template(template, mutation_type, query_code)` which prepends a "consider the following historically successful mutations" / "the following PyTorch documentation" section. In `no_rag`, the template is passed through unchanged.

## What we measure (per trial, per arm)

### Cheap metrics — always recorded (no GPU eval needed)

| Metric | Type | Source | What it tells us |
|---|---|---|---|
| `syntax_valid_first_try` | bool | `_validate_python_snippet` first attempt | Did RAG-augmented prompts produce parseable Python on the first shot? |
| `module_valid_first_try` | bool | `validate_module_source` first attempt | Did the generated module instantiate (catches NameError, missing imports)? |
| `n_attempts` | int (1–`LLM_GENERATION_MAX_RETRIES`) | retry loop in `augment_network` | Effort needed to converge |
| `fallback` | bool | presence of `.fallback` marker | Did all retries fail and we reverted to parent code? |
| `error_types_per_attempt` | list[str] | validation_errors.csv | Distribution of failure modes (NameError, SyntaxError, IndentationError, ...) |
| `prompt_chars` | int | `len(template_after_rag.format(parent_section))` | Token bloat from RAG |
| `response_chars` | int | length of cleaned LLM output | Output verbosity |
| `llm_latency_s` | float | wall time around the `submit_local_server` call | Latency cost of longer prompts |
| `retrieved_n_code` | int | `RetrievalStats.filtered_k` from rag_metrics.jsonl | How many code mutations RAG actually surfaced |
| `retrieved_n_text` | int | same | How many text doc chunks |
| `rag_block_chars` | int | `len(augmented_template) - len(raw_template)` | Size of the RAG insertion |
| `code_path` | str | `outputs/cases/{case_id}/{trial}/{arm}/network.py` | Reference to the generated file |
| `parent_changed` | bool | hash(generated) ≠ hash(parent) | Distinguishes a real mutation from a near-noop |

### Expensive metric — opt-in via `--full-eval`

| Metric | Type | Source |
|---|---|---|
| `test_accuracy` | float | run `train.py` with reduced epochs (e.g., 3) and parse `{gene_id}_results.txt` |
| `param_count` | int | from results.txt |
| `train_time_s` | float | wall clock around the eval submission |
| `train_invalid` | bool | did training error out (Traceback in stderr) |

`--full-eval` uses the existing slurm `PYTHON_BASH_SCRIPT_TEMPLATE` from `cfg/constants.py` to dispatch one eval job per (case, trial, arm). For our small validation set, that is 5 cases × 3 trials × 2 arms = **30 jobs** of ~3 epochs each. We will skip `--full-eval` in the initial validation since the cheap metrics suffice to demonstrate the harness works.

## Statistical analysis

Paired by `(case_id, trial_idx)`. Tests:

| Hypothesis | Test | Why |
|---|---|---|
| RAG changes proportion of syntax errors | **McNemar's** on the 2×2 paired table | Standard for paired binary outcomes |
| RAG changes proportion of fallbacks | **McNemar's** | Same |
| RAG changes mean number of attempts | **Wilcoxon signed-rank** | Paired, ordinal/skewed |
| RAG changes LLM latency | **Wilcoxon signed-rank** | Paired, skewed |
| RAG changes response length | **Wilcoxon signed-rank** | Same |
| RAG changes accuracy (if `--full-eval`) | **Wilcoxon signed-rank** | Same |

Plus per-template breakdown (so we can say "RAG helps Param but hurts Significant" if true).

Effect-size reporting: report median delta + IQR alongside p-values; do not rely on p-values alone given small N.

## Architecture

```
scripts/rag_isolation/
├── README.md                         # how to run
├── datasets/
│   └── small_validation.json         # the case list
├── run_paired_eval.py                # main entrypoint — runs trials, writes results.jsonl
├── core.py                           # case execution: load template, optionally RAG-augment, call augment_network, capture metrics
├── analyze.py                        # reads results.jsonl, computes stats, writes report.md + summary.csv
└── eval_runner.py                    # opt-in: dispatch train.py via slurm and collect results
```

### Why this layout

- **One Python entrypoint, in-process LLM calls.** No need to dispatch via slurm for the LLM step — the existing server already runs on slurm and serves HTTP. Our harness just makes HTTP requests via the same `submit_local_server` path that production uses, which guarantees we're measuring the real system.
- **Reuse `augment_network` from `src/llm_mutation.py`** — gives us the production retry/fallback/validation loop for free, so our metrics reflect what users actually experience.
- **Pre-augment the template ourselves** — we call `RagRuntime.enhance_template()` once per `(case, trial, with_rag)` and write the resulting template to a temp file, then pass that file path to `augment_network`. This way `augment_network` doesn't know about RAG at all and the only difference between arms is the template file content.
- **Write per-trial JSONL** — one row per (case, trial, arm). Easy to load with pandas, easy to extend with new metrics.

### The execution loop

```python
# pseudocode
for case in dataset.cases:
    for trial_idx in range(dataset.trials_per_case):
        seed = derive_seed(case.case_id, trial_idx)
        # Build BOTH templates first so any RAG retrieval failure doesn't bias one arm
        raw_template = build_raw_template(case)              # template + ConstantRules
        rag_template, rag_stats = build_rag_template(raw_template, case)

        # Randomize which arm runs first (avoid time-correlated server state)
        for arm in shuffled(["no_rag", "with_rag"]):
            np.random.seed(seed)                              # reset before each call
            template_path = write_temp_template(arm, raw_or_rag)
            t0 = perf_counter()
            try:
                augment_network(
                    input_filename=case.parent_path,
                    output_filename=f"{outdir}/{case.case_id}/{trial_idx}/{arm}/network.py",
                    template_txt=template_path,
                    top_p=cfg.top_p,
                    temperature=cfg.temperature,
                    gene_id=f"{case.case_id}_{trial_idx}_{arm}",
                )
            except Exception as e:
                record_error(case, trial_idx, arm, e)
            t1 = perf_counter()
            metrics = collect_metrics(case, trial_idx, arm, t1-t0, rag_stats)
            results_writer.write(metrics)
```

### Augment_idx pinning — a key correctness detail

`augment_network` currently does `augment_idx = np.random.randint(1, len(parts))` (line 25 of `src/llm_mutation.py`). This means **identical seed gives identical augment_idx**, but it's still a random pick rather than something we control. We will:

1. Either set `np.random.seed(seed)` immediately before each `augment_network` call so both arms pick the same `augment_idx`,
2. **OR** (preferred) add an optional `augment_idx` parameter to a thin wrapper around `augment_network` that we own — leaving the production `augment_network` untouched. We'll go with option (1) because it requires no changes to production code.

Verifying option 1 works: `np.random.randint` is the only `np.random` call in `augment_network` before the LLM submission. Resetting the seed pin will make `augment_idx` deterministic and identical across arms.

### How we collect per-attempt error type info

`augment_network` writes `validation_errors.csv` (run_log_dir/validation_errors.csv) per failed attempt. We point `RUN_LOG_DIR` at our experiment directory before each trial and parse that file after each call. Optionally we can monkey-patch `log_llm_interaction` to also push events into a per-trial in-memory list, but the CSV is sufficient for the metrics we care about.

### How we capture RAG retrieval stats

`prompt_enhancer.py:246–290` emits `rag_context_built` events to `utils.rag_metrics.record_metric()`, which writes to `${RUN_METRICS_DIR}/rag_metrics.jsonl`. We point `RUN_METRICS_DIR` at our experiment directory and read the most-recent line after each `with_rag` arm call.

Even simpler: we already call `enhance_template` ourselves to build the augmented template, so we can record `len(retrieved_mutations)` and the size delta directly — no need to parse the metrics file.

## Dataset format

```json
{
  "name": "small_validation_v1",
  "trials_per_case": 3,
  "temperature": 0.3,
  "top_p": 0.95,
  "max_new_tokens": 4096,
  "cases": [
    {
      "case_id": "seed_param_concise",
      "parent": "sota/ExquisiteNetV2/network.py",
      "template": "templates/FixedPrompts/concise/Param.txt",
      "include_constant_rules": true
    },
    {
      "case_id": "seed_significant_concise",
      "parent": "sota/ExquisiteNetV2/network.py",
      "template": "templates/FixedPrompts/concise/Significant.txt",
      "include_constant_rules": true
    },
    ...
  ]
}
```

Path fields are relative to `LLM_INFERENCE_ROOT_DIR`.

## Output format

### `runs/<run_id>/results.jsonl`

One row per (case, trial, arm). Example row:

```json
{
  "case_id": "seed_param_concise",
  "trial": 0,
  "arm": "no_rag",
  "gene_id": "seed_param_concise_0_no_rag",
  "parent": "sota/ExquisiteNetV2/network.py",
  "template": "templates/FixedPrompts/concise/Param.txt",
  "augment_idx": 3,
  "syntax_valid_first_try": true,
  "module_valid_first_try": true,
  "n_attempts": 1,
  "fallback": false,
  "error_types_per_attempt": [],
  "prompt_chars": 1834,
  "response_chars": 612,
  "llm_latency_s": 4.27,
  "retrieved_n_code": 0,
  "retrieved_n_text": 0,
  "rag_block_chars": 0,
  "parent_changed": true,
  "code_path": "runs/2026-04-27_v1/cases/seed_param_concise/0/no_rag/network.py",
  "wall_s_total": 4.41
}
```

### `runs/<run_id>/run_metadata.json`

```json
{
  "run_id": "2026-04-27_v1",
  "started_at": "...", "ended_at": "...",
  "dataset_path": "scripts/rag_isolation/datasets/small_validation.json",
  "llm_model": "Llama-3.3-70B-Instruct",
  "llm_server_url": "http://atl1-...:8000",
  "rag_config": { ... env vars ... },
  "git_commit": "..."
}
```

### `runs/<run_id>/report.md` (from analyze.py)

Markdown with:
- Per-metric paired test results (test, statistic, p, n_pairs, median delta, IQR)
- Per-template breakdown
- Per-arm marginal stats (mean, median, std)
- Top failures (cases where one arm broke and the other didn't) for qualitative inspection

## CLI

```bash
# Run paired evaluation
uv run python scripts/rag_isolation/run_paired_eval.py \
  --dataset scripts/rag_isolation/datasets/small_validation.json \
  --output  experiments/rag_isolation/2026-04-27_v1 \
  --server-url http://<host>:8000

# Optional flags:
#   --trials-per-case 3        # override dataset
#   --temperature 0.3          # override dataset
#   --full-eval                # also run train.py via slurm
#   --max-cases 2              # for smoke test

# Analyze
uv run python scripts/rag_isolation/analyze.py \
  experiments/rag_isolation/2026-04-27_v1
```

## Validation plan (Phase 5)

1. Build `small_validation.json` with **5 cases × 3 trials = 30 trials per arm = 60 LLM calls total**:
   - 1× Param (concise) on seed
   - 1× Significant (concise) on seed
   - 1× mutant0 (concise) on seed
   - 1× Expert_Complex (roleplay) on seed
   - 1× Param (concise) on a previously-evolved gene from `runs/auto_*/checkpoints/`
2. Submit a Llama-3.3-70B server via `server.sh` on slurm (~10 min boot).
3. Run `run_paired_eval.py` against that server. Expect ~3 sec/call × 60 = ~3 min.
4. Run `analyze.py`. Verify the report:
   - Has 30 paired rows
   - Reports a non-trivial RAG block size in `with_rag`
   - Reports zero RAG block in `no_rag`
   - Surfaces all the metric tests without crashing
5. Ship the harness with the validation report as evidence it works.

## Out of scope (deferred)

- Crossover (LLMGE crossover does NOT use RAG today, so it's nothing to test)
- EoT — the EoT template construction depends on having `TOP_N_GENES` from a current population; deferring until we want to test that flow specifically. The harness will support it via a richer case spec later.
- Reranker — the `Reranker` class is referenced but unimplemented (per `02_rag_subsystem.md` §1, §4(C)). We document this and skip.
- Power analysis to pick N — start with N=3 trials per case; if effects are small we increase later.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM server returns different temperature behavior arm-to-arm if requests interleave with other users | Run trials sequentially against a dedicated server we launched; don't share. |
| `augment_network` mutates global `RUN_LOG_DIR` etc. via cfg constants | Set env vars before `import` is enough since constants.py reads `os.environ.get`. We import lazily inside `core.py`. |
| Retrieval is empty (RAG_USE_*=true but index has nothing relevant) | Sanity-check at startup: log average retrieved_n. If it's zero across the board, fail loud — the experiment is meaningless. |
| One arm runs faster and biases server-side state (kv-cache warm-up etc.) | Randomize arm order per trial. |
| Same-seed-produces-same-augment-idx breaks if `np.random` is touched elsewhere | Set seed immediately before each call; double-check by recording `augment_idx` in metrics and asserting it matches across arms of a pair in `analyze.py`. |
