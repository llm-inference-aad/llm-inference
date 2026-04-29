# RAG Isolation — Fitness Evaluation Spec

**Status:** spec only — not yet implemented.
**Author:** drafted 2026-04-29.
**Companion docs:** `05_experiment_plan.md` (statistical design), `03_design.md` (harness internals).

## 1. Goal

Extend the paired RAG isolation harness so that for every (case, trial,
arm) we **train the resulting `network.py` on CIFAR-10 and capture a
fitness tuple** `(test_acc, num_params)`. This lets the finals report
answer the question that matters most:

> Does using RAG to seed the LLM produce networks that are *measurably
> better* — not just syntactically more often valid?

The current harness stops at "did the mutation produce a valid
module?". After this spec is implemented, the harness can also answer
"and is the resulting network more accurate / smaller / faster to
train than the no-RAG counterpart on the same prompt?".

## 2. Constraints fixed by this spec

| Knob | Value | Rationale |
|---|---|---|
| `EPOCHS` per eval | **8** | Production uses 24 (~30 min/net on H100). 8 keeps each eval ≈10 min, totals stay tractable for a paired comparison without losing the relative ordering between mutations. |
| Train seed | **21** (the existing `-seed 21` in `submit_run`) | Kept identical across both arms of every pair so fitness deltas reflect mutation differences, not RNG. |
| Number of eval seeds per network | **1** | Multi-seed re-evaluation would multiply wall time and is out of scope for finals; we accept that single-seed fitness has noise. |
| Other train flags | **unchanged from `run_improved.py:submit_run`** (`-bs 216 -end_lr 0.001 -val_r 0.2 -amp`) | Comparable to numbers from real evolution runs. |

## 3. What already exists, and what we reuse

The main LLMGE pipeline already evaluates networks. We piggyback on it
verbatim where possible.

- `src/cfg/constants.py:157` — `PYTHON_BASH_SCRIPT_TEMPLATE` (the sbatch
  script template). **Reused as-is.**
- `run_improved.py:577 submit_run` — builds the python runline and
  submits via `sbatch`. **Logic copied** into a small isolation-only
  helper; we do not call into `run_improved.py` directly because that
  module has heavy global state.
- `sota/ExquisiteNetV2/train.py` — the actual training entrypoint.
  Writes `{RUN_DIR}/results/{gene_id}_results.txt` containing
  `test_acc,total_params,val_acc,tr_time`. **Untouched.**
- `run_improved.py:688` — fitness tuple convention `(test_acc, -num_params)`.
  We mirror this in the harness.

## 4. Where the network files have to live

`train.py` imports the mutated module via `-network "models.network_{gene_id}"`.
That string forces the file to be at:

```
sota/ExquisiteNetV2/models/network_{gene_id}.py
```

Currently the harness writes the network into
`experiments/rag_isolation/<run>/cases/<case_id>/<trial>/<arm>/network.py`
to keep paired runs isolated. Two options:

- **(a) Copy** to `sota/ExquisiteNetV2/models/network_{gene_id}.py` right
  before submitting the eval job, then delete after fitness is read.
- **(b) Symlink** instead of copy. Cheaper but more fragile if the link
  outlives the experiment.

**Decision: copy.** Simpler, deletable, works on cluster filesystems
that don't always honor symlinks.

`gene_id` is already unique per (case, trial, arm) — it is set in
`core.py:execute_trial` as `f"{case_id}_{trial:02d}_{arm}"`. No
collisions.

## 5. Sequencing

The existing harness is fully sequential:

```
for case:
  for trial:
    for arm:
      mutate → write network.py → record metrics
```

Naive option: **add eval inline** after each mutation. Bad — every eval
is ~10 min and serializes against the LLM phase, multiplying wall time.

**Adopted: two-phase split.**

1. **Mutation phase** (existing). Run the full case×trial×arm sweep,
   write all `network.py` files, record mutation-level metrics.
2. **Eval phase** (new). After all mutations are done, iterate over the
   `results.jsonl`, copy each non-fallback network into the SOTA models
   dir, submit one sbatch per gene_id. Record `eval_job_id` per row.
3. **Collect phase** (new). Block (with a polling loop) until every
   eval results file appears (or timeout). Read fitness, merge into a
   second JSONL `results_with_fitness.jsonl`.

This way the LLM phase finishes on a bounded budget against a single
GPU (the LLM server), and the eval phase fans out across whatever
training-capable GPUs the cluster gives us.

For the smoke and small_validation runs, we expect ~10–30 eval jobs;
slurm should pack them in parallel without queue pressure.

## 6. Fallback handling

`augment_network` writes `network.py.fallback` when validation retries
exhaust and we use the parent code instead. In that case the resulting
network is byte-identical to the parent (or to the seed if the parent
is the seed).

Fallback genes do NOT need re-training. The main pipeline already
inherits parent fitness in this case (`run_improved.py:712`). The
harness mirrors this:

- **Fallback to seed** → fitness inherited from
  `sota/ExquisiteNetV2/results/network_results.txt` (the trained-once
  seed baseline). If that file is missing, train the seed first via
  `src/evolution/seed.py:train_seed_network_baseline()` and cache.
- **Fallback to a non-seed parent** → the harness's parents are all
  seed-or-evolved-checkpoint files that already exist on disk; we
  treat them as "seed-equivalent" for inheritance purposes and look
  up a precomputed fitness for the parent path (see §7).

If we cannot find a parent fitness (e.g. a brand-new evolved parent
introduced into the dataset), we fall back to **training the parent
once at the start of the run** and caching its fitness in
`experiments/rag_isolation/<run>/parent_fitness.json`.

## 7. New files

```
scripts/rag_isolation/
  eval_submit.py        # submit one sbatch eval job per gene
  collect_fitness.py    # poll for results files, merge into JSONL
  parent_fitness.py     # train any unevaluated parent once, cache result
```

Plus an extension of `analyze.py` to add fitness columns.

### 7.1 `eval_submit.py` (≈80 lines)

```
inputs:
  - run_dir (the experiments/rag_isolation/<run>/ root)
  - results.jsonl (the mutation-phase output)
  - epochs (default 8)
  - seed (default 21)
  - dry_run (just print the sbatch lines)
behavior:
  for each row in results.jsonl:
    if row.fallback:
      mark eval_status="inherited"; skip
    else:
      copy row.code_path → sota/ExquisiteNetV2/models/network_{gene_id}.py
      build bash script from PYTHON_BASH_SCRIPT_TEMPLATE
      sbatch it; capture job_id
      append row {"gene_id":..., "eval_job_id":..., "eval_status":"submitted"}
        to run_dir/eval_jobs.jsonl
outputs:
  run_dir/eval_jobs.jsonl
```

### 7.2 `collect_fitness.py` (≈100 lines)

```
inputs:
  - run_dir
  - timeout (default 4h, configurable)
  - poll_interval (default 60s)
behavior:
  while not all done and elapsed < timeout:
    for row in eval_jobs.jsonl that is still "submitted":
      check {SOTA_ROOT}/results/{gene_id}_results.txt
      if present: parse "{test_acc,num_params,val_acc,tr_time}"
                 record fitness=(test_acc, -num_params)
                 row.eval_status = "done"
      else:       check sacct/squeue for job_id; if FAILED, mark "failed"
    sleep poll_interval
  for fallback rows: copy parent fitness from parent_fitness.json
  emit run_dir/results_with_fitness.jsonl
       run_dir/results_with_fitness.csv  (flat for analyze.py)
outputs:
  results_with_fitness.jsonl, results_with_fitness.csv
```

### 7.3 `parent_fitness.py` (≈60 lines)

```
inputs:
  - dataset.json (so we know the unique parents in this run)
  - cache file: experiments/rag_isolation/_parent_fitness_cache.json
                (shared across runs, NOT per-run)
behavior:
  for each unique parent path in dataset:
    if parent in cache: continue
    if parent == "sota/ExquisiteNetV2/network.py":
       train via src/evolution/seed.py.train_seed_network_baseline()
    else:
       copy parent → sota/ExquisiteNetV2/models/network_PARENT_{hash}.py
       sbatch the same eval script with EPOCHS=8 -seed 21
       block until results file appears
    record fitness in cache
outputs:
  experiments/rag_isolation/_parent_fitness_cache.json
```

This caches across runs so we only ever train the seed once at 8 epochs.

## 8. Schema additions

Add to `TrialResult` (in `core.py`):

```python
# (added by collect_fitness.py, NOT by core.py)
fitness_acc: float | None = None         # 0..1
fitness_params: int | None = None        # raw param count, positive
fitness_inherited_from: str | None = None  # gene_id if inherited, else None
eval_job_id: str | None = None
eval_status: str | None = None  # submitted|done|failed|inherited|timeout
eval_train_seconds: float | None = None
eval_val_acc: float | None = None
```

These default to `None` so existing analysis is unaffected if the eval
phase is skipped.

## 9. analyze.py extensions

New paired-difference rows (when fitness columns are populated):

| Metric | Test | Direction expected if RAG helps |
|---|---|---|
| `fitness_acc` | Wilcoxon signed-rank, paired by (case, trial) | Δ > 0 |
| `fitness_params` | Wilcoxon signed-rank | Δ ≠ 0, sign uncertain |
| `fitness_acc / params_in_M` | Wilcoxon signed-rank | Δ > 0 |
| Pareto-domination rate | exact binomial: P(with_rag dominates no_rag) > 0.5 | > 0.5 |

Pareto domination: with_rag wins iff `fitness_acc_with ≥ fitness_acc_without` AND
`fitness_params_with ≤ fitness_params_without`, with at least one strict. This
is the actual decision rule for the multi-objective evolution loop.

## 10. Wall-time budget

Per network: 8 epochs ExquisiteNetV2 on CIFAR-10 ≈ **6–10 min** on H100/A100,
≈ 12–18 min on V100/RTX6000. Use 12 min as a conservative average.

| Run | Networks to train | Eval wall (sequential) | Eval wall (5-way parallel via sbatch) |
|---|---|---|---|
| smoke (8 mutations, ~50% fallback) | ~4 | ~50 min | ~15 min |
| small_validation (10 mutations, fallbacks unknown) | ~5–10 | ~1–2 h | ~25 min |
| finals (target: 30+ mutations) | ~25 | ~5 h | ~1 h |

The cluster usually gives us 4–8 parallel slots; the bottleneck is the
queue, not single-job time. Expect ~30–90 min eval-phase wall time for
the finals run.

## 11. Failure modes and mitigations

| Failure | Likelihood | Mitigation |
|---|---|---|
| `models/network_{gene_id}.py` import fails (silent type errors not caught by augment_network's validation) | Medium | `train.py` will exit non-zero; `collect_fitness.py` records `eval_status=failed` and a stderr excerpt. |
| Training hangs / OOM | Low | sbatch wall time is 8h; we set a tighter `--time=00:30:00` for 8-epoch jobs. |
| Slurm queue starvation (eval can't start before LLM server eats its budget) | Medium | Eval jobs go to a different partition than the LLM server. We do **not** depend on the LLM server during the eval phase. |
| Eval results file written but partial | Low | Parse defensively; if `len(parts) < 2`, mark failed. |
| CIFAR-10 dataset not present | High first time | `parent_fitness.py` calls `sota/ExquisiteNetV2/split.py` (existing) to download/extract once. |
| Two harness runs collide on `models/network_{gene_id}.py` | Very low (gene_id includes run timestamp implicitly via case-trial-arm-arm-only name) | Add run-stamp prefix to gene_id: `{run_id}_{case}_{trial}_{arm}`. **Decision: yes, do this.** |

## 12. Validation plan

Before trusting any finals number:

1. **Sanity round** on `smoke.json` (8 muts):
   - Train the seed once, cache fitness `S = (acc_seed, -params_seed)`.
   - Run mutation phase; expect ~50% fallback per smoke results.
   - Run eval phase. Verify:
     - Every fallback row has fitness == `S` (or parent's fitness).
     - Non-fallback rows have plausible fitness (acc 0.50–0.85 at 8 epochs is normal for CIFAR-10).
     - Pareto comparison runs without exception.
2. **Re-run identical seed twice** for one (case, trial) without changing anything else; check fitness reproducibility band. Expect ±0.5% test_acc noise; if more, log as known limitation.
3. **Compare to the seed network's published baseline** to make sure 8-epoch numbers haven't drifted from prior runs.

## 13. Out of scope for this spec

- Multi-seed evaluation (we'd want 3 seeds × 30 networks for a real
  variance estimate; ~3× the wall time).
- Re-training existing checkpoints from the cache for fairness; we
  trust whatever the cache has.
- Changing the fitness function from `(acc, -params)`.
- Latency/throughput-based fitness (the alternate `(acc, latency)`
  objective the main pipeline supports — out of scope here, can be
  added later by changing the cache lookup).
- Beating the eval queue with a long-running training daemon (a
  natural follow-up if the per-job sbatch overhead dominates).

## 14. Acceptance criteria

The implementation is "done" when:

- [ ] `eval_submit.py --run-dir <dir>` reads `results.jsonl`, copies
      networks, submits sbatch jobs, writes `eval_jobs.jsonl`.
- [ ] `collect_fitness.py --run-dir <dir>` polls and produces
      `results_with_fitness.jsonl` with all rows resolved (or marked
      failed/timeout).
- [ ] `parent_fitness.py --dataset <json>` produces / updates
      `_parent_fitness_cache.json` and trains any missing parents.
- [ ] `analyze.py` accepts a `--with-fitness` flag and emits the
      paired-fitness rows in `tests.csv` and a fitness section in
      `report.md`.
- [ ] Smoke run end-to-end (mutation + eval + analyze) takes < 90 min
      and produces a non-empty fitness section.
