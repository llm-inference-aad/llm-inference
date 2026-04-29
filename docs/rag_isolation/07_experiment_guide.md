# RAG Isolation — Experiment Guide

**Audience:** anyone running the paired RAG isolation harness end-to-end,
from a fresh checkout to a finished `report.md` with fitness numbers.
**Companion docs:**
[`05_experiment_plan.md`](05_experiment_plan.md) (statistical design),
[`06_fitness_evaluation_spec.md`](06_fitness_evaluation_spec.md)
(what the eval phase implements).

This guide is the operational counterpart to those specs: it tells you
which command to run, in which order, and what to do when something
breaks.

## 0. The five phases

The harness now runs in five phases. The first three were already
shipped; the last three implement spec §06.

| Phase | Script | What it does | Wall time |
|---|---|---|---|
| 1. Setup | `parent_fitness.py` | Train every parent in the dataset once, cache fitness | ~10 min for the seed; one-time |
| 2. Mutation | `run_paired_eval.py` | Generate paired (no_rag, with_rag) `network.py` files via LLM | ~5 min/pair |
| 3. Eval submit | `eval_submit.py` | Submit one sbatch CIFAR-10 training job per non-fallback gene | seconds |
| 4. Collect fitness | `collect_fitness.py` | Poll for results, merge into `results_with_fitness.jsonl` | ~30–90 min queue + train |
| 5. Analyze | `analyze.py --with-fitness` | Paired tests + fitness section + Pareto rate → `report.md` | seconds |

You always do them in this order. Phase 1 is *idempotent and shared*: it
populates a cross-run cache, so once the seed network is trained at 8
epochs you never train it again unless you deliberately bust the cache.

## 1. Prerequisites (one-time)

### 1.1 LLM server

Phase 2 needs a running local LLM server.

```bash
sbatch server.sh
# wait until $REPO/hostname.log is populated and the port is listening
tail -f hostname.log
```

The harness pings the server before starting and aborts cleanly if it
isn't reachable.

### 1.2 RAG data

Phase 2's `with_rag` arm needs a populated FAISS index at `rag_data/`.
If you've never set it up:

```bash
uv run python scripts/setup_rag.py --runs-dir runs/ --pdf-dir rag_corpus/
```

### 1.3 CIFAR-10 dataset

Phases 1, 4 need CIFAR-10 split into `sota/ExquisiteNetV2/cifar10/{train,test}/`.
`parent_fitness.py` will run `sota/ExquisiteNetV2/split.py` for you if
the unpacked `cifar10/` directory is missing but the raw
`cifar-10-batches-py/` archive is present. If neither is present,
download CIFAR-10 first.

### 1.4 Environment

```bash
source .env                       # sets VENV_PATH, LLM_INFERENCE_ROOT_DIR, …
echo $VENV_PATH                    # should be a real path
echo $LLM_INFERENCE_ROOT_DIR       # should be the repo root
```

The eval-phase sbatch scripts inherit these from your shell — if they're
missing, the slurm job will fail at `source "$VENV_PATH/bin/activate"`.

## 2. Phase 1 — Train parents (one-time per dataset)

```bash
uv run python scripts/rag_isolation/parent_fitness.py \
  --dataset scripts/rag_isolation/datasets/smoke.json \
  --epochs 8 --seed 21
```

This walks every unique `parent` referenced in the dataset, trains it
once at 8 epochs / seed=21, and writes:

```
experiments/rag_isolation/_parent_fitness_cache.json
```

The cache is **shared across runs** by design (per spec §7.3) so you only
ever pay for parent training once. If you change the eval regime
(epochs, seed, batch size) and want fresh numbers, pass `--force-retrain`
or delete the cache file.

The seed network is trained synchronously via
`src/evolution/seed.py:train_seed_network_baseline()` (~10 min on H100,
~20 min on V100). Non-seed parents are trained via sbatch + poll.

**You can skip this step if every parent in your dataset is already in
the cache.** The downstream tools no-op gracefully when fitness is
already cached.

## 3. Phase 2 — Mutation phase

This is unchanged from the previous shipped version. It runs the LLM
twice per (case, trial), once with RAG and once without, and writes
results to `results.jsonl`.

```bash
RUN_DIR=experiments/rag_isolation/smoke_$(date +%Y%m%d_%H%M)
uv run python scripts/rag_isolation/run_paired_eval.py \
  --dataset scripts/rag_isolation/datasets/smoke.json \
  --output  $RUN_DIR
```

What you should see:

```
experiments/rag_isolation/smoke_20260429_1534/
├── results.jsonl              # one row per (case, trial, arm)
├── run_metadata.json
├── hostname.log
└── cases/<case_id>/<trial>/<arm>/
    ├── network.py             # or network.py.fallback
    ├── prompt_template.txt
    └── logs/
```

**Quick sanity check:** confirm the `code_path` field in each
non-fallback row points to a real file.

```bash
jq -r 'select(.fallback == false) | .code_path' \
  $RUN_DIR/results.jsonl | xargs -I{} ls {}
```

## 4. Phase 3 — Submit eval jobs

```bash
uv run python scripts/rag_isolation/eval_submit.py \
  --run-dir $RUN_DIR \
  --epochs 8 --seed 21 \
  --wall-time 00:30:00
```

What this does, per row in `results.jsonl`:

- **fallback row:** marks `eval_status="inherited"`, no sbatch.
- **non-fallback row:** copies `network.py` to
  `sota/ExquisiteNetV2/models/network_<run-stamp>__<gene_id>.py`
  (run-stamp prefix avoids collisions across concurrent harness runs —
  spec §11), generates an sbatch script, and submits it. The script
  exports `RUN_DIR=$RUN_DIR` so `train.py` writes its results file at
  `$RUN_DIR/results/<run-stamp>__<gene_id>_results.txt`.

It writes:

```
$RUN_DIR/eval_jobs.jsonl       # one row per gene with eval_job_id, eval_status
$RUN_DIR/eval_scripts/         # the .sh files that were submitted
$RUN_DIR/logs/eval/            # slurm stdout
$RUN_DIR/errors/eval/          # slurm stderr
```

**Smoke first.** Use `--dry-run` to see what would happen without
actually submitting. Use `--limit 1` to submit just one job and verify
it lands in your queue with `squeue -u $USER` before submitting the
rest.

## 5. Phase 4 — Collect fitness

```bash
uv run python scripts/rag_isolation/collect_fitness.py \
  --run-dir $RUN_DIR \
  --timeout 14400 \
  --poll-interval 60
```

This polls every minute, checks each pending row, and:

1. If `$RUN_DIR/results/<eval_gene_id>_results.txt` exists, parses
   `test_acc,total_params,val_acc,tr_time` and marks the row done.
2. If the file doesn't exist, queries `squeue` and `sacct`. A row whose
   slurm job is FAILED/CANCELLED/TIMEOUT/OOM is marked `failed`.
3. If `--timeout` elapses with rows still pending, they're marked
   `timeout` (not deleted — re-run later to pick them up).

For fallback rows, fitness is read from
`_parent_fitness_cache.json` and the `fitness_inherited_from` column
records the parent path.

Outputs:

```
$RUN_DIR/results_with_fitness.jsonl
$RUN_DIR/results_with_fitness.csv     # flat, for downstream tooling
```

After collection, the run-stamp-prefixed copies in
`sota/ExquisiteNetV2/models/` are deleted (use `--no-cleanup` to keep
them for debugging).

**If you need to re-collect** (e.g., a job re-ran and now has results),
just run `collect_fitness.py` again. It rewrites both files atomically
from the current state of `eval_jobs.jsonl` plus the parent cache.

## 6. Phase 5 — Analyze

```bash
uv run python scripts/rag_isolation/analyze.py $RUN_DIR --with-fitness
```

The `--with-fitness` flag tells `analyze.py` to read
`results_with_fitness.jsonl` instead of `results.jsonl` and to emit:

- The four new paired tests defined in spec §9:
  - Δ `fitness_acc` (Wilcoxon)
  - Δ `fitness_params` (Wilcoxon)
  - Δ `fitness_acc / params_in_M` (Wilcoxon, efficiency)
  - Pareto-domination rate (exact two-sided binomial vs 0.5)
- A "Fitness comparison" section in `report.md` listing per-pair
  accuracies and parameter counts.

Without `--with-fitness` you still get the existing syntax-validity /
attempts / latency / cost analysis from `results.jsonl`.

## 7. End-to-end smoke walkthrough

A complete run on the 2-case smoke dataset, timed for the validation
plan in spec §12. Expected wall time: **<90 min**.

```bash
# (one-time) train the seed
uv run python scripts/rag_isolation/parent_fitness.py \
  --dataset scripts/rag_isolation/datasets/smoke.json \
  --epochs 8 --seed 21

# fresh run dir
RUN=experiments/rag_isolation/smoke_$(date +%Y%m%d_%H%M)

# phase 2: mutation (~10 min)
uv run python scripts/rag_isolation/run_paired_eval.py \
  --dataset scripts/rag_isolation/datasets/smoke.json \
  --output  $RUN

# phase 3: submit evals (seconds)
uv run python scripts/rag_isolation/eval_submit.py --run-dir $RUN

# phase 4: poll (15–60 min depending on queue)
uv run python scripts/rag_isolation/collect_fitness.py --run-dir $RUN

# phase 5: report
uv run python scripts/rag_isolation/analyze.py $RUN --with-fitness
cat $RUN/report.md
```

**Acceptance check** (per spec §14): the `report.md` should contain a
non-empty "Fitness comparison" section with at least one paired row.
Every row in `results_with_fitness.jsonl` should have an
`eval_status` of `done`, `inherited`, `failed`, or `timeout` — never
`submitted`.

## 8. Reading the report

```
## Fitness comparison (with_rag vs no_rag)

| Metric | Test | n_pairs | Effect | p-value |
|---|---|---|---|---|
| `fitness_acc` | Wilcoxon | 24 | median Δ = +0.012 (IQR -0.003 – +0.024) | 0.082 |
| `fitness_params` | Wilcoxon | 24 | median Δ = -120k (IQR -340k – +85k) | 0.341 |
| `fitness_acc_per_M` | Wilcoxon | 24 | median Δ = +0.014 | 0.067 |
| `pareto_domination_rate` | binomial | 24 | with_rag wins 14 / decided 21 (ties 3); rate = 0.667 | 0.189 |
```

How to read:

- **Δ fitness_acc > 0** → `with_rag` networks were on average more
  accurate. The Wilcoxon p-value tells you whether that ordering is
  unlikely by chance.
- **Δ fitness_params** is sign-uncertain. If it's near 0, RAG didn't
  meaningfully change network size; if it's negative, RAG produces
  smaller networks at similar accuracy.
- **Δ fitness_acc_per_M** is an efficiency metric. If positive, RAG
  produces more accurate-per-parameter networks.
- **Pareto-domination rate** is the most "production-relevant" number —
  it answers "in how many paired matchups does `with_rag` strictly beat
  `no_rag` on both axes?" Rate = 0.5 is no signal; >0.5 favors RAG; the
  binomial p-value tests against H₀: rate = 0.5.

With small N (smoke = 4 pairs, small_validation = 30, finals = 60+) read
*effect sizes* before *p-values*. The harness is designed for honest
descriptive comparison, not high-power inference.

## 9. Troubleshooting

### A submitted job never produced a results file

```bash
# what slurm thinks
sacct -j <job_id> -o JobID,State,ExitCode,Elapsed,Start,End

# stderr from the failed run
ls $RUN/errors/eval/
cat $RUN/errors/eval/eval-<job_id>.err
```

Common causes:

- `module not found: models.network_…` — the SOTA-models copy got
  deleted (e.g., a previous `collect_fitness.py` cleaned up). Re-run
  `eval_submit.py` to recopy.
- `cifar10/train` not found — phase 1 didn't run, or
  `sota/ExquisiteNetV2/cifar10/` is missing. Check
  `ls sota/ExquisiteNetV2/cifar10`.
- `OOM` — bump `--mem-per-gpu` in
  `src/cfg/constants.py:PYTHON_BASH_SCRIPT_TEMPLATE` (or, more cleanly,
  in `eval_common.build_bash_script` — the wall-time substitution lives
  there too).

### Eval phase queue-starves behind the LLM server

The eval phase does **not** depend on the LLM server. It only needs
training-capable GPUs. If you set up partitions, send the LLM server to
its own partition and let eval jobs flow through the default partition.

### A parent in the dataset has no entry in the cache after `parent_fitness.py`

Re-run with `--force-retrain` to bust the cache, or:

```bash
# inspect the cache
jq . experiments/rag_isolation/_parent_fitness_cache.json
```

If the seed network's results file at
`sota/ExquisiteNetV2/results/network_results.txt` exists but the cache
doesn't reflect it, just run `parent_fitness.py` again — it picks up
existing files without retraining.

### `analyze.py --with-fitness` errors with "results_with_fitness.jsonl missing"

You skipped phase 4. Run `collect_fitness.py` first.

### Fitness numbers look much lower than the baseline

The eval regime is *8 epochs*, not the production 24. Per spec §2 we
accept ~0.5pp test_acc noise as the price for paired comparability.
Compare against the seed's *8-epoch* baseline in
`_parent_fitness_cache.json`, not the 24-epoch number from a previous
run.

## 10. Re-running selectively

The four phases are independent. To re-run a phase without redoing the
others:

| Re-run | Side effects |
|---|---|
| `run_paired_eval.py` | Overwrites `results.jsonl` and `cases/`. You'll need to redo phases 3–5. |
| `eval_submit.py` | Resets `eval_jobs.jsonl` and resubmits. Existing `$RUN_DIR/results/*_results.txt` files are kept; previously-completed genes will be re-detected as `done` by `collect_fitness.py` since the file already exists. |
| `collect_fitness.py` | Idempotent. Rewrites `results_with_fitness.jsonl` from current state. Re-run any time. |
| `analyze.py` | Idempotent. Reads from disk only. |

For a finals-grade run, the typical loop is:

1. Phase 1 once.
2. Phase 2 once.
3. Phase 3 → wait → Phase 4. If some jobs failed (transient cluster
   issues), edit `eval_jobs.jsonl` to mark them `submitted` again, or
   just rerun Phase 3 with `--limit` after manually deleting the
   failed rows from `eval_jobs.jsonl`.
4. Phase 5 once you're satisfied with coverage.

## 11. What to commit when an experiment is done

```
$RUN_DIR/
├── run_metadata.json              ← always
├── results.jsonl                   ← always
├── eval_jobs.jsonl                 ← always
├── results_with_fitness.jsonl      ← always
├── results_with_fitness.csv        ← always
├── paired.csv                      ← always
├── tests.csv                       ← always
├── summary.csv                     ← always
└── report.md                       ← always
```

Don't commit:

- `cases/<case_id>/<trial>/<arm>/network.py` — replicable from
  `results.jsonl` if you really need it; bloats the repo otherwise.
- `eval_scripts/` and `logs/eval/`, `errors/eval/` — large, slurm-
  specific, regeneratable.

`experiments/rag_isolation/_parent_fitness_cache.json` is shared across
runs and *should* be committed so that everyone on the team avoids
retraining the seed.
