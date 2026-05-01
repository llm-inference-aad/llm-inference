# RAG Replay Eval — Dataset Spec, Methodology, and Worker Instructions

> Last updated: 2026-04-29
> Audience: Workers B–F implementing RAG backends behind the componentization plan (`docs/plans/05_rag_componentization_plan.md`).
> Status: This is the canonical eval contract. Pin the dataset sha before kickoff; do not regenerate it.

This document specifies the test suite that every RAG backend (FAISS, PageIndex, Graph, plus the no-RAG baseline) is evaluated against for the final stand-up. It defines what the dataset is, what each worker must consume and produce, how to run the experiment, and how the headline numbers are computed.

The goal is one-to-one comparable numbers across backends, with no per-team reconstruction.

---

## 1. Experiment goal

Quantify whether RAG-augmented mutation prompts produce measurably better neural-network mutations than the baseline (no-RAG) prompts on **the actual genes the team produced**, not synthetic test cases. The synthetic harness in `scripts/rag_isolation/` already covers 5 hand-curated cases × 3 trials; this replay covers 156 historical genes × 2 arms (or × N backends).

Three deliverables per backend:

1. A paired `journal.jsonl` over the eval set.
2. A `paired.csv` joining the no-RAG baseline arm to the backend's RAG arm by source gene.
3. A `report.md` with headline numbers (see §7).

The cross-backend comparison is then a join of those `paired.csv` files on `prompt_id`.

---

## 2. The eval dataset

### 2.1 Files (frozen)

```
scripts/rag_replay/datasets/
├── past_genes.csv                 # 156 rows + 1 header
└── prompts/<orig_gene_id>.txt     # 156 prompt files
```

Pin via:

```bash
sha256sum scripts/rag_replay/datasets/past_genes.csv
ls scripts/rag_replay/datasets/prompts/ | wc -l   # expect 156
```

Workers must verify the sha matches the one published in this doc's tracking issue before running any eval. If a worker regenerates the dataset, their numbers do not aggregate.

### 2.2 Schema

`past_genes.csv` columns (CSV with header):

| Column | Type | Role |
|---|---|---|
| `orig_gene_id` | str | **Stable prompt_id.** Use as the row key in journals and reports. |
| `orig_run_id` | str | Source run (e.g., `nemotron_baseline_20260311_022419`). Bookkeeping only. |
| `orig_rag_enabled` | str | `"false"` or null — every row in this set was originally RAG-OFF. |
| `orig_mutation_op` | str | `TEMPLATE_BASED`, `LAYER`, `NEW_LAYER`, `ATTENTION_REPLACE`, … |
| `orig_eligible_for_rag` | bool | Whether RAG augment is actually wired for this op. **Headline subset filter.** |
| `orig_parent_id` | str | DEAP ancestry GENES[-2]. Resolves to the parent network file. |
| `orig_parent_path` | str | `sota/ExquisiteNetV2/models/network_<parent_id>.py`. **Backend input.** |
| `orig_was_fallback` | bool | Did the original LLM exhaust retries and fall back to parent? **Recovery cohort filter.** |
| `orig_test_acc` | float | Historical eval. Bookkeeping only — do not condition on this. |
| `orig_params` | int | Historical eval. Bookkeeping only. |
| `orig_val_acc` | float | Historical eval. Bookkeeping only. |
| `orig_train_time_s` | float | Historical eval. Bookkeeping only. |
| `orig_prompt_path` | str | Already-substituted prompt the original LLM saw. **Backend input.** |
| `orig_prompt_chars` | int | Length of the prompt file. Sanity check. |

The four "historical eval" columns describe what happened the first time these genes were generated — they are useful in the writeup as context but **must not** be fed into the backend's augment call. Backends only see `orig_prompt_path`, `orig_parent_path`, `orig_mutation_op`, `orig_gene_id` per row.

### 2.3 Cohort splits

The full set is `N=156`. The eval reports two cohorts:

**Headline subset — `orig_eligible_for_rag=True` (130 rows).** The prompts where RAG is structurally wired in. Running RAG against the other 26 is a no-op (the augment call returns the template unchanged), so they dilute the headline. Backends report headline numbers on this subset.

**Fallback sub-cohort — `orig_was_fallback=True` (39 rows, intersected with eligible).** Prompts that failed for the LLM on the original RAG-OFF run (exhausted retries → wrote a `.fallback` marker, gene filled by copying parent). Two metrics derived from this split:

- `recovery_rate = mean(syntax_valid_first_try | with_rag, orig_was_fallback=True)` — RAG's save percentage on previously broken prompts.
- `preservation_rate = mean(syntax_valid_first_try | with_rag, orig_was_fallback=False)` — does RAG break previously-working prompts?

Reporting both prevents a flat goodput number from masking "rescues 80% of fallbacks but breaks 5% of working prompts" — those are very different products and the reviewer cares about both.

The remaining 26 non-eligible rows are kept in the CSV for total-fleet sanity (do non-RAG-eligible mutations still produce valid networks?) but are excluded from the headline.

---

## 3. Methodology

### 3.1 Per-row procedure

For each of the 130 eligible source genes, every backend runs **two arms**:

1. **`no_rag` arm.** Send `orig_prompt_path` to the LLM unchanged. Splice output into `orig_parent_path`. Submit SLURM train job.
2. **`with_rag` arm.** Pass `(orig_prompt_path, orig_parent_path, orig_mutation_op, orig_gene_id)` through the backend's `augment()`. Send the augmented prompt to the LLM. Splice output into `orig_parent_path`. Submit SLURM train job.

Both arms use identical LLM sampling, identical retry/validate/fallback logic, and identical training configuration. The only variable per row is the augment call — and the only variable across backends is which `RagBackend` adapter is wired into that call.

### 3.2 The shared `no_rag` baseline

To avoid every team regenerating the same baseline, we run the no-RAG arm **once** on the integration branch and check the resulting `journal.jsonl` + result files into `experiments/rag_replay/baseline_no_rag/`, pinned by sha. Each backend's `with_rag` arm joins against this baseline by `prompt_id`.

This saves ~130 sbatches per worker and removes a source of variance (LLM stochasticity on the no-RAG side).

### 3.3 Frozen controls

Non-negotiable across backends — different choices break the pairing.

| Knob | Value | Source |
|---|---|---|
| LLM endpoint | shared vLLM (Llama-3.3-Nemotron-49B) | `hostname.log` + `$SERVER_PORT` |
| Sampling | `top_p=0.95, temperature=0.3` | `src/cfg/constants.py` |
| Retries | `LLM_GENERATION_MAX_RETRIES=3` | `src/cfg/constants.py` |
| Splice + retry loop | `generate_augmented_code` from `src/llm_utils.py` | do not roll your own |
| Train script | `sota/ExquisiteNetV2/train.py -bs 216 -end_lr 0.001 -seed 21 -val_r 0.2 -amp -epoch 24 -data cifar10` | |
| SLURM resources | per `PYTHON_BASH_SCRIPT_TEMPLATE` | `src/cfg/constants.py:160` |

Backends supply only the augment call; everything else is held constant.

### 3.4 Backend protocol

All backends implement (signature aligns with `src/rag/api_types.py` once Worker A lands it):

```python
class RagBackend(Protocol):
    name: str  # "faiss" | "pageindex" | "graph"

    def augment(
        self,
        template: str,
        mutation_type: str | None,
        query_code: str | None,
        gene_id: str | None,
    ) -> AugmentResponse: ...
```

`AugmentResponse` (current location: `scripts/rag_replay/02_rag_service.py:8`):

```python
@dataclass
class AugmentResponse:
    augmented_template: str
    retrieved_n_code: int
    retrieved_n_text: int
    rag_block_chars: int
```

Once `src/rag/api_types.py` lands, the canonical home is there. Backends should import from `src.rag` rather than from the replay script.

The replay driver selects the backend via a `RAG_BACKEND` env var read at startup. The driver caches the runtime instance once, so the heavy embedding-model load is paid once per replay.

---

## 4. What each worker produces

One output dir per backend:

```
experiments/rag_replay/<backend>_<ts>/
├── journal.jsonl                  # one line per (orig_gene_id, arm)
├── run_metadata.json              # backend name, git sha, dataset sha, .env snapshot
├── sbatch/<new_gid>.sh            # rendered SLURM scripts (debug)
├── logs/llm/gene_<new_gid>.log    # per-call LLM transcript
├── slurm_logs/                    # eval-<jobid>.out
└── slurm_errors/                  # eval-<jobid>.err
```

Trained results land at `sota/ExquisiteNetV2/experiments/rag_replay/<backend>_<ts>/results/<new_gid>_results.txt` (relative `RUN_DIR` + `train.py` auto-chdir; addressed in next driver pass).

### 4.1 `journal.jsonl` line format (required keys)

```json
{
  "prompt_id": "xXx52Np9eh4KcVZOyUZBLN7el8n",
  "backend": "faiss",
  "arm": "with_rag",
  "new_gene_id": "xXxw1wUnCG0f8HuF",
  "n_attempts": 1,
  "syntax_valid_first_try": true,
  "was_fallback": false,
  "rag_block_chars": 6694,
  "retrieved_n_code": 3,
  "retrieved_n_text": 2,
  "augment_latency_ms": 21.4,
  "llm_wall_s": 22.5,
  "prompt_chars": 9434,
  "slurm_job_id": "5097422",
  "test_acc": null,
  "params": null,
  "val_acc": null,
  "train_time_s": null,
  "queued_at": "2026-04-29T19:31:02.531747+00:00"
}
```

Fields populate in two passes: lines 1–14 at queue time, lines 15–18 via the poll loop after SLURM completes. Set `train_invalid: true` if a job fails before producing a results file.

---

## 5. Running the eval

### 5.1 Smoke (3 prompts, 5 epochs) — verifies plumbing

```bash
cd /storage/ice1/4/5/satmuri6/llm-inference

# Verify dataset intact
sha256sum scripts/rag_replay/datasets/past_genes.csv
ls scripts/rag_replay/datasets/prompts/ | wc -l   # expect 156

# Submit smoke replay
sbatch --export=ALL,\
SMOKE_DIR=experiments/rag_replay/smoke_${USER}_$(date +%Y%m%d_%H%M),\
MAX_ROWS=3,EPOCHS=5,\
RAG_BACKEND=faiss,\
SERVER_URL=$(tail -1 hostname.log):$SERVER_PORT \
    scripts/rag_replay/03_replay.sbatch
```

Expected outcome: 6 sbatches (3 source × 2 arms), 6 result files within ~30 minutes wall (depending on GPU queue), `journal.jsonl` with 6 lines and matching `test_acc`s populated.

### 5.2 Headline run (130 eligible × 2 arms)

```bash
sbatch --export=ALL,\
SMOKE_DIR=experiments/rag_replay/${RAG_BACKEND}_$(date +%Y%m%d_%H%M),\
EPOCHS=24,ELIGIBLE_ONLY=1,\
RAG_BACKEND=faiss,\
SERVER_URL=$(tail -1 hostname.log):$SERVER_PORT \
    scripts/rag_replay/03_replay.sbatch
```

Expected outcome: ~260 sbatches queued, ~6–8 min each, total wall governed by GPU queue depth. Start overnight. The driver itself runs as an `ice-cpu` SLURM job for the full 4-hour budget so it survives login-node watchdogs.

### 5.3 Sanity assertions (built into `04_compare.py`)

After each run, `04_compare.py` fails loudly if any of these is violated:

- Every `with_rag` row has `rag_block_chars > 0` (RAG actually injected something).
- Every `no_rag` row has `rag_block_chars == 0` (baseline truly is baseline).
- Each pair has matching `orig_mutation_op` between arms.
- The dataset sha at run time matches the pinned sha.

If a backend's RAG arm has `rag_block_chars==0` for some rows, that backend silently no-op'd on those prompts — the report should call this out.

---

## 6. Inspecting individual cases (rescue stories for the writeup)

For one-page narrative evidence, pick a fallback gene where the with_rag arm was valid first try and walk through the artifacts.

### 6.1 The prompt differential

```
scripts/rag_replay/datasets/prompts/<prompt_id>.txt           # baseline prompt
experiments/rag_replay/<run>/logs/llm/gene_<no_rag_gid>.log   # baseline arm transcript
experiments/rag_replay/<run>/logs/llm/gene_<with_rag_gid>.log # RAG arm transcript
```

The two `[PROMPT TO LLM]` blocks differ by exactly the RAG prefix block. Diff to confirm:

```bash
diff <(awk '/\[PROMPT TO LLM\]/,/\[END PROMPT\]/' logs/llm/gene_<no_rag_gid>.log) \
     <(awk '/\[PROMPT TO LLM\]/,/\[END PROMPT\]/' logs/llm/gene_<with_rag_gid>.log)
```

### 6.2 The network differential

```
sota/ExquisiteNetV2/models/network_<orig_parent_id>.py   # parent (the unchanged starting point)
sota/ExquisiteNetV2/models/network_<no_rag_gid>.py       # baseline arm output (or .fallback if it failed)
sota/ExquisiteNetV2/models/network_<with_rag_gid>.py     # RAG arm output
```

Most legible reading: diff each arm against the parent rather than against each other — the changes the LLM made stand out cleanly.

```bash
diff sota/ExquisiteNetV2/models/network_<orig_parent_id>.py \
     sota/ExquisiteNetV2/models/network_<no_rag_gid>.py
diff sota/ExquisiteNetV2/models/network_<orig_parent_id>.py \
     sota/ExquisiteNetV2/models/network_<with_rag_gid>.py
```

### 6.3 The result differential

```
sota/ExquisiteNetV2/experiments/rag_replay/<run>/results/<no_rag_gid>_results.txt
sota/ExquisiteNetV2/experiments/rag_replay/<run>/results/<with_rag_gid>_results.txt
```

Each is a single comma-separated line: `test_acc,params,val_acc,train_time_s`.

For the writeup pick one fallback rescue and lay these three differentials side by side: that's the most concrete single piece of evidence the experiment produces.

---

## 7. Headline metrics

`04_compare.py` emits:

### 7.1 Goodput

```
goodput_no_rag    = mean(syntax_valid_first_try | arm=no_rag)
goodput_with_rag  = mean(syntax_valid_first_try | arm=with_rag)
delta_goodput     = goodput_with_rag - goodput_no_rag
```

Significance: paired McNemar exact (scipy.stats.binomtest) on the 2×2 table of agreement/disagreement across arms. p-value reported.

### 7.2 Recovery and preservation (fallback split)

```
recovery_rate     = mean(syntax_valid_first_try | with_rag, orig_was_fallback=True)
preservation_rate = mean(syntax_valid_first_try | with_rag, orig_was_fallback=False)
```

Plus the same metrics for `no_rag` regen and the historical original arm, so the table reads:

| Cohort | original | no_rag regen | with_rag (this backend) |
|---|---|---|---|
| All eligible | … | … | … |
| Was fallback | 0.0 | … | … |
| Was not fallback | 1.0 | … | … |

### 7.3 Accuracy delta

Paired Wilcoxon signed-rank on `test_acc` across rows where both arms produced a trained model. Median Δ + IQR. Direction matters more than significance at this N.

### 7.4 Cost delta

Median Δ on `prompt_chars` (sanity — RAG should add chars), `augment_latency_ms` (cost), `llm_wall_s` (LLM-side cost), `train_time_s` (sanity — should be ~unchanged).

### 7.5 Cross-backend roll-up

Once each backend produces a `paired.csv` in its own dir, `04_compare.py --roll-up <dir1> <dir2> …` joins them on `prompt_id` and emits a single table:

| prompt_id | no_rag.test_acc | faiss.test_acc | pageindex.test_acc | graph.test_acc |
|---|---|---|---|---|

with marginal headline rows at the bottom. That table is the final stand-up artifact.

---

## 8. From experiment to analysis — what the writeup answers

Each backend's `report.md` is one page of these questions:

1. **Does this backend improve goodput overall?** (delta_goodput, McNemar p)
2. **Does it rescue previously-broken prompts?** (recovery_rate, paired against the original-arm fallback rate of 0.0)
3. **Does it leave working prompts alone?** (preservation_rate, paired against the original-arm non-fallback rate of 1.0)
4. **Does it improve accuracy when both arms train?** (Wilcoxon median Δ test_acc, IQR)
5. **What does it cost?** (Δ chars / latency / LLM wall)
6. **One concrete rescue story** with the three differentials from §6.

The cross-backend roll-up adds:

7. **Which backend wins on each axis?** Tabulated, not narrative.
8. **Where do they disagree per-prompt?** Pareto-front-style scatter of `test_acc` deltas, colored by backend.

---

## 9. Open issues at time of writing

- **`RUN_DIR` resolution.** Train results land under `sota/ExquisiteNetV2/experiments/rag_replay/<run>/results/` rather than the run dir at repo root, because `train.py` auto-chdirs. Next driver pass: pass absolute `RUN_DIR` through the sbatch `--export`. Tracked separately; does not affect the data shape, only collection paths.
- **`RAG_BACKEND` plumbing.** The replay driver currently hardwires `augment_via_rag` to the single `RagRuntime` singleton. Wiring `RAG_BACKEND` to a backend registry is the integration step before kickoff.
- **`AugmentResponse` location.** Currently lives at `scripts/rag_replay/02_rag_service.py:8`; will move to `src/rag/api_types.py` once Worker A lands the schema.
- **`hostname.log` location.** Some configurations write it under `runs/server-only/logs/`; the driver currently expects it at repo root. Workaround: copy/symlink before kickoff.

These are addressed before the headline run. The smoke run is sufficient to verify everything else.

---

## 10. References

- Plan: `docs/plans/05_rag_componentization_plan.md`
- Replay overview: `docs/rag_replay/00_overview.md`
- Aggregator: `docs/rag_replay/01_aggregate.md`
- Replay loop: `docs/rag_replay/02_replay_loop.md`
- Metrics: `docs/rag_replay/03_metrics.md`
- Code:
  - `scripts/rag_replay/01_aggregate.py` (builds the dataset)
  - `scripts/rag_replay/02_rag_service.py` (augment shim)
  - `scripts/rag_replay/03_replay.py` (per-row driver)
  - `scripts/rag_replay/03_replay.sbatch` (driver SLURM wrapper)
  - `scripts/rag_replay/04_compare.py` (paired analysis)
