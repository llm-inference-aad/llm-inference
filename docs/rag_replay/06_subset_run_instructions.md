# RAG Replay — 30-Row Subset Run Instructions

> Last updated: 2026-05-01
> Audience: The three backend teams (FAISS, PageIndex, Graph) and the integration owner running the shared baseline.
> Companion to: `docs/rag_replay/05_eval_dataset_spec.md` (canonical eval contract).

This doc tells each team exactly what to run for the **6-hour parallel evaluation** on the stratified 30-row subset. It is a derivative of the canonical headline procedure in §5 of `05_eval_dataset_spec.md`; everything not stated here defers to that doc.

---

## 1. The subset dataset

| | |
|---|---|
| Path | `scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv` |
| SHA256 | `93ff252263ee40096ea618feaccd821616bfe2154c85c9cc829c40e7ac126c08` |
| Rows | 30 (all `TEMPLATE_BASED`, all `orig_eligible_for_rag=True`) |
| Fallback / non-fallback | 9 / 21 (preserves the 38/130 ≈ 29% ratio in the full eligible cohort) |
| Sampling | Stratified `random.seed(21)` over eligible cohort; **unique on `(orig_parent_id, orig_prompt_path)`** so each row is a distinct (parent network, prompt) tuple. Sorted by `orig_gene_id`. |
| Source | Stratified subset of `scripts/rag_replay/datasets/past_genes.csv` (sha `89f5449fc0fb6fad18c49327bb51fe2abac12afa8a50e3f505059329cecb6c6c`) |

Verify before kickoff:

```bash
cd /storage/ice1/4/5/satmuri6/llm-inference
sha256sum scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv
# expected: 93ff252263ee40096ea618feaccd821616bfe2154c85c9cc829c40e7ac126c08
```

The 30 referenced prompts live under the existing `scripts/rag_replay/datasets/prompts/` dir — no extra files needed.

---

## 2. Why this size, why these epochs

- **N = 30** preserves the fallback / non-fallback split that drives the recovery vs preservation metrics in §7.2 of the spec. Smaller N (e.g., 10) collapses the fallback cohort below the floor where McNemar / Wilcoxon say anything useful.
- **EPOCHS = 24** matches the headline value used in the full 130-row run (§3.3, frozen control). With the three backend teams running their `with_rag` arms in parallel, per-team queue load is N=30 — not 4N — so we have wall budget to keep the canonical training length. This makes the subset numbers directly comparable to a future full-fleet headline run.
- The shared `no_rag` baseline is run **once** by the integration owner (Step 0). Each backend team then joins against it, exactly as §3.2 of the spec describes.

Rough wall budget per team at ~6 concurrent ICE GPU slots, 8 min/eval at 24 epochs:
30 / 6 × 8 = ~40 min on-GPU + driver overhead. Add baseline wall (~40 min, sequential prereq) and per-team headroom — total ≤ 3 h, fits the 6-hour ceiling with cushion for queue stalls.

---

## 3. Roles

| Role | Who | Owns |
|---|---|---|
| **Integration owner** | One person (rotates) | Step 0: shared `no_rag` baseline. Step 2: cross-team roll-up. |
| **FAISS team** | Worker B | Step 1 with `RAG_BACKEND=faiss`. |
| **PageIndex team** | Worker C | Step 1 with `RAG_BACKEND=pageindex`. |
| **Graph team** | Worker D | Step 1 with `RAG_BACKEND=graph`. |

All four people work in the same repo clone tip on `feature/rag-pipeline-surya` (or a per-team rebase of it) so the backend registry, dataset, and replay driver are byte-identical.

---

## 4. Step 0 — Shared no-RAG baseline (integration owner, one-time)

> **Status (2026-05-01):** Step 0 has run. Baseline at `experiments/rag_replay/baseline_no_rag/`
> (symlink → `baseline_no_rag_20260501_1241`). Headline numbers — used as the
> denominator in every Step 1 paired comparison:
>
> | Metric | no_rag baseline (n=30) |
> |---|---|
> | Goodput (`syntax_valid_first_try`) | **25/30 = 83.3 %** |
> | Trained successfully (`status=done`) | 25 |
> | Train-invalid (LLM produced broken code, `status=failed`) | 5 |
> | LLM-side fallback (exhausted retries, `was_fallback=true`) | 4 |
> | Mean / median `test_acc` over 25 trained | 0.8460 / 0.8526 |
>
> Failure-mode quick read for the 5 train-invalid: 1× class rename
> (`ExquisiteNetV2 → ExquisiteNetV3`), 1× shape mismatch in custom attention,
> 3× `NameError` / `TypeError` from referenced-but-undefined symbols. These
> are real LLM-produced bugs, not harness issues.

This must finish before the three with_rag runs join against it. Plan ~40 min wall.

```bash
cd /storage/ice1/4/5/satmuri6/llm-inference

# Confirm a vLLM server is already up; export its host:port for the driver.
SERVER_URL="$(tail -1 hostname.log):${SERVER_PORT:-8000}"
TS=$(date +%Y%m%d_%H%M)

sbatch --export=ALL,\
SMOKE_DIR=experiments/rag_replay/baseline_no_rag_${TS},\
EPOCHS=24,ELIGIBLE_ONLY=1,\
SERVER_URL=${SERVER_URL},\
EXTRA_ARGS="--csv scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv --skip-arm with_rag" \
    scripts/rag_replay/03_replay.sbatch
```

When complete, copy/symlink the output into the canonical baseline path so the three teams can find it without coordinating timestamps:

```bash
ln -sfn baseline_no_rag_${TS} experiments/rag_replay/baseline_no_rag
```

Sanity check the baseline before announcing:

```bash
jq -r '.arm' experiments/rag_replay/baseline_no_rag/journal.jsonl | sort -u   # only "no_rag"
wc -l experiments/rag_replay/baseline_no_rag/journal.jsonl                     # expect 30
```

If `04_compare.py`'s sanity assertions (§5.3 of the spec) fire here, fix before the three backend teams kick off — every with_rag arm depends on this baseline being clean.

---

## 5. Step 1 — Per-team with_rag run (FAISS, PageIndex, Graph in parallel)

Each team runs one and only one of these, **after** the integration owner posts that the baseline is ready. The three runs are independent; submit them at the same time.

```bash
cd /storage/ice1/4/5/satmuri6/llm-inference

# Substitute your backend name: faiss | pageindex | graph
export RAG_BACKEND=faiss

SERVER_URL="$(tail -1 hostname.log):${SERVER_PORT:-8000}"
TS=$(date +%Y%m%d_%H%M)

sbatch --export=ALL,\
SMOKE_DIR=experiments/rag_replay/${RAG_BACKEND}_subset_n30_${TS},\
EPOCHS=24,ELIGIBLE_ONLY=1,\
RAG_BACKEND=${RAG_BACKEND},\
RAG_DATA_DIR=rag_data_eval,\
RAG_USE_CODE_CONTEXT=false,\
RAG_USE_TEXT_CONTEXT=true,\
RAG_MEMORY_STORE_ENABLED=false,\
SERVER_URL=${SERVER_URL},\
EXTRA_ARGS="--csv scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv --skip-arm no_rag" \
    scripts/rag_replay/03_replay.sbatch
```

`--skip-arm no_rag` is what makes this Step 0's mirror image — only the with_rag arm runs, joined later against the shared baseline. The `RAG_*` exports point retrieval at the curated `rag_data_eval/` corpus (10 PDFs, no code/memory) — see §7.8 for why this round is text-only.

While the driver runs, watch the journal grow:

```bash
RUN_DIR=experiments/rag_replay/${RAG_BACKEND}_subset_n30_${TS}
tail -f ${RUN_DIR}/journal.jsonl | jq -c '{prompt_id, arm, slurm_job_id, syntax_valid_first_try, rag_block_chars}'
```

Per-team self-check before reporting done:

```bash
# Every with_rag row injected something
jq -r 'select(.arm=="with_rag") | .rag_block_chars' ${RUN_DIR}/journal.jsonl | sort -un | head
# expect: lowest value > 0; if you see 0s, your backend silently no-op'd on those prompts
wc -l ${RUN_DIR}/journal.jsonl   # expect 30
```

---

## 6. Step 2 — Roll-up (integration owner, after all three teams finish)

```bash
cd /storage/ice1/4/5/satmuri6/llm-inference

# Per-backend pair tables + headline metrics
for BE in faiss pageindex graph; do
  RUN=$(ls -d experiments/rag_replay/${BE}_subset_n30_* | tail -1)
  python scripts/rag_replay/04_compare.py \
      --baseline experiments/rag_replay/baseline_no_rag \
      --backend  "${RUN}" \
      --out      "${RUN}/report.md"
done

# Cross-backend roll-up (joins per-backend paired.csv on prompt_id)
python scripts/rag_replay/04_compare.py --roll-up \
  experiments/rag_replay/faiss_subset_n30_* \
  experiments/rag_replay/pageindex_subset_n30_* \
  experiments/rag_replay/graph_subset_n30_* \
  --out experiments/rag_replay/rollup_subset_n30.md
```

The roll-up table is the artifact for the writeup; per-backend `report.md` files plug into §7 of the spec for the per-backend narrative.

---

## 7. Open prerequisites before kickoff

Status as of the integration owner's Step 0 launch (2026-05-01):

1. ✅ **`RAG_BACKEND` validation.** `02_rag_service.py` now reads `RAG_BACKEND` (default `faiss`), records it in `run_metadata.json`, and **fails fast at startup** if set to a stub backend (`pageindex`, `graph` raise SystemExit on `RagRuntime` warmup). FAISS team is unblocked today.
2. ✅ **`--skip-arm` flag.** `03_replay.py:362` already supports `--skip-arm no_rag` and `--skip-arm with_rag`. The Step 0 / Step 1 split above relies on this.
3. ✅ **No-RAG warmup skip.** Step 0 (no_rag-only) no longer pays the ~30s embedding-model load and won't trip `RAG_BACKEND` validation — `RagRuntime` is constructed lazily only when `with_rag` is in `arms_to_run`.
4. **Backend status.** PageIndex is implemented on `feature/rag-pipeline-ben` (commit `33735d465` — vendored `src/pageindex/`, tree-search retrieval against the local LLM). The Graph backend on this branch is **scaffolding** (`src/rag/backends/graph_backend.py`): protocol-compliant, returns a structured empty `RetrieveResponse` with `reason="scaffolding"` so a misconfigured run gets a diagnostic block instead of a traceback. The harness still fails fast at `02_rag_service.py` for `RAG_BACKEND=graph` so the eval can't accidentally consume scaffolding output. A worker checklist is in the module docstring; once retrieval is implemented, remove `"graph"` from `_STUB_BACKENDS` and flip `ImplementationStatus.SCAFFOLDING → IMPLEMENTED` to unblock the team.
5. **`hostname.log` location.** The sbatch wrapper falls back to `hostname.log` at repo root; some configs write it under `runs/server-only/logs/`. Symlink before kickoff if needed.
6. **vLLM server up.** All four runs share one vLLM endpoint to keep LLM stochasticity controlled. Confirm `$SERVER_URL` returns 200 on `/` before any team submits.
7. **Sha pinned in tracking issue.** Paste both `89f5449f…6c6c` (full dataset) and `93ff2522…6c08` (subset) into the tracking issue so re-runs are byte-verifiable.
8. ✅ **Eval-time RAG corpus (`rag_data_eval/`) — text-only this round.**

   Status (2026-05-01):
   - **Text namespace: curated to 10 PDFs** (PageIndex-aligned set), 44 chunks. The same set is the source for the PageIndex tree builder, so FAISS-vs-PageIndex compare on identical document coverage.
     - `text.index` sha256 `86ee8bf89f3d15824a823109bce11a6b7688a78b621741d4b6dae0c047608d2c`
     - `text.jsonl` sha256 `7fa241eebffe777256cb71f279c0e02cc3390491c5a243602cf3df482fc52de4`
   - **Code namespace: intentionally empty.** Of the three runs with checkpoints on disk, two are the eval source runs (`nemotron_rag_text_20260311_022245`, `nemotron_baseline_20260311_022419`) and the third (`nemotron_rag_text_20260318_162102`) has only `checkpoint_gen_0.pkl` — seed-only, no mutations to extract. Run-level holdout produces a 0-mutation corpus; including the source runs would leak the eval targets directly. **Step 1 runs text-only RAG.** Code namespace gets revisited once new runs without subset overlap exist.
   - **Memory namespace: empty + `RAG_MEMORY_STORE_ENABLED=false` for the headline.** Memory is a separate ablation, not part of this round.

   The audit trail (`rag_data_eval/holdout_dropped.jsonl`) records the rationale.

   Tooling (built earlier, kept for the next round when leak-free runs exist):
   - `scripts/rag_replay/curate_text_index.py --input rag_data --output rag_data_eval` — re-embeds the whitelist subset of PDFs into the target dir.
   - `scripts/rag_replay/build_eval_code_index.py` — auto-derives excluded runs from the CSV's `orig_run_id`, AST-hashes the 30 target network files via `ast_normalized_hash`, and runs `extract_mutations_from_checkpoints` with both holdouts.
   - `_assert_no_target_leakage` in `scripts/rag_replay/03_replay.py` — runs at driver startup when `with_rag` is scheduled. Skips silently when no code namespace exists (current state); becomes load-bearing once one is rebuilt.

   For Step 1: each team sets `RAG_DATA_DIR=rag_data_eval, RAG_USE_CODE_CONTEXT=false, RAG_USE_TEXT_CONTEXT=true, RAG_MEMORY_STORE_ENABLED=false`.

---

## 8. Failure / restart guidance

- **A single eval `sbatch` dies.** The driver writes `train_invalid:true` for that row and continues. `04_compare.py` excludes those rows from the Wilcoxon comparison but keeps them in goodput (they count as `syntax_valid_first_try=False`).
- **Driver itself OOMs / times out.** It runs as an `ice-cpu` 8h job; if it dies, re-submit with the same `SMOKE_DIR` and the journal will append. The driver is idempotent on `prompt_id × arm` — already-submitted rows are skipped.
- **Backend `augment()` raises mid-run.** The fix in `27714437e` ensures the eval-arm `request_id` is recorded before `augment()` is called, so the two-event JOIN survives. Confirm by joining `runs/<RUN_ID>/rag_ledger.jsonl` on `request_id` after the run.
- **Rate-limited LLM.** Lower the driver's `MAX_ROWS` for a partial run, then resume.

---

## 9. References

- Canonical eval contract: `docs/rag_replay/05_eval_dataset_spec.md`
- Replay overview: `docs/rag_replay/00_overview.md`
- Per-row driver: `scripts/rag_replay/03_replay.py`
- SLURM wrapper: `scripts/rag_replay/03_replay.sbatch`
- Comparator: `scripts/rag_replay/04_compare.py`
- Subset sampler: `scripts/rag_replay/subsample.py` — regenerates this CSV deterministically (`--n 30 --seed 21`); enforces unique `orig_parent_path` and stratifies fallback rows to match the eligible-pool ratio.
- Componentization plan: `docs/plans/05_rag_componentization_plan.md`
