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
SERVER_URL=${SERVER_URL},\
EXTRA_ARGS="--csv scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv --skip-arm no_rag" \
    scripts/rag_replay/03_replay.sbatch
```

`--skip-arm no_rag` is what makes this Step 0's mirror image — only the with_rag arm runs, joined later against the shared baseline.

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
4. ⚠️ **PageIndex / Graph backend ports are still required** before those teams' Step 1 runs will actually retrieve anything. Today both raise `NotImplementedError` in `src/rag/backends/{pageindex,graph}_backend.py`, and the harness fails fast with a pointed error message ("port the backend before running the replay"). The FAISS team can ignore this; the other two teams must implement their backend and wire it into `RagRuntime` before kickoff.
5. **`hostname.log` location.** The sbatch wrapper falls back to `hostname.log` at repo root; some configs write it under `runs/server-only/logs/`. Symlink before kickoff if needed.
6. **vLLM server up.** All four runs share one vLLM endpoint to keep LLM stochasticity controlled. Confirm `$SERVER_URL` returns 200 on `/` before any team submits.
7. **Sha pinned in tracking issue.** Paste both `89f5449f…6c6c` (full dataset) and `93ff2522…6c08` (subset) into the tracking issue so re-runs are byte-verifiable.

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
- Componentization plan: `docs/plans/05_rag_componentization_plan.md`
