# Ben + Sam — RAG Replay Step 1 Handoff

> Last updated: 2026-05-01
> Branch: `feature/rag-pipeline-surya` @ `1fd5c0f79` (merged Ben's PageIndex
> work into Surya's text-only Step 1 prereqs).
> Companion to `06_subset_run_instructions.md` (canonical runbook).

This is the operational hand-off doc for the PageIndex headline run on the
30-row subset. **Ben** builds the trees (Step 1.A) and announces ready;
**Sam** pulls and submits the replay (Step 1.B). Everything lives on
`feature/rag-pipeline-surya` — no other branch needed.

State of the world right now (read this once before starting):

- vLLM server is up: job `5150277` on `atl1-1-03-014-9-0:8000` (Llama-3.3-Nemotron-Super-49B). Tree-building and replay both depend on it.
- `rag_data_eval/` is seeded with the curated 10-PDF FAISS text namespace (44 chunks). Code/memory namespaces are intentionally empty this round (see §7.8 of `06_subset_run_instructions.md` for why).
- Step 0 baseline (`experiments/rag_replay/baseline_no_rag/`) is locked at 25/30 trained, 83.3% goodput, mean test_acc 0.846. Step 1 numbers are paired against this.
- PageIndex backend is implemented (`src/rag/backends/pageindex_backend.py`); Graph backend is scaffolding-only (worker checklist in module docstring).

---

## Step 1.A — Ben: build curated PageIndex trees

**Goal:** produce ten `*_structure.json` files under `rag_data_eval/pageindex_trees/`, one per PDF in the curated whitelist. Coverage matches the FAISS text namespace exactly so the cross-backend comparison is on equal footing.

### A.1 Sync the branch

```bash
cd /storage/ice1/4/5/satmuri6/llm-inference   # or your worktree of feature/rag-pipeline-surya
git fetch origin
git checkout feature/rag-pipeline-surya
git pull --ff-only
uv sync   # picks up pymupdf added in 1fd5c0f79
```

### A.2 Stage the 10-PDF curated input dir

The tree builder defaults to `rag_corpus/`, which has all 22 PDFs. We want trees for only the 10 PageIndex-aligned PDFs (same set the FAISS text index now covers), so stage a sibling input dir:

```bash
mkdir -p rag_corpus_curated

# The 10-PDF whitelist (mirrors scripts/rag_replay/curate_text_index.py).
KEEP=(
  "4388-Article Text-28466-1-10-20230612.pdf"
  "CIFAR 10 Dataset_ Everything You Need To Know - AskPython.pdf"
  "Cifar-10_Classification_using_Deep_Convolutional_Neural_Network.pdf"
  "Dropout Regularization in Deep Learning - GeeksforGeeks.pdf"
  "LLM_Guided_Evolution___The_Automation_of_Models_Advancing_Models.pdf"
  "The Emerging Science of Machine Learning Benchmarks _ SIAM.pdf"
  "activation-functions.pdf"
  "depthwise-separable-convultions.pdf"
  "efficientNet.pdf"
  "learning-rate-schedules.pdf"
)
for f in "${KEEP[@]}"; do ln -sf "../rag_corpus/$f" "rag_corpus_curated/$f"; done
ls rag_corpus_curated/ | wc -l   # expect 10
```

### A.3 Build the trees

The vLLM server (`atl1-1-03-014-9-0:8000`) needs to be reachable from wherever you run this. PageIndex's tree-builder fires LLM calls per PDF; budget ~5–15 min per tree depending on PDF length.

```bash
# Confirm the server is responding before kickoff
curl -fsS http://atl1-1-03-014-9-0.pace.gatech.edu:8000/ | head -1
# Expected: {"message":"LLM API (vLLM) is running!"}

# Build trees into rag_data_eval/ — same RAG_DATA_DIR Sam will set later
uv run python scripts/build_pageindex_trees.py \
    --corpus-dir rag_corpus_curated \
    --output-dir rag_data_eval/pageindex_trees \
    --model local_server
```

If a tree build dies mid-run, just re-invoke — the script `[skip]`s any tree that already exists. Use `--force` only if you need to rebuild from scratch.

### A.4 Sanity-check before handing off to Sam

```bash
ls rag_data_eval/pageindex_trees/*_structure.json | wc -l   # expect 10

# Static structural check + retrieval smoke (uses the live server)
uv run python scripts/pageindex_sanity_check.py \
    --trees-dir rag_data_eval/pageindex_trees

# A fake-LLM offline run also exists for byte-level reproducibility:
uv run python scripts/pageindex_sanity_check.py \
    --trees-dir rag_data_eval/pageindex_trees --fake-llm
```

The sanity-check exits non-zero if any tree fails the static contract or if the retrieval probe returns no nodes for a known in-domain query. Don't hand off until this is green.

### A.5 Announce ready

Post the tree count + a quick summary in chat. Sam needs:
- Path: `rag_data_eval/pageindex_trees/`
- Count: 10
- Sanity check exit: 0

---

## Step 1.B — Sam: launch the PageIndex replay

**Goal:** run the with_rag arm against the 30-row subset using PageIndex retrieval, generate `journal.jsonl` + 30 result files, post the comparison vs Step 0's no_rag baseline.

### B.1 Sync the branch (same as A.1)

```bash
cd /storage/ice1/4/5/satmuri6/llm-inference   # or your worktree
git fetch origin
git checkout feature/rag-pipeline-surya
git pull --ff-only
uv sync
```

### B.2 Pre-flight checks (don't skip — saves a wasted 6h)

```bash
# 1. vLLM server up?
curl -fsS http://atl1-1-03-014-9-0.pace.gatech.edu:8000/ | head -1
# Expected: {"message":"LLM API (vLLM) is running!"}

# 2. Trees populated?
ls rag_data_eval/pageindex_trees/*_structure.json | wc -l
# Expected: 10

# 3. Curated text index in place?
sha256sum rag_data_eval/metadata/text.jsonl
# Expected: 7fa241eebffe777256cb71f279c0e02cc3390491c5a243602cf3df482fc52de4

# 4. Step 0 baseline reachable through the canonical symlink?
ls experiments/rag_replay/baseline_no_rag/journal.jsonl
wc -l experiments/rag_replay/baseline_no_rag/journal.jsonl   # expect 30

# 5. Subset CSV pinned?
sha256sum scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv
# Expected: 93ff252263ee40096ea618feaccd821616bfe2154c85c9cc829c40e7ac126c08
```

If any of those is wrong, **stop and ping Surya/Ben before submitting** — debugging mid-run wastes GPU hours.

### B.3 Submit the run

```bash
TS=$(date +%Y%m%d_%H%M)
SMOKE_DIR=experiments/rag_replay/pageindex_subset_n30_${TS}
SERVER_URL=atl1-1-03-014-9-0.pace.gatech.edu:8000

sbatch --export=ALL,\
SMOKE_DIR=${SMOKE_DIR},\
EPOCHS=24,ELIGIBLE_ONLY=1,\
RAG_BACKEND=pageindex,\
RAG_DATA_DIR=rag_data_eval,\
RAG_USE_CODE_CONTEXT=false,\
RAG_USE_TEXT_CONTEXT=true,\
RAG_MEMORY_STORE_ENABLED=false,\
SERVER_URL=${SERVER_URL},\
EXTRA_ARGS="--csv scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv --skip-arm no_rag" \
    scripts/rag_replay/03_replay.sbatch

# Capture the driver job id
echo "Driver: $(squeue -u $USER -h --format=%i -n rag_replay_driver | tail -1)"
```

The driver runs on `ice-cpu` (8h budget). It does the 30 LLM calls + 30 sbatch submissions in-process, then `--no-poll` exits and returns. Each eval sbatch is its own GPU job. Plan ~40 min wall on-GPU at 24 epochs / 6 concurrent slots.

### B.4 Monitor while it runs

```bash
DRIVER_LOG=slurm-${DRIVER_JOB}.out      # or `ls -lt slurm-*.out | head -1`
RUN_DIR=${SMOKE_DIR}                    # paste from B.3

# Driver progress
tail -f ${DRIVER_LOG}

# Journal grows one line per (gene × arm)
tail -f ${RUN_DIR}/journal.jsonl | jq -c \
    '{prompt_id:.orig_gene_id, arm, slurm_job_id, syntax_valid_first_try, retrieved_n_text, rag_block_chars}'

# Eval queue depth + states
squeue -u $USER --format="%.10i %.10T %.10M %.20j" | head -15

# Per-row self-check: every with_rag row injected something
jq -r 'select(.arm=="with_rag") | .rag_block_chars' ${RUN_DIR}/journal.jsonl | sort -un | head
# Lowest value should be > 0. Zeros mean PageIndex no-op'd on that row — flag it.
```

### B.5 Poll for results once the driver exits

The driver was launched with `--no-poll` so it returns as soon as all 30 evals are queued. Run a poll pass to fill in `test_acc/params/val_acc/train_time_s` once the eval jobs finish:

```bash
uv run python scripts/rag_replay/03_replay.py \
    --output ${RUN_DIR} --poll-only --poll-timeout-hours 2
```

Eval jobs at 24 epochs ≈ 8 min/job, ~6 concurrent → ~40 min queue drain.

### B.6 Pair vs the Step 0 baseline + report

```bash
uv run python scripts/rag_replay/04_compare.py \
    --baseline experiments/rag_replay/baseline_no_rag \
    --backend  ${RUN_DIR} \
    --out      ${RUN_DIR}/report.md

cat ${RUN_DIR}/report.md
```

Post the headline (goodput delta, McNemar p, recovery_rate / preservation_rate, median Δ test_acc) in chat. The full table format is §7 of `05_eval_dataset_spec.md`.

---

## Cheat-sheet — paths, knobs, and pinned shas

| Thing | Value |
|---|---|
| Branch | `feature/rag-pipeline-surya` @ `1fd5c0f79` |
| Subset CSV | `scripts/rag_replay/datasets/past_genes_subset_n30_seed21.csv` |
| Subset sha | `93ff252263ee40096ea618feaccd821616bfe2154c85c9cc829c40e7ac126c08` |
| Curated text index | `rag_data_eval/metadata/text.jsonl` (44 chunks across 10 PDFs) |
| `text.jsonl` sha | `7fa241eebffe777256cb71f279c0e02cc3390491c5a243602cf3df482fc52de4` |
| `text.index` sha | `86ee8bf89f3d15824a823109bce11a6b7688a78b621741d4b6dae0c047608d2c` |
| PageIndex trees | `rag_data_eval/pageindex_trees/*_structure.json` (Ben writes; Sam consumes) |
| Step 0 baseline | `experiments/rag_replay/baseline_no_rag/` (symlink) |
| Step 0 goodput | 25/30 = 83.3% — denominator for paired comparison |
| vLLM server | `atl1-1-03-014-9-0.pace.gatech.edu:8000` (job `5150277`) |
| Tree builder | `scripts/build_pageindex_trees.py` |
| Tree sanity | `scripts/pageindex_sanity_check.py` (live + `--fake-llm` modes) |
| Replay driver | `scripts/rag_replay/03_replay.py` (poll-only mode supported) |
| Comparator | `scripts/rag_replay/04_compare.py` |

---

## Known gotchas

1. **The default `--output-dir` for `build_pageindex_trees.py` is `rag_data/pageindex_trees/`, not `rag_data_eval/`.** Always pass `--output-dir rag_data_eval/pageindex_trees` explicitly. Trees written to `rag_data/` are *not* picked up by the eval driver because `RAG_DATA_DIR=rag_data_eval` is hard-coded in the sbatch line.

2. **`hostname.log` at repo root** is sometimes stale. The sbatch wrapper uses `--server-url` if you pass it via `--export`; the cheat-sheet above always passes it explicitly. If you skip `SERVER_URL=...`, the driver falls back to `tail -1 hostname.log` which may point at a dead server.

3. **Driver runs `--no-poll` by default** in this configuration, so the journal is initially incomplete (rows 1–14 only). Always run B.5 (poll-only pass) before B.6 — `04_compare.py` needs the populated `test_acc` field.

4. **`--skip-arm no_rag` is required** for Step 1. Without it the driver also runs the no_rag arm, which is wasted work (we already have the baseline) and adds 30 extra eval jobs to the queue.

5. **PageIndex trees are LLM-generated**, so two builds with the same input can produce slightly different node summaries (deterministic up to vLLM stochasticity at temperature 0.3). For paired comparability, **build trees once and check them in** (or symlink to a pinned version) — don't rebuild between FAISS and PageIndex runs.

6. **Graph backend is scaffolding only.** `RAG_BACKEND=graph` will fail fast at driver startup with a pointed error message ("Port the backend before running the replay"). Don't try to launch a Graph run from this branch yet — wait for the implementing worker to flip `ImplementationStatus.SCAFFOLDING → IMPLEMENTED`. The worker checklist is in `src/rag/backends/graph_backend.py` module docstring.

---

## What's NOT ready (so you don't go looking)

- **Code namespace in `rag_data_eval/`** — empty, intentionally. The only run with leak-free history is `nemotron_rag_text_20260318_162102` and it has only `checkpoint_gen_0.pkl` (seed-only, no mutations). Code retrieval is off this round; revisit once new runs without subset overlap exist.
- **Graph backend** — scaffolding, see gotcha #6.
- **Memory namespace** — empty + `RAG_MEMORY_STORE_ENABLED=false` for the headline. Memory is a separate ablation.

---

## Escalation

If anything is off and you're not sure whether to proceed:
- `git log --oneline -10` — see what changed recently
- Ping Surya in chat with the sbatch line you were going to run + the failing pre-flight check output

Don't submit a 6h SLURM job to "see what happens." Pre-flights are cheap; bad runs are expensive.
