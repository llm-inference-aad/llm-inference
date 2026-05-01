# scripts/rag_replay/

Replay-with-RAG harness — re-evaluate every historical RAG-OFF gene with both
arms (no_rag, with_rag) and emit a paired comparison.

See `docs/rag_replay/00_overview.md` for the full design + diagrams. This
README is the operator-facing quickstart.

## What's here

| File | Purpose |
|---|---|
| `01_aggregate.py` | Walk `runs/`, build `datasets/past_genes.csv` and `datasets/prompts/<gid>.txt`. |
| `02_rag_service.py` | `augment_via_rag(req)` — wraps `RagRuntime.enhance_template` in HTTP-ready shape. Standalone smoke: `python 02_rag_service.py --template some_prompt.txt`. |
| `03_replay.py` | The for-loop driver. Regenerates both arms, sbatch-trains each, journals + polls. |
| `04_compare.py` | Reads journal.jsonl → paired.csv + report.md (goodput, recovery, accuracy delta, cost delta, McNemar + Wilcoxon). |

## Run

```bash
# 0. Make sure an LLM server is up
sbatch server.sh
# wait until hostname.log is populated

# 1. Build the source CSV (~30 sec)
.venv/bin/python scripts/rag_replay/01_aggregate.py

# 2. Smoke-test the replay loop (3 source genes × 2 arms = 6 sbatch jobs)
.venv/bin/python scripts/rag_replay/03_replay.py \
  --output experiments/rag_replay/smoke_$(date +%Y%m%d_%H%M) \
  --max-rows 3 --epochs 5 --eligible-only

# 3. Once jobs finish (or use --no-poll + --poll-only later)
.venv/bin/python scripts/rag_replay/04_compare.py \
  experiments/rag_replay/smoke_<...>

# 4. Full replay
.venv/bin/python scripts/rag_replay/03_replay.py \
  --output experiments/rag_replay/full_$(date +%Y%m%d_%H%M) \
  --eligible-only --epochs 24 --poll-timeout-hours 12
```

## Key flags on `03_replay.py`

| Flag | Effect |
|---|---|
| `--max-rows N` | Smoke mode — stop after N source genes. |
| `--eligible-only` | Filter to TEMPLATE_BASED mutations (drops crossover + initial creation). |
| `--skip-arm {no_rag\|with_rag}` | Run only one arm. Useful for resuming. |
| `--epochs N` | Train epochs per gene. Default `EPOCHS` env (24). |
| `--no-poll` | Submit and exit immediately. Re-run with `--poll-only` later. |
| `--poll-only` | Skip submission, just poll for results in an existing run dir. |
| `--server-url host:port` | Override `hostname.log`. |

## Output layout

```
experiments/rag_replay/<run_id>/
├── run_metadata.json        # config snapshot
├── hostname.log             # picked up by submit helpers
├── journal.jsonl            # one line per (orig_gene_id, arm) — overwritten on poll
├── paired.csv               # written by 04_compare.py
├── report.md                # written by 04_compare.py
├── results/                 # train.py writes <new_gid>_results.txt here
├── slurm_logs/              # SLURM stdout
├── slurm_errors/            # SLURM stderr
├── sbatch/                  # rendered .sh per job
└── logs/                    # per-gene LLM logs + validation_errors.csv
```

## Caveats

1. **Augment-idx inference** is heuristic. See
   `docs/rag_replay/02_replay_loop.md#augment-idx-inference`.
2. **Mutation type granularity** is `TEMPLATE_BASED` (no specific template).
   Per-template stratification deferred.
3. **Training stochasticity** is uncontrolled — both arms use seed 21 but
   data shuffling adds variance. Report median + IQR alongside p-values.
4. **FAISS index leakage** — the index was built from runs that may include
   the same source genes. RAG could "retrieve the gene we're regenerating".
   Future work: rebuild the index excluding `nemotron_baseline_*` and re-run
   to bound the leakage effect.
