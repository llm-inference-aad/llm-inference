# 03 — Metric definitions

`scripts/rag_replay/04_compare.py` joins the two arms of `journal.jsonl` by
`orig_gene_id` and emits `paired.csv` + `report.md`. This page is the
authoritative definition of every column and headline number.

## Definitions

Let `P_norag` and `P_rag` be the two regenerated children for the same source
gene. Both arms attempt the same parent network with the same templates; only
the `with_rag` arm prepends the RAG context block.

### Cheap (always recorded)

| Symbol | CSV column | Definition |
|---|---|---|
| `n_attempts` | `<arm>_n_attempts` | Iterations of the outer retry loop in `_generate_for_arm`. 1 means first try valid. |
| `was_fallback` | `<arm>_was_fallback` | True if all retries exhausted; `.fallback` marker written. |
| `syntax_valid_first_try` | `<arm>_syntax_valid_first` | `n_attempts == 1 and not was_fallback`. The goodput primitive. |
| `prompt_chars` | `<arm>_prompt_chars` | Length of the assembled prompt as sent to the LLM. RAG arm should be larger than no_rag arm by `rag_block_chars`. |
| `rag_block_chars` | `rag_block_chars` | `len(augmented) - len(raw)`; only present for the with_rag arm. |
| `retrieved_n_code` | `rag_retrieved_n_code` | Mutations returned by `RagRuntime.enhance_template`. |
| `retrieved_n_text` | `rag_retrieved_n_text` | Text-namespace chunks returned. |
| `llm_wall_s` | `<arm>_llm_wall_s` | Seconds spent in `_generate_for_arm` (LLM submit + validation + any retries). |

### Expensive (filled by polling SLURM results)

| Symbol | CSV column | Source |
|---|---|---|
| `test_acc` | `<arm>_test_acc` | `train.py` writes to `<RUN_DIR>/results/<gid>_results.txt`. |
| `params` | `<arm>_params` | Total trainable parameter count. |
| `train_time_s` | `<arm>_train_time_s` | Wall time inside `train.py`. |

## Headline metrics

### Goodput per arm

```
goodput_no_rag  = mean(norag_syntax_valid_first)
goodput_with_rag = mean(rag_syntax_valid_first)
delta_goodput   = goodput_with_rag - goodput_no_rag
```

`McNemar exact` is run on the 2×2 paired contingency table
`(norag_success, with_rag_success)`. Counts emitted in
`report.md`:

| | with_rag valid | with_rag fail |
|---|---|---|
| no_rag valid | `both_success` | `only_no_rag` |
| no_rag fail  | `only_with_rag` | `both_fail` |

McNemar tests whether the off-diagonal cells are symmetric (i.e., whether RAG
flipped failures to successes more often than vice versa).

### Fallback sub-cohort

The user's "goodput per LLMGE generation" framing decomposes into two ratios,
both conditioned on the **original** gene's outcome:

```
recovery_rate    = mean(rag_syntax_valid_first | orig_was_fallback == True)
preservation_rate = mean(rag_syntax_valid_first | orig_was_fallback == False)
```

`recovery_rate` answers *"did RAG rescue the prompts that originally
fell back?"*; `preservation_rate` answers *"did RAG keep the prompts that
already worked?"* (a value < 1 means RAG sometimes broke things that worked).

### Accuracy delta

For rows where both arms successfully trained:

```
deltas = [rag_test_acc - norag_test_acc for each paired row]
report median(deltas), IQR, paired Wilcoxon signed-rank
```

Effect size (median delta + IQR) is reported alongside the p-value because
N≈130 paired rows is enough for Wilcoxon to be significant on tiny effects;
the IQR communicates whether the lift is consistent.

### Cost / size deltas

Three additional Wilcoxon tests, all on `(no_rag, with_rag)` pairs:

| Metric | Expected sign | Why |
|---|---|---|
| `prompt_chars` | with_rag larger | sanity check — confirms RAG actually inflated the prompt |
| `llm_wall_s` | with_rag larger | longer prompts → longer LLM forward pass |
| `train_time_s` | mixed | RAG might bias toward heavier or lighter architectures |

## Sanity checks emitted in `report.md`

- `with_rag` rows where `rag_block_chars <= 0` — should be ≈0; non-zero count
  flags genes where the RAG corpus had no relevant retrieval and the with_rag
  arm degenerated to the no_rag prompt. These rows are still counted in the
  paired table but the cohort summary surfaces them so we can re-run the
  analysis with them filtered out.
- scipy/statsmodels availability — the report degrades gracefully when not
  installed and prints raw counts only.

## Reproducible install

```bash
uv add scipy statsmodels  # for p-values
```

Both are already in the lockfile (verified via `uv pip list`).

## Reading the report from the notebook side

`report.md` has six sections (in order):

1. Headline goodput table + delta
2. McNemar tests (`syntax_valid_first_try`, `was_fallback`)
3. Fallback sub-cohort (recovery + preservation)
4. Accuracy delta (paired Wilcoxon)
5. Cost / size deltas
6. Sanity assertions

Drop sections 1, 3, 4 verbatim into the research notebook; sections 2, 5, 6 are
diagnostics and live in the appendix.
