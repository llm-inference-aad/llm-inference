# RAG Isolation — Experiment Plan (Finals)

**Goal:** Use the paired RAG isolation harness to produce statistically
defensible claims about whether RAG improves LLM mutation quality at
the *per-prompt* level. Designed to fit a finals presentation: clear
hypotheses, transparent stats, honest about limitations.

## Research questions

1. **RQ1 — Syntax:** Does RAG reduce the proportion of LLM responses
   that fail Python syntax/module-validation on the first attempt?
2. **RQ2 — Effort:** Does RAG reduce the number of retry attempts
   `augment_network` needs before producing a valid module?
3. **RQ3 — Cost:** What is RAG's cost in prompt-token bloat and
   end-to-end latency per call?
4. **RQ4 — Heterogeneity:** Does RAG's effect vary by mutation
   template (e.g., does it help `Param` more than `Significant`)?

## Hypotheses

| ID | Hypothesis | Test |
|---|---|---|
| H1 | `syntax_valid_first_try` rate is higher with RAG than without | McNemar exact |
| H2 | `module_valid_first_try` rate is higher with RAG | McNemar exact |
| H3 | `fallback` rate is lower with RAG | McNemar exact |
| H4 | `n_attempts` median is lower with RAG | Wilcoxon signed-rank |
| H5 | `prompt_chars` median is higher with RAG (sanity check) | Wilcoxon signed-rank |
| H6 | `llm_latency_s` median is higher with RAG (cost) | Wilcoxon signed-rank |
| H7 | Per-template effect sizes (H1–H4) differ across templates | Per-template breakdown |

H5 and H6 are *expected* to be true — we want to quantify the cost,
not show it's zero.

## Design

**Unit of analysis:** the (case, trial) pair. Each pair contributes
exactly one row per arm.

**Case** = (parent_network, mutation_template, augment_idx) held
constant across both arms.

**Trial** = an LLM stochasticity sample. Same case, different trial =
same prompt, different LLM seed.

**Arm** = `no_rag` | `with_rag`. Differs only in whether the prompt
template is passed through `RagRuntime.enhance_template()`.

Per-pair pinning across arms:
- Same parent, template, augment_idx (forced by reseeding `np.random`
  immediately before each `augment_network` call)
- Same temperature, top_p, max_new_tokens, LLM server
- Arm execution order randomized per trial to avoid time-correlated
  server-side state

## Dataset (final-experiment scale)

We need enough pairs per template for a per-template Wilcoxon to have
non-trivial power. Rule of thumb: **n ≥ 20 pairs per template** for a
medium effect (d ≈ 0.5).

| Template | Source | n_pairs |
|---|---|---|
| `Param` (concise) | seed network | 20 |
| `Significant` (concise) | seed network | 20 |
| `mutant0` (concise) | seed network | 20 |
| `Expert_Complex` (roleplay) | seed network | 20 |
| `Param` (concise) | evolved gene `xXx05py…` | 20 |

**Total: 5 templates × 20 trials × 2 arms = 200 LLM calls.**

At ~3 s/call on the local server, that's ≈10 min of LLM wall time +
~30 min of CPU embedding for the RAG side. On a single GPU node the
embedding cost drops dramatically (sub-second per call), bringing the
full run to **under 20 minutes**.

We can scale up by adding cases (different parents) or templates if
the early signal is weak.

## Metrics (already wired in `core.py`)

Captured per trial per arm:

- `syntax_valid_first_try`, `module_valid_first_try`, `n_attempts`,
  `fallback`, `error_types_per_attempt`
- `prompt_chars`, `response_chars`, `llm_latency_s`
- `retrieved_n_code`, `retrieved_n_text`, `rag_block_chars`
- `parent_changed`, `augment_idx` (sanity), `wall_s_total`

## Analysis plan (already wired in `analyze.py`)

1. **Sanity checks** before any inference:
   - `augment_idx_match` is true for every pair (else discard)
   - `rag_block_chars` is 0 in `no_rag` arm and >0 in `with_rag` arm
   - `retrieved_n_code + retrieved_n_text > 0` in most `with_rag` rows
     (otherwise the RAG corpus is empty for these queries — meaningless
     experiment)

2. **Marginal stats per arm** — means, medians, IQRs.

3. **Paired tests** — McNemar (binary), Wilcoxon (continuous), with
   median Δ and IQR reported alongside p-values.

4. **Per-template breakdown** — does RAG help some templates more
   than others?

5. **Qualitative review** — `report.md` lists pairs where one arm
   produced syntax-valid code and the other didn't. We pick 3–5 of
   these for the slide deck and diff them by hand to ground the
   numbers in real examples.

## Reporting

Produced automatically by `analyze.py`:

- `report.md` — what we'll use as the source of truth
- `paired.csv`, `summary.csv`, `tests.csv` — for ad-hoc plots
- (optional) plot `n_attempts` paired-difference histogram, prompt-size
  violin plot, latency-vs-rag-block-size scatter for the slides

## Threats to validity

| Threat | Mitigation |
|---|---|
| LLM stochasticity dominates the signal | n=20/template provides decent power; report effect sizes alongside p-values |
| `augment_idx` drifts between arms (bug) | Sanity-check column emitted; analyze.py will flag mismatches |
| Empty/irrelevant RAG retrievals | Pre-flight check on retrieved_n in arm summary |
| Server-side caching warms the second arm | Randomize per-trial arm order |
| One template's seed code pathologically helps/hurts RAG | Per-template breakdown surfaces this |
| Multiple-comparison inflation across hypotheses | We pre-register H1–H6 here; per-template tests are exploratory |

## What we won't claim

- That RAG improves *trained model accuracy*. Our cheap metrics don't
  measure that. We have a designed-in `--full-eval` extension for it
  but it's not wired for ship-1.
- That these results generalize to other LLMs / corpora / SOTA seeds
  — this is a single configuration.
- That RAG is "worth it" in a cost/benefit sense without weighting
  prompt-token cost vs the syntax-error reduction.

## Timeline / deliverables for finals

| Deliverable | Source |
|---|---|
| Slide: research question + harness diagram | `docs/rag_isolation/00_summary.md` |
| Slide: hypotheses + statistical methodology | this doc |
| Slide: headline numbers (H1–H6 results) | `experiments/.../report.md` |
| Slide: per-template breakdown bar chart | `tests.csv` + plotting |
| Slide: 3 qualitative example diffs | `cases/<id>/<trial>/{no_rag,with_rag}/network.py` |
| Slide: limitations + future work | "Threats to validity" + "What we won't claim" |

## Reproduce in one command

```bash
sbatch scripts/rag_isolation/run_harness.sbatch \
  DATASET=scripts/rag_isolation/datasets/finals.json \
  OUTPUT=experiments/rag_isolation/finals_$(date +%Y%m%d)
```

(`finals.json` = the 5×20 dataset described above; build it with the
same schema as `small_validation.json` once we commit to the design.)
