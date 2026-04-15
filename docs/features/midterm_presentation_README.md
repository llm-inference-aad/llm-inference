# Speaker Script — Episodic Memory Layer for RAG in LLMGE

Midterm Research Talk · LLMGE Research Lab · Spring 2026

This README contains a full speaker script for a midterm presentation on the episodic memory feature added to the RAG stack in the LLM-guided evolution (LLMGE) pipeline.

---

## Slide 1: Title + Motivation

Approx time: ~2 minutes

Hey everyone - today I want to walk you through a new component I have been building for our LLM-guided evolution pipeline. The core question is simple: does giving the LLM a memory of what worked before help it evolve better architectures?

Quick context: LLMGE treats architectures like genes. The LLM proposes mutations (for example adding layers, changing activations, or modifying structural patterns), and we evaluate each candidate by training/testing and computing fitness.

We already have a RAG system that retrieves relevant code snippets and text notes to provide context during mutation generation. That helps, but each generation is still mostly amnesic. The model has no explicit summary of which mutation patterns historically worked or failed. It can retrieve related code, but it is not directly told, "this class of mutation improved fitness last time."

So the hypothesis is: if we add a compact episodic memory layer - a separate vector store of natural-language summaries of successful prior mutations - and inject that context into prompts, we can bias the LLM toward productive mutation patterns. Expected gains are:

- Better sample efficiency (fewer evaluations to hit target accuracy)
- Potential improvement in best-found performance

---

## Slide 2: Baseline Architecture

Approx time: ~2 minutes

Before showing the new piece, here is the baseline.

The RAG core is orchestrated by `RagRuntime` in `src/rag/runtime.py`. On initialization it creates:

- `EmbeddingService` with separate code and text embedding models
- `VectorStoreManager` in `src/rag/vector_db.py` managing FAISS namespaces

Baseline namespaces:

- `CODE_NAMESPACE` for mutation/model code retrieval
- `TEXT_NAMESPACE` for domain notes and textual context

One important guardrail: startup dimensionality checks verify FAISS index dimensions against current embedding output dimensions. If models changed but indices were not rebuilt, runtime catches this immediately instead of silently returning corrupted retrievals.

Baseline flow:

1. LLMGE requests prompt context.
2. `RetrievalService` fetches code/text docs.
3. `PromptEnhancer` assembles multi-section prompt context.
4. Augmented prompt is sent to the LLM.

---

## Slide 3: + Episodic Memory

Approx time: ~3 minutes

The architecture now includes a third namespace: `MEMORY_NAMESPACE`, alongside code and text, plus a feedback loop from evaluated mutations back into memory.

New component: `MemoryStore` in `src/rag/memory_store.py`.

Design choice: memory entries are natural-language summaries, so they use text embeddings (not code embeddings).

Typical memory content captures mutation type, parent gene context, and outcome summary.

Core API:

- `add_entry(summary, metadata)`:
  - Rejects empty summaries
  - Embeds with text model
  - Stores in `MEMORY_NAMESPACE`
- `search_similar(query, top_k=3, min_similarity=0.3)`:
  - Overfetches with `top_k * 2`
  - Applies similarity threshold
  - Returns top-k typed `MemoryEntry` objects

Write path in `run_improved.py`:

- After child evaluation, if mutation passes quality thresholds, `_log_mutation_result(...)` logs:
  - code document (existing behavior)
  - episodic summary (new behavior) through `runtime.log_memory_entry(...)`

Integration details:

- Runtime now validates dimensions for all 3 namespaces (code, text, memory)
- Entire feature is gated by `RAG_MEMORY_STORE_ENABLED`
- `PromptEnhancer` receives `memory_store` at init
- At prompt time it forms a memory query from mutation type + short code snippet, retrieves top-k memories, and prepends them ahead of text/code sections


---

## Slide 4: Testing Plan (A/B Ablation)

Approx time: ~2 minutes

Evaluation uses a controlled A/B ablation where memory is the only intended variable.

Conditions:

- `memory_off`: RAG enabled, memory store disabled
- `memory_on`: RAG enabled, memory store enabled

Held constant across arms:

- Seeds
- Population size
- Number of generations
- Code/text retrieval options
- Reranker setting

Launcher: `scripts/run_memory_ablation_matrix.py`

What it does:

- Constructs condition matrix (`memory_off` vs `memory_on`)
- Generates `run_id` per seed x condition
- Builds env var exports for each run
- Creates managed run directories (`checkpoints`, `results`, `logs`, `metrics`, `errors`)
- Writes experiment metadata including:
  - `ablation: "memory_store"`
  - condition name
  - matrix ID
  - relevant `RAG_*` settings
- Submits runs through `sbatch run.sh`
- Optionally chains per-run analysis job via dependency (`afterok`)

Per-run analysis:

- `scripts/analyze_rag_impact.py` emits `rag_impact_report.json`

Cross-run summary:

- `scripts/summarize_memory_ablation.py` groups by condition, computes means + 95% CI, and emits CSV/JSON/plots

---

## Slide 5: Metrics, Contributions, and Future Work

Approx time: ~2 minutes

Primary metrics:

1. Best test accuracy per run/condition
   - Definition: the maximum test accuracy observed in that run across all evaluated genes.
   - Why it matters: captures peak solution quality (did memory help find a better architecture at all?).
   - Interpretation: higher is better.
2. Evals to threshold (default threshold = 0.90) for sample efficiency
   - Definition: the first evaluation index where test accuracy reaches or exceeds 0.90 (`evals_to_reach_threshold`).
   - Why it matters: measures search efficiency, not just final quality.
   - Interpretation: lower is better (fewer model evaluations to hit target quality).
3. Average retrieved memory count (sanity check that memory-on is actually active)
   - Definition: mean `retrieved_memory_n` across `rag_context_built` events.
   - Interpretation: should be near zero for `memory_off` and clearly > 0 for `memory_on`.

Diagnostic metrics:

- LLM token usage and latency
- Invalid code rate
- RAG context volume by namespace (code/text/memory)
- Fallback rate

Aggregation:

- Means and 95% CI (normal approximation via `1.96 * SE`) across seeds
- Reporting suggestion:
  - Lead with effect direction and magnitude for both primary metrics, then show CI overlap.
  - Example framing: "memory_on reached the 0.90 threshold in fewer evaluations while maintaining or improving best test accuracy."

Contributions:

- Practical vector-based episodic memory on top of RAG for architecture search
- Text-embedded summaries for compact retrieval and prompt efficiency
- Isolated namespace for independent tuning and future retention policies
- Controlled ablation in a non-trivial NAS setting

Limitations:

- Currently focuses on successful episodes; negative-memory modeling is limited
- No advanced pruning/retention policy yet
- Small initial study size (3 seeds, 5 generations)

Future work:

- Learn context weighting across code/text/memory channels
- Structured key-value memory with richer metadata filters
- Cross-run and cross-task generalization studies
- Curriculum/forgetting mechanisms for long-horizon memory control

---

## Optional Closing Line

One-sentence takeaway:

"This is not just an engineering toggle - it is a testable retrieval-time memory intervention, and the ablation framework lets us isolate its causal effect on search behavior and performance."
