# RAG Subsystem Mapping

**Date:** 2026-04-27  
**Scope:** Retrieval-Augmented Generation module — public API, backends, configuration, lifecycle, evaluation harness.  
**Status:** Mapping complete; reranker class is referenced but undefined (latent feature).

---

## 1. Public API Surface

The RAG subsystem exposes three main entry points via `src/rag/runtime.py`:

### Singleton Runtime Access

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `get_runtime()` | `src/rag/runtime.py:99` | Returns global `RagRuntime` instance (thread-safe lazy singleton); returns `None` if `RAG_ENABLED=false` or on dimension mismatch. |

### Core Operations (via RagRuntime methods)

| Method | File:Line | Signature | Usage |
|--------|-----------|-----------|-------|
| `enhance_template()` | `src/rag/runtime.py:65–77` | `enhance_template(template, mutation_type, query_code, gene_id) -> (str, Sequence[RetrievedMutation])` | Retrieve RAG context, augment template, return modified prompt + retrieved mutations. Called from `src/llm_utils.py:254`. |
| `collect_context()` | `src/rag/runtime.py:86–89` | `collect_context(mutation_type, query_code) -> Sequence[RetrievedMutation]` | Retrieve context without template modification. Called from `src/llm_utils.py:281`. |
| `format_context()` | `src/rag/runtime.py:91–92` | `format_context(mutations) -> str` | Format retrieved mutations as readable text for display/logging. |
| `log_mutation_code()` | `src/rag/runtime.py:79–84` | `log_mutation_code(content, metadata) -> str | None` | Index a new mutation into the vector DB. Returns document_id or None. |

### Lower-level APIs (used internally by RagRuntime)

**RetrievalService** (`src/rag/retrieval.py:46–276`):
- `retrieve_similar_mutations(query_code, top_k, min_similarity) -> List[RetrievedMutation]` (line 90–104)
- `retrieve_similar_text(query, top_k, min_similarity) -> List[RetrievedContext]` (line 126–139)
- `retrieve_high_performers(min_accuracy, max_parameters, limit) -> List[RetrievedMutation]` (line 161–189)
- `retrieve_by_mutation_type(mutation_type, limit) -> List[RetrievedMutation]` (line 191–206)

**PromptEnhancer** (`src/rag/prompt_enhancer.py:45–293`):
- `enhance_template(template, mutation_type, query_code, gene_id) -> (str, List[RetrievedMutation])` (line 187–292)
- `build_context(mutation_type, query_code) -> List[RetrievedMutation]` (line 52–91)
- `build_text_context(query_code, mutation_type) -> List[RetrievedContext]` (line 133–161)

---

## 2. Backends

### Current Architecture: Unified FAISS

The system uses a **single FAISS-based backend** with namespace isolation:

**VectorStoreManager** (`src/rag/vector_db.py:182–216`):
- Manages multiple FAISS indices as "namespaces" (separate index + metadata per namespace).
- Two namespaces currently active:
  - **CODE_NAMESPACE** (`"code"`): Mutation code snippets (CodeBERT embeddings, 768-dim).
  - **TEXT_NAMESPACE** (`"text"`): PDFs and documentation (MiniLM embeddings, 384-dim).

**NamespaceStore** (`src/rag/vector_db.py:52–180`):
- Wraps a single FAISS index (`IndexFlatIP` — inner product for cosine similarity on normalized vectors).
- Persists to:
  - Index file: `rag_data/faiss_index/{namespace}.index`
  - Metadata: `rag_data/metadata/{namespace}.jsonl` (one JSON document per line).
- Thread-safe indexing with locks (`self.lock` at line 60).

### No Pluggable Backend System

**Note**: The code comment at `docs/features/rag_backend_abstraction.md` suggests a future backend abstraction, but it is not implemented. The search space references a non-existent `src/pageindex/` Python module (directory exists but contains no `.py` files). PageIndex retrieval is not wired into RAG retrieval at this time.

### Backend Selection

- Hardcoded: FAISS via `faiss-cpu` (imported `src/rag/vector_db.py:15`).
- No configuration knobs for backend selection; FAISS is mandatory.

---

## 3. Index Lifecycle

### Initialization & Setup

**On first run**, execute:
```bash
python scripts/setup_rag.py \
  --runs-dir runs/ \
  --pdf-dir rag_corpus/ \
  --rag-dir rag_data/
```

**File:Line**: `scripts/setup_rag.py:42–71`

Steps:
1. Extracts mutations from historical run checkpoints: `src/rag/data_ingestion.py:123–186`
   - Scans `runs/auto_*/checkpoints/checkpoint_gen_*.pkl`
   - Loads code from `sota/ExquisiteNetV2/models/network_{gene_id}.py`
   - Computes fitness improvements (accuracy/parameter deltas).
   - Returns `List[MutationRecord]`.

2. Processes PDFs from `rag_corpus/`:
   - `src/rag/data_ingestion.py:189–220`
   - Chunks text to 400 words (configurable); creates `(content, metadata)` dicts with source/chunk_index.

3. Creates `VectorStoreManager` and `RetrievalService`, indexes both mutation and text documents.

### Runtime Indexing

New mutations are automatically logged during evolution if they meet accuracy thresholds:
- **Location**: `src/rag/runtime.py:79–84` via `log_mutation_code(content, metadata)`.
- **Condition**: Content must be non-empty; metadata is indexed as-is.
- **Example usage** (from evolutionary loop): Not yet mapped in this document; refer to evolution agent.

### Disk Layout

```
rag_data/
├── faiss_index/
│   ├── code.index           # FAISS IndexFlatIP for CodeBERT embeddings (625 KB)
│   └── text.index           # FAISS IndexFlatIP for MiniLM embeddings (34 KB)
├── metadata/
│   ├── code.jsonl           # Mutation metadata (1.8 MB; one JSON per line)
│   └── text.jsonl           # Text document metadata (146 KB)
└── (legacy, unused)
    ├── code_records.jsonl
    ├── pageindex_trees/
    └── trees/

rag_corpus/
├── *.pdf                    # Input PDFs for text namespace (3 files, ~2 MB)
└── pytorch.json             # Pre-built PyTorch API corpus (used by eval scripts)
```

### Index Rebuild / Pointing to Existing Index

- **Re-run setup_rag.py** to re-index: it deduplicate by `gene_id` and only indexes new mutations.
- **Point to different index**: Set `RAG_DATA_DIR` environment variable before runtime initialization.
- **Dimension mismatch detection**: `src/rag/runtime.py:39–53` — if FAISS index dimension ≠ embedding model dimension, RAG is soft-disabled with a warning.

---

## 4. Prompt Enhancement Flow

Given a base prompt template and optional mutation context, the exact flow is:

### (A) Query Construction

**PromptEnhancer.enhance_template()** (`src/rag/prompt_enhancer.py:187–292`):
1. **Inputs**: template (string), mutation_type (str | None), query_code (str | None), gene_id (str | None).
2. **Build code context** via `build_context_with_stats()`:
   - If `query_code` is present, retrieve similar mutations: `retrieval.retrieve_similar_mutations_with_stats(query_code, top_k=RAG_TOP_K, min_similarity=RAG_MIN_SIMILARITY)`.
   - If `mutation_type` is present and fewer mutations found, retrieve by type: `retrieve_by_mutation_type()`.
   - If still fewer mutations needed, retrieve high performers: `retrieve_high_performers(min_accuracy=RAG_MIN_ACCURACY, max_parameters=RAG_MAX_PARAMETERS)`.
   - Deduplicate by gene_id; sort by score (descending).

3. **Build text context** via `build_text_context_with_stats()`:
   - Construct a natural-language query: `f"CNN architecture {mutation_type} mutation"` + code snippet (first 10 lines).
   - Search text namespace: `retrieve_similar_text_with_stats(query, top_k=RAG_TEXT_TOP_K, min_similarity=RAG_MIN_SIMILARITY)`.

### (B) Retrieval

**RetrievalService** methods used:

| Method | Namespace | Return Type |
|--------|-----------|-------------|
| `retrieve_similar_mutations_with_stats()` | code | `(List[RetrievedMutation], RetrievalStats)` |
| `retrieve_similar_text_with_stats()` | text | `(List[RetrievedContext], RetrievalStats)` |

Both perform:
1. Embed query (with 500-entry FIFO cache for code embeddings).
2. Search index: `store.search_code()` / `store.search_text()` with `top_k = max(1, requested_k * 2)` (fetch 2× to allow filtering).
3. Filter by `min_similarity` threshold.
4. Return top-k results.

**RetrievalStats** capture:
- `candidate_k`: Number of candidates fetched.
- `returned_k`: Number returned before filtering.
- `filtered_k`: Number passing threshold filter.
- `min_similarity`: Threshold used.

### (C) Optional Reranking

**If RAG_RERANKER_ENABLED=true** (`src/rag/prompt_enhancer.py:203–219`):
- Lazy-load `Reranker()` singleton (line 30–34).
- **Note**: `Reranker` class is imported from `.reranker` (line 19) but **does not exist in the codebase**. This is a latent feature (placeholder).
- If reranker exists, truncate query to ≤20 lines to avoid CPU bottleneck (line 206–210).
- Rerank both mutation and text results.

### (D) Snippet Formatting

**RetrievalService methods**:
- `format_context()` (`src/rag/retrieval.py:211–238`): Formats mutations as bullet-point text with fitness metrics and improvement deltas (ΔAcc, ΔParams).
- `format_text_context()` (`src/rag/retrieval.py:240–253`): Formats text chunks with relevance score and (truncated to ~300 words per chunk to avoid token bloat).

### (E) Injection Point

**Final augmented template** (`src/rag/prompt_enhancer.py:221–244`):
1. If text contexts exist, prepend section: *"The following PyTorch documentation and research context…"*
2. If code mutations exist, prepend section: *"Consider the following historically successful mutations…"*
3. Concatenate sections + original template (space-separated).

Return: `(augmented_template, mutations)`.

### (F) Metrics Recording

**Line 246–290**: Emit a `rag_context_built` event to `utils.rag_metrics.record_metric()`:
- Records run_id, gene_id, mutation_type, template/query hashes.
- Includes retrieval stats, selected doc IDs, context word counts, reranker status.
- Stored in `runs/<run_id>/metrics/rag_metrics.jsonl` or `metrics/rag_metrics.jsonl`.

---

## 5. Configuration / Environment Variables

All RAG configuration is centralized in `src/cfg/constants.py:79–93`.

| Variable | Type | Default | Override | Purpose |
|----------|------|---------|----------|---------|
| `RAG_ENABLED` | bool | `true` | `RAG_ENABLED=false` | Master switch; disables RAG if false. |
| `RAG_DATA_DIR` | str | `ROOT_DIR/rag_data` | `RAG_DATA_DIR=/path/to/index` | Path to FAISS index + metadata. |
| `RAG_CODE_EMBED_MODEL` | str | `microsoft/codebert-base` | `RAG_CODE_EMBED_MODEL=...` | CodeBERT model; 768-dim embeddings. |
| `RAG_TEXT_EMBED_MODEL` | str | `sentence-transformers/all-MiniLM-L6-v2` | `RAG_TEXT_EMBED_MODEL=...` | MiniLM model; 384-dim embeddings. |
| `RAG_TOP_K` | int | `5` | `RAG_TOP_K=10` | Number of code mutations to retrieve. |
| `RAG_TEXT_TOP_K` | int | `3` | `RAG_TEXT_TOP_K=5` | Number of text chunks to retrieve. |
| `RAG_MIN_ACCURACY` | float | `0.9` | `RAG_MIN_ACCURACY=0.93` | Minimum test accuracy to index/retrieve mutations. |
| `RAG_MAX_PARAMETERS` | float | `None` | `RAG_MAX_PARAMETERS=500000` | Optional max parameter count filter. |
| `RAG_MIN_SIMILARITY` | float | `0.3` | `RAG_MIN_SIMILARITY=0.5` | Min cosine similarity score for retrieval. |
| `RAG_USE_CODE_CONTEXT` | bool | `true` | `RAG_USE_CODE_CONTEXT=false` | Toggle code mutation retrieval. |
| `RAG_USE_TEXT_CONTEXT` | bool | `true` | `RAG_USE_TEXT_CONTEXT=false` | Toggle text document retrieval. |
| `RAG_RERANKER_ENABLED` | bool | `false` | `RAG_RERANKER_ENABLED=true` | Enable cross-encoder reranking (unimplemented). |
| `RAG_RERANKER_MODEL` | str | `BAAI/bge-reranker-v2-m3` | `RAG_RERANKER_MODEL=...` | Reranker model name (ignored if class unavailable). |

### Programmatic Override (PromptEnhancerConfig)

`src/rag/prompt_enhancer.py:37–42` allows per-instance config:
```python
config = PromptEnhancerConfig(
    top_k=10,
    text_top_k=5,
    min_accuracy=0.92,
    max_parameters=1000000,
)
enhancer = PromptEnhancer(retrieval_service, config)
```

---

## 6. Existing Eval / Ablation Tooling

### **scripts/run_rag_ablation_matrix.py**

**Purpose**: Launch a factorial matrix of RAG condition experiments.

**File:Line**: `scripts/run_rag_ablation_matrix.py:177–262`

**What it does**:
1. Defines four (or five with reranking) **Condition** objects (`src/rag/run_rag_ablation_matrix.py:123–174`):
   - `baseline`: RAG fully disabled.
   - `code_only`: Code context only (no text).
   - `text_only`: Text context only (no code).
   - `hybrid`: Both code + text (no reranking).
   - (optional) `hybrid_rerank`: Both + reranking enabled.

2. For each condition × seed combination, generates an `sbatch` command with environment variables set.
3. Prints dry-run by default; `--execute` actually submits jobs; `--submit-analysis` queues dependent analysis jobs per run.
4. Metadata is attached to each run's `run_metadata.json` under `experiment.condition`.

**Inputs**:
- `--seeds`: List of experiment seeds (default: 0–9).
- `--num-generations`, `--population-size`, `--start-population-size`: Override evolution hyperparameters.
- `--threshold`: Test accuracy threshold for "time-to-threshold" metric (default: 0.90).
- `--include-hybrid-rerank`: Add reranking condition.

**Outputs**:
- Queues jobs; no direct output file. Each job produces `runs/<RUN_ID>/` directory with results.

**Critical Limitation**: 
This is a **matrix submission tool only**. It does NOT directly produce per-individual A/B metrics. Each run is independent; comparison requires post-hoc aggregation.

---

### **scripts/analyze_rag_impact.py**

**Purpose**: Analyze a single run's results and extract RAG usage metrics.

**File:Line**: `scripts/analyze_rag_impact.py:321–402`

**What it does**:
1. **Load results** from `runs/<RUN_ID>/results/*_results.txt`:
   - Parse gene_id, test_acc, params, val_acc, train_time.
   - Order by file modification time (proxy for generation order).

2. **Compute metrics**:
   - `evals_to_reach_threshold`: First generation where test_acc ≥ threshold (line 126–130).
   - `best_test_acc`, `best_gene_id`: Best accuracy achieved (line 133–141).
   - `invalid_code_gene_rate`: Fraction of genes with LLM syntax errors (line 144–165).
   - `fallback_rate`: Fraction of genes that used fallback code (line 168–189).
   - **Latency**: Token counts, LLM latency percentiles from `latency-*.json` (line 200–247).
   - **RAG usage**: Parse `rag_metrics.jsonl` for retrieval stats (line 250–308):
     - `rag_nonempty_fraction`: Fraction of prompts that retrieved ≥1 context.
     - `avg_retrieved_code_n`, `avg_retrieved_text_n`: Average number of retrieved items per prompt.
     - `avg_context_words_code`, `avg_context_words_text`: Average context length in words.

3. **Output**:
   - JSON report: `runs/<RUN_ID>/metrics/rag_impact_report.json` (line 379–381).
   - Optional CSV row (line 384–401).

**Key limitation**: 
Metrics are **run-level aggregates**; does NOT provide per-gene, per-mutation paired comparisons (e.g., "with vs. without RAG for gene X").

---

### **scripts/summarize_rag_ablation.py**

**Purpose**: Aggregate per-run impact reports across an experiment matrix.

**File:Line**: `scripts/summarize_rag_ablation.py:59–154`

**What it does**:
1. Scans `runs/*/metrics/rag_impact_report.json` (created by analyze_rag_impact.py).
2. Groups reports by `metadata.experiment.condition`.
3. For each condition, computes mean ± 95% CI for:
   - `evals_to_threshold_mean`, `evals_to_threshold_ci95`
   - `best_test_acc_mean`, `best_test_acc_ci95`
   - `total_tokens_mean`, `total_tokens_ci95`
   - `rag_nonempty_fraction_mean`, `rag_nonempty_fraction_ci95`

4. **Outputs**:
   - CSV: `experiments/rag_ablation_summary.csv` (line 129–147).
   - JSON: `experiments/rag_ablation_summary.json` (line 149–151).

**Aggregation**: Condition-level means across multiple seeds. Example output row:
```
condition,n,evals_to_threshold_mean,evals_to_threshold_ci95,…
baseline,10,47.2,5.3,…
hybrid,10,35.8,4.1,…
```

---

### Critical Gap: Per-Individual A/B Metrics

**Summary**: The existing tooling does **NOT** provide per-individual paired-comparison metrics (with vs. without RAG for the same gene).

- `run_rag_ablation_matrix.py`: Launches independent runs per condition; no coupling.
- `analyze_rag_impact.py`: Aggregates to run level (evals to threshold, best accuracy, token counts).
- `summarize_rag_ablation.py`: Aggregates further to condition level (mean ± CI).

**What IS available**:
- **Run-level aggregate metrics**: Time-to-threshold, best accuracy, token cost (compare baseline vs. hybrid runs).
- **RAG usage observability**: Whether/how much RAG was actually used (via `rag_metrics.jsonl`).
- **Condition-level comparison**: e.g., "hybrid had 35.8 evals to threshold vs. baseline's 47.2" (with confidence intervals).

**What is NOT available**:
- Per-gene metrics paired by mutation (e.g., "gene 42 reached 90% accuracy in 12 evals with RAG, 18 evals without").
- Per-prompt retrieval effectiveness (e.g., "how much did the retrieved context help this specific mutation?").

**To implement per-gene paired A/B comparison**, you would need:
1. A new harness that:
   - Extracts per-gene fitness curves from both baseline and RAG-enabled runs.
   - Matches genes across runs (e.g., by mutation type + parent).
   - Computes delta in evals-to-threshold per pair.
   - Aggregates deltas for summary statistics.
2. A modified metric recording that ties RAG usage per prompt to the resulting fitness improvement.

---

## 7. Golden Queries

**File**: `src/rag/golden_queries.json`

**Format**:
```json
{
  "source": "rag_corpus/pytorch.json",
  "tiers": {
    "small": [...],
    "medium": [...]
  }
}
```

**Small tier** (13 hand-curated smoke tests):
```json
{"query": "convolutional layer 2d", "expected": ["torch.nn.Conv2d"]},
{"query": "adam optimizer", "expected": ["torch.optim.Adam"]},
...
```

**Medium tier** (200 auto-generated, deterministic):
- Generated by `scripts/build_golden_queries.py` from `rag_corpus/pytorch.json`.
- Each query is auto-generated from the API name (de-CamelCased + keyword expansion).
- Distributed across PyTorch modules (torch.nn, torch.optim, torchvision, etc.) to avoid overfitting.

**Count**: 
- Small: 13 queries.
- Medium: ~200 queries (configurable via `--medium-limit`).

**Intended use**:
- **Text namespace evaluation** (`src/rag/eval_retrieval.py`): Measure Recall@K, MRR@K for the text embedding model.
- **Chunk size tradeoff evaluation** (`src/rag/eval_chunk_tradeoff.py`): Rebuild text indices with different chunk sizes; evaluate precision/recall/MRR.

**Determinism**: Seed-based shuffling ensures reproducible medium-tier selection across runs.

---

## 8. PageIndex

**Location**: `src/pageindex/pageindex/` (directory exists; no `.py` files).

**Status**: A separate retrieval backend for structured/hierarchical document indexing is referenced in architectural docs (`docs/features/rag_backend_abstraction.md`) but is **not integrated** into the RAG subsystem.

**What it is** (inferred from directory structure):
- Likely a tree-based page-level index (from the `pageindex_trees/` artifact in `rag_data/`).
- Designed for exact-match or hierarchical retrieval of structured PDFs/docs.

**Wiring**: 
- Not wired into `RetrievalService`.
- `VectorStoreManager` does not reference it.
- No environment variable to toggle it on.

**Conclusion**: PageIndex is a reference/stub for future backend pluggability but is dormant. Code mutations and text documents use FAISS only.

---

## 9. Determinism / Cacheability

### Embedding Determinism

**Code embeddings** (CodeBERT):
- Same query code → same embedding vector (deterministic, given fixed model version).
- Model version pinned in `RAG_CODE_EMBED_MODEL` constant.
- Small FIFO cache (500 entries) for repeated queries (`src/rag/retrieval.py:52–67`).

**Text embeddings** (MiniLM):
- Same query text → same embedding vector (deterministic).
- Model version pinned in `RAG_TEXT_EMBED_MODEL` constant.
- No caching for text queries (rare reuse).

### Similarity Search Determinism

**FAISS `IndexFlatIP`** (cosine similarity on normalized vectors):
- Deterministic inner-product search; no randomness.
- Top-k results are stable if index is unchanged.
- **Caveat**: If multiple mutations have identical or near-identical embeddings, tie-breaking order is not guaranteed (FAISS may return them in insertion order or with small numerical variation). In practice, genetic populations have diverse code, so this is rare.

### Sources of Non-Determinism

1. **Embedding model version drift**: Different versions of CodeBERT or MiniLM may produce different embeddings.
   - **Mitigation**: Pin model versions in `constants.py`.
   - **Detection**: `src/rag/runtime.py:39–53` checks dimension match and warns if mismatch detected.

2. **Index contents**: If the FAISS index is rebuilt or pruned, retrieved results may differ.
   - **Mitigation**: Point to the same `RAG_DATA_DIR` across runs; do not re-index mid-experiment.

3. **Reranker** (if implemented): Would introduce non-determinism via cross-encoder scores (model-dependent).
   - Currently not implemented; placeholder code exists.

4. **Optional filtering by thresholds**: `RAG_MIN_ACCURACY`, `RAG_MAX_PARAMETERS` are configurable; different thresholds → different retrieved mutations.
   - **Mitigation**: Fix these in `constants.py` or environment before running.

### Cacheability

**Embedding cache** (`src/rag/retrieval.py:52–67`):
- Small FIFO cache (500 entries) for code query embeddings.
- Cleared on new `RetrievalService` instance (created per `RagRuntime`).
- Not persistent across process restarts.

**Index caching**:
- FAISS index is memory-mapped and cached by the OS; no explicit caching layer.

**Result caching**:
- No caching of retrieval results; each query re-computes from the index.

---

## 10. Disk Layout

### Artifact Directories

```
project_root/
├── rag_data/                          # Vector DB (created by setup_rag.py or on-demand)
│   ├── faiss_index/
│   │   ├── code.index                 # FAISS IndexFlatIP (625 KB; ~3400 mutations)
│   │   └── text.index                 # FAISS IndexFlatIP (34 KB; ~450 text chunks)
│   ├── metadata/
│   │   ├── code.jsonl                 # Mutation metadata (1 JSON per line; 1.8 MB)
│   │   └── text.jsonl                 # Text metadata (146 KB)
│   └── (legacy, unused)
│       ├── code_records.jsonl
│       ├── pageindex_trees/           # PageIndex artifacts (dormant)
│       └── trees/
│
├── rag_corpus/                        # Input data for text namespace
│   ├── 4388-Article Text*.pdf         # ~700 KB
│   ├── Cifar-10_Classification*.pdf   # ~430 KB
│   ├── survey_rag_llm.pdf             # ~818 KB
│   ├── pytorch.json                   # Pre-built PyTorch API reference
│   └── README.md
│
├── src/rag/                           # RAG subsystem implementation
│   ├── __init__.py                    # Exports (data_ingestion, embeddings, prompt_enhancer, retrieval, vector_db)
│   ├── runtime.py                     # RagRuntime singleton, public API
│   ├── retrieval.py                   # RetrievalService, retrieval logic, formatting
│   ├── prompt_enhancer.py             # PromptEnhancer, template augmentation, metrics
│   ├── embeddings.py                  # EmbeddingService (CodeBERT + MiniLM)
│   ├── vector_db.py                   # VectorStoreManager, NamespaceStore, FAISS wrappers
│   ├── data_ingestion.py              # extract_mutations_from_checkpoints, process_pdfs, MutationRecord
│   ├── eval_retrieval.py              # Recall/MRR evaluation script (text namespace)
│   ├── eval_chunk_tradeoff.py         # Chunk-size tradeoff evaluation (text namespace)
│   └── golden_queries.json            # Deterministic golden query dataset (small + medium tiers)
│
├── scripts/
│   ├── setup_rag.py                   # Initialize vector DB from checkpoints + PDFs
│   ├── run_rag_ablation_matrix.py     # Launch condition matrix experiments (sbatch)
│   ├── analyze_rag_impact.py          # Per-run impact analysis (evals to threshold, RAG metrics)
│   ├── summarize_rag_ablation.py      # Aggregate condition-level statistics (mean ± CI)
│   └── build_golden_queries.py        # Generate deterministic golden query dataset
│
├── docs/features/
│   ├── 02_rag_feature.md              # Overview and integration points
│   ├── rag_architecture_and_implementation.md
│   └── rag_backend_abstraction.md
│
└── runs/<RUN_ID>/                     # Per-run artifacts (created by evolution)
    ├── metrics/
    │   ├── rag_metrics.jsonl          # Detailed rag_context_built events per prompt
    │   └── rag_impact_report.json     # Summary from analyze_rag_impact.py
    ├── results/                       # *_results.txt files (gene_id → fitness)
    ├── checkpoints/                   # Evolution checkpoints (used by setup_rag.py)
    ├── logs/
    │   └── llm/                       # LLM interaction logs per gene
    └── run_metadata.json              # Run configuration snapshot
```

### Key Artifact Paths for Integration

- **Index location**: Check `src/cfg/constants.py:81` for `RAG_DATA_DIR`.
- **Input PDFs**: `rag_corpus/*.pdf`
- **Checkpoint source**: `runs/auto_*/checkpoints/checkpoint_gen_*.pkl` (scanned by setup_rag.py).
- **Model source**: `sota/ExquisiteNetV2/models/network_{gene_id}.py`.
- **Per-run metrics**: `runs/<RUN_ID>/metrics/rag_metrics.jsonl` (one `rag_context_built` event per prompt augmentation).

---

## Summary Tables

### Configuration Quick Reference

| Setting | Default | Purpose |
|---------|---------|---------|
| `RAG_ENABLED` | `true` | Master on/off switch. |
| `RAG_DATA_DIR` | `ROOT_DIR/rag_data` | Vector DB storage. |
| `RAG_TOP_K` | `5` | Mutations per prompt. |
| `RAG_MIN_ACCURACY` | `0.9` | Mutation eligibility threshold. |
| `RAG_MIN_SIMILARITY` | `0.3` | Retrieval score threshold. |
| `RAG_USE_CODE_CONTEXT` | `true` | Enable code retrieval. |
| `RAG_USE_TEXT_CONTEXT` | `true` | Enable text retrieval. |
| `RAG_RERANKER_ENABLED` | `false` | Enable reranking (unimplemented). |

### Public API Quick Reference

| Function | Module | Purpose |
|----------|--------|---------|
| `get_runtime()` | `src/rag/runtime.py:99` | Get singleton RAG runtime. |
| `enhance_template()` | `RagRuntime` | Augment prompt + retrieve mutations. |
| `collect_context()` | `RagRuntime` | Retrieve mutations only. |
| `log_mutation_code()` | `RagRuntime` | Index a new mutation. |

### Evaluation Scripts

| Script | Input | Output | Scope |
|--------|-------|--------|-------|
| `setup_rag.py` | Checkpoints + PDFs | FAISS index + metadata | One-time setup. |
| `run_rag_ablation_matrix.py` | Config (seeds, threshold) | Sbatch commands | Submit condition matrix. |
| `analyze_rag_impact.py` | `runs/<RUN_ID>/` | `rag_impact_report.json` | Single-run summary. |
| `summarize_rag_ablation.py` | Multiple `rag_impact_report.json` | CSV + JSON table | Condition-level aggregation. |

---

**Document Version**: 1.0 (2026-04-27)  
**Mapping Scope**: Complete. Reranker class is referenced but unimplemented (latent feature).  
**Known Gaps**: ingest_pytorch_docs function referenced in eval_chunk_tradeoff.py but not defined in data_ingestion.py.
