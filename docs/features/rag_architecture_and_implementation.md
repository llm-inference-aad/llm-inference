# Retrieval-Augmented Generation (RAG) Architecture & Implementation

> **Last updated:** February 24, 2026  
> **Status:** Active — Phase 1 implemented (dual-namespace retrieval + reranker), Phase 2 feature branches planned  
> **Canonical doc:** This file unifies all prior RAG documentation (`02_rag_feature.md`, `rag_implementation_review.md`, `rag_and_inference_refactor_summary.md`, `rag_small_improvements.md`).

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Deep Dive](#component-deep-dive)
4. [Data Flow: End to End](#data-flow-end-to-end)
5. [Configuration Reference](#configuration-reference)
6. [Setup & Usage](#setup--usage)
7. [Integration Points](#integration-points)
8. [Before & After: Prompt Comparison](#before--after-prompt-comparison)
9. [Data Storage Layout](#data-storage-layout)
10. [Filtering & Quality Control](#filtering--quality-control)
11. [Metrics & Observability](#metrics--observability)
12. [Troubleshooting](#troubleshooting)
13. [Historical Contributions & Improvements](#historical-contributions--improvements)
14. [Phase 2 Roadmap](#phase-2-roadmap)
15. [References](#references)

---

## Overview

The RAG pipeline enhances LLM prompts by retrieving and injecting context from:

- **Historically successful code mutations** (stored as vector embeddings of past CNN architectures)
- **Domain-specific text** (PyTorch API docs, research papers on CNN design — SE-blocks, EfficientNet, NAS, etc.)

This enables the LLM-guided evolution loop (LLMGE) to learn from past successes and relevant literature, improving mutation quality and convergence speed toward high-accuracy, low-parameter CNN architectures on CIFAR-10.

---

## High-Level Architecture

The system is divided into three stages: **Ingestion**, **Retrieval**, and **Generation**. The diagrams below show the data flow through each.

### Stage 1 — Ingestion & Storage

Sources are processed into two FAISS vector namespaces: one for **code** (768-dim, CodeBERT) and one for **text** (384-dim, MiniLM-L6-v2).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                  │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌────────────┐ │
│  │  Checkpoints  │  │  PDFs (25+)  │  │ pytorch.json  │  │ Mutation   │ │
│  │  (.pkl files) │  │  (research   │  │ (API docs,    │  │ source     │ │
│  │  from runs/   │  │   papers)    │  │  signatures,  │  │ code       │ │
│  │              │  │              │  │  examples)    │  │            │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘  └─────┬──────┘ │
│         │                 │                  │                 │        │
│         ▼                 ▼                  ▼                 │        │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐        │        │
│  │  Extract      │  │  pdfplumber  │  │  Parse JSON   │        │        │
│  │  mutations &  │  │  → 400-word  │  │  entries      │        │        │
│  │  fitness data │  │    chunks    │  │               │        │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘        │        │
│         │                 │                  │                 │        │
│         ▼                 ▼                  ▼                 ▼        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    EMBEDDING MODELS                              │   │
│  │                                                                  │   │
│  │  Code mutations ──► CodeBERT (768-dim) ──► code namespace        │   │
│  │  Text/PDFs      ──► MiniLM-L6-v2 (384-dim) ──► text namespace   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│                         ┌───────────────────┐                           │
│                         │     FAISS Index    │                           │
│                         │                   │                           │
│                         │  code.index (768d) │                           │
│                         │  text.index (384d) │                           │
│                         │  + metadata (.jsonl)│                          │
│                         └───────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 2 — Retrieval & Prompt Pipeline

When the evolution loop requests a mutation, the retrieval pipeline finds relevant context and builds an augmented prompt.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PIPELINE                               │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │ Parent Gene   │                                                       │
│  │ (source code) │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐        ┌────────────────────────────────────┐         │
│  │ CodeBERT      │        │         FAISS Search                │         │
│  │ encode query  │───────►│                                    │         │
│  └──────────────┘        │  code namespace ──► similarity      │         │
│                          │  text namespace ──► similarity      │         │
│                          │                                    │         │
│                          └─────────────┬──────────────────────┘         │
│                                        │                                │
│                                        ▼                                │
│                          ┌─────────────────────────┐                    │
│                          │  Filter & Threshold      │                    │
│                          │  (RAG_MIN_SIMILARITY≥0.3)│                    │
│                          └─────────────┬────────────┘                    │
│                                        │                                │
│                                        ▼                                │
│                          ┌─────────────────────────┐                    │
│                          │  Reranker (Phase 1)      │                    │
│                          │  BAAI/bge-reranker-v2-m3 │                    │
│                          │  (~300M params, CPU)     │                    │
│                          └─────────────┬────────────┘                    │
│                                        │                                │
│                                        ▼                                │
│                          ┌─────────────────────────┐                    │
│                          │  Top-K Results           │                    │
│                          │  Sorted by score (desc)  │                    │
│                          └─────────────┬────────────┘                    │
│                                        │                                │
│                                        ▼                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      PROMPT CONSTRUCTION                         │   │
│  │                                                                  │   │
│  │  SECTION 1 — Domain Knowledge (text namespace)                   │   │
│  │  "The following PyTorch documentation and research context       │   │
│  │   may be relevant to your architectural decisions:"              │   │
│  │  - [Api Reference] (relevance 0.782)                             │   │
│  │    torch.nn.Conv2d(in_channels, out_channels, ...)               │   │
│  │  - [Pdf] (relevance 0.651)                                       │   │
│  │    EfficientNet uses compound scaling to balance...              │   │
│  │                                                                  │   │
│  │  SECTION 2 — Historical Mutations (code namespace)               │   │
│  │  "Consider the following historically successful mutations."     │   │
│  │  - Gene abc123 (score 0.920) Acc 0.9200, Params 450K            │   │
│  │    | ΔAcc: +0.0150 | ΔParams: -12000                            │   │
│  │  - Gene def456 (score 0.910) Acc 0.9100, Params 380K            │   │
│  │                                                                  │   │
│  │  [Original mutation template appended here]                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 3 — LLM Server & Evolution Loop

The augmented prompt is sent to the self-hosted vLLM inference server. Generated CNN code is validated, evaluated via SLURM, and successful mutations feed back into the ingestion pipeline.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LLM SERVER & EVOLUTION                              │
│                                                                         │
│  ┌──────────────┐     HTTP POST      ┌──────────────────────┐          │
│  │  Augmented    │───────────────────►│  vLLM Engine          │          │
│  │  Prompt       │                    │  (model-agnostic)     │          │
│  └──────────────┘                    │                      │          │
│                                      │  MODEL_PATH from .env │          │
│                                      │  tensor_parallel=2    │          │
│                                      │  gpu_mem_util=0.95    │          │
│                                      └──────────┬───────────┘          │
│                                                  │                      │
│                                                  ▼                      │
│                                      ┌──────────────────┐              │
│                                      │  Generated CNN    │              │
│                                      │  Code             │              │
│                                      └────────┬─────────┘              │
│                                               │                        │
│                                       ┌───────┴───────┐                │
│                                       ▼               ▼                │
│                              ┌──────────────┐  ┌────────────┐          │
│                              │  Valid?       │  │  Fallback   │          │
│                              │  (exec test)  │  │  to parent  │          │
│                              └──────┬───────┘  └────────────┘          │
│                                     │                                   │
│                                     ▼                                   │
│                              ┌──────────────┐                           │
│                              │  SLURM eval   │                           │
│                              │  (train/test) │                           │
│                              └──────┬───────┘                           │
│                                     │                                   │
│                                     ▼                                   │
│                              ┌──────────────┐                           │
│                              │  Fitness      │                           │
│                              │  (acc, params)│                           │
│                              └──────┬───────┘                           │
│                                     │                                   │
│                                     ▼                                   │
│                              ┌──────────────────────────┐               │
│                              │  Log to Vector DB        │               │
│                              │  (if acc ≥ RAG_MIN_ACC)  │               │
│                              │  → feeds back to Stage 1 │               │
│                              └──────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Sequence Diagram — Runtime Retrieval Flow

```
  Generator (LLM)         RAG Runtime         Prompt Enhancer        Vector Store (FAISS)    Reranker (optional)
       │                       │                      │                        │                    │
       │  Request Prompt       │                      │                        │                    │
       │  (Template + Code) ──►│                      │                        │                    │
       │                       │  enhance_template()  │                        │                    │
       │                       │─────────────────────►│                        │                    │
       │                       │                      │                        │                    │
       │                       │                      │  1. build_context()    │                    │
       │                       │                      │  Code → Type → Global  │                    │
       │                       │                      │  search_code(embed)    │                    │
       │                       │                      │───────────────────────►│                    │
       │                       │                      │◄── [Mutation A, B]     │                    │
       │                       │                      │                        │                    │
       │                       │                      │  2. build_text_context()                    │
       │                       │                      │  search_text(embed)    │                    │
       │                       │                      │───────────────────────►│                    │
       │                       │                      │◄── [PyTorch doc, PDF]  │                    │
       │                       │                      │                        │                    │
       │                       │                      │  3. (if RERANKER_ENABLED)                   │
       │                       │                      │  rerank(query, items)──────────────────────►│
       │                       │                      │◄── [reranked items]────────────────────────│
       │                       │                      │                        │                    │
       │                       │                      │  4. Format sections    │                    │
       │                       │                      │     + merge template   │                    │
       │                       │◄─ Augmented Prompt ──│                        │                    │
       │◄── Final Prompt ──────│                      │                        │                    │
       │                       │                      │                        │                    │
```

---

## Component Deep Dive

### 1. Vector Database — `src/rag/vector_db.py`

| Property | Value |
|---|---|
| **Backend** | FAISS (`IndexFlatIP` — inner product on L2-normalized vectors = cosine similarity) |
| **Storage** | `rag_data/faiss_index/` (binary `.index` files) + `rag_data/metadata/` (`.jsonl`) |
| **Namespaces** | `code` (768-dim, CodeBERT) and `text` (384-dim, MiniLM-L6-v2) |
| **Persistence** | Separate vector index and metadata files — allows fast search without loading all metadata |

**Dimension initialization:** On the first `add_documents()` call, FAISS reads `embeddings.shape[1]` and initializes `IndexFlatIP(dimension)`. All subsequent additions to that namespace must match.

### 2. Embedding Service — `src/rag/embeddings.py`

| Model | Purpose | Dimensions | Used For |
|---|---|---|---|
| `microsoft/codebert-base` | Code embeddings | 768 | Mutation code indexing & retrieval |
| `sentence-transformers/all-MiniLM-L6-v2` | Text embeddings | 384 | PDF chunks, PyTorch docs |

Both models are used consistently for indexing and retrieval to ensure vector space alignment.

### 3. Data Ingestion — `scripts/setup_rag.py` & `src/rag/indexer.py`

**Checkpoint ingestion:**
- Scans `runs/*/checkpoints/*.pkl` and `sota/ExquisiteNetV2`
- Extracts mutation records with fitness data `(accuracy, parameters)`
- Only indexes mutations meeting `RAG_MIN_ACCURACY` threshold

**PyTorch documentation indexing (`src/rag/indexer.py`):**
- Uses Python's `inspect` module to extract runtime signatures and docstrings
- Recursive traversal of `torch.nn`, `torch.optim`, `torchvision.models`
- Outputs `rag_corpus/pytorch.json` with: `name`, `signature`, `docstring`, `example`, `embedding_text`

**PDF ingestion:**
- Processes all files in `rag_corpus/` using `pdfplumber`
- Chunks into **400-word segments** for fine-grained section-level retrieval
- Each chunk is independently searchable

### 4. Retrieval Service — `src/rag/retrieval.py`

Four retrieval strategies across two namespaces:

| Strategy | Method | Namespace | When Used |
|---|---|---|---|
| **Code similarity** | `retrieve_similar_mutations()` | `code` | Always (when `query_code` available) |
| **Mutation type** | `retrieve_by_mutation_type()` | `code` | Backfill when code similarity returns < `top_k` |
| **High performers** | `retrieve_high_performers()` | `code` | Final backfill to reach `top_k` results |
| **Text similarity** | `retrieve_similar_text()` | `text` | Always — retrieves PyTorch docs + PDF context |

**Data types:**
- `RetrievedMutation` — code mutations with `gene_id`, `score`, `code`, fitness metadata
- `RetrievedContext` — text documents with `document_id`, `score`, `content`, `doc_type` ("pdf", "api_reference", "code_example")

**Embedding cache:** In-memory cache (max 500 entries, FIFO eviction) avoids redundant embedding computation. ~1.5MB overhead.

### 5. Prompt Enhancer — `src/rag/prompt_enhancer.py`

- `build_context()` — Orchestrates the three code retrieval strategies, deduplicates by `gene_id`, sorts by score descending
- `build_text_context()` — Queries text namespace using mutation type + code snippet as search query
- `enhance_template()` — Produces final augmented prompt with two sections:
  1. **Domain knowledge** (PyTorch docs, research papers) — framed as informational context
  2. **Historical mutations** (past successful architectures) — framed as examples to learn from
- Optional **reranking** step (when `RAG_RERANKER_ENABLED=true`) re-orders both sections via cross-encoder before formatting

### 6. Reranker — `src/rag/reranker.py`

| Property | Value |
|---|---|
| **Model** | `BAAI/bge-reranker-v2-m3` (~300M params) |
| **Device** | CPU (no GPU required) |
| **Loading** | Lazy — only loaded on first call when enabled |
| **API** | Generic `content_fn` — works with both `RetrievedMutation` and `RetrievedContext` |
| **Toggle** | `RAG_RERANKER_ENABLED` (default: `false`) |

---

## Configuration Reference

All RAG settings live in `src/cfg/constants.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `RAG_ENABLED` | `true` | Master toggle for RAG pipeline |
| `RAG_DATA_DIR` | `rag_data/` | Vector DB storage location |
| `RAG_CODE_EMBED_MODEL` | `microsoft/codebert-base` | Code embedding model |
| `RAG_TEXT_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Text embedding model |
| `RAG_TOP_K` | `5` | Number of code mutations to retrieve |
| `RAG_TEXT_TOP_K` | `3` | Number of text chunks (PDFs, docs) to retrieve |
| `RAG_USE_CODE_CONTEXT` | `true` | Enable/disable code-namespace context (ablation toggle) |
| `RAG_USE_TEXT_CONTEXT` | `true` | Enable/disable text-namespace context (ablation toggle) |
| `RAG_MIN_ACCURACY` | `0.9` | Minimum test accuracy to index/retrieve |
| `RAG_MAX_PARAMETERS` | `None` | Optional max parameter count filter |
| `RAG_MIN_SIMILARITY` | `0.3` | Minimum cosine similarity threshold |
| `RAG_RERANKER_ENABLED` | `false` | Enable cross-encoder reranking |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model (~300M params) |

### Experiment Overrides (A/B + Ablations)

These are used by the experiment harness to keep A/B comparisons fair:

| Variable | Default | Description |
|---|---|---|
| `EXPERIMENT_SEED` | unset | Seeds RNGs in `run_improved.py` (best-effort) |
| `NUM_GENERATIONS` | `15` | Override `num_generations` in `src/cfg/constants.py` |
| `POPULATION_SIZE` | `16` | Override `population_size` in `src/cfg/constants.py` |
| `START_POPULATION_SIZE` | `16` | Override `start_population_size` in `src/cfg/constants.py` |

---

## Setup & Usage

### First-Time Setup

```bash
# Build initial vector database from historical checkpoints + PDFs + PyTorch docs
uv run python scripts/setup_rag.py

# Or with a custom pytorch.json path
uv run python scripts/setup_rag.py --pytorch-json /path/to/pytorch.json
```

### Retrieval Evaluation (Golden Queries)

```bash
# (Re)generate `src/rag/golden_queries.json` from `rag_corpus/pytorch.json`
uv run python scripts/build_golden_queries.py --medium-limit 200

# Evaluate the text namespace retrieval quality
uv run python src/rag/eval_retrieval.py --tier small --k 5
uv run python src/rag/eval_retrieval.py --tier medium --k 5 --limit 200
```

This scans all `runs/*/checkpoints/*.pkl`, extracts successful mutations, processes PDFs in `rag_corpus/`, ingests PyTorch API documentation from `rag_corpus/pytorch.json`, and builds FAISS indices in `rag_data/`. Deduplication guards prevent re-indexing already-ingested documents.

### GPU-Accelerated Ingestion

For large corpora, use the SLURM job:

```bash
sbatch src/rag/ingest_job.sh
```

### Automatic Logging During Evolution

Mutations are **automatically logged** to the vector DB during evolution runs if they:
1. Complete evaluation successfully (valid fitness)
2. Meet the `RAG_MIN_ACCURACY` threshold (default: 90%)

Location: `run_improved.py:_log_mutation_result()`

---

## Integration Points

### 1. Template Generation — `run_improved.py:generate_template()`

```python
template_txt = _apply_rag_context(template_txt, mute_type, query_code=x)
```

After selecting a template (EoT or FixedPrompts), RAG context is injected. The resulting augmented prompt is sent to the LLM.

### 2. Prompt Mutation — `src/llm_utils.py:mutate_prompts()`

```python
prompt = _prepend_rag_context_to_prompt(prompt_base, mutation_label)
```

When rephrasing templates across generations, RAG provides historical context.

### 3. Post-Evaluation Logging — `run_improved.py:check4results()`

```python
if GLOBAL_DATA[gene_id]['status'] == 'completed':
    fitness = GLOBAL_DATA[gene_id]['fitness']
    _log_mutation_result(gene_id, fitness)  # Only logs if accuracy >= RAG_MIN_ACCURACY
```

---

## Before & After: Prompt Comparison

The Phase 1 changes fundamentally altered how prompts are constructed. Below is a concrete before/after comparison.

### BEFORE (code namespace only)

The old `enhance_template()` only retrieved from the **code** namespace. Text namespace (PDFs, PyTorch docs) was indexed but **never queried** — a critical gap.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ AUGMENTED PROMPT (BEFORE)                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Consider the following historically successful mutations. Emphasize ideas   │
│  that balance accuracy gains with parameter reductions.                      │
│  - Gene a1b2c3d4 (score 0.847) Accuracy 0.9215, Params 412350               │
│    | ΔAcc: +0.0082 | ΔParams: -15200                                        │
│    import torch.nn as nn                                                     │
│    class Network(nn.Module):                                                 │
│        def __init__(self):                                                   │
│            ...                                                               │
│  - Gene e5f6g7h8 (score 0.791) Accuracy 0.9180, Params 398000               │
│    | ΔAcc: +0.0045 | ΔParams: -29500                                        │
│    import torch.nn as nn                                                     │
│    class Network(nn.Module):                                                 │
│        def __init__(self):                                                   │
│            ...                                                               │
│                                                                              │
│  You are an expert deep learning researcher. Given the CIFAR-10 code below,  │
│  propose a creative architectural modification that improves test accuracy    │
│  while keeping parameters under 800K.                                        │
│                                                                              │
│  <CODE_BLOCK>                                                                │
│  ...current network.py...                                                    │
│  </CODE_BLOCK>                                                               │
│                                                                              │
│  Rules:                                                                      │
│  1. Ensure the code is accurate and executable.                              │
│  2. Under-parameterize, strive for fewer trainable parameters.               │
│  ...                                                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Problem:** The LLM only sees past code mutations. It has no access to PyTorch API documentation (correct `Conv2d` signatures, `SiLU` activation docs) or research paper insights (EfficientNet compound scaling, SE-block design). This limits the LLM to copying patterns from past mutations rather than making informed architectural decisions.

### AFTER (dual-namespace: code + text)

The new `enhance_template()` queries **both** namespaces and produces a two-section prompt with domain knowledge first, then historical mutations.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ AUGMENTED PROMPT (AFTER)                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─ SECTION 1: Domain Knowledge (text namespace) ─────────────────────────┐ │
│  │                                                                         │ │
│  │  The following PyTorch documentation and research context may be        │ │
│  │  relevant to your architectural decisions:                              │ │
│  │                                                                         │ │
│  │  - [Api Reference] (relevance 0.782)                                    │ │
│  │    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,    │ │
│  │    padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',    │ │
│  │    device=None, dtype=None)                                             │ │
│  │    Applies a 2D convolution over an input signal composed of several    │ │
│  │    input planes. In the simplest case, the output value of the layer    │ │
│  │    with input size (N, C_in, H, W) and output (N, C_out, H_out, W_out) │ │
│  │    ...                                                                  │ │
│  │                                                                         │ │
│  │  - [Pdf] (relevance 0.651)                                              │ │
│  │    Squeeze-and-Excitation Networks: We propose a new architectural unit  │ │
│  │    that adaptively recalibrates channel-wise feature responses by       │ │
│  │    explicitly modelling interdependencies between channels. The SE      │ │
│  │    block first squeezes global spatial information into a channel        │ │
│  │    descriptor using global average pooling, then excites...             │ │
│  │                                                                         │ │
│  │  - [Api Reference] (relevance 0.603)                                    │ │
│  │    torch.nn.SiLU(inplace=False)                                         │ │
│  │    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.       │ │
│  │    The SiLU function is also known as the swish function.               │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─ SECTION 2: Historical Mutations (code namespace) ─────────────────────┐ │
│  │                                                                         │ │
│  │  Consider the following historically successful mutations. Emphasize    │ │
│  │  ideas that balance accuracy gains with parameter reductions.           │ │
│  │                                                                         │ │
│  │  - Gene a1b2c3d4 (score 0.847) Accuracy 0.9215, Params 412350          │ │
│  │    | ΔAcc: +0.0082 | ΔParams: -15200                                   │ │
│  │    import torch.nn as nn                                                │ │
│  │    class Network(nn.Module):                                            │ │
│  │        def __init__(self):                                              │ │
│  │            ...                                                          │ │
│  │                                                                         │ │
│  │  - Gene e5f6g7h8 (score 0.791) Accuracy 0.9180, Params 398000          │ │
│  │    | ΔAcc: +0.0045 | ΔParams: -29500                                   │ │
│  │    import torch.nn as nn                                                │ │
│  │    class Network(nn.Module):                                            │ │
│  │        def __init__(self):                                              │ │
│  │            ...                                                          │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  You are an expert deep learning researcher. Given the CIFAR-10 code below,  │
│  propose a creative architectural modification that improves test accuracy    │
│  while keeping parameters under 800K.                                        │
│                                                                              │
│  <CODE_BLOCK>                                                                │
│  ...current network.py...                                                    │
│  </CODE_BLOCK>                                                               │
│                                                                              │
│  Rules:                                                                      │
│  1. Ensure the code is accurate and executable.                              │
│  2. Under-parameterize, strive for fewer trainable parameters.               │
│  ...                                                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**What changed:**

| Aspect | Before | After |
|---|---|---|
| **Namespaces queried** | `code` only | `code` + `text` |
| **PyTorch docs in prompt** | ❌ Never retrieved | ✅ API signatures, docstrings, examples |
| **Research papers in prompt** | ❌ Indexed but unused | ✅ Relevant paper sections (SE-blocks, EfficientNet, etc.) |
| **Prompt structure** | Single section (mutations) | Two sections (domain knowledge → mutations → template) |
| **Reranking** | None | Optional cross-encoder reranking (`RAG_RERANKER_ENABLED`) |
| **pytorch.json ingestion** | Function existed but never called | Wired into `setup_rag.py` |

---

## Data Storage Layout

```
rag_data/
├── faiss_index/
│   ├── code.index          # FAISS binary index — code mutations (768-dim)
│   └── text.index          # FAISS binary index — text documents (384-dim)
├── metadata/
│   ├── code.jsonl          # Mutation metadata (one JSON per line)
│   └── text.jsonl          # PDF chunks + PyTorch API docs metadata
└── embeddings_cache/       # (Optional) Cached embeddings
```

**Why separate files?**
- **FAISS index** (`.index`): Binary vectors only — fast similarity search
- **Metadata** (`.jsonl`): Human-readable JSON with content, fitness, gene_id, etc.
- Allows fast search without loading all metadata, and metadata inspection without rebuilding indices

---

## Filtering & Quality Control

### `RAG_MIN_ACCURACY` — Dual-Stage Filter

| Stage | Location | Behavior |
|---|---|---|
| **At logging** | `_log_mutation_result()` | Prevents low-quality mutations from entering the vector DB |
| **At retrieval** | `retrieve_high_performers()` | Additional safety net for mutations logged with older thresholds |

### `RAG_MIN_SIMILARITY` — Noise Reduction

- Filters out retrieved mutations with cosine similarity below threshold (default: 0.3)
- Scores < 0.3 typically indicate unrelated code patterns
- Configurable: lower (0.2) for broader recall, higher (0.5) for precision

### Chunking Strategy

| Data Type | Chunking | Rationale |
|---|---|---|
| Code mutations | Complete files | Each `gene_id` is one atomic architecture |
| PDF documents | 400-word segments | Allows section-level retrieval from long papers |
| PyTorch docs | ≤500 words → single chunk; >500 words → 400-word segments | Most API docs are compact; large ones (Conv2d) get split |
| Code examples | Separate chunks | `pytorch.json` examples indexed independently for targeted code retrieval |

---

## Metrics & Observability

RAG metrics are logged to `metrics/rag_metrics.jsonl`:

```json
{
  "event_type": "rag_prompt_enhancement",
  "timestamp": 1234567890.123,
  "mutation_type": "Complex",
  "retrieval_ms": 45.2,
  "retrieved_mutations": 5,
  "prompt_tokens": 1234
}
```

| Event Type | Trigger |
|---|---|
| `rag_prompt_enhancement` | Template enhancement in `generate_template()` |
| `rag_prompt_rephrase_context` | Prompt rephrasing in `mutate_prompts()` |
| `rag_generation_context` | Generation context injection |
| `rag_mutation_logged` | Successful mutation indexed to vector DB |

### Retrieval Quality Baseline

From `src/rag/eval_retrieval.py` benchmark (13 golden queries):

| Metric | Value |
|---|---|
| **Recall@5** | 76.9% (10/13 in top-5) |
| **MRR@5** | 0.6667 |

---

## Troubleshooting

### "Embedding dimension mismatch" Error

**Cause:** Trying to add a code embedding to the text namespace (or vice versa).  
**Fix:** Ensure code uses `embed_code()` and text uses `embed_text()`. Check that embedding models match what was used during indexing.

### Empty Retrieval Results

| Possible Cause | Fix |
|---|---|
| Vector DB not initialized | Run `scripts/setup_rag.py` |
| No mutations meet accuracy threshold | Lower `RAG_MIN_ACCURACY` or wait for better mutations |
| Query code doesn't match any indexed mutations | Normal during novel exploration |

### Mutant Files Contain Code Instead of Prompts

`mutant0.txt`, `mutant1.txt`, etc. are **generated outputs** from `mutate_prompts()`. The LLM sometimes generates code when asked to rephrase templates. These files are not used as input templates and don't affect evolution.

---

## Historical Contributions & Improvements

All timestamped milestones that shaped the current RAG stack:

### November 2024 — Initial RAG Feature Base

- **FAISS dual-namespace vector database** with separate code/text indices
- **Three-stage integration:** Template Generation (`generate_template()`), Prompt Mutation (`mutate_prompts()`), Post-Evaluation Logging
- **Config-driven:** `RAG_ENABLED` toggle, `RAG_MIN_ACCURACY` quality gate
- **Retrieval evaluation:** Baseline Recall@5 = 76.9%, MRR@5 = 0.667 on 13 golden PyTorch queries

### November–December 2025 — Inference Refactor & Architecture Redesign

- **HTTP/Server-Side Batching Migration:** Replaced per-mutation `sbatch` GPU jobs with a persistent vLLM inference server (DeepSeek-R1-Distill-Qwen-32B). Centralized batching, latency logging, and metrics.
- **Direct HTTP Mode (`LLM_DIRECT_HTTP=true`):** Thread pool executor for concurrent LLM requests over HTTP, eliminating SLURM scheduling overhead for prompt generation.
- **Improved Context Building:** Refined the cascade logic: similar code → mutation type → global high-performers, with deduplication.
- **Future Tracks Planned:** PageIndex hierarchical RAG (tree-structured TOCs over PDFs), BM25 fusion retrieval, Code Knowledge Graphs (NetworkX).

### December 2025 — Pipeline Micro-Optimizations (`ajay-rag-improvements`)

1. **Embedding Cache ⚡:** In-memory cache (500 entries, FIFO) in `RetrievalService`. **~500–1000× speedup** for repeated queries (~50ms → ~0.1ms). ~1.5MB memory overhead.
2. **Enhanced Context Formatting 📊:** Injected fitness improvement deltas (`ΔAcc: +0.0150`, `ΔParams: -12000`) into LLM context so the model can correlate structural changes with performance.
3. **Score-Based Sorting 🎯:** Deterministic ordering by relevance score in `build_context()`, ensuring best examples appear first.
4. **Minimum Similarity Threshold 🎯:** `RAG_MIN_SIMILARITY=0.3` rejects sub-threshold noise from context.

### February 2026 — Phase 1: Dual-Namespace Retrieval + Reranker

Critical fix and enhancement addressing the text namespace retrieval gap:

1. **Text Namespace Retrieval Fix 🔧:** `PromptEnhancer.build_text_context()` now queries the FAISS `text` namespace alongside the existing `code` namespace. Previously, PDFs and PyTorch docs were indexed but **never retrieved** — a critical data flow gap.
2. **PyTorch API Docs Wired In 📚:** `setup_rag.py` now calls `ingest_pytorch_docs()` to index `pytorch.json` (API signatures, docstrings, code examples) into the text namespace. Added `--pytorch-json` CLI argument and deduplication guards.
3. **`RetrievedContext` Type 📋:** New dataclass for text documents — distinct from `RetrievedMutation` — with `document_id`, `content`, `source`, `doc_type` fields.
4. **Two-Section Prompt Structure 📝:** Augmented prompts now have two clearly separated sections: (1) domain knowledge from text namespace, framed as informational context, and (2) historical mutations from code namespace, framed as examples.
5. **Cross-Encoder Reranker 🎯:** New `src/rag/reranker.py` module using `BAAI/bge-reranker-v2-m3` (~300M params, CPU). Lazy-loaded, disabled by default (`RAG_RERANKER_ENABLED=false`). Generic `content_fn` API works with both `RetrievedMutation` and `RetrievedContext`.
6. **New Constants:** `RAG_TEXT_TOP_K` (default 3), `RAG_RERANKER_ENABLED`, `RAG_RERANKER_MODEL` — all env-configurable.

**Files modified:** `src/rag/retrieval.py`, `src/rag/prompt_enhancer.py`, `src/rag/reranker.py` (new), `src/rag/__init__.py`, `src/cfg/constants.py`, `scripts/setup_rag.py`

---

## Phase 2 Roadmap

### `feature/bm25-fusion` — Hybrid Search

Add BM25 keyword search alongside FAISS dense retrieval. Merge scores with Reciprocal Rank Fusion (RRF).
- **Files:** `src/rag/vector_db.py`, `src/rag/retrieval.py`
- **Reference:** [RAG_Techniques #15](https://github.com/NirDiamant/RAG_Techniques)

### `feature/pageindex-rag` — Hierarchical TOC-Based RAG

Build tree-structured table of contents from PDFs. LLM reasons over the tree to find relevant sections (vectorless reasoning).
- **Files:** New `src/rag/pageindex_retriever.py`
- **Reference:** [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)

### `feature/code-knowledge-graph` — Code KG

Extract entities from successful mutations via AST parsing. Build a lightweight graph (NetworkX) for structural pattern retrieval.
- **Files:** New `src/rag/code_kg.py`
- **Reference:** [LightRAG](https://github.com/HKUDS/LightRAG), [RAG_Techniques #30](https://github.com/NirDiamant/RAG_Techniques)

---

## References

### Libraries & Models

| Resource | Link |
|---|---|
| FAISS | https://github.com/facebookresearch/faiss |
| CodeBERT | https://github.com/microsoft/CodeBERT |
| Sentence Transformers | https://www.sbert.net/ |
| BGE Reranker v2 | https://huggingface.co/BAAI/bge-reranker-v2-m3 |

### Documents in RAG Corpus (`rag_corpus/`)

| # | Document | Local File |
|---|---|---|
| 1 | EfficientNet: Rethinking Model Scaling for CNNs | `efficientNet.pdf` |
| 2 | Squeeze-and-Excitation Networks | `squeeze-and-excitation.pdf` |
| 3 | Depthwise Separable Convolutions (Xception) | `depthwise-separable-convultions.pdf` |
| 4 | CNN Complexity Reduction on CIFAR-10 | `cnn-complexity-reduction.pdf` |
| 5 | CIFAR-10 Dataset Overview | `CIFAR 10 Dataset_ Everything You Need To Know - AskPython.pdf` |
| 6 | The Emerging Science of ML Benchmarks | `The Emerging Science of Machine Learning Benchmarks _ SIAM.pdf` |
| 7 | Dropout Regularization in Deep Learning | `Dropout Regularization in Deep Learning - GeeksforGeeks.pdf` |
| 8 | Using Learning Rate Schedules in PyTorch | `learning-rate-schedules.pdf` |
| 9 | Neural Architecture Search: A Survey | `nas-survey.pdf` |
| 10 | PyTorch API Documentation (extracted) | `pytorch.json` |
