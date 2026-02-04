# RAG Feature Documentation

**Date:** November 2024  
**Feature:** Retrieval-Augmented Generation (RAG) Pipeline for LLM-Guided Evolution  
**Status:** Implemented

## Overview

The RAG pipeline enhances LLM prompts by retrieving and injecting context from historically successful mutations. This enables the evolution loop to learn from past successes, improving mutation quality and convergence speed.

## Architecture

### Components

1. **Vector Database** (`src/rag/vector_db.py`):
   - Local FAISS index for fast similarity search
   - Separate namespaces for code mutations (`code.index`) and text documents (`text.index`)
   - Persistent storage in `rag_data/` (gitignored)

2. **Embedding Service** (`src/rag/embeddings.py`):
   - **CodeBERT** (`microsoft/codebert-base`): 768-dimensional embeddings for code
   - **sentence-transformers** (`all-MiniLM-L6-v2`): 384-dimensional embeddings for text
   - Same models used for both indexing and retrieval (ensures consistency)

3. **Data Ingestion** (`src/rag/data_ingestion.py`):
   - Extracts mutations from checkpoint files (`runs/*/checkpoints/*.pkl`)
   - Processes PDFs from `rag_corpus/` (400-word chunks)
   - Calculates fitness improvements (accuracy delta, parameter delta)

4. **Retrieval Service** (`src/rag/retrieval.py`):
   - Similarity search: Find mutations with similar code patterns
   - Performance-guided: Filter by `RAG_MIN_ACCURACY` (default: 90%)
   - Mutation-type filtering: Retrieve mutations of specific types
   - Hybrid strategies combining multiple retrieval methods

5. **Prompt Enhancer** (`src/rag/prompt_enhancer.py`):
   - Injects retrieved context into templates
   - Formats mutation examples with metrics
   - Configurable via `PromptEnhancerConfig`

## Configuration

All RAG settings live in `src/cfg/constants.py`:

```python
RAG_ENABLED = True                    # Toggle RAG on/off
RAG_DATA_DIR = "rag_data/"           # Vector DB storage location
RAG_CODE_EMBED_MODEL = "microsoft/codebert-base"
RAG_TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 5                        # Number of mutations to retrieve
RAG_MIN_ACCURACY = 0.9               # Minimum test accuracy to index/retrieve
RAG_MAX_PARAMETERS = None            # Optional max parameter count filter
```

### Environment Variable Overrides

All settings can be overridden via environment variables:

```bash
export RAG_ENABLED=false              # Disable RAG
export RAG_MIN_ACCURACY=0.93         # Only index mutations with 93%+ accuracy
export RAG_MAX_PARAMETERS=500000     # Filter mutations with >500k parameters
```

## Setup and Initialization

### First-Time Setup

Run the setup script to build the initial vector database from historical checkpoints:

```bash
uv run python scripts/setup_rag.py
```

This script:
- Scans all `runs/*/checkpoints/*.pkl` files
- Extracts successful mutations (with valid fitness)
- Processes PDFs in `rag_corpus/`
- Builds FAISS indices and metadata files
- Stores everything in `rag_data/` (auto-created, gitignored)

### Automatic Logging

During evolution runs, mutations are **automatically logged** to the vector DB if they:
1. Complete evaluation successfully (valid fitness)
2. Meet the `RAG_MIN_ACCURACY` threshold (default: 90%)

**Location:** `run_improved.py:_log_mutation_result()`

```python
# Called after fitness evaluation in check4results()
if GLOBAL_DATA[gene_id]['status'] == 'completed':
    fitness = GLOBAL_DATA[gene_id]['fitness']
    _log_mutation_result(gene_id, fitness)  # Only logs if accuracy >= RAG_MIN_ACCURACY
```

## Integration Points

### 1. Template Generation (`run_improved.py:generate_template()`)

After selecting a template (EoT or FixedPrompt), RAG context is injected:

```python
template_txt = _apply_rag_context(template_txt, mute_type, query_code=x)
```

This:
- Retrieves similar mutations based on mutation type and query code
- Formats them as examples with metrics
- Prepends context to the template before LLM submission

### 2. Prompt Mutation (`src/llm_utils.py:mutate_prompts()`)

When rephrasing templates, RAG provides historical context:

```python
prompt = _prepend_rag_context_to_prompt(prompt_base, mutation_label)
```

### 3. Post-Evaluation Logging (`run_improved.py:check4results()`)

After fitness evaluation completes:

```python
if GLOBAL_DATA[gene_id]['status'] == 'completed':
    fitness = GLOBAL_DATA[gene_id]['fitness']
    _log_mutation_result(gene_id, fitness)  # Filters by RAG_MIN_ACCURACY
```

## Data Storage

### File Structure

```
rag_data/
├── faiss_index/
│   ├── code.index          # FAISS index for code mutations (768-dim)
│   └── text.index          # FAISS index for text documents (384-dim)
├── metadata/
│   ├── code.jsonl          # Mutation metadata (one JSON per line)
│   └── text.jsonl          # PDF chunk metadata
└── embeddings_cache/       # (Optional) Cached embeddings
```

### Separate Persistence

**Why separate files?**
- **FAISS index** (`.index`): Binary file storing only vectors for fast similarity search
- **Metadata** (`.jsonl`): Human-readable JSON containing document content, fitness, gene_id, etc.

This separation allows:
- Fast vector search without loading all metadata
- Inspecting/editing metadata without rebuilding indices
- Efficient updates (append to `.jsonl`, add to FAISS index)

**Vector Dimensions:**
- **Code namespace**: 768 dimensions (CodeBERT)
- **Text namespace**: 384 dimensions (all-MiniLM-L6-v2)

**Dimension Initialization:**
When a namespace is first created, FAISS doesn't know the embedding dimension. On the first `add_documents()` call:
1. We read `normalized_embeddings.shape[1]` (e.g., 768 for CodeBERT)
2. Initialize `faiss.IndexFlatIP(self.dimension)` with that dimension
3. All subsequent additions to that namespace must match this dimension

**Example:**
```python
# First add to 'code' namespace
embeddings = embed_code(["some code"])  # Shape: (1, 768)
store.add_code_documents(["some code"], embeddings, [metadata])
# FAISS creates code.index with dimension=768

# Later, all code embeddings must be 768-dim
# If you try to add a 384-dim vector, you'll get:
# ValueError: Embedding dimension mismatch. Expected 768, received 384
```

## Input/Output Tensor Sizes

**What are tensor sizes?**
In PyTorch CNNs for CIFAR-10, tensors are multi-dimensional arrays representing image batches:

- **Input shape**: `(batch_size, channels, height, width)`
  - Example: `(32, 3, 32, 32)` = 32 images, 3 RGB channels, 32x32 pixels
- **Output shape after forward pass**: `(batch_size, num_classes)`
  - Example: `(32, 10)` = 32 images, 10 class probabilities (CIFAR-10 has 10 classes)

**Why preserve tensor sizes?**
When mutating CNN architectures, we must ensure:
1. **Input shape remains unchanged**: First layer must accept `(batch, 3, 32, 32)`
2. **Output shape remains unchanged**: Final layer must output `(batch, 10)`
3. **Intermediate shapes may change**: Hidden layers can be modified (channel counts, spatial dimensions) as long as the overall flow works

**Example from ExquisiteNetV2:**
```python
def forward(self, x):
    # x shape: (batch, 3, 32, 32)  <- Input
    x = self.FCT(x)                 # -> (batch, 12, 32, 32)
    x = self.EVE(x)                 # -> (batch, 48, 32, 32)
    # ... many layers ...
    x = self.fc(x)                  # -> (batch, 10)  <- Output
    return x
```

**In template prompts:**
When we say "ensure input/output tensor sizes remain unchanged", we mean:
- Don't change the first layer's input channels (must accept 3-channel RGB)
- Don't change the last layer's output size (must output 10 classes for CIFAR-10)
- You can modify internal layers as long as the overall pipeline remains valid

## Filtering and Quality Control

### `RAG_MIN_ACCURACY` Filter

**Purpose:** Only index mutations that meet a minimum quality threshold.

**How it works:**
1. **During logging** (`_log_mutation_result()`):
   - Checks if `fitness[0] >= RAG_MIN_ACCURACY`
   - If below threshold, mutation is **not logged** to vector DB

2. **During retrieval** (`retrieve_high_performers()`):
   - Filters stored mutations by `accuracy >= min_accuracy`
   - Only returns high-performing examples

**Default:** `0.9` (90% test accuracy)

**Why filter at both stages?**
- **At logging**: Prevents low-quality mutations from cluttering the vector DB
- **At retrieval**: Provides an additional safety net if old mutations were logged with a lower threshold

### Mutation Chunking

**Code mutations:** Stored as **complete files** (one mutation = one document)
- No chunking needed - each gene_id represents one complete architecture

**PDF documents:** Chunked into **400-word segments**
- Allows retrieving specific sections from long papers
- Each chunk is independently searchable

## Template System

### Directory Structure

```
templates/
├── ConstantRules.txt           # Rules appended to all FixedPrompts
├── EoT/
│   └── EoT.txt                 # Evolution of Thought template
└── FixedPrompts/
    ├── concise/                # Direct, concise prompts
    │   ├── Complex.txt
    │   ├── Param.txt
    │   ├── RemoveParams.txt
    │   └── ...
    └── roleplay/               # Prompts with character personas
        ├── Expert_Complex.txt
        ├── MadScientist_Param.txt
        ├── MrMagoo_ReduceParams.txt
        └── ...
```

### Template Types

**1. Concise Templates** (`concise/`):
- Direct, straightforward questions
- Example: `"What complex modifications can be explored to enhance performance?"`

**2. Roleplay Templates** (`roleplay/`):
- Same questions but wrapped in character personas
- Example: `"As a leading authority in ML, how can you apply complex modifications..."`
- **Purpose:** Uses Character Role Play (CRP) technique to increase creativity and feasibility

**3. Mutant Templates** (`mutant0.txt`, `mutant1.txt`, etc.):
- **Generated files** created by `mutate_prompts()` function
- LLM is asked to rephrase existing templates
- **Issue:** Sometimes LLM generates code examples instead of pure prompts (unexpected behavior)
- These files are **outputs**, not input templates used in evolution

### How Templates Work

1. **Template Selection** (`generate_template()`):
   - Randomly selects a template from `FixedPrompts/concise/*.txt` or `FixedPrompts/roleplay/*.txt`
   - Appends `ConstantRules.txt`
   - Applies RAG context injection

2. **RAG Enhancement**:
   - Retrieves similar successful mutations
   - Prepends context: `"Consider these successful mutations: [examples]"`
   - Final prompt = RAG context + template + rules

3. **LLM Submission**:
   - Enhanced prompt sent to LLM
   - LLM generates mutated code block
   - Code validated and saved as new gene

## Metrics Tracking

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

**Event Types:**
- `rag_prompt_enhancement`: Template enhancement in `generate_template()`
- `rag_prompt_rephrase_context`: Prompt rephrasing in `mutate_prompts()`
- `rag_generation_context`: Generation context injection
- `rag_mutation_logged`: Successful mutation indexed to vector DB

## Troubleshooting

### "Embedding dimension mismatch" Error

**Cause:** Trying to add a code embedding to the text namespace (or vice versa), or using wrong embedding model.

**Fix:** Ensure code uses `embed_code()` and text uses `embed_text()`. Check that `RAG_CODE_EMBED_MODEL` and `RAG_TEXT_EMBED_MODEL` match what was used during indexing.

### Empty Retrieval Results

**Possible causes:**
1. Vector DB not initialized: Run `scripts/setup_rag.py`
2. No mutations meet `RAG_MIN_ACCURACY` threshold: Lower threshold or wait for better mutations
3. Query code doesn't match any indexed mutations: Normal if evolution is exploring new patterns

### Mutant Files Contain Code Examples

**Issue:** `mutate0.txt`, `mutant1.txt`, etc. contain Python code blocks instead of prompts.

**Explanation:** These files are **generated outputs** from `mutate_prompts()`. The LLM sometimes generates code examples when asked to rephrase templates. This is unexpected but doesn't affect evolution (these files aren't used as input templates).

**Fix:** Consider improving the `mutate_prompts()` prompt to explicitly request "prompt text only, no code examples".

## Future Improvements

1. **Template Enhancement:**
   - Add CIFAR-10-specific guidance
   - Explicit multi-objective optimization instructions
   - RAG placeholder markers in templates

2. **Retrieval Strategies:**
   - Hybrid scoring (similarity + performance)
   - Time-decay weighting (prefer recent mutations)
   - Mutation-type-aware retrieval

3. **Data Quality:**
   - Automatic cleanup of outdated mutations
   - Validation of indexed code compiles correctly
   - Filtering fallback mutations (marked `.fallback`)

4. **Performance:**
   - Batch embedding generation
   - Caching frequently retrieved mutations
   - Incremental index updates (don't rebuild entire index)

## References

- **FAISS:** https://github.com/facebookresearch/faiss
- **CodeBERT:** https://github.com/microsoft/CodeBERT
- **Sentence Transformers:** https://www.sbert.net/
- **CIFAR-10 Dataset:** https://www.cs.toronto.edu/~kriz/cifar.html

