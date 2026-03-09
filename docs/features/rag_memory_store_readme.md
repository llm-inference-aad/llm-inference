# RAG Memory Store: Detailed Implementation Notes

This document describes **all changes made** to add a RAG memory store on top of the existing RAG system, including small supporting edits and cleanup.

## Goal

Add an episodic "memory" layer so the system can:

1. Persist summaries of successful past mutation attempts.
2. Retrieve relevant past attempts during future prompt construction.
3. Be controlled by a `.env` toggle.

---

## Complete File-Level Change Log

### 1) `src/rag/memory_store.py` (new file)

Added a new `MemoryStore` abstraction and `MemoryEntry` dataclass.

- `MemoryStore.add_entry(summary, metadata)`
  - Validates non-empty summary.
  - Embeds summary using text embeddings.
  - Stores entry in vector DB memory namespace.
  - Returns stored document id (or `None`).

- `MemoryStore.search_similar(query, top_k=3, min_similarity=0.3)`
  - Embeds query with text embeddings.
  - Searches memory namespace with overfetch (`top_k * 2`) then threshold filtering.
  - Returns typed `MemoryEntry` results.

Design choice:
- Memory entries are natural-language summaries, so memory uses the **text embedding model** (`all-MiniLM-L6-v2`), not code embeddings.

---

### 2) `src/rag/vector_db.py`

Extended `VectorStoreManager` with a dedicated memory namespace.

Changes:
- Added constant:
  - `MEMORY_NAMESPACE = "memory"`
- Added methods:
  - `add_memory_documents(...)`
  - `search_memory(...)`

Result:
- Memory vectors are stored independently from existing `code` and `text` namespaces.

---

### 3) `src/cfg/constants.py`

Added new env-driven RAG memory configuration:

- `RAG_MEMORY_STORE_ENABLED` (bool)
- `RAG_MEMORY_TOP_K` (int, default `3`)

Behavior:
- Feature is disabled unless enabled via env.
- Retrieval count is independently tunable for memory context section.

---

### 4) `src/rag/runtime.py`

Integrated memory store into runtime lifecycle.

Changes:
- Imported memory config and `MemoryStore`.
- Extended dimension-mismatch validation loop to also validate memory namespace dimensions when an index exists.
  - Memory namespace is validated against `embed_text` dimensions.
- Created runtime-level memory store when enabled:
  - `self.memory_store = MemoryStore(...) if RAG_MEMORY_STORE_ENABLED else None`
- Passed memory store into `PromptEnhancer` constructor.
- Added method:
  - `log_memory_entry(summary, metadata)`

Result:
- Memory component becomes a first-class, runtime-managed RAG service.

---

### 5) `src/rag/prompt_enhancer.py`

Integrated memory retrieval into prompt augmentation pipeline.

Changes:
- Imported:
  - `RAG_MEMORY_STORE_ENABLED`, `RAG_MEMORY_TOP_K`
  - `MemoryStore`
- Updated constructor to accept `memory_store` dependency.
- In `enhance_template(...)`, added a new first section:
  - Build memory query from mutation type and first 5 lines of current code context.
  - Retrieve top memory entries from memory store.
  - Inject section titled:
    - `Relevant past attempts from prior runs (for episodic context):`
- Kept existing text-context and code-context sections unchanged in order after memory.
- Kept fallback behavior:
  - If no sections are available, returns original template unchanged.

Prompt section order after changes:
1. Episodic memory (new)
2. Domain text context (existing)
3. Historical mutation context (existing)
4. Original template

---

### 6) `run_improved.py`

Connected memory logging to mutation result handling.

Changes in `_log_mutation_result(...)`:
- Existing behavior already logs successful mutation code into code namespace.
- Added memory logging path:
  - If `runtime.memory_store` is enabled, create a compact `memory_summary` from mutation type, parent gene, and generated description.
  - Persist it using `runtime.log_memory_entry(...)` with existing metadata.

Also updated run metadata enrichment block:
- Added `RAG_MEMORY_STORE_ENABLED` to experiment metadata output for traceability.

Result:
- Successful mutations now create both:
  - rich code documents (for code retrieval), and
  - concise episodic memory entries (for memory retrieval).

---

### 7) `.env.example`

Added memory config entries under the RAG section:

- `RAG_MEMORY_STORE_ENABLED=false`
- `RAG_MEMORY_TOP_K=3`

This is the documented template users copy from.

---

### 8) `.env` (local working env, non-template)

Added matching runtime entries:

- `RAG_MEMORY_STORE_ENABLED=false`
- `RAG_MEMORY_TOP_K=3`

This gives immediate on/off control from environment configuration.

---

### 9) `src/rag/__init__.py` (small cleanup/supporting)

Added `memory_store` to package exports:

- import list now includes `memory_store`
- `__all__` now includes `"memory_store"`

Reason:
- Keeps package surface consistent and discoverable.

---

### 10) Small cleanup fixes applied

To keep the codebase clean after integration:

- Removed unused imports in `src/rag/prompt_enhancer.py`.
- Removed unused import in `src/rag/memory_store.py`.

These are no-behavior changes but reduce noise and future lint churn.

---

## How the New Memory System Works (End-to-End)

## A) Write path (memory creation)

1. A gene is evaluated and passes existing quality filters.
2. `_log_mutation_result(...)` builds mutation metadata + description.
3. Existing code logging runs (`log_mutation_code`) -> code namespace.
4. New memory logging runs if enabled:
   - Creates short summary string.
   - Stores summary + metadata in memory namespace.

This means every successful mutation can become a retrievable memory item.

## B) Read path (memory retrieval)

1. During prompt creation, `PromptEnhancer.enhance_template(...)` runs.
2. If memory store is enabled and available:
   - Builds a query from mutation type + code snippet context.
   - Searches memory namespace with similarity thresholding.
   - Takes top-k memories (`RAG_MEMORY_TOP_K`).
3. Injects those memories into the prompt before other context blocks.

This gives the model short, experience-like hints from prior attempts.

## C) Toggle behavior

- `RAG_ENABLED=false`: whole RAG pipeline (including memory) is effectively off.
- `RAG_ENABLED=true` + `RAG_MEMORY_STORE_ENABLED=false`: standard RAG only (text + code as previously configured).
- `RAG_ENABLED=true` + `RAG_MEMORY_STORE_ENABLED=true`: standard RAG + episodic memory section.

---

## Data Model for Memory Entries

Each memory entry stores:

- **content**: compact summary sentence
- **metadata**: same mutation metadata payload used in mutation logging, including fields like:
  - `gene_id`
  - `parent_gene_id`
  - `mutation_type`
  - `fitness`
  - `improvement`
  - `description`
  - `source`

This allows future filtering/analysis without changing storage format.

---

## Similarity / Ranking Behavior

Memory search currently uses:

- embedding model: text embedding model
- candidate overfetch: `top_k * 2`
- filter: `score >= RAG_MIN_SIMILARITY`
- final cut: first `top_k`

Reranker is still applied only to code/text contexts in `PromptEnhancer`; memory entries are injected after vector similarity filtering.

---

## Operational Notes

- Memory vectors are persisted in the same `RAG_DATA_DIR` base path under separate namespace files.
- Runtime dimensionality checks cover memory namespace to prevent silent model/index mismatch issues.
- If no memory entries are found, prompt enhancement gracefully degrades to existing behavior.

---


## Quick Usage

In `.env`:

```bash
RAG_ENABLED=true
RAG_MEMORY_STORE_ENABLED=true
RAG_MEMORY_TOP_K=3
```

Then run your normal pipeline (`run_improved.py` path). No extra command is required.

---

