# RAG Componentization Plan

> Last updated: April 17, 2026
> Status: Draft for implementation and delegated execution

## Goal

Refactor RAG from an in-process implementation detail of LLMGE into a separately callable component with a stable request/response interface.

The first target is a clean package boundary. An HTTP layer is optional and should be added only after the local interface is stable.

This plan also adds:

- side-by-side comparison of multiple RAG backends
- experiment bookkeeping for one-to-one baseline vs RAG analysis
- re-evaluation of FAISS retrieval quality on the real corpus
- verification of data ingestion quality and indexing policy
- revision of mutation logging thresholds to better cover the accuracy vs parameter-count Pareto frontier

---

## Why This Refactor

The current RAG path is coupled to `run_improved.py` and mixes:

- retrieval
- prompt formatting
- backend selection
- storage
- observability

That makes it hard to:

- plug in alternative backends such as PageIndex or knowledge graph retrieval
- run controlled one-to-one comparisons
- inspect retrieval quality independently from prompt construction
- expose RAG as a local service or HTTP service later

---

## Target Architecture

### Phase 1: Local callable component

```text
LLMGE
  -> RagClient
  -> RagService
      -> RetrievalBackend adapter
      -> Reranker
      -> Retrieval policy
      -> Bookkeeping
```

### Phase 2: Optional HTTP wrapper

```text
LLMGE
  -> RagClient
      -> local package call OR HTTP call
  -> RagService
```

### Retrieval backends

All backends must implement the same interface:

- `faiss`
- `pageindex`
- `graph`
- later: `hybrid`

---

## Interface Design

### Core principle

Build `/retrieve` first. Make `/augment` a thin wrapper on top.

### `/retrieve`

Purpose:

- return retrieved blocks
- return diagnostics
- avoid hard-coding prompt wording

Request sketch:

```json
{
  "backend": "faiss",
  "mutation_type": "Complex",
  "query_code": "...",
  "top_k": 5,
  "filters": {
    "source_types": ["api", "pdf", "code"]
  }
}
```

Response sketch:

```json
{
  "backend": "faiss",
  "blocks": [
    {
      "kind": "api_doc",
      "document_id": "torch.nn.Conv2d::api_summary",
      "title": "torch.nn.Conv2d",
      "score": 0.97,
      "content": "Applies a 2D convolution..."
    }
  ],
  "diagnostics": {
    "latency_ms": 21.4,
    "candidate_count": 24,
    "reranker_used": true
  }
}
```

### `/augment`

Purpose:

- build the final prompt used by LLMGE
- use the same retrieval result structure from `/retrieve`

Request sketch:

```json
{
  "backend": "faiss",
  "template": "...",
  "mutation_type": "Complex",
  "query_code": "...",
  "formatting_policy": "default_v1"
}
```

Response sketch:

```json
{
  "augmented_prompt": "...",
  "retrieval": {
    "backend": "faiss",
    "blocks": []
  },
  "formatting": {
    "policy": "default_v1"
  }
}
```

---

## Task Streams

### Task 1: Define stable API and local component boundary

Deliverables:

- `src/rag/api_types.py`
- `src/rag/backend_protocol.py`
- `src/rag/client.py`
- `src/rag/service.py`

Required types:

- `RagRequest`
- `RagResponse`
- `RetrievedBlock`
- `RetrieveRequest`
- `RetrieveResponse`
- `AugmentRequest`
- `AugmentResponse`

Acceptance criteria:

- LLMGE can call RAG through a client interface without importing retrieval internals directly
- the service can run in-process first
- the response includes structured retrieval blocks, not just formatted text

### Task 2: Port current FAISS implementation behind the interface

Goal:

- current behavior remains available under `FaissBackend`

Deliverables:

- `src/rag/backends/faiss_backend.py`
- backend registration/config wiring

Acceptance criteria:

- current retrieval quality and logging are preserved
- no prompt construction code depends directly on `RetrievalService`

### Task 3: Add backend adapters for alternative RAG approaches

Subtasks:

- `PageIndexBackend`
- `GraphBackend`

Notes:

- adapt existing implementations from `feature/rag-pi` and `feature/knowledgeGraph`
- do not introduce parallel runtimes with separate calling patterns

Acceptance criteria:

- all backends answer the same request/response contract
- backend selection is runtime-configurable

### Task 4: Bookkeeping for one-to-one comparisons

Goal:

- record the exact artifacts needed to compare baseline vs RAG and backend vs backend

Log per mutation attempt:

- run ID
- generation
- parent gene ID
- child gene ID
- mutation type
- backend
- raw prompt before RAG
- retrieval request
- retrieval response blocks
- augmented prompt after formatting
- model request parameters
- model raw response
- parsed code artifact
- evaluation outputs
- latencies
- failure mode

Required capability:

- replay a fixed set of prompts with and without RAG
- compare the same prompt set across backends

Acceptance criteria:

- one-to-one direct comparisons are possible without manual reconstruction

### Task 5: Reinvesigate FAISS retrieval effectiveness on the real corpus

Goal:

- verify that FAISS is effective on the actual indexed data, not just the synthetic or narrow eval sets

Questions to answer:

- are embeddings aligned with the corpus structure?
- are current query-building heuristics appropriate for LLMGE prompts?
- are the selected blocks qualitatively relevant for real mutation prompts?
- is current reranking helping or masking poor initial retrieval?

Required outputs:

- corpus-aware retrieval audit
- representative query set from real LLMGE prompts
- per-source analysis: API docs, PDFs, mutation code
- recommendation on whether FAISS remains the default baseline backend

Acceptance criteria:

- FAISS baseline is justified with evidence, or replacement/tuning work is proposed

### Task 6: Verify ingestion quality and indexing effectiveness

Goal:

- ensure the indexed corpus is useful and correctly parsed

Areas to verify:

- `pytorch.json` structure and whether the current parsing preserves the right semantic sections
- PDF chunk quality and source value
- mutation code extracted from historically successful runs
- metadata fidelity and document IDs
- chunk sizes, section boundaries, and source-specific formatting

Specific checks:

- are PyTorch docs better represented as summary/parameters/examples/behavior?
- are PDFs contributing signal or mostly noise?
- are code mutations indexed with enough metadata for retrieval and analysis?
- are text chunks and code chunks aligned with the retrieval tasks we actually run?

Acceptance criteria:

- ingestion quality report exists
- concrete fixes are identified and ranked by expected impact

### Task 7: Revise mutation logging thresholds using Pareto-front-aware rules

Current concern:

- absolute thresholds such as `RAG_MIN_ACCURACY` can overfilter exploration and underrepresent low-parameter useful models

New proposal:

- log mutations that fall into:
  - top 10% of `test_accuracy` or overall fitness
  - bottom 10% of `model_params`

Intent:

- preserve both high-performing and compact architectures
- better cover the exploration region near the accuracy/size Pareto frontier

Questions to verify:

- should this be global over all runs, per run, or per generation window?
- should we log both raw metrics and Pareto eligibility status?
- should there be a cap on redundant near-duplicate low-param architectures?

Acceptance criteria:

- new logging policy is defined
- ingestion/logging code supports it
- policy is measurable in bookkeeping and downstream retrieval analysis

### Task 8: Add controlled backend selection for one-to-one experiments

Goal:

- run the same prompt set against:
  - no RAG
  - FAISS
  - PageIndex
  - Graph
  - later hybrid

Acceptance criteria:

- backend can be selected per run or per replay set
- comparisons are reproducible from the bookkeeping ledger

### Task 9: Optional HTTP service wrapper

This comes after the package boundary is stable.

Deliverables:

- thin HTTP layer around `RagService`
- `/retrieve`
- `/augment`

Acceptance criteria:

- HTTP adds no new business logic
- local and HTTP paths return the same schema

---

## Delegation Plan

This work should be split into bounded subprojects with disjoint write scopes.

### Worker A: Interface and plan owner

Owns:

- `docs/plans/05_rag_componentization_plan.md`
- `src/rag/api_types.py`
- `src/rag/backend_protocol.py`

### Worker B: FAISS adapter

Owns:

- `src/rag/backends/faiss_backend.py`
- current runtime migration to `RagService`

### Worker C: Bookkeeping

Owns:

- run-event schema
- prompt/response/eval logging
- replay support

### Worker D: PageIndex adapter

Owns:

- adaptation of `feature/rag-pi` into the common backend interface

### Worker E: Graph adapter

Owns:

- adaptation of `feature/knowledgeGraph` into the common backend interface

### Worker F: Retrieval validation

Owns:

- FAISS audit
- ingestion verification
- threshold revision analysis

### Worker G: Architecture audit and integration review

Owns:

- architecture conformance checks against this plan
- review of backend consistency across workers
- interface drift detection
- integration-risk reports before merge

Required outputs:

- periodic architecture audit notes
- design review findings on request/response stability
- merge-readiness checklist for the main integrator

---

## Version Control Pattern

### Main rule

Do not build this by having every worker commit directly on the same branch.

Use:

- one integration branch
- one worktree and one feature branch per worker

### Recommended structure

Integration branch:

- `feature/rag-componentization`

Worker branches:

- `worker/schema-api`
- `worker/faiss-adapter`
- `worker/bookkeeping`
- `worker/pageindex-adapter`
- `worker/graph-adapter`
- `worker/retrieval-audit`
- `worker/architecture-audit`

### How remote branches are used

Use the remote branches discussed earlier as reference material and adaptation sources:

- `origin/ajay-rag2`
- `origin/feature/knowledgeGraph`
- `origin/feature/rag-pi`

They should not become the new base branch for all work.

Instead:

- current branch or a fresh integration branch remains the implementation base
- remote branches are mined for ideas, code, and patterns
- backend-specific logic is ported or adapted behind the shared interface

### Why this pattern

This avoids:

- coupling your refactor to a colleague’s unfinished branch history
- dragging unrelated changes into the integration branch
- merge chaos when multiple workers edit the same runtime files

### Merge flow

1. lock the interface first
2. merge schema/API branch
3. merge FAISS adapter
4. merge bookkeeping
5. merge retrieval/ingestion audit findings where needed
6. merge PageIndex adapter
7. merge Graph adapter
8. merge optional HTTP wrapper last

---

## Strong Concurrent Practices

### Shared source of truth

Every worker must use the same:

- plan doc
- request/response schema
- backend protocol
- acceptance criteria

### File ownership

Workers should have disjoint write sets whenever possible.

Examples:

- schema worker should not edit backend implementations
- backend workers should not redefine request/response types
- bookkeeping worker should not redesign retrieval policy

### Integration discipline

- no worker invents a private interface
- no worker adds business logic to the optional HTTP layer
- no worker changes prompt formatting policy without updating the schema and docs

### Review discipline

Before merging any worker branch:

- architecture-audit worker reviews for interface drift
- main integrator checks branch write scope
- any cross-cutting change must be justified in the plan doc

### Handoff format

Each worker should produce:

- files changed
- assumptions made
- open risks
- tests run
- unresolved interface questions

---

## Consistency Rules

To keep backend comparisons meaningful:

- all backends must return the same response schema
- all backends must emit comparable diagnostics
- all backends must log into the same bookkeeping ledger
- prompt formatting should be layered above retrieval, not embedded differently per backend
- replay/comparison runs must operate on a stable prompt set with stable IDs

Required shared identifiers:

- `run_id`
- `request_id`
- `prompt_id`
- `parent_gene_id`
- `child_gene_id`
- `backend`
- `generation`
- `mutation_type`

---

## Tooling Guidance for Delegated Agents

### Preferred setup

Use an agent surface that supports:

- separate worktrees
- direct file editing
- long-running branch-local context
- your own model API keys when needed

### Recommended operational model

- one worktree per worker
- one agent per worktree
- one shared architecture doc in-repo
- one shared checklist for merge readiness

### Context to preload into each agent

- this plan doc
- backend protocol file
- schema/type file
- any backend-specific reference branch notes

### Role of the architecture-audit agent

This agent is not an implementer first.

Its job is to:

- inspect whether implementations still match the agreed architecture
- flag hidden coupling
- catch duplicated runtime logic
- ensure PageIndex/Graph are added as adapters rather than parallel systems

---

## Shared Context Artifacts For Delegation

Every worker should use the same shared context:

- this plan doc
- a stable backend protocol file
- a shared request/response schema
- explicit acceptance criteria

Recommended coordination rules:

- one git worktree per worker
- one markdown handoff note per worker
- one shared task board keyed to the task streams above
- no worker should invent a separate interface

---

## Recommended Execution Order

1. Define request/response schema
2. Define backend protocol
3. Port FAISS behind the protocol
4. Add bookkeeping
5. Add replay/comparison harness
6. Reinvestigate FAISS retrieval quality
7. Verify ingestion quality
8. Revise threshold policy for Pareto-front coverage
9. Adapt PageIndex backend
10. Adapt Graph backend
11. Add optional HTTP wrapper

---

## Success Criteria

The refactor is complete enough for broader testing when all of these hold:

- LLMGE uses a stable client/service interface for RAG
- FAISS, PageIndex, and Graph backends share one request/response contract
- retrieval outputs are structured and logged
- bookkeeping enables one-to-one baseline vs RAG comparison
- corpus ingestion has been audited and improved where necessary
- mutation logging reflects both strong accuracy and low-parameter Pareto-front candidates
- backend comparisons are reproducible and explainable
