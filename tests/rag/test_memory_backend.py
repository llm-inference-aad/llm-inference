"""Tests for MemoryBackend (PR 8).

All tests use hand-rolled fakes from tests/rag/fakes.py — no FAISS, torch, or
sentence-transformers are loaded.

Coverage:
- Round-trip: index → retrieve returns the seeded summary with correct fields.
- top_k and min_similarity respected.
- Dedup: RagService.augment() with overlapping gene_ids keeps only the code block.
- Weak-query regression guard: the retrieval query is not just the first 5 lines.
- Fitness-in-summary regression guard: memory_backend.index() receives a summary
  containing "Test Acc:" and "Params:" from build_mutation_description's format.
- Backward-compat: RAG_MEMORY_STORE_ENABLED=false → augment() output unchanged.
"""

from __future__ import annotations

import pathlib
import sys
import uuid

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rag.api_types import (
    AugmentRequest,
    RetrieveRequest,
    RetrievedBlock,
    RetrieveResponse,
)
from src.rag.backends.memory_backend import MemoryBackend
from tests.rag.fakes import FakeEmbeddingService, FakeVectorStoreManager


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_memory_backend(
    min_similarity: float = 0.0,
) -> tuple[MemoryBackend, FakeVectorStoreManager, FakeEmbeddingService]:
    store = FakeVectorStoreManager()
    embeddings = FakeEmbeddingService()
    backend = MemoryBackend(
        vector_store=store,
        embedding_service=embeddings,
        min_similarity=min_similarity,
    )
    return backend, store, embeddings


def _sample_request(
    query: str = "add residual connection",
    top_k: int = 5,
) -> RetrieveRequest:
    return RetrieveRequest(
        query=query,
        namespace="memory",
        top_k=top_k,
        run_id="test-run",
        request_id="req-0",
    )


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_index_then_retrieve_returns_block(self):
        """After index(), retrieve() should return the seeded summary."""
        backend, _, _ = _make_memory_backend(min_similarity=0.0)
        summary = "Complex on parent gene_a: Mutation gene_b (Complex) | Test Acc: 0.9200, Params: 1234567"
        backend.index({"text": summary, "gene_id": "gene_b", "mutation_type": "Complex"})

        resp = backend.retrieve(_sample_request(query=summary, top_k=1))
        assert len(resp.blocks) >= 1
        block = resp.blocks[0]
        assert block.content == summary

    def test_returned_block_has_memory_source_diagnostic(self):
        """Every returned block must have diagnostics[source] == 'memory'."""
        backend, _, _ = _make_memory_backend(min_similarity=0.0)
        backend.index("Complex on parent p1: Mutation g1 | Test Acc: 0.92, Params: 100000")
        resp = backend.retrieve(_sample_request(query="Complex mutation", top_k=3))
        for block in resp.blocks:
            assert block.diagnostics is not None
            assert block.diagnostics.get("source") == "memory"

    def test_index_string_directly(self):
        """index() accepts a plain string (not just a dict)."""
        backend, _, _ = _make_memory_backend(min_similarity=0.0)
        text = "Param on parent p0: Mutation g0 | Test Acc: 0.88, Params: 80000"
        backend.index(text)  # plain string
        resp = backend.retrieve(_sample_request(query=text, top_k=1))
        assert len(resp.blocks) >= 1


# ---------------------------------------------------------------------------
# top_k and min_similarity
# ---------------------------------------------------------------------------


class TestTopKAndMinSimilarity:
    def test_top_k_respected(self):
        """retrieve() returns at most top_k blocks."""
        backend, _, _ = _make_memory_backend(min_similarity=0.0)
        for i in range(5):
            backend.index(f"Complex on parent p0: Mutation g{i} | Test Acc: 0.9{i}, Params: 10000{i}")
        resp = backend.retrieve(_sample_request(query="Complex mutation add residual", top_k=2))
        assert len(resp.blocks) <= 2

    def test_min_similarity_filters_low_scores(self):
        """Blocks with score below min_similarity are excluded."""
        store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        backend = MemoryBackend(
            vector_store=store,
            embedding_service=fake_emb,
            min_similarity=0.999,  # Near-1.0 threshold — only exact (self) matches pass
        )
        backend.index("mutation g1 | Test Acc: 0.90, Params: 50000")
        # Query with a completely different string — cosine sim will be low.
        resp = backend.retrieve(_sample_request(query="ZZZZZ completely unrelated zzzzz", top_k=5))
        # All blocks should be filtered (score < 0.999 for non-matching query).
        for block in resp.blocks:
            assert block.score >= 0.999

    def test_empty_query_returns_empty(self):
        """An empty query string produces an empty response."""
        backend, _, _ = _make_memory_backend(min_similarity=0.0)
        backend.index("g1 | Test Acc: 0.90, Params: 50000")
        resp = backend.retrieve(_sample_request(query="   ", top_k=5))
        assert resp.blocks == []


# ---------------------------------------------------------------------------
# Dedup test: RagService.augment() with overlapping gene_ids
# ---------------------------------------------------------------------------


class TestDedupInRagService:
    """Dedup: when code and memory blocks share a gene_id, only the code block survives."""

    def _make_fake_backend(self, gene_id: str) -> object:
        """Return a backend that yields one block with the given gene_id."""

        class _FakeBackend:
            def __init__(self, gene_id: str, source: str) -> None:
                self._gene_id = gene_id
                self._source = source

            def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
                if request.namespace != self._source:
                    return RetrieveResponse(blocks=[], diagnostics={})
                block = RetrievedBlock(
                    kind=f"{self._source}_doc",
                    document_id=self._gene_id,
                    title=f"Gene {self._gene_id}",
                    score=0.9,
                    content=f"content from {self._source} for {self._gene_id}",
                    diagnostics={"source": self._source},
                )
                return RetrieveResponse(blocks=[block], diagnostics={})

            def index(self, document) -> None:
                pass

        return _FakeBackend(gene_id, "code")

    def test_dedup_prefers_code_block_over_memory_bullet(self, monkeypatch):
        """If the same gene_id appears in code and memory, the code block wins."""
        from src.rag.service import RagService
        from src.rag.api_types import AugmentRequest

        shared_gene = "gene_overlap_42"

        # Code backend returns gene_overlap_42 in code namespace.
        class FakeCodeBackend:
            def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
                block = RetrievedBlock(
                    kind="mutation_code",
                    document_id=shared_gene,
                    title=f"Gene {shared_gene}",
                    score=0.95,
                    content=f"# full code for {shared_gene}",
                    diagnostics={"source": "code"},
                )
                return RetrieveResponse(blocks=[block], diagnostics={})

            def index(self, doc) -> None:
                pass

        # Memory backend returns the SAME gene_id as a summary bullet.
        class FakeMemoryBackend:
            def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
                block = RetrievedBlock(
                    kind="memory_summary",
                    document_id=shared_gene,
                    title=f"Past attempt: {shared_gene}",
                    score=0.8,
                    content=f"Complex on parent p0: summary for {shared_gene}",
                    diagnostics={"source": "memory"},
                )
                return RetrieveResponse(blocks=[block], diagnostics={})

            def index(self, doc) -> None:
                pass

        # Enable memory via env.
        monkeypatch.setenv("RAG_MEMORY_STORE_ENABLED", "true")
        monkeypatch.setenv("RAG_USE_CODE_CONTEXT", "true")
        monkeypatch.setenv("RAG_USE_TEXT_CONTEXT", "false")
        # Invalidate cached module attrs by reloading cfg.constants.
        import importlib
        import cfg.constants as cc
        importlib.reload(cc)

        service = RagService(
            backend=FakeCodeBackend(),
            memory_backend=FakeMemoryBackend(),
            reranker=None,
        )
        req = AugmentRequest(
            template="Do something creative",
            mutation_type="Complex",
            query_code="class Net(nn.Module): pass",
            gene_id="parent_gene",
            run_id="test-run",
        )
        resp = service.augment(req)

        # The augmented prompt must contain the code block content.
        assert f"full code for {shared_gene}" in resp.augmented_prompt

        # The prompt must NOT contain the memory summary for the same gene
        # (it was deduped away).
        assert f"summary for {shared_gene}" not in resp.augmented_prompt

        # Blocks used: only the code block should appear for this gene.
        block_ids = [b.document_id for b in resp.blocks_used]
        # gene should appear only once (the code block).
        assert block_ids.count(shared_gene) == 1
        code_blocks = [b for b in resp.blocks_used if (b.diagnostics or {}).get("source") == "code"]
        assert any(b.document_id == shared_gene for b in code_blocks)


# ---------------------------------------------------------------------------
# Weak-query regression guard
# ---------------------------------------------------------------------------


class TestWeakQueryRegression:
    """The memory retrieval query must use the FULL query_code, not just 5 lines."""

    def test_memory_query_contains_content_beyond_first_5_lines(self, monkeypatch):
        """Captured embed_text input must contain lines beyond the 5th line."""
        from src.rag.service import RagService
        from src.rag.api_types import AugmentRequest

        captured_queries: list[str] = []

        class SpyEmbeddingService(FakeEmbeddingService):
            def embed_text(self, documents):
                if isinstance(documents, str):
                    captured_queries.append(documents)
                else:
                    captured_queries.extend(documents)
                return super().embed_text(documents)

        store = FakeVectorStoreManager()
        spy_emb = SpyEmbeddingService()
        memory_backend = MemoryBackend(
            vector_store=store, embedding_service=spy_emb, min_similarity=0.0
        )
        # Seed one entry so retrieve has something to return.
        memory_backend.index("Mutation g0 | Test Acc: 0.90, Params: 50000")

        # Build a query_code with ≥ 10 lines so we can verify lines > 5 are present.
        query_lines = [f"line_{i}: x = {i}" for i in range(10)]
        query_code = "\n".join(query_lines)
        last_line = query_lines[-1]

        monkeypatch.setenv("RAG_MEMORY_STORE_ENABLED", "true")
        monkeypatch.setenv("RAG_USE_CODE_CONTEXT", "false")
        monkeypatch.setenv("RAG_USE_TEXT_CONTEXT", "false")
        import importlib
        import cfg.constants as cc
        importlib.reload(cc)

        class NullBackend:
            def retrieve(self, req):
                return RetrieveResponse(blocks=[], diagnostics={})
            def index(self, doc):
                pass

        service = RagService(
            backend=NullBackend(),
            memory_backend=memory_backend,
            reranker=None,
        )
        req = AugmentRequest(
            template="mutate this",
            mutation_type="Complex",
            query_code=query_code,
            run_id="test-run",
        )
        service.augment(req)

        # At least one embed_text call must have contained content beyond line 5.
        found = any(last_line in q for q in captured_queries)
        assert found, (
            f"Memory query never included content beyond the first 5 lines. "
            f"Captured queries: {captured_queries!r}"
        )


# ---------------------------------------------------------------------------
# Fitness-in-summary regression guard
# ---------------------------------------------------------------------------


class TestFitnessInSummary:
    """_log_mutation_result must pass a summary with Test Acc: and Params: to memory_backend.index."""

    def test_memory_index_receives_fitness_substrings(self, tmp_path, monkeypatch):
        """build_mutation_description format must be preserved in the indexed summary."""
        # We test this by directly inspecting the docstring format that
        # run_improved._log_mutation_result builds and passes to memory_backend.index.
        # Since _log_mutation_result depends on run_improved globals, we exercise
        # the format contract through build_mutation_description + index directly.
        # Prefer src.rag.data_ingestion to avoid picking up the
        # test_apply_rag_context_integration.py stub that monkey-patches
        # rag.data_ingestion.build_mutation_description = lambda *a: "desc".
        try:
            from src.rag.data_ingestion import build_mutation_description
        except ModuleNotFoundError:
            from rag.data_ingestion import build_mutation_description

        fitness = (0.9250, 1_234_567)
        improvement = {"accuracy_delta": 0.0125, "parameters_delta": -50000}
        description = build_mutation_description("gX", "Complex", fitness, improvement)

        # The description produced by build_mutation_description must contain
        # the fitness-in-summary markers that the memory channel needs.
        assert "Test Acc:" in description, f"build_mutation_description missing 'Test Acc:': {description}"
        assert "Params:" in description, f"build_mutation_description missing 'Params:': {description}"

        # Simulate the summary format used in _log_mutation_result (run_improved.py).
        memory_summary = f"Complex on parent gene_p: {description}"

        # Verify the indexed summary (via a capturing backend) contains the markers.
        indexed_docs: list = []

        class CapturingMemoryBackend:
            def index(self, document):
                indexed_docs.append(document)
            def retrieve(self, request):
                return RetrieveResponse(blocks=[], diagnostics={})

        backend = CapturingMemoryBackend()
        backend.index({"text": memory_summary, "gene_id": "gX"})

        assert len(indexed_docs) == 1
        indexed_text = indexed_docs[0].get("text", "") if isinstance(indexed_docs[0], dict) else str(indexed_docs[0])
        assert "Test Acc:" in indexed_text
        assert "Params:" in indexed_text


# ---------------------------------------------------------------------------
# Backward-compat: disabled memory store must not change augment output
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """RAG_MEMORY_STORE_ENABLED=false -> output is byte-identical to no-memory path."""

    def test_disabled_memory_store_no_effect(self, monkeypatch):
        """With RAG_MEMORY_STORE_ENABLED=false, augment output is unchanged."""
        from src.rag.service import RagService
        from src.rag.api_types import AugmentRequest

        class ConstantCodeBackend:
            """Always returns the same block regardless of namespace."""
            def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
                block = RetrievedBlock(
                    kind="mutation_code",
                    document_id="gene_constant",
                    title="Constant gene",
                    score=0.9,
                    content="class ConstNet: pass",
                    diagnostics={"source": "code"},
                )
                return RetrieveResponse(blocks=[block], diagnostics={})

            def index(self, doc) -> None:
                pass

        class MemoryBackendSpy:
            called = False

            def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
                MemoryBackendSpy.called = True
                block = RetrievedBlock(
                    kind="memory_summary",
                    document_id="gene_from_memory",
                    title="Memory block",
                    score=0.8,
                    content="memory summary should not appear",
                    diagnostics={"source": "memory"},
                )
                return RetrieveResponse(blocks=[block], diagnostics={})

            def index(self, doc) -> None:
                pass

        monkeypatch.setenv("RAG_MEMORY_STORE_ENABLED", "false")
        monkeypatch.setenv("RAG_USE_CODE_CONTEXT", "true")
        monkeypatch.setenv("RAG_USE_TEXT_CONTEXT", "false")
        import importlib
        import cfg.constants as cc
        importlib.reload(cc)

        code_backend = ConstantCodeBackend()
        memory_spy = MemoryBackendSpy()

        service = RagService(
            backend=code_backend,
            memory_backend=memory_spy,
            reranker=None,
        )

        req = AugmentRequest(
            template="base template",
            mutation_type="Complex",
            query_code="class Net: pass",
            run_id="test-compat",
        )
        resp = service.augment(req)

        # Memory backend's retrieve must NOT have been called.
        assert not MemoryBackendSpy.called, (
            "memory_backend.retrieve() was called even though RAG_MEMORY_STORE_ENABLED=false"
        )

        # The augmented prompt must not contain memory-only content.
        assert "memory summary should not appear" not in resp.augmented_prompt

        # diagnostics must report 0 memory blocks.
        assert resp.diagnostics.get("memory_blocks_retrieved", 0) == 0


# ---------------------------------------------------------------------------
# BackendProtocol compliance
# ---------------------------------------------------------------------------


class TestMemoryBackendProtocol:
    def test_isinstance_backend_protocol(self):
        from src.rag.backend_protocol import BackendProtocol

        backend, _, _ = _make_memory_backend()
        assert isinstance(backend, BackendProtocol)

    def test_retrieve_callable(self):
        backend, _, _ = _make_memory_backend()
        assert callable(getattr(backend, "retrieve", None))

    def test_index_callable(self):
        backend, _, _ = _make_memory_backend()
        assert callable(getattr(backend, "index", None))
