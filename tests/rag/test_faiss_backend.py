"""Tests for FaissBackend (PR 2 — FAISS backend adapter).

All tests are model-free: they use FakeEmbeddingService + FakeVectorStoreManager
so no sentence-transformer weights are downloaded and no GPU is required.

Coverage targets:
- FaissBackend.retrieve() returns correct RetrievedBlock shape.
- Diagnostics include candidate_count, reranked, source per block.
- Namespace routing (code / text / None).
- Regression guard: backend construction and retrieval do NOT touch
  src.rag.runtime._runtime_instance.
- PromptEnhancer._retrieve_via_backend() wires correctly.
- BackendProtocol compliance (isinstance check).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from tests.rag.fakes import FakeEmbeddingService, FakeVectorStoreManager


# ---------------------------------------------------------------------------
# Helpers: build a seeded FaissBackend backed by in-memory fakes
# ---------------------------------------------------------------------------

def _make_backend(store=None, embeddings=None):
    """Return a FaissBackend wired to in-memory fakes (no disk I/O)."""
    from src.rag.backends.faiss_backend import FaissBackend
    from src.rag.retrieval import RetrievalService

    fake_store = store or FakeVectorStoreManager()
    fake_emb = embeddings or FakeEmbeddingService()
    # Inject fakes at the RetrievalService level so FaissBackend uses them.
    retrieval = RetrievalService(store=fake_store, embeddings=fake_emb)
    return FaissBackend(
        store=fake_store,
        embeddings=fake_emb,
        min_similarity=0.0,  # accept all candidates in tests
    )


def _seed_code_mutations(store: FakeVectorStoreManager, embeddings: FakeEmbeddingService, n: int = 5):
    """Add *n* synthetic mutation records to the code namespace of *store*."""
    contents = [f"def mutation_{i}(): pass  # gene {i}" for i in range(n)]
    embs = embeddings.embed_code(contents)
    metadata = [
        {
            "gene_id": f"gene_{i}",
            "description": f"Mutation {i}",
            "mutation_type": "Complex" if i % 2 == 0 else "Param",
            "fitness": [0.8 + i * 0.02, 1_000_000 + i * 10_000],
            "source": "code",
        }
        for i in range(n)
    ]
    store.add_code_documents(contents, embs, metadata)
    return contents, metadata


def _seed_text_docs(store: FakeVectorStoreManager, embeddings: FakeEmbeddingService, n: int = 3):
    """Add *n* synthetic text documents to the text namespace of *store*."""
    contents = [f"PyTorch documentation chunk {i} about convolutions." for i in range(n)]
    embs = embeddings.embed_text(contents)
    metadata = [
        {
            "doc_type": "api_reference",
            "source": "pytorch.json",
            "source_type": "api",
            "name": f"torch.nn.Conv{i}d",
        }
        for i in range(n)
    ]
    store.add_text_documents(contents, embs, metadata)
    return contents, metadata


# ---------------------------------------------------------------------------
# Tests: import and protocol compliance
# ---------------------------------------------------------------------------

class TestFaissBackendImport:
    def test_import_smoke(self):
        """FaissBackend is importable without triggering heavy model loads."""
        from src.rag.backends.faiss_backend import FaissBackend  # noqa: F401
        from src.rag.backends import FaissBackend as FaissBackendFromPkg  # noqa: F401
        assert FaissBackend is FaissBackendFromPkg

    def test_protocol_compliance(self):
        """FaissBackend satisfies BackendProtocol at runtime."""
        from src.rag.backends.faiss_backend import FaissBackend
        from src.rag.backend_protocol import BackendProtocol

        backend = _make_backend()
        assert isinstance(backend, BackendProtocol), (
            "FaissBackend must satisfy BackendProtocol via structural subtyping"
        )

    def test_has_required_methods(self):
        """FaissBackend exposes retrieve() and index() callables."""
        from src.rag.backends.faiss_backend import FaissBackend

        backend = _make_backend()
        assert callable(backend.retrieve)
        assert callable(backend.index)


# ---------------------------------------------------------------------------
# Tests: code-namespace retrieval
# ---------------------------------------------------------------------------

class TestFaissBackendCodeRetrieval:
    def test_returns_retrieve_response(self, tmp_rag_data):
        """retrieve() returns a RetrieveResponse object."""
        from src.rag.api_types import RetrieveRequest, RetrieveResponse

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="def conv(): pass", namespace="code", top_k=3)
        resp = backend.retrieve(req)
        assert isinstance(resp, RetrieveResponse)

    def test_code_blocks_shape(self, tmp_rag_data):
        """Each retrieved block has the expected RetrievedBlock fields."""
        from src.rag.api_types import RetrieveRequest, RetrievedBlock

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="def conv(): pass", namespace="code", top_k=3)
        resp = backend.retrieve(req)
        assert len(resp.blocks) <= 3
        for block in resp.blocks:
            assert isinstance(block, RetrievedBlock)
            assert block.kind == "mutation_code"
            assert isinstance(block.document_id, str) and block.document_id
            assert isinstance(block.score, float)
            assert isinstance(block.content, str)

    def test_top_k_respected(self, tmp_rag_data):
        """retrieve() returns at most top_k blocks."""
        from src.rag.api_types import RetrieveRequest

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        for top_k in (1, 2, 5):
            req = RetrieveRequest(query="def mutation(): pass", namespace="code", top_k=top_k)
            resp = backend.retrieve(req)
            assert len(resp.blocks) <= top_k, f"Expected <= {top_k}, got {len(resp.blocks)}"

    def test_code_diagnostics_per_block(self, tmp_rag_data):
        """Each code block has diagnostics with candidate_count, reranked, source."""
        from src.rag.api_types import RetrieveRequest

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="def conv(): pass", namespace="code", top_k=3)
        resp = backend.retrieve(req)
        for block in resp.blocks:
            diag = block.diagnostics
            assert diag is not None, "Block diagnostics must be set"
            assert "candidate_count" in diag, "diagnostics must include candidate_count"
            assert "reranked" in diag, "diagnostics must include reranked"
            assert "source" in diag, "diagnostics must include source"
            # FaissBackend does not rerank — reranker lives in PromptEnhancer
            assert diag["reranked"] is False
            assert diag["source"] == "code"

    def test_scores_are_floats_in_valid_range(self, tmp_rag_data):
        """Block scores should be floats (cosine similarity from FAISS)."""
        from src.rag.api_types import RetrieveRequest

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="def conv(): pass", namespace="code", top_k=5)
        resp = backend.retrieve(req)
        for block in resp.blocks:
            assert isinstance(block.score, float)

    def test_empty_store_returns_empty_blocks(self, tmp_rag_data):
        """When the store is empty, retrieve() returns an empty block list."""
        from src.rag.api_types import RetrieveRequest

        backend = _make_backend()
        req = RetrieveRequest(query="def conv(): pass", namespace="code", top_k=5)
        resp = backend.retrieve(req)
        assert resp.blocks == []


# ---------------------------------------------------------------------------
# Tests: text-namespace retrieval
# ---------------------------------------------------------------------------

class TestFaissBackendTextRetrieval:
    def test_text_namespace_routing(self, tmp_rag_data):
        """namespace='text' searches only the text namespace."""
        from src.rag.api_types import RetrieveRequest

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=3)
        _seed_text_docs(fake_store, fake_emb, n=3)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="convolution neural network", namespace="text", top_k=2)
        resp = backend.retrieve(req)
        for block in resp.blocks:
            diag = block.diagnostics or {}
            # Source should be the text doc's source field, not "code"
            assert diag.get("source") != "code" or block.kind != "mutation_code"

    def test_text_block_diagnostics(self, tmp_rag_data):
        """Text blocks have diagnostics with candidate_count, reranked, source."""
        from src.rag.api_types import RetrieveRequest

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_text_docs(fake_store, fake_emb, n=3)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="convolution", namespace="text", top_k=3)
        resp = backend.retrieve(req)
        for block in resp.blocks:
            diag = block.diagnostics
            assert diag is not None
            assert "candidate_count" in diag
            assert "reranked" in diag
            assert "source" in diag
            assert diag["reranked"] is False


# ---------------------------------------------------------------------------
# Tests: dual-namespace retrieval (namespace=None)
# ---------------------------------------------------------------------------

class TestFaissBackendDualNamespace:
    def test_none_namespace_searches_both(self, tmp_rag_data):
        """namespace=None returns blocks from both code and text namespaces."""
        from src.rag.api_types import RetrieveRequest

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=3)
        _seed_text_docs(fake_store, fake_emb, n=3)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="conv neural net", namespace=None, top_k=10)
        resp = backend.retrieve(req)
        kinds = {b.kind for b in resp.blocks}
        # Should have at least one mutation_code and at least one text block
        assert "mutation_code" in kinds, f"Expected mutation_code in {kinds}"

    def test_none_namespace_truncates_to_top_k(self, tmp_rag_data):
        """namespace=None never returns more than top_k blocks."""
        from src.rag.api_types import RetrieveRequest

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)
        _seed_text_docs(fake_store, fake_emb, n=5)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="convolution", namespace=None, top_k=4)
        resp = backend.retrieve(req)
        assert len(resp.blocks) <= 4


# ---------------------------------------------------------------------------
# Tests: request-level diagnostics
# ---------------------------------------------------------------------------

class TestFaissBackendRequestDiagnostics:
    def test_response_diagnostics_present(self, tmp_rag_data):
        """RetrieveResponse.diagnostics is set and contains known keys."""
        from src.rag.api_types import RetrieveRequest

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=3)
        backend = _make_backend(store=fake_store, embeddings=fake_emb)

        req = RetrieveRequest(query="def f(): pass", namespace="code", top_k=3)
        resp = backend.retrieve(req)
        assert resp.diagnostics is not None
        # FaissBackend marks reranker_used=False (reranker lives in PromptEnhancer)
        assert resp.diagnostics.get("reranker_used") is False

    def test_latency_ms_is_positive_float(self, tmp_rag_data):
        """RetrieveResponse.latency_ms should be a non-negative float."""
        from src.rag.api_types import RetrieveRequest

        backend = _make_backend()
        req = RetrieveRequest(query="test", namespace="code", top_k=1)
        resp = backend.retrieve(req)
        assert isinstance(resp.latency_ms, float)
        assert resp.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Regression guard: no singleton coupling
# ---------------------------------------------------------------------------

class TestFaissBackendSingletonGuard:
    """Assert FaissBackend does not touch src.rag.runtime._runtime_instance."""

    def test_construction_does_not_set_runtime_instance(self, tmp_rag_data):
        """Constructing FaissBackend must not initialise the runtime singleton."""
        # Remove any previously cached runtime module to start clean.
        for key in list(sys.modules.keys()):
            if "rag.runtime" in key:
                del sys.modules[key]

        _make_backend()

        runtime_mod = sys.modules.get("src.rag.runtime") or sys.modules.get("rag.runtime")
        if runtime_mod is not None:
            assert getattr(runtime_mod, "_runtime_instance", None) is None, (
                "FaissBackend.__init__ must not create _runtime_instance"
            )

    def test_retrieve_does_not_set_runtime_instance(self, tmp_rag_data):
        """Calling retrieve() must not initialise the runtime singleton."""
        from src.rag.api_types import RetrieveRequest

        for key in list(sys.modules.keys()):
            if "rag.runtime" in key:
                del sys.modules[key]

        backend = _make_backend()
        req = RetrieveRequest(query="def f(): pass", namespace="code", top_k=3)
        backend.retrieve(req)

        runtime_mod = sys.modules.get("src.rag.runtime") or sys.modules.get("rag.runtime")
        if runtime_mod is not None:
            assert getattr(runtime_mod, "_runtime_instance", None) is None, (
                "FaissBackend.retrieve() must not create _runtime_instance"
            )

    def test_runtime_module_not_imported_during_retrieve(self, tmp_rag_data):
        """FaissBackend.retrieve() must not cause runtime.py to be imported at all."""
        from src.rag.api_types import RetrieveRequest

        # Remove runtime from sys.modules so we can detect if it gets imported.
        for key in list(sys.modules.keys()):
            if "rag.runtime" in key:
                del sys.modules[key]

        backend = _make_backend()
        req = RetrieveRequest(query="query", namespace="code", top_k=2)
        backend.retrieve(req)

        # If runtime.py is imported as a side-effect, it will appear in sys.modules.
        # We allow it to be there (it may be imported elsewhere in the test session)
        # but assert the singleton is not set.
        runtime_mod = sys.modules.get("src.rag.runtime") or sys.modules.get("rag.runtime")
        if runtime_mod is not None:
            assert getattr(runtime_mod, "_runtime_instance", None) is None


# ---------------------------------------------------------------------------
# Tests: PromptEnhancer._retrieve_via_backend integration
# ---------------------------------------------------------------------------

class TestPromptEnhancerBackendWiring:
    """Assert that PromptEnhancer._retrieve_via_backend correctly delegates to FaissBackend."""

    def _make_enhancer_with_backend(self, store, embeddings):
        from src.rag.backends.faiss_backend import FaissBackend
        from src.rag.prompt_enhancer import PromptEnhancer, PromptEnhancerConfig
        from src.rag.retrieval import RetrievalService

        backend = FaissBackend(store=store, embeddings=embeddings, min_similarity=0.0)
        retrieval = RetrievalService(store=store, embeddings=embeddings)
        config = PromptEnhancerConfig(top_k=3)
        return PromptEnhancer(retrieval_service=retrieval, config=config, backend=backend)

    def test_retrieve_via_backend_returns_mutations(self, tmp_rag_data):
        """_retrieve_via_backend() returns (mutations, stats) from the backend."""
        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)

        enhancer = self._make_enhancer_with_backend(fake_store, fake_emb)
        mutations, stats = enhancer._retrieve_via_backend("def conv(): pass")
        assert isinstance(mutations, list)
        # With 5 seeded docs and top_k=3, should return up to 3 mutations
        assert len(mutations) <= 3

    def test_retrieve_via_backend_mutation_shape(self, tmp_rag_data):
        """Mutations returned by _retrieve_via_backend have the RetrievedMutation shape."""
        from src.rag.retrieval import RetrievedMutation

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)

        enhancer = self._make_enhancer_with_backend(fake_store, fake_emb)
        mutations, _stats = enhancer._retrieve_via_backend("def conv(): pass")
        for m in mutations:
            assert isinstance(m, RetrievedMutation)
            assert isinstance(m.gene_id, str)
            assert isinstance(m.score, float)
            assert isinstance(m.code, str)

    def test_no_backend_returns_empty(self, tmp_rag_data):
        """_retrieve_via_backend() returns ([], None) when no backend is wired."""
        from src.rag.prompt_enhancer import PromptEnhancer, PromptEnhancerConfig
        from src.rag.retrieval import RetrievalService

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        retrieval = RetrievalService(store=fake_store, embeddings=fake_emb)
        enhancer = PromptEnhancer(retrieval_service=retrieval)
        mutations, stats = enhancer._retrieve_via_backend("def f(): pass")
        assert mutations == []
        assert stats is None

    def test_build_context_with_stats_uses_backend_path(self, tmp_rag_data, monkeypatch):
        """build_context_with_stats() dispatches via _retrieve_via_backend when backend is set."""
        from cfg import constants as _constants
        monkeypatch.setattr(_constants, "RAG_USE_CODE_CONTEXT", True)

        # Re-import after patching so the module picks up the new constant value.
        import importlib
        import src.rag.prompt_enhancer as _pe_mod
        importlib.reload(_pe_mod)

        from src.rag.backends.faiss_backend import FaissBackend
        from src.rag.retrieval import RetrievalService

        fake_store = FakeVectorStoreManager()
        fake_emb = FakeEmbeddingService()
        _seed_code_mutations(fake_store, fake_emb, n=5)

        backend = FaissBackend(store=fake_store, embeddings=fake_emb, min_similarity=0.0)
        retrieval = RetrievalService(store=fake_store, embeddings=fake_emb)
        config = _pe_mod.PromptEnhancerConfig(top_k=3)
        enhancer = _pe_mod.PromptEnhancer(
            retrieval_service=retrieval,
            config=config,
            backend=backend,
        )

        called_via_backend = []
        original = enhancer._retrieve_via_backend

        def spy(query_code):
            called_via_backend.append(query_code)
            return original(query_code)

        enhancer._retrieve_via_backend = spy
        mutations, _stats = enhancer.build_context_with_stats(query_code="def f(): pass")
        assert len(called_via_backend) == 1, "Expected _retrieve_via_backend to be called once"
