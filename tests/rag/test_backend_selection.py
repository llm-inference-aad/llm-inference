from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

import rag.runtime as rag_runtime


class _DummyFaissBackend:
    backend_name = "faiss"

    def __init__(self, *args, **kwargs):
        pass

    def index_mutations(self, records):
        return []

    def index_text_documents(self, documents):
        return []

    def log_mutation_code(self, content, metadata):
        return "dummy"

    def retrieve_similar_mutations(self, query_code, top_k=5, min_similarity=0.3):
        return []

    def retrieve_high_performers(self, min_accuracy=0.9, max_parameters=None, limit=5):
        return []

    def retrieve_by_mutation_type(self, mutation_type, limit=5):
        return []

    def format_context(self, mutations):
        return ""


def _patch_runtime_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_runtime, "EmbeddingService", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rag_runtime, "VectorStoreManager", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rag_runtime, "FaissRetrievalBackend", _DummyFaissBackend)


def test_runtime_selects_faiss_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime_primitives(monkeypatch)
    monkeypatch.setattr(rag_runtime, "RAG_BACKEND", "faiss")
    monkeypatch.setattr(rag_runtime, "RAG_FAIL_OPEN", True)

    runtime = rag_runtime.RagRuntime()

    assert runtime.backend_name == "faiss"
    assert runtime.retrieval.backend_name == "faiss"


def test_runtime_selects_pageindex_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime_primitives(monkeypatch)
    monkeypatch.setattr(rag_runtime, "RAG_BACKEND", "pageindex")
    monkeypatch.setattr(rag_runtime, "RAG_FAIL_OPEN", True)

    module = types.ModuleType("rag.pageindex_backend")

    class _DummyPageIndexBackend(_DummyFaissBackend):
        backend_name = "pageindex"

        def __init__(self, *args, **kwargs):
            self.client = object()

    module.PageIndexRetrievalBackend = _DummyPageIndexBackend
    monkeypatch.setitem(sys.modules, "rag.pageindex_backend", module)

    runtime = rag_runtime.RagRuntime()

    assert runtime.backend_name == "pageindex"
    assert runtime.retrieval.backend_name == "pageindex"


def test_runtime_pageindex_fail_open_falls_back_to_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_runtime_primitives(monkeypatch)
    monkeypatch.setattr(rag_runtime, "RAG_BACKEND", "pageindex")
    monkeypatch.setattr(rag_runtime, "RAG_FAIL_OPEN", True)

    module = types.ModuleType("rag.pageindex_backend")

    class _BrokenPageIndexBackend:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PageIndex init failure")

    module.PageIndexRetrievalBackend = _BrokenPageIndexBackend
    monkeypatch.setitem(sys.modules, "rag.pageindex_backend", module)

    runtime = rag_runtime.RagRuntime()

    assert runtime.backend_name == "faiss"
    assert runtime.retrieval.backend_name == "faiss"
