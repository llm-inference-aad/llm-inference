"""RAG pipeline public exports.

Submodule imports are lazy (PEP 562) so that lightweight consumers can do
``from src.rag import api_types`` or ``from src.rag import backend_protocol``
without paying the cost of importing torch / faiss / sentence_transformers
(pulled in by embeddings, retrieval, vector_db, etc.).
"""

from __future__ import annotations

_LAZY_SUBMODULES = {
    "data_ingestion",
    "embeddings",
    "prompt_enhancer",
    "reranker",
    "retrieval",
    "vector_db",
    # Componentization-era surfaces (PRs 3-8). Listed here so callers can do
    # `from src.rag import service` / `from src.rag import client` per the
    # migration guide in docs/features/04_rag_componentization.md.
    "service",
    "client",
    "bookkeeping",
    "runtime",
    "backends",
    "pareto_policy",
    # Light surfaces — no heavy deps, but listed here so plain attribute
    # access (`from src.rag import api_types`) works without relying on a
    # sibling submodule having imported them as a side effect first.
    "api_types",
    "backend_protocol",
}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_LAZY_SUBMODULES)
