"""RAG pipeline public exports."""

from __future__ import annotations

from . import data_ingestion, embeddings, retrieval, vector_db

# Prompt enhancer (and its optional reranker) may be unavailable in some
# environments (missing optional dependencies). Import it lazily with a
# safe fallback so tooling like `scripts/setup_rag.py` can run without the
# full runtime present.
try:
    from . import prompt_enhancer
    __all__ = [
        "data_ingestion",
        "embeddings",
        "prompt_enhancer",
        "retrieval",
        "vector_db",
    ]
except Exception:
    prompt_enhancer = None
    __all__ = [
        "data_ingestion",
        "embeddings",
        "retrieval",
        "vector_db",
    ]



