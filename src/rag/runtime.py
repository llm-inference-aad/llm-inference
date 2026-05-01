"""Runtime helpers for initializing and reusing RAG components.

.. deprecated::
    This module is a compatibility shim. New code should use
    :class:`~src.rag.client.RagClient` directly instead of calling
    :func:`get_runtime`. The ``RagRuntime`` class will be removed once
    ``run_improved.py`` has fully migrated to the ``RagClient`` API.

    Migration guide::

        # Old (deprecated):
        from rag.runtime import get_runtime
        runtime = get_runtime()
        augmented, mutations = runtime.enhance_template(template, ...)

        # New:
        from rag.client import RagClient
        from rag.api_types import AugmentRequest
        client = RagClient()
        resp = client.augment(AugmentRequest(template=template, ...))
        augmented = resp.augmented_prompt
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Sequence

from cfg.constants import (
    RAG_CODE_EMBED_MODEL,
    RAG_DATA_DIR,
    RAG_ENABLED,
    RAG_MAX_PARAMETERS,
    RAG_MIN_ACCURACY,
    RAG_TEXT_EMBED_MODEL,
    RAG_TEXT_CANDIDATE_K,
    RAG_TEXT_TOP_K,
    RAG_TEXT_TOP_K_API,
    RAG_TEXT_TOP_K_PDF,
    RAG_TOP_K,
)
from utils.rag_metrics import record_metric

from .embeddings import EmbeddingConfig, EmbeddingService
from .prompt_enhancer import PromptEnhancer, PromptEnhancerConfig
from .retrieval import RetrievedMutation, RetrievalService
from .vector_db import VectorStoreManager


class _DimensionMismatchError(RuntimeError):
    """Internal: raised when FAISS index dims don't match embedding model dims."""


class RagRuntime:
    """Encapsulates the long-lived RAG services."""

    def __init__(self) -> None:
        embedding_config = EmbeddingConfig(
            code_model_name=RAG_CODE_EMBED_MODEL,
            text_model_name=RAG_TEXT_EMBED_MODEL,
        )
        self.store = VectorStoreManager(RAG_DATA_DIR)
        self.embeddings = EmbeddingService(embedding_config)

        # H1: Verify FAISS index dimensions match the configured embedding models.
        # Detects model drift between ingestion (setup_rag.py) and runtime.
        for ns_name, embed_fn, label in [
            (VectorStoreManager.CODE_NAMESPACE, self.embeddings.embed_code, "code"),
            (VectorStoreManager.TEXT_NAMESPACE, self.embeddings.embed_text, "text"),
        ]:
            ns = self.store._namespace(ns_name)
            if ns.index is not None:
                probe = embed_fn("dimension probe")
                if ns.index.d != probe.shape[-1]:
                    raise _DimensionMismatchError(
                        f"FAISS {label} index dimension ({ns.index.d}) != "
                        f"embedding model dimension ({probe.shape[-1]}). "
                        f"Re-run setup_rag.py to re-index with the current models."
                    )

        self.retrieval = RetrievalService(self.store, self.embeddings)
        self.prompt_enhancer = PromptEnhancer(
            self.retrieval,
            PromptEnhancerConfig(
                top_k=RAG_TOP_K,
                text_candidate_k=RAG_TEXT_CANDIDATE_K,
                text_top_k=RAG_TEXT_TOP_K,
                text_top_k_api=RAG_TEXT_TOP_K_API,
                text_top_k_pdf=RAG_TEXT_TOP_K_PDF,
                min_accuracy=RAG_MIN_ACCURACY,
                max_parameters=RAG_MAX_PARAMETERS,
            ),
        )

    def enhance_template(
        self,
        template: str,
        mutation_type: str | None = None,
        query_code: str | None = None,
        gene_id: str | None = None,
    ) -> tuple[str, Sequence[RetrievedMutation]]:
        return self.prompt_enhancer.enhance_template(
            template=template,
            mutation_type=mutation_type,
            query_code=query_code,
            gene_id=gene_id,
        )

    def log_mutation_code(self, content: str, metadata: dict) -> str | None:
        if not content.strip():
            return None
        embeddings = self.embeddings.embed_code(content)
        document_ids = self.store.add_code_documents([content], embeddings, [metadata])
        return document_ids[0] if document_ids else None

    def collect_context(
        self, mutation_type: str | None = None, query_code: str | None = None
    ) -> Sequence[RetrievedMutation]:
        return self.prompt_enhancer.build_context(mutation_type=mutation_type, query_code=query_code)

    def format_context(self, mutations: Sequence[RetrievedMutation]) -> str:
        return self.retrieval.format_context(mutations)


_runtime_lock = threading.Lock()
_runtime_instance: RagRuntime | None = None
_runtime_status: Dict[str, Any] = {
    "state": "unknown",
    "reason": None,
    "details": {},
}
_runtime_status_signature: tuple[Any, ...] | None = None


def _emit_runtime_status(state: str, reason: str | None = None, **details: Any) -> None:
    global _runtime_status, _runtime_status_signature
    _runtime_status = {
        "state": state,
        "reason": reason,
        "details": details,
    }
    signature = (
        state,
        reason,
        tuple(sorted((str(k), str(v)) for k, v in details.items())),
    )
    if signature == _runtime_status_signature:
        return
    _runtime_status_signature = signature
    record_metric(
        "rag_runtime_status",
        {
            "run_id": os.getenv("RUN_ID"),
            "state": state,
            "reason": reason,
            **details,
        },
    )


def get_runtime_status() -> Dict[str, Any]:
    return {
        "state": _runtime_status.get("state"),
        "reason": _runtime_status.get("reason"),
        "details": dict(_runtime_status.get("details") or {}),
    }


def get_runtime() -> RagRuntime | None:
    """Return the singleton RAG runtime if enabled.

    Soft-disables RAG (returns None) if the FAISS index dimensions don't match
    the configured embedding models, preventing silent retrieval corruption.
    """
    if not RAG_ENABLED:
        _emit_runtime_status(
            "disabled",
            "rag_disabled_by_config",
            rag_enabled=False,
            rag_data_dir=RAG_DATA_DIR,
        )
        return None

    global _runtime_instance
    if _runtime_instance is None:
        with _runtime_lock:
            if _runtime_instance is None:
                try:
                    _runtime_instance = RagRuntime()
                    _emit_runtime_status(
                        "ready",
                        None,
                        rag_enabled=True,
                        rag_data_dir=RAG_DATA_DIR,
                        code_model=RAG_CODE_EMBED_MODEL,
                        text_model=RAG_TEXT_EMBED_MODEL,
                    )
                except _DimensionMismatchError as exc:
                    import warnings

                    _emit_runtime_status(
                        "disabled",
                        "dimension_mismatch",
                        rag_enabled=True,
                        rag_data_dir=RAG_DATA_DIR,
                        error=str(exc),
                    )
                    warnings.warn(
                        f"[RAG] {exc} — RAG disabled for this session.",
                        stacklevel=2,
                    )
                    return None
                except Exception as exc:
                    _emit_runtime_status(
                        "failed",
                        "runtime_init_exception",
                        rag_enabled=True,
                        rag_data_dir=RAG_DATA_DIR,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise
    return _runtime_instance
