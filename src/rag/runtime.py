"""Runtime helpers for initializing and reusing RAG components."""

from __future__ import annotations

import threading
import warnings
from typing import Sequence

from cfg.constants import (
    RAG_BACKEND,
    RAG_CODE_EMBED_MODEL,
    RAG_DATA_DIR,
    RAG_ENABLED,
    RAG_FAIL_OPEN,
    RAG_MEMORY_STORE_ENABLED,
    RAG_MAX_PARAMETERS,
    RAG_MIN_ACCURACY,
    RAG_TEXT_EMBED_MODEL,
    RAG_TOP_K,
)

from .backends.base import RetrievalBackend
from .memory_store import MemoryStore
from .prompt_enhancer import PromptEnhancer, PromptEnhancerConfig
from .retrieval import RetrievedMutation


def _build_backend() -> RetrievalBackend:
    """Construct the retrieval backend selected by ``RAG_BACKEND``."""
    selected = RAG_BACKEND.lower().strip()

    if selected == "pageindex":
        try:
            from .backends.pageindex_backend import PageIndexRetrievalBackend

            return PageIndexRetrievalBackend()
        except Exception as exc:
            if not RAG_FAIL_OPEN:
                raise RuntimeError(
                    f"PageIndex backend failed to initialise: {exc}"
                ) from exc
            warnings.warn(
                f"[RAG] PageIndex backend unavailable ({exc}). "
                "Falling back to FAISS backend.",
                stacklevel=2,
            )
            # Fall through to FAISS

    if selected not in {"faiss", "pageindex"}:
        raise ValueError(
            f"Unknown RAG_BACKEND={selected!r}. "
            "Supported values: 'faiss', 'pageindex'."
        )

    from .backends.faiss_backend import FaissRetrievalBackend

    return FaissRetrievalBackend(
        rag_data_dir=RAG_DATA_DIR,
        code_embed_model=RAG_CODE_EMBED_MODEL,
        text_embed_model=RAG_TEXT_EMBED_MODEL,
    )


class RagRuntime:
    """Encapsulates the long-lived RAG services."""

    def __init__(self) -> None:
        self.backend = _build_backend()

        memory_store: MemoryStore | None = None
        if RAG_MEMORY_STORE_ENABLED:
            from .backends.faiss_backend import FaissRetrievalBackend

            if isinstance(self.backend, FaissRetrievalBackend):
                memory_store = MemoryStore(self.backend._store, self.backend._embeddings)
            else:
                warnings.warn(
                    f"[RAG] Memory store requires FAISS backend but "
                    f"'{self.backend.backend_name}' is active — memory disabled.",
                    stacklevel=2,
                )
        self.memory_store = memory_store

        self.prompt_enhancer = PromptEnhancer(
            self.backend,
            PromptEnhancerConfig(
                top_k=RAG_TOP_K,
                min_accuracy=RAG_MIN_ACCURACY,
                max_parameters=RAG_MAX_PARAMETERS,
            ),
            memory_store=memory_store,
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
        return self.backend.log_mutation_code(content, metadata)

    def log_memory_entry(self, summary: str, metadata: dict) -> str | None:
        """Log a memory entry (past interaction) when RAG memory store is enabled."""
        if self.memory_store is None:
            return None
        return self.memory_store.add_entry(summary=summary, metadata=metadata)

    def collect_context(
        self, mutation_type: str | None = None, query_code: str | None = None
    ) -> Sequence[RetrievedMutation]:
        return self.prompt_enhancer.build_context(mutation_type=mutation_type, query_code=query_code)

    def format_context(self, mutations: Sequence[RetrievedMutation]) -> str:
        return self.backend.format_context(mutations)


_runtime_lock = threading.Lock()
_runtime_instance: RagRuntime | None = None


def get_runtime() -> RagRuntime | None:
    """Return the singleton RAG runtime if enabled.

    Soft-disables RAG (returns None) if the selected backend fails to
    initialise, preventing silent retrieval corruption.
    """
    if not RAG_ENABLED:
        return None

    global _runtime_instance
    if _runtime_instance is None:
        with _runtime_lock:
            if _runtime_instance is None:
                try:
                    _runtime_instance = RagRuntime()
                except RuntimeError as exc:
                    warnings.warn(
                        f"[RAG] {exc} — RAG disabled for this session.",
                        stacklevel=2,
                    )
                    return None
    return _runtime_instance
