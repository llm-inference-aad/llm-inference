"""Cross-encoder reranker for RAG retrieval results.

Reranking improves retrieval quality by scoring (query, document) pairs with a
cross-encoder model, which is more accurate than bi-encoder similarity but
slower. We apply it *after* FAISS retrieval to re-order the top candidates.

Usage:
    reranker = Reranker()  # loads model on first use
    reranked = reranker.rerank(query, candidates, top_k=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import List, Sequence, TypeVar, Union

from cfg.constants import RAG_RERANKER_ENABLED, RAG_RERANKER_MODEL

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RerankResult:
    """A reranked item with its cross-encoder score."""
    item: object
    original_score: float
    rerank_score: float


class Reranker:
    """Cross-encoder reranker using a BAAI/bge-reranker model.

    The model is loaded lazily on the first call to `rerank()` to avoid
    startup overhead when reranking is disabled.
    """

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self.model_name = model_name or RAG_RERANKER_MODEL
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info("[RAG Reranker] Loading %s on %s", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("[RAG Reranker] Model loaded successfully")
        except ImportError:
            logger.warning(
                "[RAG Reranker] sentence-transformers not found. "
                "Install it to enable reranking."
            )
            self._model = None
        except Exception as exc:
            logger.warning("[RAG Reranker] Failed to load model: %s", exc)
            self._model = None

    def rerank(
        self,
        query: str,
        items: Sequence[T],
        content_fn,
        top_k: int = 5,
    ) -> List[T]:
        """Rerank items by cross-encoder relevance to query.

        Args:
            query: The search query (code snippet or natural language).
            items: Sequence of items to rerank (RetrievedMutation, RetrievedContext, etc.)
            content_fn: Callable that extracts text content from an item for scoring.
                        e.g. `lambda m: m.code` for mutations or `lambda c: c.content` for text.
            top_k: Number of top items to return after reranking.

        Returns:
            Reranked list of items (same type as input), sorted by cross-encoder score.
            Each item's `score` field is updated with the cross-encoder score.
            Falls back to returning items unchanged if model isn't available.
        """
        if not RAG_RERANKER_ENABLED:
            return list(items[:top_k])

        if not items:
            return []

        self._load_model()
        if self._model is None:
            logger.debug("[RAG Reranker] Model unavailable, returning original order")
            return list(items[:top_k])

        # Build (query, document) pairs for cross-encoder scoring
        pairs = [(query, content_fn(item)) for item in items]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("[RAG Reranker] Scoring failed: %s", exc)
            return list(items[:top_k])

        # Sort by cross-encoder score (descending)
        scored = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)

        logger.debug(
            "[RAG Reranker] Reranked %d items → top score %.3f, bottom score %.3f",
            len(scored),
            scored[0][1] if scored else 0,
            scored[-1][1] if scored else 0,
        )

        # M3: Propagate cross-encoder score into the item's score field
        result = []
        for item, rerank_score in scored[:top_k]:
            try:
                result.append(replace(item, score=float(rerank_score)))
            except (TypeError, AttributeError):
                result.append(item)
        return result
