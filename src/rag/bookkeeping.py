"""Bookkeeping ledger for mutation events.

Every mutation attempt produces one or two :class:`MutationEvent` records:

1. **Augment event** — emitted by :class:`~src.rag.service.RagService` when the
   RAG pipeline finishes.  Contains ``retrieval_request``, ``retrieval_response``,
   ``augmented_prompt``, and retrieval-side latencies.  ``eval_outputs`` and
   ``model_response`` are ``None`` at this point.

2. **Eval event** — emitted by ``run_improved.py::_log_mutation_result()`` after
   the gene has been evaluated.  Contains ``eval_outputs``, ``model_response``,
   ``parsed_artifact``, ``failure_mode``, and eval-side latency.

The two events for the same mutation attempt share the same ``request_id`` (UUID4
injected by :class:`~src.rag.client.RagClient`) so a downstream reader can JOIN
them with ``replay_ledger``.  This **two-event approach** was chosen over carrying
a mutable partial-event through the call stack because it requires no shared
in-memory state between the RAG subsystem and the evolution loop — each side is
independently serializable and the reader reconciles them.

Callers must never crash because of ledger failures; all write errors are
swallowed and logged, matching the contract of ``utils/rag_metrics.record_metric``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MutationEvent dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutationEvent:
    """Immutable record of a single mutation attempt.

    All fields are JSON-serializable via :func:`dataclasses.asdict`.  Fields
    that are not yet known at emit time are ``None``.

    Emit-time matrix
    ----------------
    - Augment event (from RagService.augment): retrieval_request,
      retrieval_response, augmented_prompt, backend, latencies[retrieve_ms],
      latencies[rerank_ms], latencies[augment_ms].
    - Eval event (from _log_mutation_result): model_response, parsed_artifact,
      eval_outputs, failure_mode, latencies[llm_ms], latencies[eval_ms].
    """

    # --- Identity fields (8 required comparison identifiers) --------------- #
    run_id: str
    generation: int
    parent_gene_id: str
    child_gene_id: str
    mutation_type: str
    backend: str  # e.g. "faiss", "baseline"
    request_id: str  # UUID4; stable across all ledger entries for same attempt
    prompt_id: str  # SHA-256 of raw_prompt, for replay-grouping

    # --- Prompt fields ----------------------------------------------------- #
    raw_prompt: str  # pre-RAG template
    augmented_prompt: Optional[str] = None  # post-RAG; None for baseline

    # --- Retrieval fields -------------------------------------------------- #
    retrieval_request: Optional[dict] = None   # serialized RetrieveRequest / AugmentRequest
    retrieval_response: Optional[dict] = None  # serialized RetrieveResponse / AugmentResponse

    # --- Model fields ------------------------------------------------------ #
    model_request: Optional[dict] = None   # prompt, temperature, top_p, model_id
    model_response: Optional[str] = None   # raw LLM output text
    parsed_artifact: Optional[str] = None  # extracted code

    # --- Evaluation fields ------------------------------------------------- #
    eval_outputs: Optional[dict] = None  # test_acc, total_params, val_acc, train_time

    # --- Observability fields ---------------------------------------------- #
    latencies: dict = field(default_factory=dict)  # ms per stage: retrieve, rerank, augment, llm, eval
    failure_mode: Optional[str] = None  # e.g. "syntax_error", "retrieval_empty", None for success
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ---------------------------------------------------------------------- #
    # Factory helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def make_request_id() -> str:
        """Generate a new UUID4 request ID."""
        return str(uuid.uuid4())

    @staticmethod
    def make_prompt_id(raw_prompt: str) -> str:
        """Return a stable short hash of *raw_prompt* for grouping."""
        return hashlib.sha256(raw_prompt.encode("utf-8", errors="replace")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict) -> "MutationEvent":
        """Reconstruct a MutationEvent from a plain dict (e.g. from JSON).

        Unknown keys are silently ignored so old ledger files with fewer fields
        can be read after schema evolution.
        """
        known: set[str] = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        # Fill in any fields that are missing (schema evolution forward-compat).
        for f in fields(cls):
            if f.name not in filtered:
                if f.default is not dataclasses.MISSING:
                    filtered[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                    filtered[f.name] = f.default_factory()  # type: ignore[misc]
        return cls(**filtered)


# ---------------------------------------------------------------------------
# RunLedger
# ---------------------------------------------------------------------------

class RunLedger:
    """Append-only JSONL writer for :class:`MutationEvent` records.

    One event per line.  Writes are atomic for small payloads (< PIPE_BUF on
    Linux) when the file is opened with ``O_APPEND``; for safety we additionally
    try ``fcntl.flock`` to serialize concurrent writers from different threads.
    If locking is unavailable (non-POSIX) or fails, we fall back to plain
    ``O_APPEND`` which is still atomic for single-line writes on Linux.

    **Never crashes the caller** — any write error is logged and swallowed.

    Args:
        run_id: The run identifier, used to build the default ledger path.
        ledger_path: Explicit override for the JSONL file path.  When ``None``,
            the path is ``runs/<run_id>/metrics/rag_ledger.jsonl`` resolved from
            ``cfg.constants.RUN_METRICS_DIR``.
    """

    def __init__(self, run_id: str, ledger_path: Optional[Path] = None) -> None:
        self._run_id = run_id
        self._path: Path = ledger_path if ledger_path is not None else self._default_path(run_id)

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def append(self, event: MutationEvent) -> None:
        """Write one event to the ledger.

        Thread-safe on POSIX via ``fcntl.flock``.  Swallows write errors.
        """
        try:
            line = json.dumps(asdict(event), ensure_ascii=False) + "\n"
            self._write_line(line)
        except Exception as exc:
            logger.warning("[RunLedger] Failed to append event: %s", exc)

    def append_many(self, events: Iterable[MutationEvent]) -> None:
        """Write multiple events under a single lock acquisition."""
        try:
            lines = [json.dumps(asdict(e), ensure_ascii=False) + "\n" for e in events]
            if not lines:
                return
            self._write_lines(lines)
        except Exception as exc:
            logger.warning("[RunLedger] Failed to append_many events: %s", exc)

    @property
    def path(self) -> Path:
        return self._path

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _default_path(run_id: str) -> Path:
        """Resolve the default ledger path from cfg.constants or env."""
        try:
            # Lazy import to avoid pulling in torch at module load time.
            metrics_dir = os.environ.get("RUN_METRICS_DIR")
            if not metrics_dir:
                from cfg.constants import RUN_METRICS_DIR  # noqa: PLC0415
                metrics_dir = RUN_METRICS_DIR
            return Path(metrics_dir) / "rag_ledger.jsonl"
        except Exception:
            # Ultimate fallback: next to the cwd under runs/<run_id>/metrics/
            return Path("runs") / run_id / "metrics" / "rag_ledger.jsonl"

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write_line(self, line: str) -> None:
        self._ensure_parent()
        self._write_lines([line])

    def _write_lines(self, lines: list[str]) -> None:
        """Write *lines* to the ledger file, holding an flock if available."""
        self._ensure_parent()
        payload = "".join(lines)
        fd = os.open(
            str(self._path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            self._flock_acquire(fd)
            try:
                os.write(fd, payload.encode("utf-8"))
            finally:
                self._flock_release(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _flock_acquire(fd: int) -> None:
        try:
            import fcntl  # noqa: PLC0415  (POSIX only)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass

    @staticmethod
    def _flock_release(fd: int) -> None:
        try:
            import fcntl  # noqa: PLC0415
            fcntl.flock(fd, fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass


# ---------------------------------------------------------------------------
# replay_ledger
# ---------------------------------------------------------------------------

def replay_ledger(
    run_id: str,
    ledger_path: Optional[Path] = None,
) -> Iterator[MutationEvent]:
    """Stream :class:`MutationEvent` records from a ledger JSONL file.

    Skips malformed lines with a warning.  Returns an empty iterator when the
    file does not exist (no crash).

    Args:
        run_id: The run identifier, used to resolve the default ledger path.
        ledger_path: Explicit override for the JSONL file path.

    Yields:
        :class:`MutationEvent` instances in file order.
    """
    path: Path = ledger_path if ledger_path is not None else RunLedger._default_path(run_id)

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield MutationEvent.from_dict(data)
            except Exception as exc:
                logger.warning(
                    "[replay_ledger] Skipping malformed line %d in %s: %s",
                    lineno,
                    path,
                    exc,
                )
