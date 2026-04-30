"""Lightweight RAG metric logging utilities."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from cfg.constants import RUN_METRICS_DIR

METRICS_FILE = Path(RUN_METRICS_DIR) / "rag_metrics.jsonl"


def record_metric(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Append a structured metric entry to the rag_metrics log.
    """
    try:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "event_type": event_type,
            "timestamp": time.time(),
            **payload,
        }
        with METRICS_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + os.linesep)
    except Exception as exc:  # pragma: no cover - metrics must not break core loop
        print(f"[RAG][Metrics] Failed to record event {event_type}: {exc}")


