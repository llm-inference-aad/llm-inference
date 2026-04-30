#!/usr/bin/env python3
"""Inspect RAG context-building events for a single run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_events(run_dir: Path) -> List[Dict[str, Any]]:
    metrics_path = run_dir / "metrics" / "rag_metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")

    events: List[Dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("event_type") == "rag_context_built":
            events.append(payload)
    return events


def _matches(event: Dict[str, Any], gene_id: str | None, mutation_type: str | None) -> bool:
    if gene_id and str(event.get("gene_id") or "") != gene_id:
        return False
    if mutation_type and str(event.get("mutation_type") or "") != mutation_type:
        return False
    return True


def _print_candidates(title: str, candidates: Iterable[Dict[str, Any]]) -> None:
    print(title)
    found = False
    for idx, candidate in enumerate(candidates, start=1):
        found = True
        print(
            f"  {idx}. id={candidate.get('document_id')} "
            f"type={candidate.get('doc_type')} "
            f"name={candidate.get('name')} "
            f"pre={candidate.get('pre_score')} "
            f"post={candidate.get('post_score')}"
        )
        preview = str(candidate.get("content_preview") or "").replace("\n", " ")
        if preview:
            print(f"     preview: {preview}")
    if not found:
        print("  <none>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RAG context events for a run.")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<RUN_ID>")
    parser.add_argument("--gene-id", default=None, help="Optional gene id to filter on.")
    parser.add_argument("--mutation-type", default=None, help="Optional mutation type to filter on.")
    parser.add_argument("--event-index", type=int, default=None, help="Optional zero-based event index to inspect.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    events = _load_events(run_dir)
    filtered = [event for event in events if _matches(event, args.gene_id, args.mutation_type)]
    if args.event_index is not None:
        if args.event_index < 0 or args.event_index >= len(filtered):
            raise IndexError(f"event-index {args.event_index} out of range for {len(filtered)} events")
        filtered = [filtered[args.event_index]]

    if not filtered:
        print("No matching rag_context_built events found.")
        return

    for idx, event in enumerate(filtered):
        print("=" * 80)
        print(f"Event {idx}")
        print(f"run_id={event.get('run_id')}")
        print(f"gene_id={event.get('gene_id')}")
        print(f"mutation_type={event.get('mutation_type')}")
        print(f"query_code_present={event.get('query_code_present')}")
        print(f"text_query_present={event.get('text_query_present')}")
        print(f"text_query_preview={event.get('text_query_preview')}")
        print(f"text_selection_mode={event.get('text_selection_mode')}")
        print(f"selected_doc_ids_text={event.get('selected_doc_ids_text')}")
        print(f"template_augmented={event.get('template_augmented')}")
        print(f"reranker_used={event.get('reranker_used')}")
        _print_candidates("Text candidates before rerank:", event.get("text_candidates_pre_rerank") or [])
        _print_candidates("Text candidates after rerank:", event.get("text_candidates_post_rerank") or [])
        _print_candidates("Code candidates before rerank:", event.get("code_candidates_pre_rerank") or [])
        _print_candidates("Code candidates after rerank:", event.get("code_candidates_post_rerank") or [])


if __name__ == "__main__":
    main()
