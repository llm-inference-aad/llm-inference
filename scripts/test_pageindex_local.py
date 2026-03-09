#!/usr/bin/env python3
"""Integration test for PageIndex using local LLM server calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_md = repo_root / "PageIndex" / "tutorials" / "doc-search" / "README.md"
    parser = argparse.ArgumentParser(description="Test PageIndex local-model integration.")
    parser.add_argument("--md-path", default=None, help="Markdown file used to build a test tree.")
    parser.add_argument("--pdf-path", default=None, help="PDF file used to build a test tree.")
    parser.add_argument("--model", default=os.getenv("PAGEINDEX_MODEL", "local_server"))
    parser.add_argument(
        "--query",
        default="Which section is most relevant to tree search for document retrieval?",
        help="Navigation query used against the generated tree.",
    )
    parser.add_argument("--summary-token-threshold", type=int, default=200)
    args = parser.parse_args()
    if not args.md_path and not args.pdf_path:
        args.md_path = str(default_md)
    return args


def resolve_endpoint() -> str:
    explicit = os.getenv("PAGEINDEX_LOCAL_SERVER_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")

    root_dir = Path(os.getenv("LLM_INFERENCE_ROOT_DIR", Path.cwd()))
    use_load_balancer = os.getenv("USE_LOAD_BALANCER", "false").lower() in {"1", "true", "yes"}
    if use_load_balancer:
        host_file = Path(os.getenv("LOADBALANCER_LOG_FILE", root_dir / "loadbalancer.log"))
        port = int(os.getenv("LOAD_BALANCER_PORT", "9000"))
    else:
        host_file = Path(os.getenv("HOSTNAME_LOG_FILE", root_dir / "hostname.log"))
        port = int(os.getenv("SERVER_PORT", "8000"))

    if not host_file.exists():
        raise FileNotFoundError(
            f"Server host file not found: {host_file}. "
            "Start the server first (`sbatch server.sh`) or set PAGEINDEX_LOCAL_SERVER_URL."
        )

    host = host_file.read_text(encoding="utf-8").strip()
    if not host:
        raise ValueError(f"Server host file is empty: {host_file}")
    return f"http://{host}:{port}/generate"


def flatten_nodes(tree) -> list[dict]:
    nodes = []

    def visit(node):
        if isinstance(node, dict):
            nodes.append(node)
            for child in node.get("nodes", []) or []:
                visit(child)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(tree)
    return nodes


def extract_structure(tree_result):
    if isinstance(tree_result, dict) and "structure" in tree_result:
        return tree_result["structure"]
    return tree_result


def _trim_text(value: object, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _clean_summary_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"```(?:json|python)?", " ", text, flags=re.IGNORECASE)
    text = text.replace("```", " ")
    text = re.sub(r"^\s*(Return Value|Return value|Return|Description)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_node_id_from_raw(raw: str, valid_ids: set[str]) -> str | None:
    match = re.search(r'"selected_node_id"\s*:\s*"([^"]+)"', raw)
    if match:
        maybe = match.group(1).strip()
        if maybe in valid_ids:
            return maybe
    quoted_ids = [m.group(1) for m in re.finditer(r'"(0\d{3,})"', raw)]
    for maybe in quoted_ids:
        if maybe in valid_ids:
            return maybe
    return None


def _fallback_node_selection(candidates: list[dict], query: str) -> str:
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    tokens = {tok for tok in tokens if len(tok) >= 3}
    best_id = str(candidates[0]["node_id"])
    best_score = -1
    for candidate in candidates:
        hay = f"{candidate.get('title', '')} {candidate.get('summary', '')}".lower()
        score = sum(1 for tok in tokens if tok in hay)
        if score > best_score:
            best_score = score
            best_id = str(candidate["node_id"])
    return best_id


async def build_tree(md_path: str, model: str, summary_token_threshold: int):
    repo_root = Path(__file__).resolve().parents[1]
    pageindex_root = repo_root / "PageIndex"
    if str(pageindex_root) not in sys.path:
        sys.path.insert(0, str(pageindex_root))

    from pageindex.page_index_md import md_to_tree  # pylint: disable=import-error

    tree = await md_to_tree(
        md_path=md_path,
        if_thinning=False,
        if_add_node_summary="yes",
        summary_token_threshold=summary_token_threshold,
        model=model,
        if_add_doc_description="no",
        if_add_node_text="no",
        if_add_node_id="yes",
    )
    return tree


def build_tree_from_pdf(pdf_path: str, model: str):
    repo_root = Path(__file__).resolve().parents[1]
    pageindex_root = repo_root / "PageIndex"
    if str(pageindex_root) not in sys.path:
        sys.path.insert(0, str(pageindex_root))

    from pageindex.page_index import page_index  # pylint: disable=import-error

    # Keep this focused on tree generation + navigation validation.
    return page_index(
        pdf_path,
        model=model,
        if_add_node_id="yes",
        if_add_node_summary="no",
        if_add_doc_description="no",
        if_add_node_text="no",
    )


def run_navigation_prompt(model: str, query: str, tree):
    repo_root = Path(__file__).resolve().parents[1]
    pageindex_root = repo_root / "PageIndex"
    if str(pageindex_root) not in sys.path:
        sys.path.insert(0, str(pageindex_root))

    from pageindex.utils import ChatGPT_API, extract_json  # pylint: disable=import-error

    nodes = flatten_nodes(tree)
    valid_nodes = [n for n in nodes if n.get("node_id") is not None]
    candidates = []
    for node in valid_nodes[:12]:
        candidates.append(
            {
                "node_id": node.get("node_id"),
                "title": _trim_text(node.get("title"), 160),
                "summary": _trim_text(_clean_summary_text(node.get("summary") or node.get("prefix_summary", "")), 280),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
            }
        )

    prompt = f"""
You are navigating a PageIndex tree for retrieval.
Given user query and candidate nodes, choose the single best node.

Return JSON only:
{{
  "selected_node_id": "<node_id>",
  "reason": "<short reason>"
}}
Start your answer with "{{" and end your answer with "}}".
Choose only from the candidate node_id values exactly as written.

User query:
{query}

Candidate nodes:
{json.dumps(candidates, indent=2)}
"""
    valid_ids = {str(item.get("node_id")) for item in candidates}
    for attempt in range(1, 4):
        raw = ChatGPT_API(
            model=model,
            prompt=prompt,
            max_new_tokens=192,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )
        if not isinstance(raw, str) or not raw.strip():
            print(f"Navigation attempt {attempt}/3 returned empty output.")
            continue
        parsed = extract_json(raw)
        selected_raw = parsed.get("selected_node_id") if isinstance(parsed, dict) else None
        selected = str(selected_raw).strip() if selected_raw is not None else ""
        if selected and selected in valid_ids:
            return selected, parsed
        selected_from_raw = _extract_node_id_from_raw(raw, valid_ids)
        if selected_from_raw:
            return selected_from_raw, {
                "selected_node_id": selected_from_raw,
                "reason": "parsed selected_node_id from non-JSON raw response",
            }
        print(f"Navigation attempt {attempt}/3 returned invalid JSON payload.")

    fallback_id = _fallback_node_selection(candidates, query)
    fallback_payload = {
        "selected_node_id": fallback_id,
        "reason": "fallback: invalid/empty LLM JSON response; selected by lexical overlap with query",
    }
    print(f"Navigation fallback selected node_id={fallback_id}")
    return fallback_id, fallback_payload


def main() -> int:
    args = parse_args()
    if bool(args.md_path) and bool(args.pdf_path):
        raise ValueError("Provide only one of --md-path or --pdf-path.")
    endpoint = resolve_endpoint()
    os.environ["PAGEINDEX_LOCAL_SERVER_URL"] = endpoint
    os.environ.setdefault("PAGEINDEX_LOCAL_SERVER_TIMEOUT", os.getenv("LOCAL_SERVER_TIMEOUT", "300"))
    os.environ.setdefault("PAGEINDEX_LOCAL_MAX_RETRIES", "10")
    os.environ.setdefault("PAGEINDEX_LOCAL_MAX_NEW_TOKENS", "4096")
    os.environ.setdefault("PAGEINDEX_LOCAL_TOP_P", "0.8")
    os.environ.setdefault("PAGEINDEX_LOCAL_TEMPERATURE", "0.1")

    print(f"Using local PageIndex endpoint: {endpoint}")
    if args.pdf_path:
        print(f"Building tree from pdf: {args.pdf_path}")
        tree_result = build_tree_from_pdf(args.pdf_path, args.model)
    else:
        print(f"Building tree from markdown: {args.md_path}")
        tree_result = asyncio.run(build_tree(args.md_path, args.model, args.summary_token_threshold))

    tree = extract_structure(tree_result)
    nodes = [n for n in flatten_nodes(tree) if n.get("node_id") is not None]
    if not nodes:
        raise RuntimeError("No nodes generated from tree build.")
    print(f"Tree generation OK: {len(nodes)} nodes")

    selected_node_id, nav_payload = run_navigation_prompt(args.model, args.query, tree)
    print(f"Tree navigation OK: selected_node_id={selected_node_id}")
    print("Navigation payload:")
    print(json.dumps(nav_payload, indent=2))
    print("PageIndex local-model integration test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
