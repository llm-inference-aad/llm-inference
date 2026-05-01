"""Unit tests for PageIndexBackend.

These tests use a synthetic on-disk tree and an injected fake LLM callable,
so they neither hit the local server nor depend on the corpus PDFs.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import List

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rag.api_types import RetrieveRequest, RetrieveResponse  # noqa: E402
from src.rag.backends.pageindex_backend import PageIndexBackend  # noqa: E402


def _write_tree(trees_dir: pathlib.Path, stem: str) -> None:
    """Write a 3-node synthetic tree to *trees_dir*/<stem>_structure.json.

    Layout:
        Document Title
        ├── 0001 Introduction (parent)
        │       └── 0002 Sub-introduction (leaf)
        └── 0003 Methods (leaf)
    """
    structure = [
        {
            "node_id": "0001",
            "title": "Introduction",
            "summary": "Sets up the problem and motivation.",
            "text": "Introduction full text — overview of the paper.",
            "nodes": [
                {
                    "node_id": "0002",
                    "title": "Sub-introduction",
                    "summary": "A specific subsection inside the introduction.",
                    "text": "Subsection full text — details about the motivation.",
                },
            ],
        },
        {
            "node_id": "0003",
            "title": "Methods",
            "summary": "Describes the experimental methodology.",
            "text": "Methods full text — describes the CNN training procedure.",
        },
    ]
    payload = {"doc_name": "Synthetic Document", "structure": structure}
    out_path = trees_dir / f"{stem}_structure.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def synthetic_trees_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    _write_tree(tmp_path, "synthetic_doc")
    return tmp_path


def _request(query: str = "How does the method train the CNN?", top_k: int = 5) -> RetrieveRequest:
    return RetrieveRequest(
        query=query, namespace=None, top_k=top_k, filters={},
        run_id="test-run", request_id="req-1",
    )


class TestPageIndexBackendRetrieval:
    def test_returns_selected_leaf_node(self, synthetic_trees_dir):
        captured_prompts: List[str] = []

        def fake_llm(model: str, prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "thinking": "Methods covers CNN training.",
                "selected_nodes": [
                    {"node_id": "0003", "relevance": 5},
                ],
            })

        backend = PageIndexBackend(
            trees_dir=str(synthetic_trees_dir),
            llm_call=fake_llm,
        )
        resp = backend.retrieve(_request())

        assert isinstance(resp, RetrieveResponse)
        assert len(resp.blocks) == 1
        block = resp.blocks[0]
        assert block.kind == "pageindex_node"
        assert block.document_id == "synthetic_doc:0003"
        assert block.title == "Methods"
        assert block.score == 5.0
        assert "training procedure" in block.content
        assert (block.diagnostics or {}).get("source") == "pageindex"
        assert (block.diagnostics or {}).get("is_leaf") is True
        assert (block.diagnostics or {}).get("doc_name") == "Synthetic Document"
        assert (resp.diagnostics or {}).get("trees_searched") == 1

        # The prompt sent to the LLM must include the query and the doc name,
        # but must NOT include the leaf "text" field (it should be stripped
        # by remove_fields to keep the prompt compact).
        assert len(captured_prompts) == 1
        sent = captured_prompts[0]
        assert "How does the method train the CNN?" in sent
        assert "Synthetic Document" in sent
        assert "Methods full text" not in sent
        assert "training procedure" not in sent

    def test_filters_low_relevance_nodes(self, synthetic_trees_dir):
        def fake_llm(model: str, prompt: str) -> str:
            return json.dumps({
                "thinking": "Mixed.",
                "selected_nodes": [
                    {"node_id": "0001", "relevance": 2},  # below default min=3
                    {"node_id": "0003", "relevance": 4},
                ],
            })

        backend = PageIndexBackend(
            trees_dir=str(synthetic_trees_dir),
            llm_call=fake_llm,
        )
        resp = backend.retrieve(_request())

        assert [b.document_id for b in resp.blocks] == ["synthetic_doc:0003"]

    def test_unknown_node_id_is_dropped(self, synthetic_trees_dir):
        def fake_llm(model: str, prompt: str) -> str:
            return json.dumps({
                "selected_nodes": [
                    {"node_id": "9999", "relevance": 5},  # absent
                    {"node_id": "0002", "relevance": 5},  # leaf
                ],
            })

        backend = PageIndexBackend(
            trees_dir=str(synthetic_trees_dir),
            llm_call=fake_llm,
        )
        resp = backend.retrieve(_request())

        assert [b.document_id for b in resp.blocks] == ["synthetic_doc:0002"]

    def test_top_k_truncates_after_cross_doc_sort(self, tmp_path):
        # Two docs, both selected; top_k=1 must keep the higher-scoring one.
        _write_tree(tmp_path, "doc_a")
        _write_tree(tmp_path, "doc_b")

        def fake_llm(model: str, prompt: str) -> str:
            # doc_a → relevance 4 on node 0003
            # doc_b → relevance 5 on node 0003 (must win)
            score = 4 if "doc_a" in prompt or "synthetic" in prompt else 5
            # Disambiguate by stem in prompt — doc_name is identical for both
            # because we re-use the same synthetic payload, so vary by which
            # call we're on instead.
            return json.dumps({
                "selected_nodes": [{"node_id": "0003", "relevance": score}]
            })

        # Easier: switch the score per call deterministically.
        call_count = {"n": 0}

        def fake_llm_seq(model: str, prompt: str) -> str:
            call_count["n"] += 1
            score = 5 if call_count["n"] == 1 else 3
            return json.dumps({
                "selected_nodes": [{"node_id": "0003", "relevance": score}]
            })

        backend = PageIndexBackend(
            trees_dir=str(tmp_path),
            llm_call=fake_llm_seq,
        )
        resp = backend.retrieve(_request(top_k=1))

        assert len(resp.blocks) == 1
        assert resp.blocks[0].score == 5.0

    def test_empty_query_short_circuits(self, synthetic_trees_dir):
        def fake_llm(model: str, prompt: str) -> str:
            raise AssertionError("LLM should not be called for empty query")

        backend = PageIndexBackend(
            trees_dir=str(synthetic_trees_dir),
            llm_call=fake_llm,
        )
        resp = backend.retrieve(_request(query="   "))
        assert resp.blocks == []
        assert (resp.diagnostics or {}).get("reason") == "empty_query"

    def test_missing_trees_dir_returns_empty_response(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        backend = PageIndexBackend(
            trees_dir=str(missing),
            llm_call=lambda model, prompt: '{"selected_nodes": []}',
        )
        resp = backend.retrieve(_request())
        assert resp.blocks == []
        assert (resp.diagnostics or {}).get("reason") == "no_trees"

    def test_llm_error_string_returns_empty_blocks(self, synthetic_trees_dir):
        backend = PageIndexBackend(
            trees_dir=str(synthetic_trees_dir),
            llm_call=lambda model, prompt: "Error",
        )
        resp = backend.retrieve(_request())
        assert resp.blocks == []
        per_tree = (resp.diagnostics or {}).get("per_tree") or []
        assert per_tree and per_tree[0].get("error") == "llm_error_response"

    def test_index_is_noop(self, synthetic_trees_dir):
        backend = PageIndexBackend(
            trees_dir=str(synthetic_trees_dir),
            llm_call=lambda model, prompt: "",
        )
        assert backend.index({"text": "irrelevant"}) is None
        assert backend.index("plain string") is None
