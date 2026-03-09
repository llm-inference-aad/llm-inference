from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from rag.pageindex_backend import PageIndexRetrievalBackend


def test_pageindex_backend_logs_and_retrieves_by_type(tmp_path: Path) -> None:
    backend = PageIndexRetrievalBackend(data_dir=str(tmp_path), model_name="gpt-test")

    backend.log_mutation_code(
        content="class Net: pass",
        metadata={
            "gene_id": "g1",
            "mutation_type": "Conservative",
            "fitness": [0.91, 12345],
            "description": "stable mutation",
        },
    )

    matches = backend.retrieve_by_mutation_type("conservative", limit=5)

    assert len(matches) == 1
    assert matches[0].gene_id == "g1"
    assert matches[0].metadata["mutation_type"] == "Conservative"


def test_pageindex_backend_lexical_similarity_retrieval(tmp_path: Path) -> None:
    backend = PageIndexRetrievalBackend(data_dir=str(tmp_path), model_name="gpt-test")

    backend.log_mutation_code(
        content="def conv_block(x):\n    return x\n",
        metadata={
            "gene_id": "g2",
            "mutation_type": "Structural",
            "fitness": [0.88, 22000],
            "description": "conv block mutation",
        },
    )
    backend.log_mutation_code(
        content="def attention_block(x):\n    return x\n",
        metadata={
            "gene_id": "g3",
            "mutation_type": "Structural",
            "fitness": [0.89, 20000],
            "description": "attention mutation",
        },
    )

    results = backend.retrieve_similar_mutations(
        query_code="def conv_block(y):\n    return y\n",
        top_k=3,
        min_similarity=0.2,
    )

    assert results
    assert results[0].gene_id == "g2"


def test_pageindex_backend_generates_tree_and_uses_tree_search(tmp_path: Path) -> None:
    class _FakeClient:
        def submit_document(self, file_path: str):
            return {"doc_id": "doc-1"}

        def get_tree(self, doc_id: str, node_summary: bool = True):
            return {
                "status": "completed",
                "retrieval_ready": True,
                "tree": {
                    "node_id": "0001",
                    "title": "Optimization Strategies",
                    "summary": "Methods for balancing accuracy and parameters",
                    "start_index": 3,
                    "end_index": 6,
                    "nodes": [],
                },
            }

        def submit_query(self, doc_id: str, query: str, thinking: bool = True):
            return {"retrieval_id": "retrieval-1"}

        def get_retrieval(self, retrieval_id: str):
            return {
                "status": "completed",
                "node_list": ["0001"],
            }

    backend = PageIndexRetrievalBackend(
        data_dir=str(tmp_path),
        model_name="gpt-test",
        api_key="fake-key",
        tree_timeout_sec=5,
        poll_interval_sec=0.01,
    )
    backend._sdk_available = True
    backend.client = _FakeClient()

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    indexed = backend.index_pdf_documents([str(pdf_path)])
    assert indexed == 1
    assert backend.tree_manifest_path.exists()
    assert any(path.suffix == ".json" for path in backend.tree_dir.iterdir())

    retrieved = backend.retrieve_similar_mutations(
        query_code="find sections about parameter-efficient optimization",
        top_k=3,
        min_similarity=0.0,
    )
    assert retrieved
    assert retrieved[0].metadata.get("source") == "pageindex_tree"
    assert retrieved[0].metadata.get("doc_id") == "doc-1"
