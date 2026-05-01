"""Tests for src/rag/api_types.py and src/rag/backend_protocol.py.

Covers:
- Dataclass round-trip: asdict -> json.dumps -> json.loads -> reconstruct -> equality.
- Required field validation: omitting a required field raises TypeError.
- Diagnostics dict accepts arbitrary JSON-safe nested keys.
- BackendProtocol compliance: a toy class implementing both methods passes
  isinstance(toy, BackendProtocol).
- Import smoke: neither api_types nor backend_protocol pulls in torch, faiss,
  or sentence_transformers.

Import strategy: we load api_types and backend_protocol via importlib.util so
we bypass src/rag/__init__.py, which imports heavy ML deps (sentence_transformers,
pdfplumber, etc.) not needed for pure-type tests.  This is intentional: the
acceptance criterion for PR 1 is that the *modules themselves* are importable
without heavy deps, not that the package __init__ is light.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import sys
import pathlib
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load api_types and backend_protocol directly from their files,
# bypassing src/rag/__init__.py which eagerly imports heavy ML dependencies.
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = pathlib.Path(__file__).parent.parent.parent

def _load_module(rel_path: str, mod_name: str):
    """Load a module from a file path, injecting it into sys.modules."""
    full_path = _WORKTREE_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, full_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Ensure src.rag exists as a stub namespace in sys.modules so that
# relative imports inside backend_protocol.py (from .api_types import ...)
# can resolve.  We only need the two new modules; we do NOT run __init__.py.
if "src" not in sys.modules:
    import types
    _src_pkg = types.ModuleType("src")
    _src_pkg.__path__ = [str(_WORKTREE_ROOT / "src")]  # type: ignore[attr-defined]
    _src_pkg.__package__ = "src"
    sys.modules["src"] = _src_pkg

if "src.rag" not in sys.modules:
    import types
    _rag_pkg = types.ModuleType("src.rag")
    _rag_pkg.__path__ = [str(_WORKTREE_ROOT / "src" / "rag")]  # type: ignore[attr-defined]
    _rag_pkg.__package__ = "src.rag"
    sys.modules["src.rag"] = _rag_pkg

# Load the two new modules.
_api_types_mod = _load_module("src/rag/api_types.py", "src.rag.api_types")
_backend_protocol_mod = _load_module(
    "src/rag/backend_protocol.py", "src.rag.backend_protocol"
)

# Bind the exported names so the rest of the test file can use normal imports.
AugmentRequest = _api_types_mod.AugmentRequest
AugmentResponse = _api_types_mod.AugmentResponse
RetrieveRequest = _api_types_mod.RetrieveRequest
RetrievedBlock = _api_types_mod.RetrievedBlock
RetrieveResponse = _api_types_mod.RetrieveResponse
augment_response_from_dict = _api_types_mod.augment_response_from_dict
retrieve_response_from_dict = _api_types_mod.retrieve_response_from_dict
RagRequest = _api_types_mod.RagRequest
RagResponse = _api_types_mod.RagResponse

BackendProtocol = _backend_protocol_mod.BackendProtocol


# ---------------------------------------------------------------------------
# Helper: build canonical instances for each dataclass
# ---------------------------------------------------------------------------

def _make_retrieve_request(**overrides) -> RetrieveRequest:
    defaults = dict(
        query="test query",
        namespace="pytorch_api",
        top_k=3,
        filters={"source_types": ["api", "pdf"]},
        run_id="run-001",
        request_id="req-abc",
    )
    defaults.update(overrides)
    return RetrieveRequest(**defaults)


def _make_retrieved_block(**overrides) -> RetrievedBlock:
    defaults = dict(
        kind="api_doc",
        document_id="torch.nn.Conv2d::api_summary",
        title="torch.nn.Conv2d",
        score=0.97,
        content="Applies a 2D convolution over an input signal.",
        diagnostics={"candidate_count": 24, "reranked": True, "source": "faiss"},
    )
    defaults.update(overrides)
    return RetrievedBlock(**defaults)


def _make_retrieve_response(**overrides) -> RetrieveResponse:
    defaults = dict(
        blocks=[_make_retrieved_block()],
        diagnostics={"reranker_used": True},
        latency_ms=21.4,
    )
    defaults.update(overrides)
    return RetrieveResponse(**defaults)


def _make_augment_request(**overrides) -> AugmentRequest:
    defaults = dict(
        template="Mutate the following code: {code}",
        mutation_type="Complex",
        query_code="class Net(nn.Module): pass",
        gene_id="gene-007",
        run_id="run-001",
        request_id="req-xyz",
    )
    defaults.update(overrides)
    return AugmentRequest(**defaults)


def _make_augment_response(**overrides) -> AugmentResponse:
    defaults = dict(
        augmented_prompt="Mutate the following code with context: ...",
        blocks_used=[_make_retrieved_block()],
        diagnostics={"blocks_injected": 1},
        latency_ms=42.0,
    )
    defaults.update(overrides)
    return AugmentResponse(**defaults)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

class TestRetrieveRequestRoundTrip:
    def test_asdict_json_reconstruct_equality(self):
        original = _make_retrieve_request()
        d = dataclasses.asdict(original)
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        reconstructed = RetrieveRequest(**loaded)
        assert reconstructed == original

    def test_minimal_fields_round_trip(self):
        """Only the required field (query) is set; all others default."""
        original = RetrieveRequest(query="minimal")
        d = dataclasses.asdict(original)
        reconstructed = RetrieveRequest(**json.loads(json.dumps(d)))
        assert reconstructed == original
        assert reconstructed.namespace is None
        assert reconstructed.top_k == 5
        assert reconstructed.filters is None


class TestRetrievedBlockRoundTrip:
    def test_asdict_json_reconstruct_equality(self):
        original = _make_retrieved_block()
        d = dataclasses.asdict(original)
        reconstructed = RetrievedBlock(**json.loads(json.dumps(d)))
        assert reconstructed == original

    def test_diagnostics_none_round_trip(self):
        original = RetrievedBlock(
            kind="code",
            document_id="doc-1",
            title="example",
            score=0.5,
            content="pass",
        )
        d = dataclasses.asdict(original)
        reconstructed = RetrievedBlock(**json.loads(json.dumps(d)))
        assert reconstructed == original
        assert reconstructed.diagnostics is None


class TestRetrieveResponseRoundTrip:
    def test_asdict_json_reconstruct_equality(self):
        original = _make_retrieve_response()
        d = dataclasses.asdict(original)
        reconstructed = retrieve_response_from_dict(json.loads(json.dumps(d)))
        assert reconstructed == original

    def test_empty_blocks_round_trip(self):
        original = RetrieveResponse(blocks=[], latency_ms=0.0)
        d = dataclasses.asdict(original)
        reconstructed = retrieve_response_from_dict(json.loads(json.dumps(d)))
        assert reconstructed == original

    def test_multiple_blocks_preserved_order(self):
        blocks = [
            _make_retrieved_block(score=0.9, document_id="doc-1"),
            _make_retrieved_block(score=0.7, document_id="doc-2"),
            _make_retrieved_block(score=0.5, document_id="doc-3"),
        ]
        original = RetrieveResponse(blocks=blocks)
        d = dataclasses.asdict(original)
        reconstructed = retrieve_response_from_dict(json.loads(json.dumps(d)))
        assert [b.document_id for b in reconstructed.blocks] == ["doc-1", "doc-2", "doc-3"]


class TestAugmentRequestRoundTrip:
    def test_asdict_json_reconstruct_equality(self):
        original = _make_augment_request()
        d = dataclasses.asdict(original)
        reconstructed = AugmentRequest(**json.loads(json.dumps(d)))
        assert reconstructed == original

    def test_minimal_fields_round_trip(self):
        original = AugmentRequest(
            template="template",
            mutation_type="Param",
            query_code="x = 1",
        )
        d = dataclasses.asdict(original)
        reconstructed = AugmentRequest(**json.loads(json.dumps(d)))
        assert reconstructed == original
        assert reconstructed.gene_id is None
        assert reconstructed.run_id is None


class TestAugmentResponseRoundTrip:
    def test_asdict_json_reconstruct_equality(self):
        original = _make_augment_response()
        d = dataclasses.asdict(original)
        reconstructed = augment_response_from_dict(json.loads(json.dumps(d)))
        assert reconstructed == original

    def test_empty_blocks_used_round_trip(self):
        original = AugmentResponse(
            augmented_prompt="prompt without context",
            blocks_used=[],
        )
        d = dataclasses.asdict(original)
        reconstructed = augment_response_from_dict(json.loads(json.dumps(d)))
        assert reconstructed == original


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------

class TestRequiredFieldValidation:
    def test_retrieve_request_missing_query_raises(self):
        with pytest.raises(TypeError):
            RetrieveRequest()  # type: ignore[call-arg]

    def test_retrieved_block_missing_kind_raises(self):
        with pytest.raises(TypeError):
            RetrievedBlock(  # type: ignore[call-arg]
                document_id="d", title="t", score=1.0, content="c"
            )

    def test_retrieved_block_missing_document_id_raises(self):
        with pytest.raises(TypeError):
            RetrievedBlock(  # type: ignore[call-arg]
                kind="api_doc", title="t", score=1.0, content="c"
            )

    def test_retrieved_block_missing_score_raises(self):
        with pytest.raises(TypeError):
            RetrievedBlock(  # type: ignore[call-arg]
                kind="api_doc", document_id="d", title="t", content="c"
            )

    def test_retrieve_response_missing_blocks_raises(self):
        with pytest.raises(TypeError):
            RetrieveResponse()  # type: ignore[call-arg]

    def test_augment_request_missing_template_raises(self):
        with pytest.raises(TypeError):
            AugmentRequest(mutation_type="Complex", query_code="x")  # type: ignore[call-arg]

    def test_augment_request_missing_mutation_type_raises(self):
        with pytest.raises(TypeError):
            AugmentRequest(template="t", query_code="x")  # type: ignore[call-arg]

    def test_augment_request_missing_query_code_raises(self):
        with pytest.raises(TypeError):
            AugmentRequest(template="t", mutation_type="Complex")  # type: ignore[call-arg]

    def test_augment_response_missing_augmented_prompt_raises(self):
        with pytest.raises(TypeError):
            AugmentResponse(blocks_used=[])  # type: ignore[call-arg]

    def test_augment_response_missing_blocks_used_raises(self):
        with pytest.raises(TypeError):
            AugmentResponse(augmented_prompt="p")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------

class TestFrozenDataclasses:
    def test_retrieve_request_is_frozen(self):
        req = _make_retrieve_request()
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.query = "mutated"  # type: ignore[misc]

    def test_retrieved_block_is_frozen(self):
        block = _make_retrieved_block()
        with pytest.raises(dataclasses.FrozenInstanceError):
            block.score = 0.0  # type: ignore[misc]

    def test_retrieve_response_is_frozen(self):
        resp = _make_retrieve_response()
        with pytest.raises(dataclasses.FrozenInstanceError):
            resp.latency_ms = 999.0  # type: ignore[misc]

    def test_augment_request_is_frozen(self):
        req = _make_augment_request()
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.template = "mutated"  # type: ignore[misc]

    def test_augment_response_is_frozen(self):
        resp = _make_augment_response()
        with pytest.raises(dataclasses.FrozenInstanceError):
            resp.augmented_prompt = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Diagnostics dict accepts arbitrary JSON-safe nested keys
# ---------------------------------------------------------------------------

class TestDiagnosticsFlexibility:
    def test_diagnostics_arbitrary_nested_keys(self):
        diag: dict[str, Any] = {
            "reranker_used": True,
            "candidate_count": 42,
            "nested": {"a": 1, "b": [2, 3]},
            "unicode_key_\u00e9": "ok",
            "float_val": 3.14,
            "null_val": None,
        }
        block = RetrievedBlock(
            kind="code",
            document_id="doc-x",
            title="test",
            score=0.8,
            content="content",
            diagnostics=diag,
        )
        # Should round-trip through JSON without error
        d = dataclasses.asdict(block)
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        reconstructed = RetrievedBlock(**loaded)
        assert reconstructed.diagnostics == diag

    def test_response_diagnostics_arbitrary_keys(self):
        diag = {"latency_breakdown": {"embed_ms": 5.0, "search_ms": 12.0}, "hits": 7}
        resp = RetrieveResponse(blocks=[], diagnostics=diag, latency_ms=17.0)
        d = dataclasses.asdict(resp)
        reconstructed = retrieve_response_from_dict(json.loads(json.dumps(d)))
        assert reconstructed.diagnostics == diag

    def test_augment_response_diagnostics_arbitrary_keys(self):
        diag = {"blocks_injected": 3, "policy": "default_v1", "extra": {"flag": False}}
        resp = AugmentResponse(
            augmented_prompt="p",
            blocks_used=[],
            diagnostics=diag,
            latency_ms=55.0,
        )
        d = dataclasses.asdict(resp)
        reconstructed = augment_response_from_dict(json.loads(json.dumps(d)))
        assert reconstructed.diagnostics == diag


# ---------------------------------------------------------------------------
# BackendProtocol compliance
# ---------------------------------------------------------------------------

class _ToyBackend:
    """Minimal class that structurally satisfies BackendProtocol."""

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        return RetrieveResponse(blocks=[], latency_ms=0.0)

    def index(self, document: Any) -> None:
        pass


class _MissingIndexBackend:
    """Class that has retrieve but NOT index — should not satisfy the protocol."""

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        return RetrieveResponse(blocks=[], latency_ms=0.0)


class _MissingRetrieveBackend:
    """Class that has index but NOT retrieve."""

    def index(self, document: Any) -> None:
        pass


class TestBackendProtocolCompliance:
    def test_toy_backend_is_instance_of_protocol(self):
        toy = _ToyBackend()
        assert isinstance(toy, BackendProtocol)

    def test_missing_index_fails_isinstance(self):
        obj = _MissingIndexBackend()
        assert not isinstance(obj, BackendProtocol)

    def test_missing_retrieve_fails_isinstance(self):
        obj = _MissingRetrieveBackend()
        assert not isinstance(obj, BackendProtocol)

    def test_plain_object_fails_isinstance(self):
        assert not isinstance(object(), BackendProtocol)

    def test_toy_backend_retrieve_returns_response(self):
        toy = _ToyBackend()
        req = RetrieveRequest(query="hello")
        resp = toy.retrieve(req)
        assert isinstance(resp, RetrieveResponse)
        assert resp.blocks == []

    def test_toy_backend_index_does_not_raise(self):
        toy = _ToyBackend()
        toy.index({"any": "document"})  # should complete without error


# ---------------------------------------------------------------------------
# Import smoke: verify the module FILES themselves have no heavy ML deps
# ---------------------------------------------------------------------------

class TestNoHeavyImports:
    """Verify that api_types.py and backend_protocol.py do not transitively
    pull in torch, faiss, or sentence_transformers when loaded in isolation.

    We use the already-loaded modules from module-level setup above, which
    loaded them via importlib.util bypassing src/rag/__init__.py.  If those
    files imported heavy deps they would already be in sys.modules here.
    """

    def test_torch_not_imported(self):
        # If api_types or backend_protocol imported torch, it would be present.
        assert "torch" not in sys.modules, (
            "api_types.py or backend_protocol.py pulled in torch — "
            "these files must be importable in zero-ML-dep environments"
        )

    def test_faiss_not_imported(self):
        assert "faiss" not in sys.modules, (
            "api_types.py or backend_protocol.py pulled in faiss"
        )

    def test_sentence_transformers_not_imported(self):
        assert "sentence_transformers" not in sys.modules, (
            "api_types.py or backend_protocol.py pulled in sentence_transformers"
        )

    def test_api_types_exports_all_expected_names(self):
        """api_types module exports the full set of types specified in the plan."""
        mod = _api_types_mod
        assert hasattr(mod, "RetrieveRequest")
        assert hasattr(mod, "RetrievedBlock")
        assert hasattr(mod, "RetrieveResponse")
        assert hasattr(mod, "AugmentRequest")
        assert hasattr(mod, "AugmentResponse")
        assert hasattr(mod, "RagRequest")
        assert hasattr(mod, "RagResponse")
        assert hasattr(mod, "retrieve_response_from_dict")
        assert hasattr(mod, "augment_response_from_dict")

    def test_backend_protocol_exports_protocol_class(self):
        mod = _backend_protocol_mod
        assert hasattr(mod, "BackendProtocol")

    def test_union_type_aliases_defined(self):
        """RagRequest and RagResponse are Union aliases (not class types)."""
        import typing
        # get_args works on Union types in Python 3.10+; on 3.9 we check __args__
        args = getattr(RagRequest, "__args__", None)
        assert args is not None, "RagRequest should be a Union type alias"
        assert AugmentRequest in args
        assert RetrieveRequest in args

        args2 = getattr(RagResponse, "__args__", None)
        assert args2 is not None, "RagResponse should be a Union type alias"
        assert AugmentResponse in args2
        assert RetrieveResponse in args2
