"""Inspection tests for what the augmented prompt actually looks like when the
code namespace is populated.

These tests build a small in-memory code corpus (no FAISS, no torch — uses the
existing FakeVectorStoreManager / FakeEmbeddingService from tests/rag/fakes.py),
run a real `PromptEnhancer.enhance_template` call against it, and capture the
exact augmented prompt that would be sent to the LLM.

Two purposes:

1. Snapshot what the prompt looks like end-to-end so we can read it (the file
   ``tests/rag/snapshots/augmented_prompt.txt`` is the canonical reference for
   "this is what the LLM sees with RAG on").
2. Lock in invariants: every retrieved mutation's *code* must appear in the
   output, the gene_id must appear, and the accuracy/improvement metadata must
   be surfaced. Together they guard against regressions where the code index
   silently goes from "shows code" to "shows only summaries" (a real bug we
   fixed when first writing these tests).

Also covers the build-time hygiene that makes the code index a reliable
reference for the eval replay:

3. AST-normalized hash dedup collapses reformatted/whitespace-only clones.
4. ``extract_mutations_from_checkpoints`` honors ``excluded_runs`` and
   ``excluded_code_hashes`` so the eval-time corpus can hold out the source
   runs of the eval subset.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Fixtures: a tiny code corpus + PromptEnhancer wired to it
# ---------------------------------------------------------------------------

# Sample mutation codes — small, syntactically valid, distinct enough to
# survive cosine-similarity ordering through the FakeEmbeddingService hash.
_CODE_GOOD_A = """\
class ConvBlockSiLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
"""

_CODE_GOOD_B = """\
class DepthwiseSeparable(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.pw(self.dw(x))
"""

_CODE_BAD = """\
class WideExplosion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c * 64, 1)

    def forward(self, x):
        return self.conv(x)
"""


def _build_enhancer():
    """Construct a PromptEnhancer wired to a FakeVectorStoreManager seeded with
    three mutations: two high-acc (EXEMPLAR), one regression (BAD).
    """
    # Local imports — keep heavy deps lazy and only inside fixture so the test
    # file is cheap to collect.
    from src.rag.retrieval import RetrievalService
    from src.rag.prompt_enhancer import PromptEnhancer, PromptEnhancerConfig
    from tests.rag.fakes import FakeEmbeddingService, FakeVectorStoreManager

    embeddings = FakeEmbeddingService()
    store = FakeVectorStoreManager()

    # Manually populate three mutations into the code namespace, using the
    # same content/metadata shape that data_ingestion.MutationRecord.to_document
    # produces.
    fixtures = [
        {
            "gene_id": "xXxGOODA",
            "code": _CODE_GOOD_A,
            "fitness": (0.821, 234567, 0.815, 102.3),
            "improvement": {"accuracy_delta": 0.012, "parameters_delta": 1234},
            "mutation_type": "LAYER",
            "quality_label": "EXEMPLAR",
        },
        {
            "gene_id": "xXxGOODB",
            "code": _CODE_GOOD_B,
            "fitness": (0.812, 198432, 0.806, 98.1),
            "improvement": {"accuracy_delta": 0.005, "parameters_delta": -5012},
            "mutation_type": "LAYER",
            "quality_label": "EXEMPLAR",
        },
        {
            "gene_id": "xXxBADXX",
            "code": _CODE_BAD,
            "fitness": (0.612, 9_876_543, 0.605, 145.0),
            "improvement": {"accuracy_delta": -0.024, "parameters_delta": 9_500_000},
            "mutation_type": "LAYER",
            "quality_label": "REGRESSION",
        },
    ]

    contents = []
    metadata = []
    for f in fixtures:
        # Shape that matches MutationRecord.to_document().
        content = f"Mutation {f['gene_id']} ({f['mutation_type']}) | Test Acc: {f['fitness'][0]:.4f}\n\nCode:\n{f['code'].strip()}"
        meta = {
            "document_id": f["gene_id"],
            "gene_id": f["gene_id"],
            "mutation_type": f["mutation_type"],
            "fitness": f["fitness"],
            "improvement": f["improvement"],
            "quality_label": f["quality_label"],
            "description": f"Mutation {f['gene_id']} ({f['mutation_type']}) | Test Acc: {f['fitness'][0]:.4f} | ΔAcc: {f['improvement']['accuracy_delta']:+.4f}",
        }
        contents.append(content)
        metadata.append(meta)

    code_embeddings = embeddings.embed_code(contents)
    store.add_code_documents(contents, code_embeddings, metadata)

    retrieval = RetrievalService(store=store, embeddings=embeddings)
    enhancer = PromptEnhancer(
        retrieval_service=retrieval,
        config=PromptEnhancerConfig(top_k=3, text_top_k=0, text_candidate_k=0),
    )
    return enhancer, fixtures


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Query code chosen to be lexically near _CODE_GOOD_A so cosine on the fake
# hash embeddings yields it first. The hash-based fakes don't model real
# semantic similarity — we're testing structure, not retrieval quality.
_QUERY = """\
class MyConvBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
"""


_TEMPLATE = """\
You are mutating a small CIFAR-10 classifier. Replace the highlighted block
with a new variant that improves test accuracy without exploding parameter
count. Output only the replacement class.

```python
class ConvBlock(nn.Module):
    pass
```
"""


def test_format_context_includes_retrieved_code(monkeypatch):
    """Every retrieved mutation's *code* must appear in format_context output.

    Regression guard: at one point format_context emitted only the
    description summary line, omitting the actual code despite the surrounding
    prompt claiming "code blocks follow." That silently degraded the with_rag
    arm to "see one-liners about historical mutations" with zero exemplar code.
    """
    monkeypatch.setenv("RAG_USE_CODE_CONTEXT", "true")
    monkeypatch.setenv("RAG_USE_TEXT_CONTEXT", "false")
    # Reload constants so the env override takes effect for this test.
    import src.cfg.constants as _c
    importlib.reload(_c)

    enhancer, fixtures = _build_enhancer()
    mutations = enhancer.retrieval.retrieve_similar_mutations(
        _QUERY, top_k=3, min_similarity=0.0,
    )
    assert len(mutations) >= 2, "fake retrieval should return at least 2 candidates"

    rendered = enhancer.retrieval.format_context(mutations)

    # Each retrieved mutation must contribute a recognizable code snippet
    # (the first 'class XxxName' line of its source) into the rendered block.
    for m in mutations:
        first_class_line = next(
            (l for l in m.code.splitlines() if l.startswith("class ")),
            None,
        )
        assert first_class_line is not None, f"fixture {m.gene_id} has no class line"
        assert first_class_line in rendered, (
            f"format_context output is missing the actual code for {m.gene_id}.\n"
            f"Expected to find: {first_class_line!r}\n"
            f"Rendered:\n{rendered}"
        )


def test_enhance_template_renders_full_prompt_with_code_and_metadata(monkeypatch, tmp_path):
    """End-to-end: enhance_template emits a prompt that contains the original
    template, the retrieved gene_ids, the ΔAcc metadata, AND the actual code.

    The full augmented prompt is also dumped to ``tmp_path/augmented_prompt.txt``
    so that running this test produces a copy of "what the LLM sees" for
    inspection.
    """
    monkeypatch.setenv("RAG_USE_CODE_CONTEXT", "true")
    monkeypatch.setenv("RAG_USE_TEXT_CONTEXT", "false")
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "false")
    import src.cfg.constants as _c
    importlib.reload(_c)

    enhancer, fixtures = _build_enhancer()
    augmented, mutations = enhancer.enhance_template(
        template=_TEMPLATE,
        mutation_type="LAYER",
        query_code=_QUERY,
        gene_id="testGene",
    )

    # Dump for human inspection (runs every time; cheap).
    out = tmp_path / "augmented_prompt.txt"
    out.write_text(augmented)
    print(f"\n[snapshot] full augmented prompt written to {out}\n")

    # Original template body still present.
    assert "Replace the highlighted block" in augmented

    # Retrieved gene_ids labeled.
    retrieved_ids = {m.gene_id for m in mutations}
    assert retrieved_ids, "expected at least one retrieved mutation"
    for gid in retrieved_ids:
        assert gid in augmented, f"missing gene_id {gid} from augmented prompt"

    # ΔAcc metadata visible.
    assert "ΔAcc" in augmented or "accuracy_delta" in augmented

    # Actual code blocks (not just descriptions) must be in the augmented prompt.
    for m in mutations:
        first_class_line = next(
            (l for l in m.code.splitlines() if l.startswith("class ")),
            None,
        )
        assert first_class_line is not None
        assert first_class_line in augmented, (
            f"augmented prompt is missing code for {m.gene_id}; "
            f"expected the line {first_class_line!r}.\n"
            f"--- prompt ---\n{augmented}\n--- end ---"
        )


# ---------------------------------------------------------------------------
# Build-time hygiene — AST hash + holdout
# ---------------------------------------------------------------------------

_CODE_REFORMATTED_CLONE = """\
# A reformatted clone — same semantics as _CODE_GOOD_A, different whitespace
# and an inserted comment.

class ConvBlockSiLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        # Same layers, different formatting.
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.bn   = nn.BatchNorm2d(c)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
"""


def test_ast_normalized_hash_collapses_reformatted_clones():
    """Two textually-different but AST-equivalent code strings must hash equal.

    Drives the dedup step that prevents an "excluded run" gene from leaking
    via a reformatted copy in a "kept" run.
    """
    from src.rag.data_ingestion import ast_normalized_hash

    h1 = ast_normalized_hash(_CODE_GOOD_A)
    h2 = ast_normalized_hash(_CODE_REFORMATTED_CLONE)
    assert h1 == h2, (
        f"AST-normalized hashes diverged for clones: {h1[:12]} vs {h2[:12]}"
    )

    # Negative control: a structurally different module hashes differently.
    h3 = ast_normalized_hash(_CODE_GOOD_B)
    assert h1 != h3


def test_extract_mutations_holdout_excludes_runs_and_hashes(tmp_path):
    """Build a synthetic two-run checkpoint layout and verify that
    ``excluded_runs`` and ``excluded_code_hashes`` both filter the output.
    """
    import pickle

    from src.rag.data_ingestion import (
        ast_normalized_hash,
        extract_mutations_from_checkpoints,
    )

    runs_dir = tmp_path / "runs"
    models_dir = tmp_path / "models" / "models"
    models_dir.mkdir(parents=True)

    def _write_run(run_name: str, gene_id: str, code: str, fitness):
        run_dir = runs_dir / run_name / "checkpoints"
        run_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / f"network_{gene_id}.py").write_text(code)
        chk = {
            "global_data": {
                gene_id: {"fitness": fitness, "fallback": False, "status": "ok"}
            },
            "ancestry": {
                gene_id: {"GENES": [gene_id], "MUTATE_TYPE": ["LAYER"]}
            },
        }
        with (run_dir / "checkpoint_gen_001.pkl").open("wb") as f:
            pickle.dump(chk, f)

    _write_run("runA_kept",     "xXxAlpha", _CODE_GOOD_A, (0.85, 100_000))
    _write_run("runB_excluded", "xXxBeta",  _CODE_GOOD_B, (0.84,  90_000))
    _write_run("runC_kept",     "xXxGamma", _CODE_REFORMATTED_CLONE, (0.83, 100_000))

    target_hashes = {ast_normalized_hash(_CODE_GOOD_A)}

    records = extract_mutations_from_checkpoints(
        runs_dir=str(runs_dir),
        models_dir=str(tmp_path / "models"),
        excluded_runs={"runB_excluded"},
        excluded_code_hashes=target_hashes,
    )
    gene_ids = {r.gene_id for r in records}

    # runB excluded by run-level holdout
    assert "xXxBeta"  not in gene_ids, "runB should be excluded by excluded_runs"
    # runA's xXxAlpha excluded by hash holdout (its hash IS in target_hashes)
    assert "xXxAlpha" not in gene_ids, "alpha should be excluded by hash holdout"
    # runC's xXxGamma is a reformatted clone of alpha → same AST hash → excluded
    assert "xXxGamma" not in gene_ids, (
        "gamma is an AST-normalized clone of alpha; must be excluded by hash holdout"
    )
