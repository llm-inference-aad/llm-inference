"""Data ingestion utilities for building the RAG corpus."""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set

import pdfplumber


FitnessTuple = Sequence[float]


def _safe_load_checkpoint(path: Path) -> dict | None:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # pragma: no cover - logging handled by caller
        print(f"[RAG] Failed to load checkpoint '{path}': {exc}")
        return None


def _discover_checkpoint_files(runs_dir: Path) -> Iterator[Path]:
    """Yield ``checkpoint_gen_*.pkl`` files under any direct subdir of *runs_dir*
    that owns a ``checkpoints/`` directory.

    Previously restricted to ``auto_*`` runs; widened so the eval-time replay
    can ingest from any historical run directory (``nemotron_*``, etc.) and so
    the holdout filter in :func:`extract_mutations_from_checkpoints` has a
    surface to act on.
    """
    if not runs_dir.exists():
        return
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.is_dir():
            continue
        for checkpoint_file in sorted(checkpoint_dir.glob("checkpoint_gen_*.pkl")):
            yield checkpoint_file


def ast_normalized_hash(code: str) -> str:
    """Return a deterministic sha256 hash over the AST of *code*.

    Whitespace, comments, and docstrings do not contribute to the hash, so
    two textually-different but semantically-identical modules collide. This
    is the dedup key used to prevent eval-time leakage when a "kept" run
    contains a reformatted clone of a gene from an "excluded" run.

    Falls back to a byte-level sha256 if *code* is not parseable as Python —
    in that case we still want a hash, just one that doesn't collapse
    formatting differences.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    # Strip module-/class-/function-level docstrings — they are pure prose
    # and shouldn't drive identity. Also strip the location attributes that
    # ast.dump bakes in by default; those carry whitespace info via column
    # offsets.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]

    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _chunk_text(text: str, max_words: int = 400) -> Iterator[str]:
    words = text.split()
    for start in range(0, len(words), max_words):
        yield " ".join(words[start : start + max_words]).strip()


def _stable_document_id(*parts: str) -> str:
    joined = "|".join(part or "" for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(len(text), 1)


def _alpha_word_ratio(text: str) -> float:
    words = [word for word in re.findall(r"\b\w+\b", text)]
    if not words:
        return 0.0
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    return len(alpha_words) / len(words)


def _looks_like_low_value_pdf_chunk(text: str) -> bool:
    normalized = _normalize_whitespace(text).lower()
    if not normalized:
        return True
    if len(normalized.split()) < 40:
        return True
    if _digit_ratio(normalized) > 0.30:
        return True
    if _alpha_word_ratio(normalized) < 0.55:
        return True

    noisy_markers = (
        "precision recall f1-score support",
        "macro avg",
        "weighted avg",
        "confusion matrix",
        "latency (ms)",
    )
    if any(marker in normalized for marker in noisy_markers):
        return True
    if normalized.count("table") >= 2:
        return True
    return False


def _extract_doc_section(docstring: str, headings: Sequence[str]) -> str:
    if not docstring:
        return ""
    pattern = r"(?im)^(" + "|".join(re.escape(heading) for heading in headings) + r"):\s*$"
    matches = list(re.finditer(pattern, docstring))
    if not matches:
        return ""
    start_match = matches[0]
    start = start_match.end()
    next_heading_pattern = r"(?im)^(Args|Arguments|Parameters|Shape|Returns|Return|Examples?|Example):\s*$"
    next_match = re.search(next_heading_pattern, docstring[start:])
    end = start + next_match.start() if next_match else len(docstring)
    return docstring[start:end].strip()


def _docstring_summary(docstring: str) -> str:
    if not docstring:
        return ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", docstring) if part.strip()]
    return paragraphs[0] if paragraphs else docstring.strip()


def _append_section_documents(
    documents: list[dict],
    content: str,
    base_metadata: dict,
    *,
    doc_type: str,
    section_name: str,
    chunk_words: int,
    chunk_threshold_words: int,
) -> None:
    cleaned = content.strip()
    if not cleaned:
        return

    if len(cleaned.split()) > chunk_threshold_words:
        chunks = list(_chunk_text(cleaned, chunk_words))
    else:
        chunks = [cleaned]

    for chunk_index, chunk in enumerate(chunks):
        documents.append(
            {
                "content": chunk,
                "metadata": {
                    **base_metadata,
                    "doc_type": doc_type,
                    "section_name": section_name,
                    "chunk_index": chunk_index,
                    "document_id": _stable_document_id(
                        "pytorch",
                        base_metadata.get("name", ""),
                        doc_type,
                        section_name,
                        str(chunk_index),
                    ),
                },
            }
        )


def calculate_fitness_improvement(
    current: FitnessTuple | None, parent: FitnessTuple | None
) -> dict | None:
    if not current or not parent:
        return None
    try:
        accuracy_delta = float(current[0]) - float(parent[0])
        params_delta = float(current[1]) - float(parent[1]) if len(current) > 1 else None
        return {
            "accuracy_delta": accuracy_delta,
            "parameters_delta": params_delta,
        }
    except (TypeError, ValueError, IndexError):
        return None


def build_mutation_description(
    gene_id: str,
    mutation_type: str | None,
    fitness: FitnessTuple | None,
    improvement: dict | None,
) -> str:
    headline = f"Mutation {gene_id}"
    if mutation_type:
        headline = f"{headline} ({mutation_type})"

    metrics_line = ""
    if fitness:
        metrics = ", ".join(
            [
                f"Test Acc: {fitness[0]:.4f}" if len(fitness) > 0 else "",
                f"Params: {int(fitness[1])}" if len(fitness) > 1 else "",
                f"Val Acc: {fitness[2]:.4f}" if len(fitness) > 2 else "",
                f"Train Time: {fitness[3]:.2f}s" if len(fitness) > 3 else "",
            ]
        )
        metrics_line = metrics.replace(",,", ",").strip(", ")

    improvement_line = ""
    if improvement:
        acc_delta = improvement.get("accuracy_delta")
        params_delta = improvement.get("parameters_delta")
        parts = []
        if acc_delta is not None:
            parts.append(f"ΔAcc: {acc_delta:+.4f}")
        if params_delta is not None:
            parts.append(f"ΔParams: {params_delta:+.0f}")
        improvement_line = ", ".join(parts)

    summary_parts = [headline]
    if metrics_line:
        summary_parts.append(metrics_line)
    if improvement_line:
        summary_parts.append(improvement_line)
    return " | ".join(summary_parts)


@dataclass(frozen=True)
class MutationRecord:
    gene_id: str
    parent_gene_id: str | None
    mutation_type: str | None
    code: str
    fitness: FitnessTuple | None
    improvement: dict | None
    description: str
    metadata: dict

    def to_document(self) -> tuple[str, dict]:
        metadata = {
            "document_id": self.gene_id,
            "gene_id": self.gene_id,
            "parent_gene_id": self.parent_gene_id,
            "mutation_type": self.mutation_type,
            "fitness": self.fitness,
            "improvement": self.improvement,
            **self.metadata,
        }
        content = f"{self.description}\n\nCode:\n{self.code.strip()}"
        return content, metadata


def extract_mutations_from_checkpoints(
    runs_dir: str,
    models_dir: str,
    limit: int | None = None,
    min_accuracy: float = 0.0,
    *,
    excluded_runs: Optional[Iterable[str]] = None,
    excluded_code_hashes: Optional[Iterable[str]] = None,
    require_quality_gate: bool = True,
) -> List[MutationRecord]:
    """Parse DEAP checkpoints across run directories and extract mutation pairs.

    Reads checkpoint_gen_*.pkl files to reconstruct the genealogy and extracts
    the before/after code states for each successful mutation.

    Holdout knobs (used by the eval-time replay to prevent leakage):

    * ``excluded_runs`` — set of run-dir names whose checkpoints are skipped
      entirely. Use this to drop the runs that produced the eval subset.
    * ``excluded_code_hashes`` — set of AST-normalized code hashes; any gene
      whose code matches is dropped, even if its run is "kept". Catches
      reformatted clones that survive the run-level filter.
    * ``require_quality_gate`` — when True (default), the seed-baseline
      quality gate filters out genes that didn't beat the seed on either
      accuracy or parameters. Set False to keep regressions / fallbacks for
      use as "avoid" exemplars in a partitioned-retrieval setup.

    Each emitted ``MutationRecord`` carries a ``quality_label`` in its
    metadata: one of ``EXEMPLAR``, ``BASELINE_OK``, ``REGRESSION``,
    ``FALLBACK``, or ``INVALID``. The label is computed at index time from
    the gene's fitness vs its parent's; downstream prompt-formatting code
    surfaces it so the LLM knows whether a retrieved snippet is a
    "model-after" or "avoid" example.
    """
    try:
        from evolution.seed import train_seed_network_baseline
        seed_stats = train_seed_network_baseline()
    except Exception:
        seed_stats = None

    seed_accuracy = seed_stats[0] if seed_stats else min_accuracy
    seed_params = seed_stats[1] if seed_stats else float('inf')

    runs_path = Path(runs_dir)
    models_path = Path(models_dir)

    excluded_run_set: Set[str] = set(excluded_runs or ())
    excluded_hash_set: Set[str] = set(excluded_code_hashes or ())

    mutations: dict[str, MutationRecord] = {}
    seen_genes: set[str] = set()
    seen_code_hashes: set[str] = set()

    holdout_drops_run = 0
    holdout_drops_hash = 0

    print(f"[RAG] Scanning for checkpoints in {runs_path}")
    print(f"[RAG] Base seed threshold: accuracy > {seed_accuracy} OR params < {seed_params}")
    if excluded_run_set:
        print(f"[RAG] Run-level holdout: excluding {len(excluded_run_set)} run(s)")
    if excluded_hash_set:
        print(f"[RAG] Hash-level holdout: excluding {len(excluded_hash_set)} target code hash(es)")

    for idx, checkpoint_file in enumerate(_discover_checkpoint_files(runs_path)):
        # checkpoint_file is .../<run_name>/checkpoints/checkpoint_gen_*.pkl
        run_name = checkpoint_file.parent.parent.name
        if run_name in excluded_run_set:
            holdout_drops_run += 1
            continue

        if limit and idx >= limit:
            break
        chk = _safe_load_checkpoint(checkpoint_file)
        if not chk:
            continue

        global_data: Dict[str, dict] = chk.get("global_data", {})
        ancestry: Dict[str, dict] = chk.get("ancestry", {})

        for gene_id, record in global_data.items():
            # Skip if already processed from another checkpoint
            if gene_id in seen_genes:
                continue
            seen_genes.add(gene_id)

            code_path = models_path / f"models/network_{gene_id}.py"
            if not code_path.exists():
                continue

            try:
                code = code_path.read_text(encoding="utf-8")
            except OSError:
                continue

            code_hash = ast_normalized_hash(code)
            if code_hash in excluded_hash_set:
                holdout_drops_hash += 1
                continue
            if code_hash in seen_code_hashes:
                continue
            seen_code_hashes.add(code_hash)

            fitness = record.get("fitness")
            if fitness in (None, ()):
                continue

            try:
                test_acc = float(fitness[0])
                params_count = float(fitness[1]) if len(fitness) > 1 else float('inf')
            except (TypeError, ValueError, IndexError):
                continue

            ancestry_info = ancestry.get(gene_id, {})
            genes = ancestry_info.get("GENES") or []
            mutation_types = ancestry_info.get("MUTATE_TYPE") or []
            parent_gene_id = genes[-2] if len(genes) >= 2 else (genes[0] if genes else None)
            mutation_type = mutation_types[-1] if mutation_types else None

            parent_fitness = None
            if parent_gene_id and parent_gene_id in global_data:
                parent_fitness = global_data[parent_gene_id].get("fitness")

            improvement = calculate_fitness_improvement(fitness, parent_fitness)

            quality_label = _classify_quality(
                test_acc=test_acc,
                seed_accuracy=seed_accuracy,
                params_count=params_count,
                seed_params=seed_params,
                improvement=improvement,
                fallback=bool(record.get("fallback", False)),
                status=record.get("status"),
            )

            if require_quality_gate and quality_label in {"REGRESSION", "FALLBACK", "INVALID"}:
                continue

            description = build_mutation_description(gene_id, mutation_type, fitness, improvement)

            metadata = {
                "run_name": run_name,
                "run_checkpoint": str(checkpoint_file),
                "code_ast_hash": code_hash,
                "quality_label": quality_label,
                "fallback": bool(record.get("fallback", False)),
                "status": record.get("status"),
            }

            mutations[gene_id] = MutationRecord(
                gene_id=gene_id,
                parent_gene_id=parent_gene_id,
                mutation_type=mutation_type,
                code=code,
                fitness=fitness,
                improvement=improvement,
                description=description,
                metadata=metadata,
            )

    if excluded_run_set or excluded_hash_set:
        print(
            f"[RAG] Holdout drops: {holdout_drops_run} checkpoint(s) by run, "
            f"{holdout_drops_hash} gene(s) by hash"
        )
    return list(mutations.values())


def _classify_quality(
    *,
    test_acc: float,
    seed_accuracy: float,
    params_count: float,
    seed_params: float,
    improvement: dict | None,
    fallback: bool,
    status,
) -> str:
    """Assign a coarse quality label used for prompt-time good/bad framing.

    Five categories:
      - ``EXEMPLAR``    — beats seed on accuracy *and* improved over parent
      - ``BASELINE_OK`` — beats seed (or matches it) without parent regression
      - ``REGRESSION``  — significant accuracy regression vs parent
      - ``FALLBACK``    — LLM exhausted retries during generation
      - ``INVALID``     — training crashed / no fitness
    """
    if fallback:
        return "FALLBACK"
    if status in (None, "error", "failed"):
        return "INVALID"

    acc_delta = (improvement or {}).get("accuracy_delta")
    has_better_acc = test_acc > seed_accuracy
    has_fewer_params = params_count < seed_params

    if acc_delta is not None and acc_delta < -0.01:
        return "REGRESSION"
    if has_better_acc and (acc_delta is None or acc_delta > 0):
        return "EXEMPLAR"
    if has_better_acc or has_fewer_params:
        return "BASELINE_OK"
    return "REGRESSION"


def process_pdfs(pdf_dir: str, max_pages: int | None = None) -> List[dict]:
    documents: list[dict] = []
    pdf_path = Path(pdf_dir)
    for pdf_file in pdf_path.glob("*.pdf"):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                page_texts = []
                for idx, page in enumerate(pdf.pages):
                    if max_pages and idx >= max_pages:
                        break
                    text = page.extract_text() or ""
                    page_texts.append(text)
                full_text = "\n".join(page_texts)
        except Exception as exc:
            print(f"[RAG] Failed to parse PDF '{pdf_file}': {exc}")
            continue

        for chunk_idx, chunk in enumerate(_chunk_text(full_text, max_words=400)):
            if not chunk or _looks_like_low_value_pdf_chunk(chunk):
                continue
            documents.append(
                {
                    "content": chunk,
                    "metadata": {
                        "document_id": f"{pdf_file.stem}-{chunk_idx}",
                        "source": str(pdf_file),
                        "chunk_index": chunk_idx,
                        "type": "pdf",
                        "doc_type": "pdf_chunk",
                        "source_type": "pdf",
                    },
                }
            )
    return documents


def load_manifest(manifest_path: str) -> List[dict]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ingest_extracted_json(json_path: str) -> List[dict]:
    """
    Load and enrich documentation from the extracted JSON format.

    This function reads the specifically formatted JSON (containing module, 
    section, and content fields), contextually enriches the content by 
    prepending the module and section headers, and structures the result 
    for the retrieval service.
    """
    path = Path(json_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    documents = []
    for entry in raw_data:
        # Contextual Enrichment: Prepend module and section to the content
        # so embedding models understand the context of independent chunks.
        module = entry.get("module", "Unknown")
        section = entry.get("section", "General")
        raw_content = entry.get("content", "")
        
        enriched_content = f"{module} {section}:\n{raw_content}"
        
        # Preserve all original metadata and add identification
        metadata = entry.get("metadata", {}).copy()
        metadata.update({
            "source": str(path),
            "module": module,
            "section": section,
            "type": "documentation",
            "original_content_type": entry.get("content_type")
        })
        metadata.setdefault(
            "document_id",
            _stable_document_id(
                str(path),
                module,
                section,
                str(entry.get("content_type", "")),
                raw_content[:200],
            ),
        )

        documents.append({
            "content": enriched_content,
            "metadata": metadata
        })

    return documents

def ingest_pytorch_docs(
    json_path: str,
    *,
    chunk_words: int = 400,
    chunk_threshold_words: int = 500,
) -> List[dict]:
    """
    Ingest documentation generated by src/rag/indexer.py.
    
    Expected schema:
    [
      {
        "name": "torch.nn.Conv2d",
        "type": "class",
        "signature": "...",
        "docstring": "...",
        "example": "...",
        "embedding_text": "...",
        "metadata": {...}
      }
    ]
    """
    path = Path(json_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    documents = []
    for entry in raw_data:
        name = entry.get("name", "Unknown")
        sig = entry.get("signature", "")
        doc = entry.get("docstring", "")
        meta = entry.get("metadata", {}).copy()
        meta.update(
            {
                "source": str(path),
                "source_type": "api",
                "name": name,
                "entity_type": entry.get("type", "unknown"),
            }
        )

        summary = _docstring_summary(doc)
        summary_content = _normalize_whitespace(f"{name}{sig}\n{summary}")
        _append_section_documents(
            documents,
            summary_content,
            meta,
            doc_type="api_summary",
            section_name="summary",
            chunk_words=chunk_words,
            chunk_threshold_words=chunk_threshold_words,
        )

        parameters = _extract_doc_section(doc, ("Args", "Arguments", "Parameters"))
        if parameters:
            _append_section_documents(
                documents,
                _normalize_whitespace(f"{name}{sig}\nParameters:\n{parameters}"),
                meta,
                doc_type="api_parameters",
                section_name="parameters",
                chunk_words=chunk_words,
                chunk_threshold_words=chunk_threshold_words,
            )

        behavior_parts: list[str] = []
        returns = _extract_doc_section(doc, ("Returns", "Return"))
        shape = _extract_doc_section(doc, ("Shape",))
        if returns:
            behavior_parts.append(f"Returns:\n{returns}")
        if shape:
            behavior_parts.append(f"Shape:\n{shape}")
        if behavior_parts:
            _append_section_documents(
                documents,
                _normalize_whitespace(f"{name}{sig}\n" + "\n\n".join(behavior_parts)),
                meta,
                doc_type="api_behavior",
                section_name="behavior",
                chunk_words=chunk_words,
                chunk_threshold_words=chunk_threshold_words,
            )

        example = entry.get("example", "") or _extract_doc_section(doc, ("Examples", "Example"))
        if example and len(example) > 10:
            _append_section_documents(
                documents,
                _normalize_whitespace(f"Example for {name}:\n{example}"),
                meta,
                doc_type="code_example",
                section_name="example",
                chunk_words=chunk_words,
                chunk_threshold_words=chunk_threshold_words,
            )

    return documents
