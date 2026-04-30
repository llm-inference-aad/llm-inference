"""Data ingestion utilities for building the RAG corpus."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence
import math

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
    for run_dir in runs_dir.glob("auto_*"):
        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue
        for checkpoint_file in checkpoint_dir.glob("checkpoint_gen_*.pkl"):
            yield checkpoint_file


def _chunk_text(text: str, max_words: int = 400) -> Iterator[str]:
    words = text.split()
    for start in range(0, len(words), max_words):
        yield " ".join(words[start : start + max_words]).strip()


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
        metrics_parts = []
        if len(fitness) > 0:
            try:
                metrics_parts.append(f"Test Acc: {float(fitness[0]):.4f}")
            except Exception:
                metrics_parts.append("Test Acc: N/A")
        if len(fitness) > 1:
            try:
                p = fitness[1]
                if isinstance(p, (int, float)) and math.isfinite(float(p)):
                    metrics_parts.append(f"Params: {int(p)}")
                else:
                    metrics_parts.append(f"Params: {p}")
            except Exception:
                metrics_parts.append("Params: N/A")
        if len(fitness) > 2:
            try:
                metrics_parts.append(f"Val Acc: {float(fitness[2]):.4f}")
            except Exception:
                metrics_parts.append("Val Acc: N/A")
        if len(fitness) > 3:
            try:
                metrics_parts.append(f"Train Time: {float(fitness[3]):.2f}s")
            except Exception:
                metrics_parts.append("Train Time: N/A")

        metrics = ", ".join(metrics_parts)
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
) -> List[MutationRecord]:
    runs_path = Path(runs_dir)
    models_path = Path(models_dir)
    records: dict[str, MutationRecord] = {}

    for idx, checkpoint_file in enumerate(_discover_checkpoint_files(runs_path)):
        if limit and idx >= limit:
            break
        checkpoint = _safe_load_checkpoint(checkpoint_file)
        if not checkpoint:
            continue

        global_data: Dict[str, dict] = checkpoint.get("GLOBAL_DATA", {})
        ancestry: Dict[str, dict] = checkpoint.get("GLOBAL_DATA_ANCESTRY", {})

        for gene_id, record in global_data.items():
            code_path = models_path / f"models/network_{gene_id}.py"
            if not code_path.exists():
                continue

            try:
                code = code_path.read_text(encoding="utf-8")
            except OSError:
                continue

            fitness = record.get("fitness")
            if fitness in (None, ()):
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
            description = build_mutation_description(gene_id, mutation_type, fitness, improvement)

            metadata = {
                "run_checkpoint": str(checkpoint_file),
                "fallback": record.get("fallback", False),
                "status": record.get("status"),
            }

            records[gene_id] = MutationRecord(
                gene_id=gene_id,
                parent_gene_id=parent_gene_id,
                mutation_type=mutation_type,
                code=code,
                fitness=fitness,
                improvement=improvement,
                description=description,
                metadata=metadata,
            )

    return list(records.values())


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
            if not chunk:
                continue
            documents.append(
                {
                    "content": chunk,
                    "metadata": {
                        "document_id": f"{pdf_file.stem}-{chunk_idx}",
                        "source": str(pdf_file),
                        "chunk_index": chunk_idx,
                        "type": "pdf",
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



