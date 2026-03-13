"""Lightweight code knowledge graph built from successful mutation artifacts."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
from networkx.readwrite import json_graph

from .vector_db import StoredDocument


KNOWN_LAYER_TYPES = {
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "Linear",
    "ReLU",
    "LeakyReLU",
    "GELU",
    "SiLU",
    "Dropout",
    "Dropout2d",
    "AvgPool2d",
    "MaxPool2d",
    "AdaptiveAvgPool2d",
    "GroupNorm",
    "LayerNorm",
}

CUSTOM_LAYER_HINTS = {
    "SEBlock",
    "SqueezeExcite",
    "MBConv",
    "Residual",
    "SkipConnection",
}


@dataclass(frozen=True)
class ExtractedCodeEntities:
    # Canonicalized layer/type names observed in AST call sites.
    layer_types: tuple[str, ...]
    # High-level architecture motifs inferred from syntax/class naming.
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class KGGeneHit:
    # Candidate mutation gene id.
    gene_id: str
    # Aggregate traversal score.
    score: float
    # Human-readable explanation of why this candidate matched.
    reason: str
    # Snapshot of candidate node metadata.
    metadata: dict


def _resolve_name(node: ast.AST) -> str:
    """Resolve nested AST call/attribute names (e.g., nn.Conv2d)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _resolve_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _resolve_name(node.func)
    return ""


class _EntityVisitor(ast.NodeVisitor):
    """Heuristic AST visitor to extract layer usage + architecture patterns."""
    def __init__(self) -> None:
        self.layer_types: set[str] = set()
        self.patterns: set[str] = set()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Pattern inference from class naming conventions.
        class_name = node.name.lower()
        if "seblock" in class_name or "squeeze" in class_name:
            self.patterns.add("squeeze_excite")
        if "skip" in class_name:
            self.patterns.add("skip_connection")
        if "residual" in class_name or class_name.startswith("res"):
            self.patterns.add("residual_connection")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Pattern hints from variable names.
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id.lower()
                if "skip" in name:
                    self.patterns.add("skip_connection")
                if "res" in name:
                    self.patterns.add("residual_connection")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # x + y often indicates a residual merge.
        if isinstance(node.op, ast.Add):
            self.patterns.add("residual_connection")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Capture known primitive/custom layers from call expressions.
        call_name = _resolve_name(node.func)
        leaf_name = call_name.split(".")[-1] if call_name else ""
        if call_name.startswith("nn.") and leaf_name in KNOWN_LAYER_TYPES:
            self.layer_types.add(f"nn.{leaf_name}")
        elif leaf_name in KNOWN_LAYER_TYPES:
            self.layer_types.add(leaf_name)
        elif leaf_name in CUSTOM_LAYER_HINTS:
            self.layer_types.add(leaf_name)

        lowered = leaf_name.lower()
        if "seblock" in lowered or "squeeze" in lowered:
            self.patterns.add("squeeze_excite")
        if "skip" in lowered:
            self.patterns.add("skip_connection")
        self.generic_visit(node)


def extract_entities_from_source(source_code: str) -> ExtractedCodeEntities:
    """Parse source and return normalized layer/pattern entities for graph indexing."""
    if not source_code.strip():
        return ExtractedCodeEntities(layer_types=(), patterns=())
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return ExtractedCodeEntities(layer_types=(), patterns=())

    visitor = _EntityVisitor()
    visitor.visit(tree)
    return ExtractedCodeEntities(
        layer_types=tuple(sorted(visitor.layer_types)),
        patterns=tuple(sorted(visitor.patterns)),
    )


def _parse_fitness_tuple(raw_fitness: Sequence[float] | None) -> tuple[float | None, float | None]:
    """Extract (accuracy, parameter_count) safely from fitness tuples/lists."""
    if not raw_fitness:
        return None, None
    try:
        accuracy = float(raw_fitness[0]) if len(raw_fitness) > 0 else None
    except (TypeError, ValueError):
        accuracy = None
    try:
        params = float(raw_fitness[1]) if len(raw_fitness) > 1 else None
    except (TypeError, ValueError):
        params = None
    return accuracy, params


def _extract_code_from_document(content: str) -> str:
    """Split mutation artifacts and keep only the code payload segment."""
    marker = "\n\nCode:\n"
    if marker not in content:
        return content
    return content.split(marker, 1)[1].strip()


class CodeKnowledgeGraph:
    """File-backed NetworkX graph for mutation architecture knowledge."""

    def __init__(self, rag_data_dir: str | Path):
        # Persist graph independently from vector index under rag_data/kg.
        self._base_dir = Path(rag_data_dir) / "kg"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self._base_dir / "code_kg.json"
        self.graph = nx.MultiDiGraph()
        self._last_signature: tuple[int, int] | None = None
        self._load_if_available()

    def _load_if_available(self) -> None:
        """Best-effort load of prior persisted graph snapshot."""
        if not self._graph_path.exists():
            return
        try:
            payload = json.loads(self._graph_path.read_text(encoding="utf-8"))
            self.graph = json_graph.node_link_graph(payload, edges="edges")
        except Exception:
            self.graph = nx.MultiDiGraph()

    def _persist(self) -> None:
        """Write current graph snapshot to disk as JSON node-link format."""
        payload = json_graph.node_link_data(self.graph, edges="edges")
        self._graph_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _documents_signature(self, documents: Sequence[StoredDocument]) -> tuple[int, int]:
        """Cheap change detector so unchanged corpora skip rebuild."""
        doc_hash = hash(tuple(sorted(doc.document_id for doc in documents))) if documents else 0
        return len(documents), doc_hash

    def build_from_documents(self, documents: Sequence[StoredDocument], force: bool = False) -> None:
        """Rebuild KG from mutation artifacts and persist typed nodes/edges."""
        signature = self._documents_signature(documents)
        if not force and self._last_signature == signature and self.graph.number_of_nodes() > 0:
            return

        graph = nx.MultiDiGraph()
        for doc in documents:
            metadata = doc.metadata or {}
            gene_id = str(metadata.get("gene_id") or metadata.get("document_id") or doc.document_id)
            fitness = metadata.get("fitness") or []
            accuracy, parameters = _parse_fitness_tuple(fitness)

            gene_node = f"gene:{gene_id}"
            # Gene nodes carry fitness and quality metadata for filtering.
            graph.add_node(
                gene_node,
                node_type="gene",
                gene_id=gene_id,
                accuracy=accuracy,
                parameters=parameters,
                fallback=bool(metadata.get("fallback", False)),
            )

            source_code = _extract_code_from_document(doc.content)
            entities = extract_entities_from_source(source_code)

            for layer in entities.layer_types:
                layer_node = f"layer:{layer}"
                graph.add_node(layer_node, node_type="layer", layer_type=layer)
                graph.add_edge(gene_node, layer_node, relation="uses")

            for pattern in entities.patterns:
                pattern_node = f"pattern:{pattern}"
                graph.add_node(pattern_node, node_type="pattern", pattern=pattern)
                graph.add_edge(gene_node, pattern_node, relation="uses")

            parent_gene_id = metadata.get("parent_gene_id")
            if parent_gene_id:
                parent_node = f"gene:{parent_gene_id}"
                graph.add_node(parent_node, node_type="gene", gene_id=str(parent_gene_id))
                graph.add_edge(gene_node, parent_node, relation="replaces")

                improvement = metadata.get("improvement") or {}
                acc_delta = improvement.get("accuracy_delta")
                params_delta = improvement.get("parameters_delta")
                if isinstance(acc_delta, (int, float)) and acc_delta > 0:
                    # Positive accuracy deltas become explicit optimization edges.
                    graph.add_edge(
                        gene_node,
                        parent_node,
                        relation="improves_accuracy",
                        delta=float(acc_delta),
                    )
                if isinstance(params_delta, (int, float)) and params_delta < 0:
                    # Negative parameter deltas indicate smaller/leaner models.
                    graph.add_edge(
                        gene_node,
                        parent_node,
                        relation="reduces_params",
                        delta=float(params_delta),
                    )

        self.graph = graph
        self._last_signature = signature
        self._persist()

    def related_successful_genes(
        self,
        parent_gene_id: str,
        top_k: int = 5,
        min_accuracy: float = 0.0,
    ) -> list[KGGeneHit]:
        """Rank related candidates from genealogy + shared feature traversal."""
        parent_node = f"gene:{parent_gene_id}"
        if parent_node not in self.graph:
            return []

        candidate_scores: dict[str, float] = {}
        candidate_reasons: dict[str, list[str]] = {}

        def register(gene_node: str, score: float, reason: str) -> None:
            if gene_node == parent_node:
                return
            candidate_scores[gene_node] = candidate_scores.get(gene_node, 0.0) + score
            candidate_reasons.setdefault(gene_node, []).append(reason)

        for child_node in self.graph.predecessors(parent_node):
            # Score direct descendants and optimization edges from the parent.
            edge_bundle = self.graph.get_edge_data(child_node, parent_node, default={})
            for edge_data in edge_bundle.values():
                relation = edge_data.get("relation")
                if relation == "replaces":
                    register(child_node, 1.0, "direct descendant")
                elif relation == "improves_accuracy":
                    delta = float(edge_data.get("delta", 0.0))
                    register(child_node, 0.5 + max(0.0, delta), f"improved accuracy {delta:+.4f}")
                elif relation == "reduces_params":
                    delta = abs(float(edge_data.get("delta", 0.0)))
                    register(child_node, 0.5 + min(1.0, delta / 100_000.0), "reduced params")

        shared_features = [
            node
            for node in self.graph.successors(parent_node)
            if self.graph.nodes[node].get("node_type") in {"layer", "pattern"}
        ]
        for feature_node in shared_features:
            # Also score candidates that share architectural features with parent.
            for gene_node in self.graph.predecessors(feature_node):
                edge_bundle = self.graph.get_edge_data(gene_node, feature_node, default={})
                for edge_data in edge_bundle.values():
                    if edge_data.get("relation") == "uses":
                        feature_kind = self.graph.nodes[feature_node].get("node_type")
                        feature_name = (
                            self.graph.nodes[feature_node].get("layer_type")
                            or self.graph.nodes[feature_node].get("pattern")
                            or feature_node
                        )
                        register(gene_node, 0.2, f"shares {feature_kind} {feature_name}")

        hits: list[KGGeneHit] = []
        for gene_node, score in candidate_scores.items():
            node_data = self.graph.nodes[gene_node]
            accuracy = node_data.get("accuracy")
            is_fallback = bool(node_data.get("fallback", False))
            if is_fallback:
                continue
            if accuracy is not None and float(accuracy) < min_accuracy:
                continue
            gene_id = str(node_data.get("gene_id", gene_node.replace("gene:", "")))
            reason = "; ".join(dict.fromkeys(candidate_reasons.get(gene_node, [])))
            hits.append(KGGeneHit(gene_id=gene_id, score=score, reason=reason, metadata=dict(node_data)))

        hits.sort(
            key=lambda hit: (
                hit.score,
                float(hit.metadata.get("accuracy") or 0.0),
            ),
            reverse=True,
        )
        return hits[:top_k]
