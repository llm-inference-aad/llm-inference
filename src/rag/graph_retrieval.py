"""Standalone graph-based retrieval service for experimental Graph-RAG runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .code_kg import CodeKnowledgeGraph, extract_entities_from_source
from .vector_db import StoredDocument, VectorStoreManager


@dataclass(frozen=True)
class GraphRetrievedMutation:
    # Unique mutation/gene identifier.
    gene_id: str
    # Aggregated graph relevance score.
    score: float
    # Human-readable mutation summary.
    description: str
    # Full stored code payload.
    code: str
    # Extended metadata copied from vector store plus graph annotations.
    metadata: dict


class GraphRetrievalService:
    """Retrieve mutation exemplars using a file-backed code knowledge graph."""

    def __init__(self, store: VectorStoreManager):
        self.store = store
        # KG is persisted under <rag_data_dir>/kg and rebuilt from current docs
        # when signatures change.
        self.kg = CodeKnowledgeGraph(self.store.base_dir)

    def _documents(self) -> list[StoredDocument]:
        """Load all indexed mutation documents from the code namespace."""
        return self.store.list_documents(VectorStoreManager.CODE_NAMESPACE)

    def _doc_by_gene_id(self, documents: Sequence[StoredDocument]) -> dict[str, StoredDocument]:
        """Map graph gene_id -> stored document for fast payload hydration."""
        return {
            str(doc.metadata.get("gene_id") or doc.metadata.get("document_id") or doc.document_id): doc
            for doc in documents
        }

    def retrieve_by_parent_gene(
        self,
        parent_gene_id: str,
        top_k: int = 5,
        min_accuracy: float = 0.0,
    ) -> List[GraphRetrievedMutation]:
        """Retrieve top descendants/related genes anchored on a parent gene node."""
        documents = self._documents()
        self.kg.build_from_documents(documents)
        by_gene_id = self._doc_by_gene_id(documents)

        hits = self.kg.related_successful_genes(
            parent_gene_id=parent_gene_id,
            top_k=top_k,
            min_accuracy=min_accuracy,
        )
        results: list[GraphRetrievedMutation] = []
        for hit in hits:
            # Convert KG hit -> full mutation payload from vector metadata/content.
            doc = by_gene_id.get(hit.gene_id)
            if not doc:
                continue
            metadata = {**doc.metadata}
            metadata["retrieval_source"] = "graph"
            metadata["graph_reason"] = hit.reason
            metadata.setdefault("description", metadata.get("gene_id", hit.gene_id))
            results.append(
                GraphRetrievedMutation(
                    gene_id=hit.gene_id,
                    score=hit.score,
                    description=metadata.get("description", ""),
                    code=doc.content,
                    metadata=metadata,
                )
            )
        return results

    def retrieve_by_pattern_overlap(
        self,
        query_code: str,
        top_k: int = 5,
        min_accuracy: float = 0.0,
    ) -> List[GraphRetrievedMutation]:
        """Retrieve by structural overlap (shared layers/patterns) with query code."""
        documents = self._documents()
        self.kg.build_from_documents(documents)
        by_gene_id = self._doc_by_gene_id(documents)

        entities = extract_entities_from_source(query_code)
        if not entities.layer_types and not entities.patterns:
            return []

        candidate_scores: dict[str, float] = {}
        candidate_reasons: dict[str, list[str]] = {}

        def register(gene_node: str, score: float, reason: str) -> None:
            # Simple additive scoring over multiple overlap signals.
            candidate_scores[gene_node] = candidate_scores.get(gene_node, 0.0) + score
            candidate_reasons.setdefault(gene_node, []).append(reason)

        for layer in entities.layer_types:
            layer_node = f"layer:{layer}"
            if layer_node not in self.kg.graph:
                continue
            for gene_node in self.kg.graph.predecessors(layer_node):
                edge_bundle = self.kg.graph.get_edge_data(gene_node, layer_node, default={})
                if any(edge.get("relation") == "uses" for edge in edge_bundle.values()):
                    # Layer overlap is informative, but weaker than pattern overlap.
                    register(gene_node, 0.3, f"shares layer {layer}")

        for pattern in entities.patterns:
            pattern_node = f"pattern:{pattern}"
            if pattern_node not in self.kg.graph:
                continue
            for gene_node in self.kg.graph.predecessors(pattern_node):
                edge_bundle = self.kg.graph.get_edge_data(gene_node, pattern_node, default={})
                if any(edge.get("relation") == "uses" for edge in edge_bundle.values()):
                    # Architectural pattern overlap gets higher weight.
                    register(gene_node, 0.5, f"shares pattern {pattern}")

        results: list[GraphRetrievedMutation] = []
        for gene_node, score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True):
            gene_data = self.kg.graph.nodes.get(gene_node, {})
            if gene_data.get("node_type") != "gene":
                continue
            accuracy = gene_data.get("accuracy")
            if accuracy is not None and float(accuracy) < min_accuracy:
                continue
            gene_id = str(gene_data.get("gene_id", gene_node.replace("gene:", "")))
            doc = by_gene_id.get(gene_id)
            if not doc:
                continue
            metadata = {**doc.metadata}
            metadata["retrieval_source"] = "graph"
            metadata["graph_reason"] = "; ".join(dict.fromkeys(candidate_reasons.get(gene_node, [])))
            metadata.setdefault("description", metadata.get("gene_id", gene_id))
            results.append(
                GraphRetrievedMutation(
                    gene_id=gene_id,
                    score=float(score),
                    description=metadata.get("description", ""),
                    code=doc.content,
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def format_context(self, mutations: Sequence[GraphRetrievedMutation]) -> str:
        """Format graph-retrieved exemplars into prompt-ready bullet text."""
        lines: list[str] = []
        for mutation in mutations:
            fitness = mutation.metadata.get("fitness") or []
            accuracy = f"{fitness[0]:.4f}" if fitness else "unknown"
            params = f"{int(fitness[1])}" if len(fitness) > 1 else "unknown"
            reason = mutation.metadata.get("graph_reason")
            reason_text = f" | Graph: {reason}" if reason else ""
            lines.append(
                f"- Gene {mutation.gene_id} (score {mutation.score:.3f}) "
                f"Accuracy {accuracy}, Params {params}{reason_text}\n"
                f"{mutation.description}"
            )
        return "\n".join(lines)
