"""PageIndex-backed retrieval with real tree generation and tree search."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Sequence

from .data_ingestion import MutationRecord
from .retrieval import RetrievedMutation

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PageIndexRetrievalBackend:
    """PageIndex retrieval backend with tree indexing and query retrieval.

    The backend keeps mutation-code retrieval behavior for operator continuity,
    while adding real PageIndex tree generation + tree-search retrieval for PDFs.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        api_key: str | None = None,
        tree_timeout_sec: int = 900,
        poll_interval_sec: float = 5.0,
        query_thinking: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.api_key = (api_key or "").strip()
        self.tree_timeout_sec = tree_timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.query_thinking = query_thinking

        self._sdk_available = False
        self.client = None
        self._api_error_type = Exception
        try:
            from pageindex import PageIndexAPIError, PageIndexClient  # type: ignore

            self._sdk_available = True
            self._api_error_type = PageIndexAPIError
            if self.api_key:
                self.client = PageIndexClient(api_key=self.api_key)
        except Exception:
            self.client = None

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tree_dir = self.data_dir / "trees"
        self.tree_dir.mkdir(parents=True, exist_ok=True)

        self.code_store_path = self.data_dir / "code_records.jsonl"
        self.tree_manifest_path = self.data_dir / "tree_manifest.json"

        self._records: list[dict] = []
        self._tree_manifest: dict[str, dict] = {}
        self._trees_by_doc: dict[str, dict] = {}
        self._load_records()
        self._load_tree_manifest()

    @property
    def backend_name(self) -> str:
        return "pageindex"

    def _load_records(self) -> None:
        if not self.code_store_path.exists():
            return
        with self.code_store_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def _append_records(self, records: list[dict]) -> None:
        if not records:
            return
        with self.code_store_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_tree_manifest(self) -> None:
        if not self.tree_manifest_path.exists():
            return
        try:
            payload = json.loads(self.tree_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        self._tree_manifest = payload
        for entry in self._tree_manifest.values():
            tree_path = entry.get("tree_path")
            doc_id = entry.get("doc_id")
            if not tree_path or not doc_id:
                continue
            path = Path(tree_path)
            if not path.exists():
                continue
            try:
                self._trees_by_doc[doc_id] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

    def _persist_tree_manifest(self) -> None:
        self.tree_manifest_path.write_text(
            json.dumps(self._tree_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _extract_tree_payload(self, response: dict) -> dict | list | None:
        if not isinstance(response, dict):
            return None
        for key in ("tree", "structure", "result", "data"):
            value = response.get(key)
            if isinstance(value, (dict, list)):
                return value
        if "nodes" in response and isinstance(response.get("nodes"), list):
            return response
        return None

    def _flatten_tree_nodes(self, tree_payload: dict | list) -> list[dict]:
        nodes: list[dict] = []

        def visit(item: dict) -> None:
            node_id = str(item.get("node_id") or item.get("id") or "")
            if node_id:
                nodes.append(item)
            for child in item.get("nodes", []) or []:
                if isinstance(child, dict):
                    visit(child)

        if isinstance(tree_payload, list):
            for candidate in tree_payload:
                if isinstance(candidate, dict):
                    visit(candidate)
        elif isinstance(tree_payload, dict):
            visit(tree_payload)
        return nodes

    def _is_tree_ready_response(self, payload: dict) -> bool:
        status = str(payload.get("status", "")).lower()
        if status in {"completed", "done", "success", "succeeded", "ready"}:
            return True
        if payload.get("retrieval_ready") is True:
            return True
        return self._extract_tree_payload(payload) is not None

    def _require_tree_client(self) -> None:
        if not self._sdk_available:
            raise RuntimeError("pageindex SDK is not installed")
        if self.client is None:
            raise RuntimeError("PAGEINDEX_API_KEY is required for PageIndex tree operations")

    def index_pdf_documents(self, pdf_paths: Sequence[str]) -> int:
        self._require_tree_client()
        client = self.client
        if client is None:
            raise RuntimeError("PAGEINDEX_API_KEY is required for PageIndex tree operations")
        indexed = 0
        for pdf_path in pdf_paths:
            source_path = str(Path(pdf_path).resolve())
            existing = self._tree_manifest.get(source_path)
            if (
                existing
                and existing.get("doc_id")
                and existing.get("tree_path")
                and Path(existing["tree_path"]).exists()
            ):
                continue

            submit_result = client.submit_document(source_path)
            doc_id = submit_result.get("doc_id")
            if not doc_id:
                raise RuntimeError(f"PageIndex did not return doc_id for {source_path}")

            deadline = time.time() + self.tree_timeout_sec
            tree_response: dict | None = None
            while time.time() < deadline:
                tree_response = client.get_tree(doc_id, node_summary=True)
                if isinstance(tree_response, dict) and self._is_tree_ready_response(tree_response):
                    break
                time.sleep(self.poll_interval_sec)

            if not isinstance(tree_response, dict) or not self._is_tree_ready_response(tree_response):
                raise RuntimeError(f"Timed out waiting for PageIndex tree generation for doc_id={doc_id}")

            tree_path = self.tree_dir / f"{Path(source_path).stem}_{doc_id}.json"
            tree_path.write_text(json.dumps(tree_response, ensure_ascii=False, indent=2), encoding="utf-8")

            self._tree_manifest[source_path] = {
                "doc_id": doc_id,
                "tree_path": str(tree_path),
                "indexed_at": time.time(),
            }
            self._trees_by_doc[doc_id] = tree_response
            indexed += 1

        self._persist_tree_manifest()
        return indexed

    def _extract_retrieval_node_ids(self, retrieval_payload: dict) -> list[str]:
        candidates: list[str] = []

        def walk(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    lowered = key.lower()
                    if lowered in {"node_id", "nodeid"} and isinstance(value, str):
                        candidates.append(value)
                    elif lowered in {"node_list", "node_ids", "retrieved_nodes", "nodes"} and isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                candidates.append(item)
                            elif isinstance(item, dict):
                                maybe = item.get("node_id") or item.get("id")
                                if isinstance(maybe, str):
                                    candidates.append(maybe)
                    else:
                        walk(value)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(retrieval_payload)
        deduped: list[str] = []
        seen: set[str] = set()
        for node_id in candidates:
            if node_id in seen:
                continue
            seen.add(node_id)
            deduped.append(node_id)
        return deduped

    def _tree_node_to_mutation(self, doc_id: str, node: dict, rank: int) -> RetrievedMutation:
        node_id = str(node.get("node_id") or node.get("id") or f"rank-{rank}")
        title = str(node.get("title") or "Untitled node")
        summary = str(node.get("summary") or "")
        start_index = node.get("start_index")
        end_index = node.get("end_index")
        description = title if not summary else f"{title}: {summary}"

        content_lines = [
            f"Document: {doc_id}",
            f"Node ID: {node_id}",
            f"Title: {title}",
        ]
        if start_index is not None or end_index is not None:
            content_lines.append(f"Pages: {start_index} - {end_index}")
        if summary:
            content_lines.append(f"Summary: {summary}")

        metadata = {
            "gene_id": f"pageindex:{doc_id}:{node_id}",
            "source": "pageindex_tree",
            "doc_id": doc_id,
            "node_id": node_id,
            "start_index": start_index,
            "end_index": end_index,
            "description": description,
        }
        return RetrievedMutation(
            gene_id=metadata["gene_id"],
            score=max(0.0, 1.0 - (rank * 0.05)),
            description=description,
            code="\n".join(content_lines),
            metadata=metadata,
        )

    def _retrieve_tree_nodes(self, query_text: str, top_k: int) -> list[RetrievedMutation]:
        if not query_text.strip() or not self.client or not self._tree_manifest:
            return []
        client = self.client

        results: list[RetrievedMutation] = []
        for entry in self._tree_manifest.values():
            doc_id = entry.get("doc_id")
            if not doc_id:
                continue

            submit = client.submit_query(doc_id=doc_id, query=query_text, thinking=self.query_thinking)
            retrieval_id = submit.get("retrieval_id")
            if not retrieval_id:
                continue

            deadline = time.time() + self.tree_timeout_sec
            retrieval_payload: dict | None = None
            while time.time() < deadline:
                retrieval_payload = client.get_retrieval(retrieval_id)
                if not isinstance(retrieval_payload, dict):
                    retrieval_payload = None
                    break
                status = str(retrieval_payload.get("status", "")).lower()
                if status in {"completed", "done", "success", "succeeded", "ready"}:
                    break
                if status in {"failed", "error"}:
                    retrieval_payload = None
                    break
                time.sleep(self.poll_interval_sec)

            if not retrieval_payload:
                continue

            node_ids = self._extract_retrieval_node_ids(retrieval_payload)
            tree_response = self._trees_by_doc.get(doc_id)
            tree_payload = self._extract_tree_payload(tree_response or {})
            if not tree_payload:
                continue

            nodes = self._flatten_tree_nodes(tree_payload)
            node_map = {str(node.get("node_id") or node.get("id")): node for node in nodes}
            if node_ids:
                ranked_nodes = [node_map[node_id] for node_id in node_ids if node_id in node_map]
            else:
                ranked_nodes = nodes[:top_k]

            for rank, node in enumerate(ranked_nodes[:top_k]):
                results.append(self._tree_node_to_mutation(doc_id=doc_id, node=node, rank=rank))

        results.sort(key=lambda item: item.score, reverse=True)
        deduped: dict[str, RetrievedMutation] = {}
        for item in results:
            deduped.setdefault(item.gene_id, item)
            if len(deduped) >= top_k:
                break
        return list(deduped.values())

    def _score_overlap(self, query_text: str, content: str) -> float:
        query_tokens = set(_TOKEN_PATTERN.findall(query_text.lower()))
        if not query_tokens:
            return 0.0
        content_tokens = set(_TOKEN_PATTERN.findall(content.lower()))
        if not content_tokens:
            return 0.0
        intersection = len(query_tokens & content_tokens)
        if intersection == 0:
            return 0.0
        return intersection / max(1, len(query_tokens))

    def _record_to_mutation(self, record: dict, score: float) -> RetrievedMutation:
        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("description", record.get("description", metadata.get("gene_id", "")))
        return RetrievedMutation(
            gene_id=metadata.get("gene_id", record.get("document_id", "unknown")),
            score=score,
            description=metadata.get("description", ""),
            code=record.get("content", ""),
            metadata=metadata,
        )

    def index_mutations(self, records: Sequence[MutationRecord]) -> List[str]:
        if not records:
            return []
        to_write: list[dict] = []
        ids: list[str] = []
        for record in records:
            content, metadata = record.to_document()
            doc_id = metadata.get("gene_id", record.gene_id)
            payload = {
                "document_id": doc_id,
                "content": content,
                "metadata": metadata,
            }
            to_write.append(payload)
            self._records.append(payload)
            ids.append(doc_id)
        self._append_records(to_write)
        return ids

    def index_text_documents(self, documents: Sequence[dict]) -> List[str]:
        # Maintained for backward compatibility; tree indexing is handled by index_pdf_documents.
        if not documents:
            return []
        to_write: list[dict] = []
        ids: list[str] = []
        start = len(self._records)
        for index, document in enumerate(documents, start=1):
            metadata = dict(document.get("metadata") or {})
            doc_id = metadata.get("document_id") or f"text-{start + index}"
            payload = {
                "document_id": doc_id,
                "content": document.get("content", ""),
                "metadata": metadata,
            }
            to_write.append(payload)
            self._records.append(payload)
            ids.append(doc_id)
        self._append_records(to_write)
        return ids

    def log_mutation_code(self, content: str, metadata: dict) -> str | None:
        if not content.strip():
            return None
        doc_id = metadata.get("gene_id") or metadata.get("document_id") or f"gene-{len(self._records)+1}"
        payload = {
            "document_id": doc_id,
            "content": content,
            "metadata": dict(metadata),
        }
        self._records.append(payload)
        self._append_records([payload])
        return doc_id

    def retrieve_similar_mutations(
        self, query_code: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> List[RetrievedMutation]:
        if not query_code.strip():
            return []

        if self.client and self._tree_manifest:
            try:
                tree_results = self._retrieve_tree_nodes(query_text=query_code, top_k=top_k)
                if tree_results:
                    return tree_results
            except self._api_error_type as exc:
                print(f"[RAG][PageIndex] Tree retrieval failed, falling back to local lexical retrieval: {exc}")
            except Exception as exc:
                print(f"[RAG][PageIndex] Unexpected tree retrieval failure, fallback enabled: {exc}")

        scored = []
        for record in self._records:
            metadata = record.get("metadata") or {}
            if "gene_id" not in metadata:
                continue
            score = self._score_overlap(query_code, record.get("content", ""))
            if score >= min_similarity:
                scored.append(self._record_to_mutation(record, score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def retrieve_high_performers(
        self,
        min_accuracy: float = 0.9,
        max_parameters: float | None = None,
        limit: int = 5,
    ) -> List[RetrievedMutation]:
        filtered: list[RetrievedMutation] = []
        for record in self._records:
            metadata = record.get("metadata") or {}
            if "gene_id" not in metadata:
                continue
            fitness = metadata.get("fitness") or []
            accuracy = float(fitness[0]) if fitness else 0.0
            parameters = float(fitness[1]) if len(fitness) > 1 else None
            if accuracy < min_accuracy:
                continue
            if max_parameters is not None and parameters is not None and parameters > max_parameters:
                continue
            filtered.append(self._record_to_mutation(record, accuracy))
        filtered.sort(key=lambda item: item.score, reverse=True)
        return filtered[:limit]

    def retrieve_by_mutation_type(self, mutation_type: str, limit: int = 5) -> List[RetrievedMutation]:
        matches: list[RetrievedMutation] = []
        for record in self._records:
            metadata = record.get("metadata") or {}
            if "gene_id" not in metadata:
                continue
            if metadata.get("mutation_type", "").lower() != mutation_type.lower():
                continue
            fitness = metadata.get("fitness") or []
            score = float(fitness[0]) if fitness else 0.0
            matches.append(self._record_to_mutation(record, score))
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def format_context(self, mutations: Sequence[RetrievedMutation]) -> str:
        lines: list[str] = []
        for mutation in mutations:
            source = mutation.metadata.get("source")
            if source == "pageindex_tree":
                pages = "unknown"
                start = mutation.metadata.get("start_index")
                end = mutation.metadata.get("end_index")
                if start is not None or end is not None:
                    pages = f"{start} - {end}"
                lines.append(
                    f"- PageIndex {mutation.metadata.get('doc_id')}::{mutation.metadata.get('node_id')} "
                    f"(score {mutation.score:.3f}, pages {pages})\n"
                    f"{mutation.description}"
                )
                continue

            fitness = mutation.metadata.get("fitness") or []
            accuracy = f"{fitness[0]:.4f}" if fitness else "unknown"
            params = f"{int(fitness[1])}" if len(fitness) > 1 else "unknown"
            lines.append(
                f"- Gene {mutation.gene_id} (score {mutation.score:.3f}) "
                f"Accuracy {accuracy}, Params {params}\n"
                f"{mutation.description}"
            )
        return "\n".join(lines)
