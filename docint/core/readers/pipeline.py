"""Core PDF ingestion reader backed by the document processing pipeline."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode
from loguru import logger

from docint.core.pipeline import DocumentPipelineOrchestrator
from docint.utils.hashing import compute_file_hash
from docint.utils.mimetype import get_mimetype


@dataclass(slots=True)
class CorePDFPipelineReader:
    """Read PDF content through the core document processing pipeline.

    Attributes:
        data_dir: Root ingestion path (directory or single file).
        discovered_hashes: File hashes observed while scanning PDF sources.
    """

    data_dir: Path
    entity_extractor: Callable[[str], tuple[list[dict], list[dict]]] | None = None
    ner_max_workers: int = 1
    discovered_hashes: set[str] = field(default_factory=set, init=False)

    def _apply_ner(
        self,
        nodes: list[BaseNode],
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Attach entity/relation metadata to nodes in-place.

        Args:
            nodes: Nodes to enrich.
            progress_callback: Optional callback for progress updates.
        """
        if not self.entity_extractor or not nodes:
            return

        total_nodes = len(nodes)

        def _process_node(idx: int, node: BaseNode) -> None:
            text_value = getattr(node, "text", "") or ""
            if not text_value.strip():
                return
            try:
                if self.entity_extractor:
                    ents, rels = self.entity_extractor(text_value)
                    if ents or rels:
                        meta = dict(getattr(node, "metadata", {}) or {})
                        if ents:
                            meta["entities"] = ents
                        if rels:
                            meta["relations"] = rels
                        node.metadata = meta
            except Exception as exc:
                logger.warning("Entity extractor failed on chunk {}: {}", idx, exc)

        if self.ner_max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.ner_max_workers) as executor:
                futures = [
                    executor.submit(_process_node, i, node)
                    for i, node in enumerate(nodes)
                ]
                for i, _ in enumerate(as_completed(futures)):
                    if progress_callback:
                        progress_callback(
                            f"Extracting entities: {i + 1}/{total_nodes} chunks processed"
                        )
        else:
            for i, node in enumerate(nodes):
                _process_node(i, node)
                if progress_callback:
                    progress_callback(
                        f"Extracting entities: {i + 1}/{total_nodes} chunks processed"
                    )

    @staticmethod
    def _iter_pdf_files(data_dir: Path) -> list[Path]:
        """Return sorted PDF files under *data_dir*.

        Args:
            data_dir: Root ingestion path, which may be a directory or file.

        Returns:
            Sorted list of PDF file paths.
        """
        if data_dir.is_file():
            return [data_dir] if data_dir.suffix.lower() == ".pdf" else []
        if not data_dir.exists():
            return []
        return sorted(
            p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
        )

    @staticmethod
    def _load_pipeline_chunks(doc_id: str, artifacts_dir: Path) -> list[dict[str, Any]]:
        """Load chunk records produced by the core PDF pipeline.

        Args:
            doc_id: Document hash identifier.
            artifacts_dir: Root artifacts directory.

        Returns:
            Parsed list of chunk dictionaries from ``chunks.jsonl``.
        """
        chunks_path = artifacts_dir / doc_id / "chunks.jsonl"
        if not chunks_path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        rows.append(data)
                except Exception as exc:
                    logger.warning(
                        "Skipping malformed chunk row for doc {}: {}",
                        doc_id,
                        exc,
                    )
        return rows

    @staticmethod
    def _build_nodes(
        *,
        file_path: Path,
        doc_id: str,
        pipeline_version: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[Document], list[BaseNode]]:
        """Convert pipeline chunks into LlamaIndex documents and nodes.

        Args:
            file_path: Source PDF path.
            doc_id: Deterministic document hash.
            pipeline_version: Version reported by the pipeline manifest.
            chunks: Raw chunk payloads loaded from artifacts.

        Returns:
            Tuple of generated documents and nodes.
        """
        mimetype = get_mimetype(file_path)
        docs: list[Document] = []
        nodes: list[BaseNode] = []
        base_origin: dict[str, Any] = {
            "filename": file_path.name,
            "mimetype": mimetype,
            "file_hash": doc_id,
        }

        for idx, chunk in enumerate(chunks):
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue

            chunk_id = str(chunk.get("chunk_id") or f"{doc_id}-chunk-{idx}")
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_id}"))
            raw_pages = chunk.get("page_range")
            pages: list[int] = []
            if isinstance(raw_pages, list):
                for page in raw_pages:
                    try:
                        # Core pipeline stores zero-based page indexes; expose
                        # one-based pages to downstream query payloads/UI.
                        pages.append(int(page) + 1)
                    except (TypeError, ValueError):
                        continue
            page_number = pages[0] if pages else None

            metadata: dict[str, Any] = {
                "chunk_id": chunk_id,
                "point_id": point_id,
                "doc_id": doc_id,
                "file_hash": doc_id,
                "file_path": str(file_path),
                "file_name": file_path.name,
                "filename": file_path.name,
                "file_type": mimetype,
                "mimetype": mimetype,
                "source": "document",
                "origin": dict(base_origin),
                "pipeline_name": "core_pipeline",
                "pipeline_version": pipeline_version,
                "page_range": pages,
                "block_ids": chunk.get("block_ids") or [],
                "section_path": chunk.get("section_path") or [],
                "table_ids": chunk.get("table_ids") or [],
                "image_ids": chunk.get("image_ids") or [],
                "source_mix": chunk.get("source_mix"),
                "bbox_refs": chunk.get("bbox_refs") or [],
            }
            if page_number is not None:
                metadata["page"] = page_number
                metadata["page_number"] = page_number
                metadata["origin"]["page_number"] = page_number

            extra_metadata = chunk.get("metadata")
            if isinstance(extra_metadata, dict):
                metadata.update(extra_metadata)

            docs.append(Document(text=text, metadata=metadata, id_=point_id))
            nodes.append(TextNode(text=text, metadata=dict(metadata), id_=point_id))

        return docs, nodes

    def build(
        self,
        existing_hashes: set[str],
        progress_callback: Callable[[str], None] | None = None,
    ) -> Iterable[tuple[list[Document], list[BaseNode], str]]:
        """Yield batches produced by the core PDF ingestion pipeline.

        Args:
            existing_hashes: Hashes already present in the destination collection.
            progress_callback: Optional callback for ingestion progress events.

        Yields:
            Tuples of ``(docs, nodes, file_hash)`` for each successfully-processed PDF.
        """
        self.discovered_hashes.clear()
        pdf_files = self._iter_pdf_files(self.data_dir)
        if not pdf_files:
            return

        orchestrator = DocumentPipelineOrchestrator()
        artifacts_dir = Path(orchestrator.config.artifacts_dir)
        emitted_hashes: set[str] = set()

        for index, pdf_path in enumerate(pdf_files, start=1):
            file_hash = compute_file_hash(pdf_path)
            self.discovered_hashes.add(file_hash)

            if file_hash in existing_hashes or file_hash in emitted_hashes:
                if progress_callback:
                    progress_callback(
                        "Skipping already ingested PDF "
                        f"({index}/{len(pdf_files)}): {pdf_path.name}"
                    )
                continue

            if progress_callback:
                progress_callback(
                    "Core pipeline processing PDF "
                    f"({index}/{len(pdf_files)}): {pdf_path.name}"
                )

            try:
                manifest = orchestrator.process(pdf_path)
            except Exception as exc:
                logger.warning(
                    "Core pipeline failed for {}: {}",
                    pdf_path.name,
                    exc,
                )
                continue

            if manifest.status != "completed":
                logger.warning(
                    "Core pipeline incomplete for {} (status={}): {}",
                    pdf_path.name,
                    manifest.status,
                    manifest.error,
                )
                continue

            chunks = self._load_pipeline_chunks(manifest.doc_id, artifacts_dir)
            docs, nodes = self._build_nodes(
                file_path=pdf_path,
                doc_id=manifest.doc_id,
                pipeline_version=manifest.pipeline_version,
                chunks=chunks,
            )
            if not nodes:
                logger.warning("Core pipeline produced no chunks for {}", pdf_path.name)
                continue

            self._apply_ner(nodes, progress_callback=progress_callback)
            emitted_hashes.add(manifest.doc_id)
            if progress_callback:
                progress_callback(
                    f"Core pipeline indexed {len(nodes)} chunks: {pdf_path.name}"
                )
            yield docs, nodes, manifest.doc_id
