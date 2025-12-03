from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import BaseNode
from llama_index.node_parser.docling import DoclingNodeParser
from loguru import logger

from docint.core.readers.audio import AudioReader
from docint.core.readers.documents import CustomDoclingReader
from docint.core.readers.images import ImageReader
from docint.core.readers.json import CustomJSONReader
from docint.core.readers.tables import TableReader
from docint.utils.hashing import compute_file_hash

CleanFn = Callable[[str], str]


@dataclass(slots=True)
class DocumentIngestionPipeline:
    """Encapsulates document loading, cleaning, and node construction."""

    data_dir: Path
    clean_fn: CleanFn
    sentence_splitter: SentenceSplitter
    embed_model_factory: Callable[[], BaseEmbedding]
    device: str = "cpu"
    reader_errors: str = "ignore"
    reader_recursive: bool = True
    reader_encoding: str = "utf-8"
    reader_required_exts: list[str] | None = None
    table_text_cols: list[str] | None = None
    table_metadata_cols: list[str] | str | None = None
    table_id_col: str | None = None
    table_excel_sheet: str | int | None = None
    table_row_limit: int | None = None
    table_row_filter: str | None = None
    buffer_size: int = 5
    breakpoint_percentile_threshold: int = 90
    chunk_size: int = 1024
    chunk_overlap: int = 0
    semantic_splitter_char_limit: int = 20000

    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    md_node_parser: MarkdownNodeParser | None = field(default=None, init=False)
    docling_node_parser: DoclingNodeParser | None = field(default=None, init=False)
    semantic_node_parser: SemanticSplitterNodeParser | None = field(
        default=None, init=False
    )
    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)

    def build(
        self, existing_hashes: set[str] | None = None
    ) -> tuple[list[Document], list[BaseNode]]:
        """Execute the full ingestion pipeline and return cleaned docs + nodes."""

        self._load_doc_readers()
        self._load_node_parsers()

        if self.dir_reader is None:
            raise RuntimeError("Directory reader failed to initialize.")

        docs = self.dir_reader.load_data()
        docs = self._attach_clean_text(docs)
        docs = self._filter_docs_by_existing_hashes(docs, existing_hashes)
        nodes = self._create_nodes(docs)

        self.docs = docs
        self.nodes = nodes
        return docs, nodes

    def _attach_clean_text(self, docs: Iterable[Document]) -> list[Document]:
        cleaned: list[Document] = []
        for doc in docs:
            if hasattr(doc, "text") and isinstance(doc.text, str):
                cleaned.append(
                    Document(text=self.clean_fn(doc.text), metadata=doc.metadata)
                )
            else:
                cleaned.append(doc)
        return cleaned

    def _load_doc_readers(self) -> None:
        audio_reader = AudioReader(device=self.device)
        image_reader = ImageReader()
        table_reader = TableReader(
            text_cols=self.table_text_cols,
            metadata_cols=self.table_metadata_cols
            if self.table_metadata_cols
            else None,
            id_col=self.table_id_col,
            excel_sheet=self.table_excel_sheet,
            limit=self.table_row_limit,
            row_query=self.table_row_filter,
        )

        def _metadata(path: str | Path) -> dict[str, str]:
            resolved = path if isinstance(path, Path) else Path(path)
            file_hash = compute_file_hash(resolved)
            filename = resolved.name
            return {
                "file_path": str(resolved),
                "file_name": filename,
                "filename": filename,
                "file_hash": file_hash,
            }

        self.dir_reader = SimpleDirectoryReader(
            input_dir=self.data_dir,
            errors=self.reader_errors,
            recursive=self.reader_recursive,
            encoding=self.reader_encoding,
            required_exts=self.reader_required_exts,
            file_metadata=_metadata,
            file_extractor={
                ".mpeg": audio_reader,
                ".mp3": audio_reader,
                ".m4a": audio_reader,
                ".ogg": audio_reader,
                ".wav": audio_reader,
                ".webm": audio_reader,
                ".avi": audio_reader,
                ".flv": audio_reader,
                ".mkv": audio_reader,
                ".mov": audio_reader,
                ".mpg": audio_reader,
                ".mp4": audio_reader,
                ".m4v": audio_reader,
                ".wmv": audio_reader,
                ".json": CustomJSONReader(),
                ".docx": CustomDoclingReader(),
                ".pdf": CustomDoclingReader(),
                ".gif": image_reader,
                ".jpeg": image_reader,
                ".jpg": image_reader,
                ".png": image_reader,
                ".csv": table_reader,
                ".parquet": TableReader(
                    text_cols=self.table_text_cols or ["text"],
                    metadata_cols=set(self.table_metadata_cols)
                    if self.table_metadata_cols
                    else None,
                    id_col=self.table_id_col,
                    limit=self.table_row_limit,
                    row_query=self.table_row_filter,
                ),
                ".tsv": TableReader(
                    csv_sep="\t",
                    text_cols=self.table_text_cols,
                    metadata_cols=set(self.table_metadata_cols)
                    if self.table_metadata_cols
                    else None,
                    id_col=self.table_id_col,
                    limit=self.table_row_limit,
                    row_query=self.table_row_filter,
                ),
                ".xls": table_reader,
                ".xlsx": table_reader,
            },
        )

    def _load_node_parsers(self) -> None:
        self.md_node_parser = MarkdownNodeParser()
        self.docling_node_parser = DoclingNodeParser()
        if self.embed_model_factory is None:
            raise RuntimeError("Embed model factory must be provided for ingestion.")
        self.semantic_node_parser = SemanticSplitterNodeParser(
            embed_model=self.embed_model_factory(),
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
        )

    def _filter_docs_by_existing_hashes(
        self, docs: Iterable[Document], existing_hashes: set[str] | None
    ) -> list[Document]:
        if not existing_hashes:
            return list(docs)

        filtered: list[Document] = []
        skipped: dict[str, str] = {}
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            file_hash = metadata.get("file_hash") or self._extract_file_hash(metadata)
            if not file_hash or file_hash not in existing_hashes:
                filtered.append(doc)
                continue

            filename = (
                metadata.get("file_name")
                or metadata.get("filename")
                or metadata.get("file_path")
                or metadata.get("path")
                or metadata.get("source")
                or ""
            )
            if not filename:
                origin = metadata.get("origin")
                if isinstance(origin, dict):
                    filename = (
                        origin.get("filename")
                        or origin.get("file_path")
                        or origin.get("path")
                        or ""
                    )
            skipped[file_hash] = filename

        if skipped:
            display = [name or h[:12] for h, name in skipped.items()]
            logger.info(
                "Skipping {} file(s) already ingested: {}",
                len(skipped),
                ", ".join(sorted(display)),
            )
        return filtered

    def _create_nodes(self, docs: list[Document]) -> list[BaseNode]:
        if (
            self.md_node_parser is None
            or self.docling_node_parser is None
            or self.semantic_node_parser is None
        ):
            raise RuntimeError("Node parsers are not initialized.")

        audio_docs, document_docs, img_docs, json_docs, table_docs, text_docs = [
            [] for _ in range(6)
        ]
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            file_type = (meta.get("file_type") or "").lower()
            source_kind = meta.get("source", "") or ""
            file_path = str(meta.get("file_path") or meta.get("file_name") or "")
            ext = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else ""

            if source_kind == "audio" or ext in {
                ".avi",
                ".flv",
                ".mkv",
                ".mov",
                ".mpeg",
                ".mpg",
                ".mp3",
                ".mp4",
                ".m4v",
                ".ogg",
                ".wav",
                ".webm",
                ".wmv",
            }:
                audio_docs.append(d)
            elif source_kind == "image" or ext in {"gif", "jpeg", "jpg", "png"}:
                img_docs.append(d)
            elif source_kind == "table" or ext in {"csv", "tsv"}:
                table_docs.append(d)
            elif file_type.endswith(("json", "jsonl")) or ext in {"json", "jsonl"}:
                json_docs.append(d)
            elif file_type.endswith(("docx", "pdf")) or ext in {"docx", "pdf"}:
                document_docs.append(d)
            elif file_type.startswith("text/") or ext in {"txt", "md", "rst"}:
                text_docs.append(d)
            else:
                if file_type.startswith("text/") or ext in {"txt", "md", "rst"}:
                    text_docs.append(d)
                else:
                    logger.warning(
                        "Unrecognized document type for file '{}'; treating as plain text.",
                        file_path,
                    )
                    text_docs.append(d)

        nodes: list[BaseNode] = []
        if audio_docs:
            logger.info(
                "Parsing {} audio documents with SemanticSplitterNodeParser",
                len(audio_docs),
            )
            nodes.extend(self._semantic_nodes_with_fallback(audio_docs, "audio"))

        if img_docs:
            logger.info(
                "Parsing {} image documents with SentenceSplitter",
                len(img_docs),
            )
            nodes.extend(self.sentence_splitter.get_nodes_from_documents(img_docs))

        if json_docs:
            logger.info(
                "Parsing {} JSON documents with SemanticSplitterNodeParser",
                len(json_docs),
            )
            nodes.extend(self._semantic_nodes_with_fallback(json_docs, "JSON"))

        if document_docs:

            def _is_docling_json(doc: Document) -> bool:
                try:
                    json.loads(getattr(doc, "text", "") or "")
                    return True
                except Exception:
                    return False

            pdf_docs_docling = [d for d in document_docs if _is_docling_json(d)]
            pdf_docs_md = [d for d in document_docs if not _is_docling_json(d)]

            if pdf_docs_docling:
                logger.info(
                    "Parsing {} Docling JSON PDFs with DoclingNodeParser",
                    len(pdf_docs_docling),
                )
                nodes.extend(
                    self.docling_node_parser.get_nodes_from_documents(pdf_docs_docling)
                )
            if pdf_docs_md:
                logger.info(
                    "Parsing {} Markdown PDFs with MarkdownNodeParser",
                    len(pdf_docs_md),
                )
                nodes.extend(self.md_node_parser.get_nodes_from_documents(pdf_docs_md))

        if table_docs:
            logger.info(
                "Parsing {} table documents with SemanticSplitterNodeParser",
                len(table_docs),
            )
            nodes.extend(self._semantic_nodes_with_fallback(table_docs, "table"))

        if text_docs:
            markdown_docs = [
                d
                for d in text_docs
                if str(d.metadata.get("file_path", "")).endswith(
                    (".md", ".markdown", ".rst")
                )
                or (d.text.strip().startswith("#"))
            ]
            plain_docs = [d for d in text_docs if d not in markdown_docs]

            if markdown_docs:
                logger.info(
                    "Parsing {} markdown documents with MarkdownNodeParser",
                    len(markdown_docs),
                )
                nodes.extend(
                    self.md_node_parser.get_nodes_from_documents(markdown_docs)
                )
            if plain_docs:
                logger.info(
                    "Parsing {} plain text documents with SemanticSplitterNodeParser",
                    len(plain_docs),
                )
                nodes.extend(self._semantic_nodes_with_fallback(plain_docs, "text"))

        return nodes

    def _partition_large_docs(
        self, docs: list[Document]
    ) -> tuple[list[Document], list[Document]]:
        if not docs:
            return [], []
        limit = max(self.semantic_splitter_char_limit, self.chunk_size)
        semantic_docs: list[Document] = []
        oversized_docs: list[Document] = []
        for doc in docs:
            text = getattr(doc, "text", None)
            if text and len(text) > limit:
                oversized_docs.append(doc)
            else:
                semantic_docs.append(doc)
        return semantic_docs, oversized_docs

    def _explode_oversized_documents(self, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
        limit = max(self.semantic_splitter_char_limit, self.chunk_size)
        overlap = max(int(limit * 0.05), 0)
        stride = max(limit - overlap, 1)
        exploded: list[Document] = []
        for doc in docs:
            text = getattr(doc, "text", None)
            if not text or len(text) <= limit:
                exploded.append(doc)
                continue
            meta = dict(getattr(doc, "metadata", {}) or {})
            for start in range(0, len(text), stride):
                end = min(len(text), start + limit)
                segment_meta = dict(meta)
                segment_meta["segment_start"] = start
                segment_meta["segment_end"] = end
                exploded.append(Document(text=text[start:end], metadata=segment_meta))
                if end >= len(text):
                    break
        return exploded

    def _semantic_nodes_with_fallback(
        self, docs: list[Document], doc_label: str
    ) -> list[BaseNode]:
        if not docs:
            return []
        if self.semantic_node_parser is None:
            raise RuntimeError("Semantic splitter is not initialized.")

        semantic_docs, oversized_docs = self._partition_large_docs(docs)
        nodes: list[BaseNode] = []

        if semantic_docs:
            try:
                nodes.extend(
                    self.semantic_node_parser.get_nodes_from_documents(semantic_docs)
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if "buffer size" not in message and "mps" not in message:
                    raise
                logger.warning(
                    (
                        "Semantic splitter failed for {} {} document(s); retrying with sentence-based chunks. Error: {}"
                    ),
                    len(semantic_docs),
                    doc_label,
                    exc,
                )
                fallback_docs = self._explode_oversized_documents(semantic_docs)
                nodes.extend(
                    self.sentence_splitter.get_nodes_from_documents(fallback_docs)
                )

        if oversized_docs:
            limit = max(self.semantic_splitter_char_limit, self.chunk_size)
            exploded_docs = self._explode_oversized_documents(oversized_docs)
            logger.info(
                (
                    "Chunking {} {} document(s) ({} expanded segments) over {} chars with SentenceSplitter"
                ),
                len(oversized_docs),
                doc_label,
                len(exploded_docs),
                limit,
            )
            nodes.extend(self.sentence_splitter.get_nodes_from_documents(exploded_docs))

        return nodes

    @staticmethod
    def _extract_file_hash(data: dict | None) -> str | None:
        if not isinstance(data, dict):
            return None
        candidate = data.get("file_hash")
        if isinstance(candidate, str) and candidate:
            return candidate
        origin = data.get("origin")
        if isinstance(origin, dict):
            candidate = origin.get("file_hash")
            if isinstance(candidate, str) and candidate:
                return candidate
        for key in ("metadata", "meta", "extra_info"):
            nested = data.get(key)
            if isinstance(nested, dict):
                nested_hash = DocumentIngestionPipeline._extract_file_hash(nested)
                if nested_hash:
                    return nested_hash
        for value in data.values():
            if isinstance(value, dict):
                nested_hash = DocumentIngestionPipeline._extract_file_hash(value)
                if nested_hash:
                    return nested_hash
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        nested_hash = DocumentIngestionPipeline._extract_file_hash(item)
                        if nested_hash:
                            return nested_hash
        return None
