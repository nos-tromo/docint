from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    NodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import BaseNode
from llama_index.llms.ollama import Ollama
from llama_index.node_parser.docling import DoclingNodeParser
from loguru import logger

from docint.core.readers.audio import AudioReader
from docint.core.readers.documents import CustomDoclingReader
from docint.core.readers.images import ImageReader
from docint.core.readers.json import CustomJSONReader
from docint.core.readers.tables import TableReader
from docint.core.storage.hierarchical import HierarchicalNodeParser
from docint.utils.clean_text import basic_clean
from docint.utils.env_cfg import load_ie_env, load_path_env, load_rag_env
from docint.utils.hashing import compute_file_hash
from docint.utils.ie_extractor import build_ie_extractor
from docint.utils.ollama_cfg import OllamaPipeline

CleanFn = Callable[[str], str]


@dataclass(slots=True)
class DocumentIngestionPipeline:
    """
    Encapsulates document loading, cleaning, and node construction.
    """

    # --- Constructor args ---
    data_dir: Path
    device: str
    ie_model: Ollama | None
    progress_callback: Callable[[str], None] | None

    # --- Cleaning config ---
    clean_fn: CleanFn = basic_clean

    # --- Directory reader config ---
    reader_errors: str = "ignore"
    reader_recursive: bool = True
    reader_encoding: str = "utf-8"
    reader_required_exts: list[str] = field(default_factory=list, init=False)

    # --- Ingestion config ---
    ingestion_batch_size: int = field(default=5, init=False)

    # --- Table reader config ---
    table_text_cols: list[str] | None = None
    table_metadata_cols: list[str] | str | None = None
    table_id_col: str | None = None
    table_excel_sheet: str | int | None = None

    # --- Information extraction ---
    ie_max_workers: int = field(default=4, init=False)
    entity_extractor: Callable[[str], tuple[list[dict], list[dict]]] | None = None

    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    md_node_parser: MarkdownNodeParser | None = field(default=None, init=False)
    docling_node_parser: DoclingNodeParser | None = field(default=None, init=False)
    sentence_splitter: SentenceSplitter = field(
        default_factory=SentenceSplitter, init=False
    )
    hierarchical_node_parser: HierarchicalNodeParser | None = field(
        default=None, init=False
    )
    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)
    file_hash_cache: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to load configurations and set up components.
        """
        # --- Path config ---
        path_config = load_path_env()
        reader_required_exts_path = path_config.required_exts
        with open(reader_required_exts_path, "r", encoding="utf-8") as f:
            self.reader_required_exts = [f".{line.strip()}" for line in f]

        # --- Information Extraction config ---
        ie_config = load_ie_env()
        ie_enabled = ie_config.ie_enabled
        ie_max_chars = ie_config.ie_max_chars
        self.ie_max_workers = ie_config.ie_max_workers
        ie_prompt = OllamaPipeline().load_prompt(kw="ner")

        # Initialize IE extractor if runtime pieces are provided
        if self.ie_model and ie_enabled and ie_max_chars and ie_prompt:
            self.entity_extractor = build_ie_extractor(
                model=self.ie_model,
                prompt=ie_prompt,
                max_chars=ie_max_chars,
            )

        # --- RAG config ---
        rag_config = load_rag_env()
        self.ingestion_batch_size = rag_config.ingestion_batch_size
        sentence_splitter_chunk_size = rag_config.sentence_splitter_chunk_size
        sentence_splitter_chunk_overlap = rag_config.sentence_splitter_chunk_overlap
        self.sentence_splitter = SentenceSplitter(
            chunk_size=sentence_splitter_chunk_size,
            chunk_overlap=sentence_splitter_chunk_overlap,
        )
        if rag_config.hierarchical_chunking_enabled:
            logger.info("Hierarchical chunking is ENABLED.")
            self.hierarchical_node_parser = HierarchicalNodeParser(
                coarse_chunk_size=rag_config.coarse_chunk_size,
                fine_chunk_size=rag_config.fine_chunk_size,
                fine_chunk_overlap=rag_config.fine_chunk_overlap,
            )
        else:
            self.hierarchical_node_parser = None

    def build(
        self, existing_hashes: set[str] | None = None
    ) -> Iterable[tuple[list[Document], list[BaseNode]]]:
        """
        Execute the full ingestion pipeline and yield batches of cleaned docs + nodes.

        Args:
            existing_hashes (set[str] | None): A set of existing document hashes to filter out already processed documents.

        Yields:
            tuple[list[Document], list[BaseNode]]: A batch of cleaned documents and their corresponding nodes.

        Raises:
            RuntimeError: If the directory reader fails to initialize.
        """

        self._load_doc_readers()
        self._load_node_parsers()

        if self.dir_reader is None:
            raise RuntimeError("Directory reader failed to initialize.")

        # Pre-filter files based on existing hashes to avoid unnecessary processing
        if existing_hashes:
            self._filter_input_files(existing_hashes)

        # Process in batches
        current_docs: list[Document] = []
        files_processed = 0

        # iter_data() yields List[Document] per file
        for file_docs in self.dir_reader.iter_data():
            current_docs.extend(file_docs)
            files_processed += 1

            if files_processed >= self.ingestion_batch_size:
                yield self._process_batch(current_docs, existing_hashes)
                current_docs = []
                files_processed = 0

        if current_docs:
            yield self._process_batch(current_docs, existing_hashes)

    def _process_batch(
        self, docs: list[Document], existing_hashes: set[str] | None
    ) -> tuple[list[Document], list[BaseNode]]:
        """
        Process a batch of documents through cleaning, hashing, filtering, and node creation.

        Args:
            docs (list[Document]): The list of documents to process.
            existing_hashes (set[str] | None): A set of existing document hashes to filter out already processed documents.

        Returns:
            tuple[list[Document], list[BaseNode]]: A tuple containing the processed documents and their corresponding nodes.
        """
        docs = self._attach_clean_text(docs)
        docs = self._ensure_file_hashes(docs)
        # We still keep this filter as a safety net, though pre-filtering should catch most
        docs = self._filter_docs_by_existing_hashes(docs, existing_hashes)
        nodes = self._create_nodes(docs)

        # Update internal state (optional, but useful for debugging last batch)
        self.docs = docs
        self.nodes = nodes

        return docs, nodes

    def _filter_input_files(self, existing_hashes: set[str]) -> None:
        """
        Filter self.dir_reader.input_files based on existing hashes. Populates self.file_hash_cache.

        Args:
            existing_hashes (set[str]): A set of existing document hashes to filter out already processed documents.
        """
        if not self.dir_reader or not self.dir_reader.input_files:
            return

        filtered_files: list[Path | PurePosixPath] = []
        skipped_count = 0

        for file_path in self.dir_reader.input_files:
            path_obj: Path = Path(file_path)
            path_str = str(path_obj)
            try:
                # Compute hash (or get from cache if we ever re-run)
                if path_str in self.file_hash_cache:
                    f_hash = self.file_hash_cache[path_str]
                else:
                    f_hash = compute_file_hash(path_obj)
                    self.file_hash_cache[path_str] = f_hash

                if f_hash in existing_hashes:
                    skipped_count += 1
                    continue

                filtered_files.append(path_obj)
            except Exception as e:
                logger.warning(
                    f"Failed to compute hash for {file_path}, skipping pre-filter: {e}"
                )
                filtered_files.append(path_obj)

        if skipped_count > 0:
            logger.info(
                f"Skipping {skipped_count} files that already exist in the collection."
            )
            self.dir_reader.input_files = filtered_files

    def _ensure_file_hashes(self, docs: list[Document]) -> list[Document]:
        """
        Ensure every document has a file_hash in its metadata. Computes it from the file path if missing.

        Args:
            docs (list[Document]): The list of documents to process.

        Returns:
            list[Document]: The list of documents with ensured file_hash metadata.

        Raises:
            RuntimeError: If the file hash computation fails.
        """
        # Cache hashes by path to avoid re-reading the same file multiple times
        path_hash_map: dict[str, str] = {}

        for doc in docs:
            if "file_hash" in doc.metadata and doc.metadata["file_hash"]:
                continue

            # Try to find the file path
            file_path = (
                doc.metadata.get("file_path")
                or doc.metadata.get("path")
                or doc.metadata.get("filename")
            )

            if not file_path:
                continue

            file_path_str = str(file_path)

            # Use cached hash if available
            if file_path_str in path_hash_map:
                doc.metadata["file_hash"] = path_hash_map[file_path_str]
                continue

            # Compute and cache
            try:
                # Only compute if file exists
                p = Path(file_path_str)
                if p.is_file():
                    f_hash = compute_file_hash(p)
                    doc.metadata["file_hash"] = f_hash
                    path_hash_map[file_path_str] = f_hash
            except Exception as e:
                logger.warning("Could not compute hash for {}: {}", file_path_str, e)
                raise RuntimeError(
                    f"Failed to compute file hash for {file_path}"
                ) from e

        return docs

    def _attach_clean_text(self, docs: Iterable[Document]) -> list[Document]:
        """
        Attach cleaned text to each document.

        Args:
            docs (Iterable[Document]): The documents to process.

        Returns:
            list[Document]: The documents with cleaned text.
        """
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
        """
        Load document readers for various file types.
        """
        audio_reader = AudioReader(device=self.device)
        image_reader = ImageReader()
        table_reader = TableReader(
            text_cols=self.table_text_cols,
            metadata_cols=self.table_metadata_cols
            if self.table_metadata_cols
            else None,
            id_col=self.table_id_col,
            excel_sheet=self.table_excel_sheet,
        )

        def _metadata(path: str | Path) -> dict[str, str]:
            """
            Get metadata for a file.

            Args:
                path (str | Path): The path to the file.

            Returns:
                dict[str, str]: Metadata including file path, name, and hash.
            """
            resolved = path if isinstance(path, Path) else Path(path)
            path_str = str(resolved)

            if path_str in self.file_hash_cache:
                file_hash = self.file_hash_cache[path_str]
            else:
                file_hash = compute_file_hash(resolved)
                self.file_hash_cache[path_str] = file_hash

            filename = resolved.name
            return {
                "file_path": path_str,
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
                ),
                ".tsv": TableReader(
                    csv_sep="\t",
                    text_cols=self.table_text_cols,
                    metadata_cols=set(self.table_metadata_cols)
                    if self.table_metadata_cols
                    else None,
                    id_col=self.table_id_col,
                ),
                ".xls": table_reader,
                ".xlsx": table_reader,
            },
        )

    def _load_node_parsers(self) -> None:
        """
        Load document parsers for various file types.
        """
        self.md_node_parser = MarkdownNodeParser()
        self.docling_node_parser = DoclingNodeParser()

    @staticmethod
    def _extract_file_hash(data: dict | None) -> str | None:
        """
        Extract the file hash from the given data.

        Args:
            data (dict | None): The input data.

        Returns:
            str | None: The extracted file hash, or None if not found.
        """
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

    def _filter_docs_by_existing_hashes(
        self, docs: Iterable[Document], existing_hashes: set[str] | None
    ) -> list[Document]:
        """
        Filter documents by their existing hashes.

        Args:
            docs (Iterable[Document]): The documents to filter.
            existing_hashes (set[str] | None): The set of existing document hashes.

        Returns:
            list[Document]: The filtered list of documents.
        """
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

    def _process_docs_hierarchical(
        self, docs: list[Document], base_parser: NodeParser | None = None
    ) -> list[BaseNode]:
        """
        Process documents possibly using hierarchical chunking.

        Args:
            docs (list[Document]): Documents to process.
            base_parser (NodeParser | None): The base parser to use for coarse chunking.
                                             If None, uses the internal logic of HierarchicalNodeParser (defaulting to SentenceSplitter).

        Returns:
            list[BaseNode]: The processed nodes.
        """
        if not docs:
            return []

        if self.hierarchical_node_parser:
            # If a specific base parser is provided (and it's not the default sentence splitter),
            # use it to generate Level 1 chunks, then refine.
            if base_parser and base_parser != self.sentence_splitter:
                # Use base parser for Coarse L1
                # Note: get_nodes_from_documents returns list[BaseNode]
                coarse = base_parser.get_nodes_from_documents(docs)
                # Refine to Level 2
                return self.hierarchical_node_parser._parse_nodes(coarse)
            else:
                # Use hierarchical parser from scratch (L0 -> L1 -> L2)
                # This uses SentenceSplitter internally for L1
                return self.hierarchical_node_parser.get_nodes_from_documents(docs)

        parser = base_parser or self.sentence_splitter
        return parser.get_nodes_from_documents(docs)

    def _create_nodes(self, docs: list[Document]) -> list[BaseNode]:
        """
        Create nodes from the provided documents.

        Args:
            docs (list[Document]): The documents to process.

        Returns:
            list[BaseNode]: The created nodes.

        Raises:
            RuntimeError: If node parsers are not initialized.
        """
        if self.md_node_parser is None or self.docling_node_parser is None:
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

            if source_kind in {"audio", "video"} or ext in {
                "avi",
                "flv",
                "mkv",
                "mov",
                "mpeg",
                "mpg",
                "mp3",
                "mp4",
                "m4v",
                "ogg",
                "wav",
                "webm",
                "wmv",
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
                "Parsing {} audio documents with SentenceSplitter",
                len(audio_docs),
            )
            nodes.extend(self._process_docs_hierarchical(audio_docs))

        if img_docs:
            logger.info(
                "Parsing {} image documents with SentenceSplitter",
                len(img_docs),
            )
            nodes.extend(self._process_docs_hierarchical(img_docs))

        if json_docs:
            logger.info(
                "Parsing {} JSON documents with SentenceSplitter",
                len(json_docs),
            )
            nodes.extend(self._process_docs_hierarchical(json_docs))

        if document_docs:

            def _is_docling_json(doc: Document) -> bool:
                """
                Check if the document text is valid Docling JSON.

                Args:
                    doc (Document): The document to check.

                Returns:
                    bool: True if the document text is valid Docling JSON, False otherwise.
                """
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
                    self._process_docs_hierarchical(
                        pdf_docs_docling, self.docling_node_parser
                    )
                )
            if pdf_docs_md:
                logger.info(
                    "Parsing {} Markdown PDFs with MarkdownNodeParser",
                    len(pdf_docs_md),
                )
                nodes.extend(
                    self._process_docs_hierarchical(pdf_docs_md, self.md_node_parser)
                )

        if table_docs:
            logger.info(
                "Parsing {} table documents with SentenceSplitter (one node per document)",
                len(table_docs),
            )
            table_splitter = SentenceSplitter(chunk_size=10_000_000, chunk_overlap=0)
            nodes.extend(table_splitter.get_nodes_from_documents(table_docs))

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
                    self._process_docs_hierarchical(markdown_docs, self.md_node_parser)
                )
            if plain_docs:
                logger.info(
                    "Parsing {} plain text documents with SentenceSplitter",
                    len(plain_docs),
                )
                nodes.extend(self._process_docs_hierarchical(plain_docs))

        if self.entity_extractor:
            total_nodes = len(nodes)

            def _process_node(args: tuple[int, BaseNode]) -> None:
                """
                Helper to process a single node for entity extraction.

                Args:
                    args (tuple[int, BaseNode]): The index and node to process.
                """
                idx, node = args
                text_value = getattr(node, "text", "") or ""
                if not text_value.strip():
                    return
                try:
                    # Provide type hint if self.entity_extractor is not None
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

            # Use ThreadPoolExecutor to run extraction in parallel
            # We limit workers to avoid overwhelming the local inference server
            with ThreadPoolExecutor(max_workers=self.ie_max_workers) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(_process_node, (i, node))
                    for i, node in enumerate(nodes)
                ]

                # Wait for completion and update progress
                for i, _ in enumerate(as_completed(futures)):
                    if self.progress_callback:
                        self.progress_callback(
                            f"Extracting entities: {i + 1}/{total_nodes} chunks processed"
                        )

        return nodes
