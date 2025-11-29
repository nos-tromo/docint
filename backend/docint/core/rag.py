from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import pandas as pd
import torch
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from llama_index.core import (
    Response,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import BaseNode, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from sqlalchemy.orm import Session

from docint.core.state.base import _make_session_maker
from docint.core.state.citation import Citation
from docint.core.state.conversation import Conversation
from docint.core.state.turn import Turn
from docint.core.readers.audio import AudioReader
from docint.core.readers.documents import HybridPDFReader
from docint.core.readers.images import ImageReader
from docint.core.readers.json import CustomJSONReader
from docint.core.readers.tables import TableReader
from docint.utils.clean_text import basic_clean
from docint.utils.hashing import compute_file_hash

# --- Environment variables ---
load_dotenv()
DATA_PATH: Path = Path(os.getenv("DATA_PATH", Path.home() / "docint" / "data"))
PROMPT_DIR: Path = Path(__file__).parents[1].resolve() / "utils" / "prompts"
REQUIRED_EXTS_PATH: Path = (
    Path(__file__).parent.resolve() / "readers" / "required_exts.txt"
)
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_THINKING: str = os.getenv("OLLAMA_THINKING", "false")
QDRANT_COL_DIR: str = os.getenv("QDRANT_COL_DIR", "qdrant_collections")
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "http://127.0.0.1:6333")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
SPARSE_MODEL: str = os.getenv("SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
GEN_MODEL: str = os.getenv("LLM", "granite4:7b-a1b-h")
RETRIEVE_SIMILARITY_TOP_K: int = int(os.getenv("RETRIEVE_SIMILARITY_TOP_K", "20"))

CleanFn = Callable[[str], str]


@dataclass(slots=True)
class RAG:
    """
    Represents a Retrieval-Augmented Generation (RAG) model.
    """

    # --- Data path & cleaning setup ---
    data_dir: Path = Path(DATA_PATH) if not isinstance(DATA_PATH, Path) else DATA_PATH
    clean_fn: CleanFn = basic_clean

    # --- Models ---
    embed_model_id: str = EMBED_MODEL
    sparse_model_id: str = SPARSE_MODEL
    rerank_model_id: str = RERANK_MODEL
    gen_model_id: str = GEN_MODEL

    # --- Qdrant controls ---
    qdrant_host: str = QDRANT_HOST
    _qdrant_host_dir: Path | None = field(default=None, init=False, repr=False)
    qdrant_collection: str = "default"

    # --- Ollama Parameters ---
    base_url: str = OLLAMA_HOST
    context_window: int = -1
    temperature: float = 0.2
    request_timeout: int = 1200
    thinking: bool = bool(OLLAMA_THINKING)
    ollama_options: dict[str, Any] | None = None

    # --- Reranking / retrieval ---
    enable_hybrid: bool = True
    embed_batch_size: int = 64
    retrieve_similarity_top_k: int = RETRIEVE_SIMILARITY_TOP_K
    rerank_top_n: int = int(retrieve_similarity_top_k // 5)

    # --- Prompt config ---
    prompt_template_path: Path | None = PROMPT_DIR
    if prompt_template_path:
        summarize_prompt_path: Path = PROMPT_DIR / "summarize.txt"
    summarize_prompt: str = field(default="", init=False)

    # --- Directory reader config ---
    reader_errors: str = "ignore"
    reader_recursive: bool = True
    reader_encoding: str = "utf-8"
    reader_required_exts: list[str] = field(default_factory=list, init=False)
    reader_required_exts_path: Path = field(default=REQUIRED_EXTS_PATH, init=False)

    # --- TableReader config ---
    table_text_cols: list[str] | None = None
    table_metadata_cols: list[str] | str | None = None
    table_id_col: str | None = None
    table_excel_sheet: str | int | None = None
    table_row_limit: int | None = None
    table_row_filter: str | None = None

    # --- SemanticSplitterNodeParser config ---
    buffer_size: int = 5
    breakpoint_percentile_threshold: int = 90

    # --- SentenceSplitter config ---
    chunk_size: int = 1024
    chunk_overlap: int = 0
    semantic_splitter_char_limit: int = 20000

    # --- Runtime (lazy caches / not in repr) ---
    _device: str | None = field(default=None, init=False, repr=False)
    _embed_model: BaseEmbedding | None = field(default=None, init=False, repr=False)
    _gen_model: Ollama | None = field(default=None, init=False, repr=False)
    _reranker: SentenceTransformerRerank | None = field(
        default=None, init=False, repr=False
    )
    _qdrant_client: QdrantClient | None = field(default=None, init=False, repr=False)
    _qdrant_aclient: AsyncQdrantClient | None = field(
        default=None, init=False, repr=False
    )

    pdf_reader: HybridPDFReader | None = field(default=None, init=False)
    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    docling_node_parser: DoclingNodeParser | None = field(default=None, init=False)
    md_node_parser: MarkdownNodeParser | None = field(default=None, init=False)
    semantic_node_parser: SemanticSplitterNodeParser | None = field(
        default=None, init=False
    )
    table_node_parser: SemanticSplitterNodeParser | None = field(
        default=None, init=False
    )
    sentence_splitter: SentenceSplitter = field(
        default_factory=SentenceSplitter, init=False
    )

    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)

    index: VectorStoreIndex | None = field(default=None, init=False)
    query_engine: RetrieverQueryEngine | None = field(default=None, init=False)

    # Chat/session runtime
    chat_engine: RetrieverQueryEngine | CondenseQuestionChatEngine | None = field(
        default=None, init=False
    )
    chat_memory: Any | None = field(default=None, init=False)
    _SessionMaker: Any | None = field(default=None, init=False, repr=False)
    session_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to set up any necessary components.
        """
        with open(self.reader_required_exts_path, "r", encoding="utf-8") as f:
            self.reader_required_exts = [f".{line.strip()}" for line in f]

        with open(self.summarize_prompt_path, "r", encoding="utf-8") as f:
            self.summarize_prompt = f.read()

        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    # --- Static methods ---
    @staticmethod
    def _list_supported_sparse_models() -> list[str]:
        """
        Lists all supported sparse models.

        Returns:
            list[str]: A list of supported sparse model IDs.

        Raises:
            ImportError: If fastembed is not installed.
        """
        try:
            return [m["model"] for m in SparseTextEmbedding.list_supported_models()]
        except ImportError:
            logger.warning(
                "ImportError: fastembed is not installed; cannot list sparse models."
            )
            return []

    # --- Properties (lazy loading) ---
    @property
    def qdrant_host_dir(self) -> Path:
        """
        Best-effort resolution of the host directory where Qdrant stores data.
        Used only as a *fallback* when we cannot reach the Qdrant API.
        Priority: explicit field -> env var -> platform default under home.

        Returns:
            The Path representing the Qdrant host directory.

        Raises:
            ValueError: If the Qdrant host directory is not set.
        """
        if self._qdrant_host_dir is None:
            env = os.getenv("QDRANT_COL_DIR")
            if env:
                self._qdrant_host_dir = Path(env)
            else:
                home = os.getenv("HOME") or os.getenv("USERPROFILE")
                if home:
                    self._qdrant_host_dir = Path(home) / ".qdrant" / "storage"
        if self._qdrant_host_dir is None:
            logger.error("ValueError: Qdrant host directory is not set.")
            raise ValueError("Qdrant host directory is not set.")
        return self._qdrant_host_dir

    @property
    def device(self) -> str:
        """
        Returns the device being used for computation.

        Returns:
            str: The device being used ("cpu", "cuda", or "mps").
        """
        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("Using CUDA for GPU acceleration.")
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                self._device = "mps"
                logger.info("Using MPS for GPU acceleration.")
            else:
                self._device = "cpu"
                logger.info("Using CPU for computation.")
        return self._device

    @property
    def embed_model(self) -> BaseEmbedding:
        """
        Lazily initializes and returns the embedding model.

        Returns:
            BaseEmbedding: The initialized embedding model.
        """
        if self._embed_model is None:
            try:
                model = HuggingFaceEmbedding(
                    model_name=self.embed_model_id,
                    normalize=True,
                    device=self.device,
                )
                # Trigger warmup to detect potential MPS/meta-tensor issues immediately
                model.get_text_embedding("warmup")
                self._embed_model = model
                logger.info("Initializing embedding model: {}", self.embed_model_id)
            except Exception as e:
                if self.device == "mps" and "meta tensor" in str(e):
                    logger.warning(
                        "MPS meta-tensor error detected. Falling back to CPU for embeddings. Error: {}",
                        e,
                    )
                    self._embed_model = HuggingFaceEmbedding(
                        model_name=self.embed_model_id,
                        normalize=True,
                        device="cpu",
                    )
                else:
                    raise
        return self._embed_model

    @property
    def sparse_model(self) -> str | None:
        """
        Returns the configured sparse model id for hybrid retrieval.

        Returns:
            str | None: The sparse model id or None if not enabled.

        Raises:
            ValueError: If the sparse model is not supported.
        """
        if not self.enable_hybrid:
            return None
        if self.sparse_model_id not in self._list_supported_sparse_models():
            logger.error(
                "ValueError: Sparse model {} not supported. Supported: {}",
                self.sparse_model_id,
                self._list_supported_sparse_models(),
            )
            raise ValueError(
                f"Sparse model {self.sparse_model_id!r} not supported. "
                f"Supported: {self._list_supported_sparse_models()}"
            )
        logger.info("Initializing sparse model: {}", self.sparse_model_id)
        return self.sparse_model_id

    @property
    def reranker(self) -> SentenceTransformerRerank:
        """
        Lazily initializes and returns the reranker model (SentenceTransformerRerank).

        Returns:
            SentenceTransformerRerank: The initialized reranker model.
        """
        if self._reranker is None:
            self._reranker = SentenceTransformerRerank(
                top_n=self.rerank_top_n,
                model=self.rerank_model_id,
                device=self.device,
            )
            logger.info("Initializing reranker model: {}", self.rerank_model_id)
        return self._reranker

    @property
    def gen_model(self) -> Ollama:
        """
        Lazily initializes and returns the generation model (Ollama).

        Returns:
            Ollama: The initialized generation model.
        """
        if self._gen_model is None:
            self._gen_model = Ollama(
                model=self.gen_model_id,
                base_url=self.base_url,
                temperature=self.temperature,
                context_window=self.context_window,
                request_timeout=self.request_timeout,
                thinking=self.thinking,
                additional_kwargs=self.ollama_options,
            )
            logger.info("Initializing generator model: {}", self.gen_model_id)
        return self._gen_model

    @property
    def qdrant_client(self) -> QdrantClient:
        """
        Lazily initializes and returns the Qdrant client.

        Returns:
            QdrantClient: The initialized Qdrant client.
        """
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(url=self.qdrant_host)
            logger.info(
                "Qdrant client initialized: {}",
                self.qdrant_host,
            )
        return self._qdrant_client

    @property
    def qdrant_aclient(self) -> AsyncQdrantClient:
        """
        Lazily initializes and returns the Qdrant async client.

        Returns:
            AsyncQdrantClient: The initialized Qdrant async client.
        """
        if self._qdrant_aclient is None:
            self._qdrant_aclient = AsyncQdrantClient(url=self.qdrant_host)
            logger.info(
                "Qdrant async client initialized: {}",
                self.qdrant_host,
            )
        return self._qdrant_aclient

    # --- Build pieces ---
    def _load_doc_readers(self) -> None:
        """
        Loads the document readers for various file types.
        """
        # Audio reader for audio files
        audio_reader = AudioReader(device=self.device)

        # Image reader for image files
        image_reader = ImageReader()

        # Table reader for CSV/TSV/XLSX/Parquet
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
                # audio files
                ".mpeg": audio_reader,
                ".mp3": audio_reader,
                ".m4a": audio_reader,
                ".ogg": audio_reader,
                ".wav": audio_reader,
                ".webm": audio_reader,
                # video files
                ".avi": audio_reader,
                ".flv": audio_reader,
                ".mkv": audio_reader,
                ".mov": audio_reader,
                ".mpg": audio_reader,
                ".mp4": audio_reader,
                ".m4v": audio_reader,
                ".wmv": audio_reader,
                # json files
                ".json": CustomJSONReader(),
                # document files
                ".docx": HybridPDFReader(),
                ".pdf": HybridPDFReader(),
                # image files
                ".gif": image_reader,
                ".jpeg": image_reader,
                ".jpg": image_reader,
                ".png": image_reader,
                # table files
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
                    csv_sep="\t",  # allow explicit TSV sep
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
        """
        Initializes advanced, multilingual-aware node parsers for different document types.
        """
        # Markdown parser (for .txt, .md, .rst)
        self.md_node_parser = MarkdownNodeParser()

        # Layout-aware for Docling JSON
        self.docling_node_parser = DoclingNodeParser()

        # Semantic parser for tables, text, and json
        self.semantic_node_parser = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
        )

    def _vector_store(self) -> QdrantVectorStore:
        """
        Creates the vector store for document embeddings.

        Returns:
            QdrantVectorStore: The initialized vector store.
        """
        return QdrantVectorStore(
            collection_name=self.qdrant_collection,
            client=self.qdrant_client,
            aclient=self.qdrant_aclient,
            enable_hybrid=self.enable_hybrid,
            fastembed_sparse_model=self.sparse_model,
        )

    def _storage_context(self, vector_store: QdrantVectorStore) -> StorageContext:
        """
        Creates the storage context for document embeddings.

        Args:
            vector_store (QdrantVectorStore): The vector store for document embeddings.

        Returns:
            StorageContext: The created storage context.
        """
        return StorageContext.from_defaults(vector_store=vector_store)

    def _index(self, storage_ctx: StorageContext) -> VectorStoreIndex:
        """
        Creates the vector store index for document embeddings.

        Args:
            storage_ctx (StorageContext): The storage context for document embeddings.

        Returns:
            VectorStoreIndex: The created vector store index.
        """
        return VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_ctx,
            embed_model=self.embed_model,
        )

    @staticmethod
    def _extract_file_hash(data: Any) -> str | None:
        """
        Best-effort extraction of a ``file_hash`` value from nested payloads.

        Args:
            data (Any): The data dictionary to search for a file hash.
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
                nested_hash = RAG._extract_file_hash(nested)
                if nested_hash:
                    return nested_hash

        for value in data.values():
            if isinstance(value, dict):
                nested_hash = RAG._extract_file_hash(value)
                if nested_hash:
                    return nested_hash
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        nested_hash = RAG._extract_file_hash(item)
                        if nested_hash:
                            return nested_hash
        return None

    def _get_existing_file_hashes(self) -> set[str]:
        """
        Fetch file hashes already stored in the active Qdrant collection.

        Returns:
            set[str]: A set of existing file hashes.
        """

        existing: set[str] = set()

        try:
            _ = self.qdrant_client
        except Exception as exc:
            logger.warning(
                "Unable to initialize Qdrant client for hash lookup: {}", exc
            )
            return existing

        offset: Any = None
        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    offset=offset,
                    limit=256,
                    with_vectors=False,
                    with_payload=True,
                )
            except Exception as exc:
                # Qdrant may return a 404 when the collection does not exist;
                # treat that case as non-fatal and log at debug level to avoid
                # cluttering logs with expected messages for new collections.
                msg = str(exc)
                not_found = (
                    "Not found" in msg
                    or "doesn't exist" in msg
                    or "does not exist" in msg
                    or f"Collection `{self.qdrant_collection}`" in msg
                )
                if not_found:
                    logger.debug(
                        "Qdrant collection '%s' not found; skipping existing-hash check: %s",
                        self.qdrant_collection,
                        exc,
                    )
                else:
                    logger.warning(
                        "Failed to fetch existing hashes from collection '{}': {}",
                        self.qdrant_collection,
                        exc,
                    )
                break

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", None)
                if isinstance(payload, dict):
                    file_hash = self._extract_file_hash(payload)
                    if file_hash:
                        existing.add(file_hash)

            if offset is None:
                break

        return existing

    def _filter_docs_by_existing_hashes(self) -> None:
        """
        Remove documents whose hashes already exist in the target collection.
        """

        if not self.docs:
            return

        existing_hashes = self._get_existing_file_hashes()
        if not existing_hashes:
            return

        filtered_docs: list[Document] = []
        skipped: dict[str, str] = {}

        for doc in self.docs:
            metadata = getattr(doc, "metadata", {}) or {}
            file_hash = metadata.get("file_hash") or self._extract_file_hash(metadata)
            if not file_hash or file_hash not in existing_hashes:
                filtered_docs.append(doc)
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
                "Skipping {} file(s) already ingested in collection '{}': {}",
                len(skipped),
                self.qdrant_collection,
                ", ".join(sorted(display)),
            )

        self.docs = filtered_docs

    def _partition_large_docs(
        self, docs: list[Document]
    ) -> tuple[list[Document], list[Document]]:
        """
        Split documents into ones that are safe for semantic splitting and
        ones that must be chunked aggressively to avoid huge embeddings.

        Args:
            docs (list[Document]): The documents to partition.

        Returns:
            tuple[list[Document], list[Document]]: A tuple containing two lists:
                - The first list contains documents safe for semantic splitting.
                - The second list contains documents that exceed the character limit
                  and should be chunked with the sentence splitter.
        """

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
        """Split each oversized document into smaller pseudo-documents.

        This avoids giving the sentence splitter multi-megabyte blobs that can
        stall the tokenizer, while preserving the original metadata for later
        attribution.
        """

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
        """
        Use the semantic splitter when possible, otherwise fall back to
        the sentence splitter for oversized or error-prone documents.

        Args:
            docs (list[Document]): The documents to process.
            doc_label (str): A label for the document type (for logging).

        Returns:
            list[BaseNode]: The resulting list of nodes.

        Raises:
            RuntimeError: If the semantic splitter is not initialized.
        """

        if not docs:
            return []

        if self.semantic_node_parser is None:
            logger.error("Semantic splitter is not initialized.")
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
                        "Semantic splitter failed for {} {} document(s); "
                        "retrying with sentence-based chunks. Error: {}"
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
                    "Chunking {} {} document(s) ({} expanded segments) over {} chars "
                    "with SentenceSplitter"
                ),
                len(oversized_docs),
                doc_label,
                len(exploded_docs),
                limit,
            )
            nodes.extend(self.sentence_splitter.get_nodes_from_documents(exploded_docs))

        return nodes

    def _create_nodes(self) -> None:
        """
        Converts loaded documents into nodes using the appropriate parsers.

        Raises:
            RuntimeError: If the directory reader or node parsers are not initialized.
        """
        if self.dir_reader is None:
            logger.error("RuntimeError: Directory reader is not initialized.")
            raise RuntimeError("Directory reader is not initialized.")
        self.docs = self.dir_reader.load_data()
        self._filter_docs_by_existing_hashes()
        cleaned_docs = []
        for doc in self.docs:
            if hasattr(doc, "text") and isinstance(doc.text, str):
                cleaned_docs.append(
                    Document(text=self.clean_fn(doc.text), metadata=doc.metadata)
                )
            else:
                cleaned_docs.append(doc)
        self.docs = cleaned_docs
        if (
            self.md_node_parser is None
            or self.docling_node_parser is None
            or self.semantic_node_parser is None
        ):
            logger.error("RuntimeError: Node parsers are not initialized.")
            raise RuntimeError("Node parsers are not initialized.")

        audio_docs, document_docs, img_docs, json_docs, table_docs, text_docs = [
            [] for _ in range(6)
        ]
        for d in self.docs:
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
            elif file_type.endswith(("json", "jsonl")) or ext in {
                "json",
                "jsonl",
            }:
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
                "Parsing {} image documents with SemanticSplitterNodeParser",
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

            def _is_docling_json(doc):
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
                    "Parsing {} Markdown PDFs with MarkdownNodeParser", len(pdf_docs_md)
                )
                nodes.extend(self.md_node_parser.get_nodes_from_documents(pdf_docs_md))

        if table_docs:
            logger.info(
                "Parsing {} table documents with SemanticSplitterNodeParser",
                len(table_docs),
            )
            nodes.extend(self._semantic_nodes_with_fallback(table_docs, "table"))

        if text_docs:
            # detect markdown by file extension or text content
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

        self.nodes = nodes

    def create_index(self) -> None:
        """
        Create the full index by loading documents, converting to nodes, and
        setting up the Qdrant collection and vector store.
        """
        vector_store = self._vector_store()
        storage_ctx = self._storage_context(vector_store)
        self.index = self._index(storage_ctx)

    def create_query_engine(self) -> None:
        """
        Create the query engine with a retriever and reranker.

        Raises:
            RuntimeError: If the index is not initialized.
        """
        if self.index is None:
            logger.error("RuntimeError: Index is not initialized.")
            raise RuntimeError("Index is not initialized. Cannot create query engine.")
        k = min(max(self.retrieve_similarity_top_k, self.rerank_top_n * 8), 64)
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.index.as_retriever(similarity_top_k=k),
            llm=self.gen_model,
            node_postprocessors=[self.reranker],
        )

    def _normalize_response_data(
        self, query: str, result: Any, reason: str | None = None
    ) -> dict[str, Any]:
        """
        Normalize both llama_index.core.Response and AgentChatResponse into a single payload.
        Handles:
        - response text (result.response or result.text)
        - source_nodes (list[NodeWithScore])
        - metadata differences

        Returns:
            dict[str, Any]: A dictionary containing:
            - 'query': The original query string.
            - 'reasoning': The reasoning behind the response, if available.
            - 'response': The normalized response text.
            - 'sources': A list of source metadata dictionaries, each containing:
                - 'text': The text content of the source.
                - 'filename': The name of the file where the source was found.
                - 'filetype': The type of the file (e.g., PDF, CSV).
                - 'source': The source kind (e.g., "table" for TableReader).
                - 'page': Optional page number if the source is a PDF.
                - 'row': Optional row index if the source is a table.
            - 'table_info': Optional dictionary with 'n_rows' and 'n_cols' for table sources.
        """

        # --- normalize response text ---
        resp_text = None
        if hasattr(result, "response") and isinstance(result.response, str):
            resp_text = result.response
        elif hasattr(result, "text") and isinstance(result.text, str):
            resp_text = result.text
        elif hasattr(result, "message") and hasattr(result.message, "content"):
            resp_text = str(result.message.content)
        else:
            resp_text = ""

        # strip <think>…</think> (optional)
        if resp_text.startswith("<think>"):
            m = re.search(r"<think>(.*?)</think>", resp_text, flags=re.DOTALL)
            reason = (m.group(1).strip() if m else None) or reason
            resp_text = re.sub(
                r"<think>.*?</think>", "", resp_text, flags=re.DOTALL
            ).strip()

        # --- normalize source_nodes ---
        source_nodes = getattr(result, "source_nodes", None)
        if source_nodes is None and hasattr(result, "metadata"):
            # some Response variants tuck nodes under metadata
            meta = getattr(result, "metadata", {}) or {}
            source_nodes = meta.get("source_nodes")
        if not isinstance(source_nodes, list):
            source_nodes = []

        sources = []
        for nws in source_nodes:
            # nws: NodeWithScore; the node itself:
            node = getattr(nws, "node", None)
            if node is None:
                continue
            meta = getattr(node, "metadata", {}) or {}

            # common file info (Docling puts origin here)
            origin = meta.get("origin") or {}
            filename = (
                origin.get("filename")
                or meta.get("file_name")
                or meta.get("filename")
                or meta.get("file_path")
                or ""
            )
            filetype = (
                origin.get("mimetype")
                or meta.get("mimetype")
                or meta.get("file_type")
                or ""
            )
            file_hash = origin.get("file_hash") or meta.get("file_hash")
            source_kind = meta.get("source", "unknown")

            # --- page detection (PDF / Docling) ---
            page = meta.get("page_number") or meta.get("page_label") or meta.get("page")
            if page is None:
                # try doc_items → prov → page_no
                doc_items = meta.get("doc_items") or []
                for item in doc_items:
                    for prov in item.get("prov", []) or []:
                        if isinstance(prov, dict) and "page_no" in prov:
                            page = prov.get("page_no")
                            break
                    if page is not None:
                        break
            try:
                page = int(page) if page is not None else None
            except Exception:
                pass

            # --- row detection (tables) ---
            table_meta = meta.get("table") or {}
            row_index = table_meta.get("row_index")
            try:
                row_index = int(row_index) if row_index is not None else None
            except Exception:
                pass

            location_label = (
                "page"
                if page is not None
                else ("row" if row_index is not None else None)
            )
            location_value = page if page is not None else row_index

            text_value = getattr(node, "text", "") or ""
            preview_url: str | None = None
            if file_hash:
                preview_url = (
                    f"/sources/preview?collection={self.qdrant_collection}"
                    f"&file_hash={file_hash}"
                )

            src = {
                "text": text_value,
                "preview_text": text_value[:280].strip(),
                "filename": filename,
                "filetype": filetype,
                "source": source_kind,
            }
            if file_hash:
                src["file_hash"] = file_hash
            if preview_url:
                src["preview_url"] = preview_url
                src["document_url"] = preview_url
            if location_label:
                src[location_label] = location_value

            if source_kind == "table":
                n_rows = table_meta.get("n_rows")
                n_cols = table_meta.get("n_cols")
                src["table_info"] = {"n_rows": n_rows, "n_cols": n_cols}

            sources.append(src)

        return {
            "query": query,
            "reasoning": reason,
            "response": resp_text,
            "sources": sources,
        }

    # --- Collection discovery / selection ---
    def list_collections(self, prefer_api: bool = True) -> list[str]:
        """
        Return a list of collection names. Uses Qdrant API when available; falls back to listing the host storage path.

        Args:
            prefer_api (bool): Whether to prefer the Qdrant API over filesystem access.

        Returns:
            list[str]: A list of collection names.
        """
        if prefer_api:
            try:
                resp = self.qdrant_client.get_collections()
                names = [c.name for c in getattr(resp, "collections", []) or []]
                if names:
                    return sorted(names)
            except Exception as e:
                logger.warning(
                    "Qdrant API list_collections failed, will try FS fallback: {}",
                    e,
                )
        base = self.qdrant_host_dir
        if base is None:
            return []
        collections_dir = base / "collections"
        try:
            if not collections_dir.exists():
                return []
            return sorted([p.name for p in collections_dir.iterdir() if p.is_dir()])
        except Exception as e:
            logger.warning("FS fallback list_collections failed: {}", e)
            return []

    def select_collection(self, name: str) -> None:
        """Switch active collection, ensuring it already exists.

        Args:
            name: Name of the collection to select.

        Raises:
            ValueError: If the name is empty or the collection does not exist.
        """
        if not name or not name.strip():
            logger.error("ValueError: Collection name cannot be empty.")
            raise ValueError("Collection name cannot be empty.")
        name = name.strip()
        if name not in self.list_collections():
            logger.error("ValueError: Collection '{}' does not exist.", name)
            raise ValueError(f"Collection '{name}' does not exist.")

        self.qdrant_collection = name

        # Reset any state tied to the previously selected collection so that
        # future queries do not use stale indexes or conversations.
        self.docs.clear()
        self.nodes.clear()
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.chat_memory = None
        self.session_id = None

    # --- Public API ---
    def ingest_docs(self, data_dir: str | Path) -> None:
        """
        Ingest documents from the specified directory into the Qdrant collection.

        Args:
            data_dir (str | Path): The directory containing the documents to ingest.
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self._load_doc_readers()
        self._load_node_parsers()
        self._create_nodes()
        self.create_index()
        try:
            eff_k = None
            if self.query_engine is not None and hasattr(
                self.query_engine, "retriever"
            ):
                eff_k = None
                if self.query_engine is not None and hasattr(
                    self.query_engine, "retriever"
                ):
                    eff_k = (
                        getattr(self.query_engine.retriever, "similarity_top_k", None)
                        if self.query_engine
                        else None
                    )
        except Exception:
            eff_k = None
        logger.info(
            "Effective retrieval k={} | top_n={} | embed_device={} | rerank_device={}",
            eff_k,
            self.rerank_top_n,
            self.device,
            self.device,
        )
        logger.info("Documents ingested successfully.")

    async def asingest_docs(self, data_dir: str | Path) -> None:
        """
        Asynchronously ingest documents from the specified directory into the Qdrant collection.

        Args:
            data_dir (str | Path): The directory containing the documents to ingest.

        Raises:
            RuntimeError: If the index is not initialized for async ingestion.
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self._load_doc_readers()
        self._load_node_parsers()
        self._create_nodes()
        if self.index is None:
            logger.error("RuntimeError: Index is not initialized for async ingestion.")
            raise RuntimeError("Index is not initialized for async ingestion.")
        # Concurrent, non-blocking upsert into Qdrant via aclient
        await self.index.ainsert_nodes(self.nodes)
        try:
            eff_k = None
            if self.query_engine is not None and hasattr(
                self.query_engine, "retriever"
            ):
                eff_k = getattr(self.query_engine.retriever, "similarity_top_k", None)
        except Exception:
            eff_k = None
        logger.info(
            "Effective retrieval k={} | top_n={} | embed_device={} | rerank_device={}",
            eff_k,
            self.rerank_top_n,
            self.device,
            self.device,
        )
        logger.info("Documents ingested successfully (async path).")

    def run_query(self, prompt: str) -> dict[str, Any]:
        """
        Run a query against the Qdrant collection.

        Args:
            prompt (str): The query prompt.

        Returns:
            dict[str, Any]: The query results.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the query engine is not initialized.
            TypeError: If the response is not of the expected type.
        """
        if not prompt.strip():
            logger.error("ValueError: Query prompt cannot be empty.")
            raise ValueError("Query prompt cannot be empty.")
        engine = self.query_engine
        if engine is None:
            logger.error("RuntimeError: Query engine has not been initialized.")
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs() first."
            )
        result = engine.query(prompt)
        if not isinstance(result, Response):
            logger.error("TypeError: Expected Response, got {}.", type(result).__name__)
            raise TypeError(f"Expected Response, got {type(result).__name__}")
        return self._normalize_response_data(prompt, result)

    async def run_query_async(self, prompt: str) -> dict[str, Any]:
        """
        Run a query against the Qdrant collection asynchronously.

        Args:
            prompt (str): The query prompt.

        Returns:
            dict[str, Any]: The query results.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the query engine is not initialized.
            TypeError: If the response is not of the expected type.
        """
        if not prompt.strip():
            logger.error("ValueError: Query prompt cannot be empty.")
            raise ValueError("Query prompt cannot be empty.")
        engine = self.query_engine
        if engine is None:
            logger.error("RuntimeError: Query engine has not been initialized.")
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs()/asingest_docs() first."
            )
        result = await engine.aquery(prompt)
        if not isinstance(result, Response):
            logger.error("TypeError: Expected Response, got {}.", type(result).__name__)
            raise TypeError(f"Expected Response, got {type(result).__name__}")
        return self._normalize_response_data(prompt, result)

    # --- Session store wiring ---
    def init_session_store(self, db_url: str = "sqlite:///rag_sessions.db") -> None:
        """
        Initialize (or reinitialize) the SQLAlchemy session factory.

        Args:
            db_url (str, optional): The database URL. Defaults to "sqlite:///rag_sessions.db".
        """
        self._SessionMaker = _make_session_maker(db_url)

    def _ensure_store(self) -> None:
        """
        Ensure the session store is initialized.
        """
        if self._SessionMaker is None:
            self.init_session_store()

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """Context manager that yields a SQLAlchemy session and closes it on exit."""
        self._ensure_store()
        if self._SessionMaker is None:
            logger.error("RuntimeError: SessionMaker is not initialized.")
            raise RuntimeError("SessionMaker is not initialized.")
        session = self._SessionMaker()
        try:
            yield session
        finally:
            session.close()

    def _load_or_create_convo(self, session: Session, session_id: str) -> Conversation:
        """
        Load an existing conversation or create a new one using the provided session.

        Args:
            session (Session): Active SQLAlchemy session.
            session_id (str): The ID of the session.

        Returns:
            Conversation: The conversation row for the provided session id.

        """
        conv = session.get(Conversation, session_id)
        if conv is None:
            conv = Conversation(id=session_id)
            session.add(conv)
            session.commit()
        return conv

    def _get_rolling_summary(self, session_id: str) -> str:
        """
        Get the rolling summary for a conversation.

        Args:
            session_id (str): The ID of the session.

        Returns:
            str: The rolling summary for the conversation.

        Raises:
            RuntimeError: If the SessionMaker is not initialized.
        """
        with self._session_scope() as s:
            conv = s.get(Conversation, session_id)
            if conv is None:
                return ""
            summary_text = cast(str | None, conv.rolling_summary)
            return summary_text or ""

    # --- Chat session lifecycle ---
    def _persist_turn(
        self, session_id: str, user_msg: str, resp: Any, data: dict
    ) -> None:
        """
        Persist the conversation turn and its citations in the relational store.

        Args:
            session_id (str): The ID of the session.
            user_msg (str): The user's message.
            resp (Any): The response object.
            data (dict): The additional data to persist.
        """
        with self._session_scope() as s:
            conv = self._load_or_create_convo(s, session_id)

            # try to capture the condensed query from response metadata
            meta = getattr(resp, "metadata", {}) or {}
            rewritten = meta.get("query_str") or meta.get("compressed_query_str")

            reasoning = data.get("reasoning")
            next_idx = len(conv.turns)
            t = Turn(
                conversation_id=conv.id,
                idx=next_idx,
                user_text=user_msg,
                rewritten_query=rewritten,
                model_response=data.get("response") or "",
                reasoning=reasoning,
            )
            s.add(t)
            s.flush()

            # citations
            for src_node in getattr(resp, "source_nodes", []) or []:
                # Prefer node-attached metadata
                node = getattr(src_node, "node", None)
                meta_node = getattr(node, "metadata", {}) or {}

                # Robust filename/filetype extraction across readers
                filename = (
                    meta_node.get("file_name")
                    or meta_node.get("filename")
                    or meta_node.get("file_path")
                    or meta_node.get("source")
                    or meta_node.get("document_id")
                    or ""
                )
                filetype = (
                    meta_node.get("mimetype")
                    or meta_node.get("filetype")
                    or meta_node.get("content_type")
                    or ""
                )
                source_kind = meta_node.get("source", "")

                # Common page/row hints
                page = meta_node.get("page_label") or meta_node.get("page") or None
                table_meta = meta_node.get("table") or {}
                row_index = table_meta.get("row_index")

                # Capture a stable node id strictly from the node object
                node_id = None
                if node is not None:
                    node_id = getattr(node, "node_id", None) or getattr(
                        node, "id_", None
                    )

                score = (
                    float(getattr(src_node, "score", 0.0))
                    if hasattr(src_node, "score")
                    else None
                )

                s.add(
                    Citation(
                        turn_id=t.id,
                        node_id=str(node_id) if node_id is not None else None,
                        score=score,
                        filename=filename,
                        filetype=filetype,
                        source=source_kind,
                        page=int(page) if page is not None else None,
                        row=int(row_index) if row_index is not None else None,
                    )
                )

            s.commit()

    def _maybe_update_summary(self, session_id: str, every_n_turns: int = 5) -> None:
        """
        Check if the conversation summary should be updated and perform the update if necessary.

        Args:
            session_id (str): The ID of the session.
            every_n_turns (int): The interval of turns after which to update the summary.

        Raises:
            RuntimeError: If the SessionMaker is not initialized.
        """
        with self._session_scope() as s:
            conv = s.get(Conversation, session_id)
            if (
                not conv
                or len(conv.turns) == 0
                or (len(conv.turns) % every_n_turns) != 0
            ):
                return

            # Build a concise slice of the last N turns
            slice_text = []
            for turn in conv.turns[-every_n_turns:]:
                slice_text.append(
                    f"User: {turn.user_text}\nAssistant: {turn.model_response}"
                )
            prompt = self.summarize_prompt + "\n\n".join(slice_text)

            # Use the same LLM to summarize
            summary_resp = self.gen_model.complete(prompt)
            existing_summary = cast(str | None, conv.rolling_summary) or ""
            new_summary = (existing_summary + "\n" + summary_resp.text).strip()
            conv.rolling_summary = new_summary
            s.commit()

    def _get_node_text_by_id(self, node_id: str) -> str | None:
        """
        Best-effort fetch of a node's text from the index docstore given its id.

        Args:
            node_id (str): The ID of the node.

        Returns:
            str | None: The text content of the node, or None if not found.
        """
        try:
            if self.index is None:
                return None
            # Prefer storage_context.docstore when available
            docstore = getattr(self.index, "storage_context", None)
            if docstore is not None:
                docstore = getattr(docstore, "docstore", None)
            else:
                docstore = getattr(self.index, "docstore", None)
            if docstore is None:
                return None

            # Try common getters across versions
            for getter in ("get_node", "get", "get_document"):
                fn = getattr(docstore, getter, None)
                if callable(fn):
                    try:
                        node = fn(node_id)
                        if node is None:
                            continue
                        text = getattr(node, "text", None)
                        if isinstance(text, str) and text:
                            return text
                        # Some versions store content on helper methods
                        if (
                            isinstance(node, BaseNode)
                            and hasattr(node, "get_content")
                            and callable(node.get_content)
                        ):
                            content = node.get_content()
                            if isinstance(content, str) and content:
                                return content
                        if isinstance(node, BaseNode):
                            text = getattr(node, "text", None)
                            if isinstance(text, str) and text:
                                return text
                    except Exception:
                        continue
        except Exception:
            return None

        # Fallback to Qdrant payload when docstore text is unavailable
        try:
            recs = self.qdrant_client.retrieve(
                collection_name=self.qdrant_collection, ids=[node_id]
            )
            if recs:
                payload = getattr(recs[0], "payload", None)
                if isinstance(payload, dict):
                    txt = (
                        payload.get("text")
                        or payload.get("chunk")
                        or payload.get("content")
                    )
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
        except Exception:
            pass
        return None

    # Export session data as JSON
    def export_session(
        self, session_id: str | None = None, out_dir: str | Path = "session"
    ) -> Path:
        """
        Export the session data to the specified output directory.

        Args:
            session_id (str | None, optional): The ID of the session. Defaults to None.
            out_dir (str | Path, optional): The output directory for the exported session. Defaults to "session".

        Returns:
            Path: The path to the exported session directory.

        Raises:
            RuntimeError: If the SessionMaker is not initialized.
            ValueError: If no conversation is found for the given session ID or the session ID is invalid.
        """
        with self._session_scope() as s:
            if not session_id and self.session_id is not None:
                session_id = self.session_id

            if session_id is None:
                raise ValueError("Session ID cannot be None.")

            conv = s.get(Conversation, session_id)
            if conv is None:
                logger.error(
                    "ValueError: No conversation found for session_id={}", session_id
                )
                raise ValueError(f"No conversation found for session_id={session_id}")

            out_dir = Path(out_dir) / session_id
            if not out_dir.exists():
                out_dir.mkdir(parents=True, exist_ok=True)

            rolling_summary = cast(str | None, conv.rolling_summary) or ""

            # 1) session.json
            session_meta = {
                "schema_version": "1.0.0",
                "session_id": conv.id,
                "created_at": conv.created_at.replace(tzinfo=timezone.utc).isoformat(),
                "turn_count": len(conv.turns),
                "rolling_summary": rolling_summary,
                "models": {
                    "embed_model_id": self.embed_model_id,
                    "rerank_model_id": self.rerank_model_id,
                    "gen_model_id": self.gen_model_id,
                },
                "retrieval": {
                    "similarity_top_k": self.retrieve_similarity_top_k,
                    "top_n": self.rerank_top_n,
                },
                "vector_store": {
                    "type": "qdrant",
                    "url": self.qdrant_host,
                    "collection": self.qdrant_collection,
                    "host_dir": str(self.qdrant_host_dir or ""),
                },
            }
            (out_dir / "session.json").write_text(
                json.dumps(session_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # 2) messages.jsonl
            with (out_dir / "messages.jsonl").open("w", encoding="utf-8") as f:
                for t in conv.turns:
                    obj = {
                        "turn_idx": t.idx,
                        "created_at": t.created_at.replace(
                            tzinfo=timezone.utc
                        ).isoformat(),
                        "user_text": t.user_text,
                        "rewritten_query": t.rewritten_query,
                        "assistant_text": t.model_response,
                        "reasoning": t.reasoning,
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # 3) citations.parquet (optional if pandas/pyarrow present)
            try:
                rows: list[dict[str, Any]] = []
                for t in conv.turns:
                    for c in t.citations:
                        rows.append(
                            {
                                "turn_idx": t.idx,
                                "node_id": c.node_id,
                                "score": c.score,
                                "filename": c.filename,
                                "filetype": c.filetype,
                                "source": c.source,
                                "page": c.page,
                                "row": c.row,
                            }
                        )
                if rows:
                    df = pd.DataFrame(rows)
                    df.to_parquet(out_dir / "citations.parquet", index=False)
                else:
                    # write empty parquet to preserve schema
                    empty_columns: list[str] = [
                        "turn_idx",
                        "node_id",
                        "score",
                        "filename",
                        "filetype",
                        "source",
                        "page",
                        "row",
                    ]
                    empty_source: dict[str, list[Any]] = {
                        col: [] for col in empty_columns
                    }
                    pd.DataFrame(empty_source).to_parquet(
                        out_dir / "citations.parquet", index=False
                    )
            except Exception as e:
                logger.warning(
                    "Skipping citations.parquet export (pandas/pyarrow not available?): {}",
                    e,
                )

            # 4) transcript.md (human-readable)
            conv_id = str(conv.id)
            lines = ["# Transcript", f"Session: `{conv_id}`", ""]
            if rolling_summary:
                lines += ["## Rolling Summary", "", rolling_summary, ""]
            for t in conv.turns:
                lines += [
                    f"## Turn {t.idx}",
                    f"**User**: {t.user_text}",
                    f"**Rewritten**: {t.rewritten_query or ''}",
                    f"**Assistant**: {t.model_response}",
                ]
                if t.reasoning:
                    lines += [
                        f"<details><summary>Reasoning</summary>\n\n{t.reasoning}\n\n</details>"
                    ]
                # citations with embedded source text
                if t.citations:
                    lines += ["**Citations (with source excerpts):**"]
                    for c in t.citations:
                        loc = (
                            f"page {c.page}"
                            if c.page is not None
                            else (f"row {c.row}" if c.row is not None else "")
                        )
                        header = f"- {c.filename} {loc} (score={c.score})"
                        excerpt = None
                        if c.node_id:
                            excerpt = self._get_node_text_by_id(c.node_id)
                        if excerpt is None:
                            excerpt = "[source text unavailable]"
                        else:
                            excerpt = excerpt.strip()
                            max_chars = 800
                            if len(excerpt) > max_chars:
                                excerpt = excerpt[:max_chars].rstrip() + " …"
                        lines += [
                            header,
                            ">\n> " + "\n> ".join(excerpt.splitlines()) + "\n>",
                        ]
                lines += [""]
            (out_dir / "transcript.md").write_text("\n".join(lines), encoding="utf-8")

            # 5) manifest.json with checksums
            def sha256_file(p: Path) -> str:
                h = hashlib.sha256()
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        h.update(chunk)
                return h.hexdigest()

            manifest = {}
            for name in ["session.json", "messages.jsonl", "transcript.md"]:
                fp = out_dir / name
                if fp.exists():
                    manifest[name] = {
                        "sha256": sha256_file(fp),
                        "bytes": fp.stat().st_size,
                    }
            parquet_fp = out_dir / "citations.parquet"
            if parquet_fp.exists():
                manifest["citations.parquet"] = {
                    "sha256": sha256_file(parquet_fp),
                    "bytes": parquet_fp.stat().st_size,
                }

            (out_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            return out_dir

    def summarize_collection(self, prompt: str | None = None) -> dict[str, Any]:
        """Generate a summary of the currently selected collection.

        Args:
            prompt (str | None): Optional override for the summarization prompt.

        Returns:
            dict[str, Any]: Normalized response data containing the summary and sources.

        Raises:
            ValueError: If no collection is selected.
        """

        if not self.qdrant_collection:
            raise ValueError("No collection selected.")

        if self.query_engine is None:
            if self.index is None:
                self.create_index()
            self.create_query_engine()
        engine = self.query_engine
        if engine is None:
            raise RuntimeError("Query engine failed to initialize for summarization.")

        summary_prompt = prompt or (
            "Provide a concise overview of the active collection. Highlight the main "
            "topics, document types, and notable findings. Limit the response to 8 "
            "sentences."
        )

        resp = engine.query(summary_prompt)
        return self._normalize_response_data(summary_prompt, resp)

    def start_session(self, session_id: str | None = None) -> str:
        """
        Start a new chat session or continue an existing one.

        Args:
            session_id (str | None, optional): The ID of the session to continue. Defaults to None.

        Returns:
            str: The ID of the session.

        Raises:
            RuntimeError: If the query engine has not been initialized.
        """
        # 1) ensure a valid session id
        if not session_id:
            session_id = str(uuid.uuid4())
        self.session_id = session_id

        # 2) ensure a Conversation row exists (idempotent)
        with self._session_scope() as s:
            self._load_or_create_convo(s, session_id)

        # 3) seed memory from rolling summary
        rolling = self._get_rolling_summary(session_id)
        self.chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=2000, chat_history=[]
        )
        if rolling:
            if self.chat_memory is not None:
                self.chat_memory.put(
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=f"Conversation summary so far:\n{rolling}",
                    )
                )

        engine = self.query_engine
        if engine is None:
            logger.error("RuntimeError: Query engine has not been initialized.")
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs() first."
            )

        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=engine,  # reuse retriever + reranker
            memory=self.chat_memory,
            llm=self.gen_model,
        )
        return session_id

    def chat(self, user_msg: str) -> dict[str, Any]:
        """
        Run one conversational turn, persist it, and return your normalized payload.

        Args:
            user_msg (str): The user message to process.

        Returns:
            str: The normalized response from the chat engine.

        Raises:
            ValueError: If the user message is empty or the session ID is invalid.
            RuntimeError: If the query engine has not been initialized.
        """
        if not user_msg.strip():
            logger.error("ValueError: Chat prompt cannot be empty.")
            raise ValueError("Chat prompt cannot be empty.")
        engine = self.query_engine
        if engine is None:
            logger.error("RuntimeError: Query engine has not been initialized.")
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs() first."
            )

        # Ensure we have a session and conversation storage
        session_id = self.session_id
        if self.chat_engine is None or getattr(self, "_session_id", None) != session_id:
            session_id = self.start_session(session_id)

        # Build a retrieval query that includes the rolling conversation summary
        if session_id is None:
            logger.error("ValueError: Session ID cannot be None.")
            raise ValueError("Session ID cannot be None.")
        summary = self._get_rolling_summary(session_id)
        if summary:
            retrieval_query = f"{summary}\n\nUser question: {user_msg}"
        else:
            retrieval_query = user_msg

        resp = engine.query(retrieval_query)
        response = self._normalize_response_data(user_msg, resp)
        self._persist_turn(session_id, user_msg, resp, response)
        self._maybe_update_summary(session_id)
        return response
