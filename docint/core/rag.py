from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from llama_index.core import (
    Response,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import BaseNode, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from docint.core.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.session_manager import SessionManager
from docint.utils.env_cfg import (
    load_host_env,
    load_ie_env,
    load_model_env,
    load_path_env,
)
from docint.utils.clean_text import basic_clean
from docint.utils.model_cfg import resolve_model_path
from docint.utils.ollama_cfg import OllamaPipeline
from docint.utils.storage import stage_sources_to_qdrant

# --- Environment variables ---
load_dotenv()

OLLAMA_THINKING: bool = os.getenv("OLLAMA_THINKING", "true").lower() == "true"
RETRIEVE_SIMILARITY_TOP_K: int = int(os.getenv("RETRIEVE_SIMILARITY_TOP_K", "20"))

CleanFn = Callable[[str], str]
NER_PROMPT = OllamaPipeline().load_prompt(kw="ner")


@dataclass
class RAG:
    """
    Represents a Retrieval-Augmented Generation (RAG) model.
    """

    # --- Data path & cleaning setup ---
    data_dir: Path | None = field(default=None, init=False)
    clean_fn: CleanFn = basic_clean

    # --- Path setup ---
    hf_hub_cache: Path | None = field(default=None, init=False)
    xdg_cache_home: Path | None = field(default=None, init=False)

    # --- Models ---
    embed_model_id: str | None = field(default=None)
    sparse_model_id: str | None = field(default=None)
    gen_model_id: str | None = field(default=None)

    # --- Qdrant controls ---
    qdrant_host: str | None = field(default=None, init=False)
    _qdrant_col_dir: Path | None = field(default=None, init=False, repr=False)
    _qdrant_src_dir: Path | None = field(default=None, init=False, repr=False)
    qdrant_collection: str = "default"

    # --- Ollama parameters ---
    base_url: str | None = field(default=None, init=False)
    context_window: int = -1
    temperature: float = 0.2
    request_timeout: int = 1200
    thinking: bool = OLLAMA_THINKING
    ollama_options: dict[str, Any] | None = None

    # --- Information extraction ---
    enable_ie: bool = False
    ie_max_chars: int = 800
    ner_prompt: str = NER_PROMPT

    # --- Reranking / retrieval ---
    enable_hybrid: bool = True
    embed_batch_size: int = 64
    retrieve_similarity_top_k: int = RETRIEVE_SIMILARITY_TOP_K
    rerank_top_n: int = int(retrieve_similarity_top_k // 5)

    # --- Prompt config ---
    prompt_dir: Path | None = field(default=None, init=False)
    summarize_prompt_path: Path | None = field(default=None, init=False)
    summarize_prompt: str = field(default="", init=False)

    # --- Directory reader config ---
    reader_errors: str = "ignore"
    reader_recursive: bool = True
    reader_encoding: str = "utf-8"
    reader_required_exts: list[str] = field(default_factory=list, init=False)
    reader_required_exts_path: Path | None = field(default=None, init=False)

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
    _reranker: LLMRerank | None = field(default=None, init=False, repr=False)
    _qdrant_client: QdrantClient | None = field(default=None, init=False, repr=False)
    _qdrant_aclient: AsyncQdrantClient | None = field(
        default=None, init=False, repr=False
    )

    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    sentence_splitter: SentenceSplitter = field(
        default_factory=SentenceSplitter, init=False
    )

    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)

    index: VectorStoreIndex | None = field(default=None, init=False)
    query_engine: RetrieverQueryEngine | None = field(default=None, init=False)
    sessions: SessionManager | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to set up any necessary components.

        Raises:
            ValueError: If summarize_prompt_path is not set.
        """
        # --- Host config ---
        host_config = load_host_env()
        self.base_url = host_config.ollama_host
        self.qdrant_host = host_config.qdrant_host

        # --- Path config ---
        path_config = load_path_env()
        self.data_dir = path_config.data
        self.prompt_dir = path_config.prompts
        self._qdrant_col_dir = path_config.qdrant_collections
        self._qdrant_src_dir = path_config.qdrant_sources
        self.reader_required_exts_path = path_config.required_exts
        self.hf_hub_cache = path_config.hf_hub_cache
        self.xdg_cache_home = path_config.xdg_cache_home

        # --- Model config ---
        model_config = load_model_env()
        self.embed_model_id = model_config.embed_model
        self.sparse_model_id = model_config.sparse_model
        self.gen_model_id = model_config.gen_model
        self.context_window = model_config.ollama_ctx_window

        ie_config = load_ie_env()
        self.enable_ie = ie_config.enabled
        self.ie_max_chars = ie_config.max_chars

        with open(self.reader_required_exts_path, "r", encoding="utf-8") as f:
            self.reader_required_exts = [f".{line.strip()}" for line in f]

        if self.prompt_dir:
            self.summarize_prompt_path = self.prompt_dir / "summarize.txt"

        if self.summarize_prompt_path is None:
            logger.error(
                "ValueError: summarize_prompt_path is not set. Cannot load summarize prompt."
            )
            raise ValueError(
                "summarize_prompt_path is not set. Cannot load summarize prompt."
            )
        with open(self.summarize_prompt_path, "r", encoding="utf-8") as f:
            self.summarize_prompt = f.read()

        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.sessions = SessionManager(self)

    @property
    def session_id(self) -> str | None:
        """
        Get the current session ID.

        Returns:
            str | None: The current session ID.
        """
        return self.sessions.session_id if self.sessions else None

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        """
        Set the current session ID.

        Args:
            value (str | None): The new session ID.
        """
        if self.sessions is not None:
            self.sessions.session_id = value

    @property
    def chat_engine(self) -> Any | None:
        """
        Get the current chat engine.

        Returns:
            Any | None: The current chat engine.
        """
        return self.sessions.chat_engine if self.sessions else None

    @chat_engine.setter
    def chat_engine(self, value: Any | None) -> None:
        """
        Set the current chat engine.

        Args:
            value (Any | None): The new chat engine.
        """
        if self.sessions is not None:
            self.sessions.chat_engine = value

    @property
    def chat_memory(self) -> Any | None:
        """
        Get the current chat memory.

        Returns:
            Any | None: The current chat memory.
        """
        return self.sessions.chat_memory if self.sessions else None

    @chat_memory.setter
    def chat_memory(self, value: Any | None) -> None:
        """
        Set the current chat memory.

        Args:
            value (Any | None): The new chat memory.
        """
        if self.sessions is not None:
            self.sessions.chat_memory = value

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
    def qdrant_col_dir(self) -> Path:
        """
        Best-effort resolution of the host directory where Qdrant stores data.
        Used only as a *fallback* when we cannot reach the Qdrant API.
        Priority: explicit field -> env var -> platform default under home.

        Returns:
            The Path representing the Qdrant host directory.

        Raises:
            ValueError: If the Qdrant host directory is not set.
        """
        if self._qdrant_col_dir is None:
            env = load_path_env().qdrant_collections
            if env:
                self._qdrant_col_dir = Path(env) if not env.is_absolute() else env
            else:
                home = os.getenv("HOME") or os.getenv("USERPROFILE")
                if home:
                    self._qdrant_col_dir = (
                        Path(home) / ".qdrant" / "storage" / "collections"
                    )
        if self._qdrant_col_dir is None:
            logger.error("ValueError: Qdrant host directory is not set.")
            raise ValueError("Qdrant host directory is not set.")
        return self._qdrant_col_dir

    @property
    def qdrant_src_dir(self) -> Path:
        """
        Best-effort resolution of the host directory where Qdrant stores source data.
        Used only as a *fallback* when we cannot reach the Qdrant API.
        Priority: explicit field -> env var -> platform default under home.

        Returns:
            The Path representing the Qdrant source host directory.

        Raises:
            ValueError: If the Qdrant source host directory is not set.
        """
        if self._qdrant_src_dir is None:
            env = load_path_env().qdrant_sources
            if env:
                self._qdrant_src_dir = Path(env) if not env.is_absolute() else env
            else:
                home = os.getenv("HOME") or os.getenv("USERPROFILE")
                if home:
                    self._qdrant_src_dir = (
                        Path(home) / ".qdrant" / "storage" / "sources"
                    )
        if self._qdrant_src_dir is None:
            logger.error("ValueError: Qdrant source host directory is not set.")
            raise ValueError("Qdrant source host directory is not set.")
        return self._qdrant_src_dir

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

        Raises:
            ValueError: If embed_model_id is None.
        """
        if self._embed_model is None:
            if self.embed_model_id is None:
                raise ValueError("embed_model_id cannot be None")
            resolved_model = resolve_model_path(
                self.embed_model_id, self.hf_hub_cache or Path()
            )
            if resolved_model != self.embed_model_id:
                logger.info("Using local model path: {}", resolved_model)

            try:
                model = HuggingFaceEmbedding(
                    model_name=resolved_model,
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
                    self._device = "cpu"
                    self._embed_model = HuggingFaceEmbedding(
                        model_name=resolved_model,
                        normalize=True,
                        device="cpu",
                    )
                    # Trigger warmup to ensure CPU fallback is working
                    self._embed_model.get_text_embedding("warmup")
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
            ValueError: If the sparse model is None or not supported.
            ImportError: If fastembed is not installed when hybrid search is enabled.
        """
        if not self.enable_hybrid:
            return None

        if self.sparse_model_id is None:
            raise ValueError("sparse_model_id is None")

        try:
            supported_models = SparseTextEmbedding.list_supported_models()
        except ImportError:
            raise ImportError(
                "fastembed is not installed, but hybrid search is enabled."
            )

        # Check if the configured ID is directly supported
        supported_ids = [m["model"] for m in supported_models]
        if self.sparse_model_id in supported_ids:
            return self.sparse_model_id

        # Check if it matches a source HF repo (mapping logic)
        for model_desc in supported_models:
            sources = model_desc.get("sources")
            if sources and sources.get("hf") == self.sparse_model_id:
                logger.info(
                    "Mapped sparse model {} to its source {}",
                    self.sparse_model_id,
                    model_desc["model"],
                )
                return model_desc["model"]

        logger.error(
            "ValueError: Sparse model {} not supported. Supported: {}",
            self.sparse_model_id,
            supported_ids,
        )
        raise ValueError(
            f"Sparse model {self.sparse_model_id!r} not supported. "
            f"Supported: {supported_ids}"
        )

    @property
    def gen_model(self) -> Ollama:
        """
        Lazily initializes and returns the generation model (Ollama).

        Returns:
            Ollama: The initialized generation model.

        Raises:
            ValueError: If gen_model_id or base_url is None.
        """
        if self._gen_model is None:
            if self.gen_model_id is None:
                raise ValueError("gen_model_id cannot be None")

            if self.base_url is None:
                raise ValueError("base_url cannot be None for Ollama model")

            # Ensure base_url is clean (no trailing slash)
            base_url = self.base_url.rstrip("/")

            self._gen_model = Ollama(
                model=self.gen_model_id,
                base_url=base_url,
                temperature=self.temperature,
                context_window=self.context_window,
                request_timeout=self.request_timeout,
                thinking=self.thinking,
                additional_kwargs=self.ollama_options,
            )
            logger.info("Initializing generator model: {}", self.gen_model_id)
        return self._gen_model

    @property
    def reranker(self) -> LLMRerank:
        """
        Lazily initializes and returns the reranker model (LLMRerank).

        Returns:
            LLMRerank: The initialized reranker model.
        """
        if self._reranker is None:
            self._reranker = LLMRerank(
                top_n=self.rerank_top_n,
                llm=self.gen_model,
            )
            logger.info("Initializing LLM reranker with model: {}", self.gen_model_id)
        return self._reranker

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

    # --- Information extraction helpers ---
    @staticmethod
    def _parse_ie_payload(raw: str) -> dict[str, Any]:
        """
        Parses the raw text response from the IE model into a JSON payload.

        Args:
            raw (str): Raw text response from the IE model.

        Returns:
            dict[str, Any]: Parsed JSON payload.
        """
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
        except Exception:
            return {}
        return {}

    def _ie_extract(self, text: str) -> tuple[list[dict], list[dict]]:
        """
        Performs information extraction on the given text.

        Args:
            text (str): Text to perform information extraction on.

        Returns:
            tuple[list[dict], list[dict]]: Extracted entities and relations.
        """
        snippet = text[: self.ie_max_chars]
        prompt = self.ner_prompt.format(text=snippet)

        try:
            resp = self.gen_model.complete(prompt)
            raw = resp.text if hasattr(resp, "text") else str(resp)
        except Exception as exc:  # pragma: no cover - model failures are runtime
            logger.warning("IE extraction request failed: {}", exc)
            return [], []

        payload = self._parse_ie_payload(raw) if isinstance(raw, str) else {}
        entities_raw = payload.get("entities") if isinstance(payload, dict) else []
        relations_raw = payload.get("relations") if isinstance(payload, dict) else []

        entities: list[dict] = []
        for ent in entities_raw or []:
            if not isinstance(ent, dict):
                continue
            text_val = str(ent.get("text") or ent.get("name") or "").strip()
            if not text_val:
                continue
            entities.append(
                {
                    "text": text_val,
                    "type": ent.get("type") or ent.get("label"),
                    "score": ent.get("score"),
                }
            )

        relations: list[dict] = []
        for rel in relations_raw or []:
            if not isinstance(rel, dict):
                continue
            head = str(rel.get("head") or rel.get("subject") or "").strip()
            tail = str(rel.get("tail") or rel.get("object") or "").strip()
            if not head or not tail:
                continue
            relations.append(
                {
                    "head": head,
                    "tail": tail,
                    "label": rel.get("label") or rel.get("type"),
                    "score": rel.get("score"),
                }
            )

        return entities, relations

    def _build_ingestion_pipeline(
        self, progress_callback: Callable[[str], None] | None = None
    ) -> DocumentIngestionPipeline:
        """
        Instantiate a document ingestion pipeline using current settings.

        Args:
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.

        Returns:
            DocumentIngestionPipeline: The instantiated ingestion pipeline.

        Raises:
            ValueError: If data_dir is None.
        """
        if self.data_dir is None:
            logger.error("ValueError: data_dir cannot be None for ingestion pipeline.")
            raise ValueError("data_dir cannot be None for ingestion pipeline.")

        return DocumentIngestionPipeline(
            data_dir=self.data_dir,
            clean_fn=self.clean_fn,
            sentence_splitter=self.sentence_splitter,
            embed_model_factory=lambda: self.embed_model,
            device=self.device,
            reader_errors=self.reader_errors,
            reader_recursive=self.reader_recursive,
            reader_encoding=self.reader_encoding,
            reader_required_exts=list(self.reader_required_exts),
            table_text_cols=self.table_text_cols,
            table_metadata_cols=self.table_metadata_cols,
            table_id_col=self.table_id_col,
            table_excel_sheet=self.table_excel_sheet,
            table_row_limit=self.table_row_limit,
            table_row_filter=self.table_row_filter,
            buffer_size=self.buffer_size,
            breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            semantic_splitter_char_limit=self.semantic_splitter_char_limit,
            entity_extractor=self._ie_extract if self.enable_ie else None,
            progress_callback=progress_callback,
        )

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
                        "Qdrant collection '{}' not found; skipping existing-hash check: {}",
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

    def create_index(self) -> None:
        """
        Materialize a VectorStoreIndex.
        If nodes are present in memory, create from nodes.
        Otherwise, load from vector store.
        """
        vector_store = self._vector_store()
        storage_ctx = self._storage_context(vector_store)

        if self.nodes:
            self.index = self._index(storage_ctx)
        else:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=self.embed_model
            )

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

        Args:
            query (str): The original query string.
            result (Any): The response object from the query engine.
            reason (str | None): Optional reasoning string.

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
        m = re.search(
            r"<think>(.*?)</think>", resp_text, flags=re.DOTALL | re.IGNORECASE
        )
        if m:
            reason = (m.group(1).strip() if m else None) or reason
            resp_text = re.sub(
                r"<think>.*?</think>", "", resp_text, flags=re.DOTALL | re.IGNORECASE
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
            )
            filetype = (
                origin.get("filetype")
                or origin.get("mimetype")
                or meta.get("filetype")
                or meta.get("mimetype")
                or meta.get("file_type")
                or meta.get("file_format")
            )
            source_kind = (
                meta.get("source") or meta.get("source_type") or meta.get("reader")
            )
            file_hash = (
                origin.get("file_hash")
                or meta.get("file_hash")
                or self._extract_file_hash(meta)
            )

            # page detection
            page = (
                meta.get("page")
                or meta.get("page_number")
                or origin.get("page")
                or origin.get("page_number")
            )
            provenance = meta.get("provenance") or meta.get("provenances") or []
            if page is None and isinstance(provenance, list):
                for prov in provenance:
                    if isinstance(prov, dict):
                        page = prov.get("page_no")
                        if page is not None:
                            break

            # doc_items detection (Docling)
            if page is None:
                doc_items = meta.get("doc_items")
                if isinstance(doc_items, list):
                    for item in doc_items:
                        if isinstance(item, dict):
                            provs = item.get("prov")
                            if isinstance(provs, list):
                                for p in provs:
                                    if isinstance(p, dict):
                                        page = p.get("page_no")
                                        if page is not None:
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
                "score": getattr(nws, "score", None),
            }
            entities = meta.get("entities") or origin.get("entities")
            relations = meta.get("relations") or origin.get("relations")
            if entities:
                src["entities"] = entities
            if relations:
                src["relations"] = relations
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
        base = self.qdrant_col_dir
        if base is None:
            return []
        collections_dir = base
        try:
            if not collections_dir.exists():
                return []
            return sorted([p.name for p in collections_dir.iterdir() if p.is_dir()])
        except Exception as e:
            logger.warning("FS fallback list_collections failed: {}", e)
            return []

    def select_collection(self, name: str) -> None:
        """
        Switch active collection, ensuring it already exists.

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
        self.reset_session_state()

    def _prepare_sources_dir(self, data_dir: Path) -> Path:
        """
        Ensure source files live under qdrant_src_dir/<collection> for preview and persistence.

        If the provided data_dir is already under that path, it is returned as-is.
        Otherwise, files/directories are copied into the target.

        Args:
            data_dir (Path): The original data directory.

        Returns:
            Path: The path to the staged sources directory.
        """
        if not self.qdrant_collection:
            return data_dir
        return stage_sources_to_qdrant(
            data_dir, self.qdrant_collection, self.qdrant_src_dir
        )

    # --- Public API ---
    def ingest_docs(
        self,
        data_dir: str | Path,
        *,
        build_query_engine: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """
        Ingest documents from the specified directory into the Qdrant collection.

        Args:
            data_dir (str | Path): The directory containing the documents to ingest.
            build_query_engine (bool): Whether to eagerly build the query engine after
                ingestion. Disable when running headless ingestion jobs to avoid
                loading large reranker/generation models. Defaults to True.
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.
        """
        prepared_dir = self._prepare_sources_dir(
            Path(data_dir) if isinstance(data_dir, str) else data_dir
        )
        self.data_dir = prepared_dir
        pipeline = self._build_ingestion_pipeline(progress_callback=progress_callback)

        # Initialize index (load existing or create new wrapper)
        vector_store = self._vector_store()
        # We use from_vector_store to attach to the store.
        # If the store is empty, it will be populated as we insert nodes.
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=self.embed_model
        )

        # Process batches from the pipeline generator
        for docs, nodes in pipeline.build(self._get_existing_file_hashes()):
            if nodes:
                self.index.insert_nodes(nodes)

        self.dir_reader = pipeline.dir_reader
        # Clear memory-heavy lists as we've persisted them to the vector store
        self.docs = []
        self.nodes = []

        if build_query_engine:
            self.create_query_engine()
        else:
            # Ensure downstream callers recreate a fresh query engine as needed.
            self.query_engine = None

        self.reset_session_state()

        eff_k = None
        if self.query_engine is not None and hasattr(self.query_engine, "retriever"):
            try:
                eff_k = getattr(self.query_engine.retriever, "similarity_top_k", None)
            except Exception:
                eff_k = None

        if self.query_engine is not None:
            logger.info(
                "Effective retrieval k={} | top_n={} | embed_device={} | rerank_device={}",
                eff_k,
                self.rerank_top_n,
                self.device,
                self.device,
            )
        logger.info("Documents ingested successfully.")

    async def asingest_docs(
        self,
        data_dir: str | Path,
        *,
        build_query_engine: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """
        Asynchronously ingest documents from the specified directory into the Qdrant collection.

        Args:
            data_dir (str | Path): The directory containing the documents to ingest.
            build_query_engine (bool): Whether to build the query engine immediately
                after ingestion. Defaults to True.
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.

        Raises:
            RuntimeError: If the index is not initialized for async ingestion.
        """
        prepared_dir = self._prepare_sources_dir(
            Path(data_dir) if isinstance(data_dir, str) else data_dir
        )
        self.data_dir = prepared_dir
        pipeline = self._build_ingestion_pipeline(progress_callback=progress_callback)

        # Initialize index
        vector_store = self._vector_store()
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=self.embed_model
        )

        # Process batches
        for docs, nodes in pipeline.build(self._get_existing_file_hashes()):
            if nodes:
                await self.index.ainsert_nodes(nodes)

        self.dir_reader = pipeline.dir_reader
        self.docs = []
        self.nodes = []

        if build_query_engine:
            if self.query_engine is None:
                self.create_query_engine()
        else:
            self.query_engine = None

        self.reset_session_state()

        eff_k = None
        if self.query_engine is not None and hasattr(self.query_engine, "retriever"):
            try:
                eff_k = getattr(self.query_engine.retriever, "similarity_top_k", None)
            except Exception:
                eff_k = None

        if self.query_engine is not None:
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

    # --- Session integration ---
    def init_session_store(self, db_url: str = "sqlite:///rag_sessions.db") -> None:
        """
        Initialize the relational session store via SessionManager.

        Args:
            db_url (str): The database URL for the session store.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        self.sessions.init_session_store(db_url)

    def reset_session_state(self) -> None:
        """
        Clear cached chat state so future sessions start fresh.
        """
        if self.sessions is not None:
            self.sessions.reset_runtime()

    def export_session(
        self, session_id: str | None = None, out_dir: str | Path = "session"
    ) -> Path:
        """
        Delegate session export to SessionManager.

        Args:
            session_id (str | None): The session ID to export. If None, exports the
                current session.
            out_dir (str | Path): The output directory for the exported session.

        Returns:
            Path: The path to the exported session file.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions.export_session(session_id=session_id, out_dir=out_dir)

    def start_session(self, session_id: str | None = None) -> str:
        """
        Start or resume a chat session through SessionManager.

        Args:
            session_id (str | None): The session ID to start or resume. If None,
                a new session is created.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions.start_session(session_id)

    def chat(self, user_msg: str) -> dict[str, Any]:
        """
        Proxy chat turns to SessionManager.

        Args:
            user_msg (str): The user's chat message.

        Returns:
            dict[str, Any]: The chat response data.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions.chat(user_msg)

    def stream_chat(self, user_msg: str) -> Any:
        """
        Proxy stream chat turns to SessionManager.

        Args:
            user_msg (str): The user's chat message.

        Returns:
            Any: A generator yielding response chunks.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions.stream_chat(user_msg)

    def summarize_collection(self) -> dict[str, Any]:
        """
        Generate a summary of the currently selected collection.

        Returns:
            dict[str, Any]: Normalized response data containing the summary and sources.

        Raises:
            ValueError: If no collection is selected.
            RuntimeError: If the query engine is not initialized for summarization.
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

        resp = engine.query(self.summarize_prompt)
        return self._normalize_response_data(self.summarize_prompt, resp)

    def stream_summarize_collection(self) -> Any:
        """
        Generate a streaming summary of the currently selected collection.

        Yields:
            str | dict: Chunks of text, followed by a dict with metadata.

        Raises:
            ValueError: If no collection is selected.
            RuntimeError: If the index is not initialized for streaming summarization.
        """
        if not self.qdrant_collection:
            raise ValueError("No collection selected.")

        if self.index is None:
            self.create_index()
        if self.index is None:
            raise RuntimeError(
                "Index failed to initialize for streaming summarization."
            )

        # Create a temporary streaming engine for summarization
        k = min(max(self.retrieve_similarity_top_k, self.rerank_top_n * 8), 64)
        retriever = self.index.as_retriever(similarity_top_k=k)
        streaming_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=self.gen_model,
            node_postprocessors=[self.reranker],
            streaming=True,
        )

        response = streaming_engine.query(self.summarize_prompt)

        full_text = ""
        tokens = getattr(response, "response_gen", None)
        if tokens is not None:
            for token in tokens:
                full_text += token
                yield token
        else:
            resp_value = getattr(response, "response", None)
            if isinstance(resp_value, str):
                full_text = resp_value

        # Create a Response object to reuse normalization logic
        final_response = Response(
            response=full_text,
            source_nodes=getattr(response, "source_nodes", []) or [],
            metadata=getattr(response, "metadata", {}) or {},
        )
        normalized = self._normalize_response_data(
            self.summarize_prompt, final_response
        )
        yield normalized

    def get_collection_ie(self) -> list[dict[str, Any]]:
        """
        Fetch all nodes from the current collection and return their IE metadata.

        Returns:
            list[dict[str, Any]]: A list of source metadata dictionaries containing IE data.
        """
        if not self.qdrant_collection:
            return []

        sources = []
        offset = None
        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.error(
                    "Failed to scroll collection '{}': {}", self.qdrant_collection, exc
                )
                break

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", None)
                if isinstance(payload, dict):
                    # We only care about nodes that have IE data
                    if "entities" in payload or "relations" in payload:
                        # Normalize payload to match what _normalize_response_data produces
                        # so that _render_ie_overview in app.py can handle it.

                        # Extract filename/filetype/etc.
                        origin = payload.get("origin") or {}
                        filename = (
                            origin.get("filename")
                            or payload.get("file_name")
                            or payload.get("filename")
                            or payload.get("file_path")
                        )

                        # Extract page number
                        page = (
                            payload.get("page")
                            or payload.get("page_number")
                            or origin.get("page_no")
                        )
                        if page is None:
                            doc_items = payload.get("doc_items")
                            if isinstance(doc_items, list):
                                for item in doc_items:
                                    if isinstance(item, dict):
                                        provs = item.get("prov")
                                        if isinstance(provs, list):
                                            for p in provs:
                                                if isinstance(p, dict):
                                                    page = p.get("page_no")
                                                    if page is not None:
                                                        break
                                    if page is not None:
                                        break

                        sources.append(
                            {
                                "filename": filename,
                                "entities": payload.get("entities", []),
                                "relations": payload.get("relations", []),
                                "page": page,
                                "row": payload.get("table", {}).get("row_index"),
                            }
                        )

            if offset is None:
                break

        return sources

    def unload_models(self) -> None:
        """
        Unload models to free up memory.
        """
        self._embed_model = None
        self._gen_model = None
        self._reranker = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Models unloaded and memory cleared.")
