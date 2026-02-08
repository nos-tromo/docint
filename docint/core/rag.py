from __future__ import annotations

import gc
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from fastembed import SparseTextEmbedding
from llama_index.core import (
    Response,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from docint.core.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.session_manager import SessionManager
from docint.core.storage.docstore import QdrantKVStore
from docint.core.storage.sources import stage_sources_to_qdrant
from docint.utils.env_cfg import (
    load_host_env,
    load_ie_env,
    load_model_env,
    load_llama_cpp_env,
    load_path_env,
    load_rag_env,
    load_session_env,
)
from docint.utils.model_cfg import resolve_model_path


@dataclass(slots=True)
class RAG:
    """
    Represents a Retrieval-Augmented Generation (RAG) model. Handles configuration,
    initialization, and interaction with underlying components like embedding models,
    generation models, and vector stores. Provides methods to start sessions,
    retrieve information, and manage document ingestion.
    """

    # --- Constructor args ---
    qdrant_collection: str
    enable_hybrid: bool = field(default=True)

    # --- Path setup ---
    data_dir: Path | None = field(default=None, init=False)
    hf_hub_cache: Path | None = field(default=None, init=False)

    # --- Session config ---
    session_store: str = field(default="", init=False)

    # --- Models ---
    embed_model_id: str | None = field(default=None, init=False)
    sparse_model_id: str | None = field(default=None, init=False)
    gen_model_id: str | None = field(default=None, init=False)

    # --- Qdrant controls ---
    docstore_batch_size: int = field(default=100, init=False)
    qdrant_host: str | None = field(default=None, init=False)
    _qdrant_col_dir: Path | None = field(default=None, init=False, repr=False)
    _qdrant_src_dir: Path | None = field(default=None, init=False, repr=False)

    # --- Llama.cpp parameters ---
    llama_cpp_ctx_window: int | None = field(default=None, init=False)
    llama_cpp_request_timeout: int | None = field(default=None, init=False)
    llama_cpp_seed: int | None = field(default=None, init=False)
    llama_cpp_temperature: float | None = field(default=None, init=False)
    llama_cpp_n_gpu_layers: int = field(default=-1, init=False)
    llama_cpp_top_k: int | None = field(default=None, init=False)
    llama_cpp_top_p: float | None = field(default=None, init=False)
    llama_cpp_repeat_penalty: float | None = field(default=None, init=False)
    llama_cpp_options: dict[str, Any] | None = field(default=None, init=False)

    # --- Information extraction ---
    ie_enabled: bool = field(default=False, init=False)
    ie_sources: list[dict[str, Any]] = field(default_factory=list, init=False)

    # --- Reranking / retrieval ---
    retrieve_similarity_top_k: int = field(default=20, init=False)
    rerank_top_n: int = field(default=5, init=False)

    # --- Prompt config ---
    prompt_dir: Path | None = field(default=None, init=False)
    summarize_prompt_path: Path | None = field(default=None, init=False)
    summarize_prompt: str = field(default="", init=False)

    # --- Runtime (lazy caches / not in repr) ---
    _device: str | None = field(default=None, init=False, repr=False)
    _embed_model: BaseEmbedding | None = field(default=None, init=False, repr=False)
    _gen_model: LlamaCPP | None = field(default=None, init=False, repr=False)
    _reranker: LLMRerank | None = field(default=None, init=False, repr=False)
    _qdrant_client: QdrantClient | None = field(default=None, init=False, repr=False)
    _qdrant_aclient: AsyncQdrantClient | None = field(
        default=None, init=False, repr=False
    )

    # -- Ingested data ---
    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)

    # --- Built components (lazy loaded) ---
    index: VectorStoreIndex | None = field(default=None, init=False)
    query_engine: RetrieverQueryEngine | None = field(default=None, init=False)
    sessions: SessionManager | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to set up any necessary components.

        Raises:
            ValueError: If summarize_prompt_path is not set.
        """
        # --- Host config ---
        host_config = load_host_env()
        self.ollama_host = host_config.ollama_host
        self.qdrant_host = host_config.qdrant_host

        # --- Path config ---
        path_config = load_path_env()
        self.data_dir = path_config.data
        self.prompt_dir = path_config.prompts
        self._qdrant_col_dir = path_config.qdrant_collections
        self._qdrant_src_dir = path_config.qdrant_sources
        self.hf_hub_cache = path_config.hf_hub_cache

        # --- Model config ---
        model_config = load_model_env()
        self.embed_model_id = model_config.embed_model
        self.sparse_model_id = model_config.sparse_model
        self.gen_model_id = model_config.gen_model

        # --- Information Extraction config ---
        self.ie_enabled = load_ie_env().ie_enabled

        # --- Llama.cpp config ---
        llama_cpp_config = load_llama_cpp_env()
        self.llama_cpp_ctx_window = llama_cpp_config.ctx_window
        self.llama_cpp_request_timeout = llama_cpp_config.request_timeout
        self.llama_cpp_seed = llama_cpp_config.seed
        self.llama_cpp_temperature = llama_cpp_config.temperature
        self.llama_cpp_n_gpu_layers = llama_cpp_config.n_gpu_layers
        self.llama_cpp_top_k = llama_cpp_config.top_k
        self.llama_cpp_top_p = llama_cpp_config.top_p
        self.llama_cpp_repeat_penalty = llama_cpp_config.repeat_penalty
        self.llama_cpp_options = {
            "seed": self.llama_cpp_seed,
            "top_k": self.llama_cpp_top_k,
            "top_p": self.llama_cpp_top_p,
            "repeat_penalty": self.llama_cpp_repeat_penalty,
        }

        # --- RAG config ---
        rag_config = load_rag_env()
        self.docstore_batch_size = rag_config.docstore_batch_size
        self.retrieve_similarity_top_k = rag_config.retrieve_top_k
        self.rerank_top_n = int(self.retrieve_similarity_top_k // 4)

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

        # --- Session config ---
        self.session_store = load_session_env().session_store
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
        chosen = self.sparse_model_id
        if self.sparse_model_id in supported_ids:
            chosen = self.sparse_model_id
        else:
            # Check if it matches a source HF repo (mapping logic)
            for model_desc in supported_models:
                sources = model_desc.get("sources")
                if sources and sources.get("hf") == self.sparse_model_id:
                    logger.info(
                        "Mapped sparse model {} to its source {}",
                        self.sparse_model_id,
                        model_desc["model"],
                    )
                    chosen = model_desc["model"]
                    break
            else:
                logger.error(
                    "ValueError: Sparse model {} not supported. Supported: {}",
                    self.sparse_model_id,
                    supported_ids,
                )
                raise ValueError(
                    f"Sparse model {self.sparse_model_id!r} not supported. "
                    f"Supported: {supported_ids}"
                )

        cache_dir = self.hf_hub_cache or Path.home() / ".cache" / "huggingface" / "hub"
        resolved = resolve_model_path(chosen, cache_dir)
        if resolved != chosen:
            logger.info("Using local sparse model path: {}", resolved)
            return resolved

        return chosen

    def _create_gen_model(self, enable_json: bool = False) -> LlamaCPP:
        """
        Helper to create a Llama.cpp model instance with specific settings.

        Args:
            enable_json (bool): Whether to enforce JSON output mode.

        Returns:
            LlamaCPP: The initialized model.

        Raises:
            ValueError: If required configuration is missing.
        """
        if self.gen_model_id is None:
            raise ValueError("gen_model_id cannot be None")
        if self.llama_cpp_ctx_window is None:
            raise ValueError("llama_cpp_ctx_window cannot be None for Llama.cpp model")

        # Resolve model path
        from docint.utils.env_cfg import load_path_env
        model_cache = load_path_env().llama_cpp_cache
        model_path = model_cache / self.gen_model_id
        
        if not model_path.exists():
            logger.error("Model file not found: {}", model_path)
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Ensure options dict exists
        if self.llama_cpp_options is None:
            self.llama_cpp_options = {}

        # Prepare options copy to avoid side effects
        options = self.llama_cpp_options.copy()

        # Consistent seed behavior
        if self.llama_cpp_seed is not None:
            options["seed"] = self.llama_cpp_seed

        model = LlamaCPP(
            model_path=str(model_path),
            temperature=self.llama_cpp_temperature,
            max_new_tokens=2048,
            context_window=self.llama_cpp_ctx_window,
            model_kwargs={
                "n_gpu_layers": self.llama_cpp_n_gpu_layers,
            },
            generate_kwargs=options,
        )
        logger.info(
            "Initializing generator model: {} (json={})",
            self.gen_model_id,
            enable_json,
        )
        return model

    @property
    def gen_model(self) -> LlamaCPP:
        """
        Lazily initializes and returns the generation model (Llama.cpp).

        Returns:
            LlamaCPP: The initialized generation model.

        Raises:
            ValueError: If gen_model_id or llama_cpp_ctx_window is None.
        """
        if self._gen_model is None:
            self._gen_model = self._create_gen_model()
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

        Raises:
            ValueError: If qdrant_collection is None.
        """
        if self.qdrant_collection is None:
            logger.error("ValueError: qdrant_collection cannot be None")
            raise ValueError("qdrant_collection cannot be None")

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
        kv_collection = f"{self.qdrant_collection}_dockv"
        kv_store = QdrantKVStore(
            client=self.qdrant_client,
            collection_name=kv_collection,
        )
        # Use a reasonable batch size to encourage batch operations
        doc_store = KVDocumentStore(
            kvstore=kv_store, batch_size=self.docstore_batch_size
        )

        return StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=doc_store,
        )

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

        ie_model = None
        if self.ie_enabled:
            # Enforce JSON for IE tasks to ensure structured output
            ie_model = self._create_gen_model(enable_json=True)

        return DocumentIngestionPipeline(
            data_dir=self.data_dir,
            ie_model=ie_model,
            device=self.device,
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
        Materialize a VectorStoreIndex. If nodes are present in memory, create from nodes.
        Otherwise, load from vector store.
        """
        vector_store = self._vector_store()
        storage_ctx = self._storage_context(vector_store)

        if self.nodes:
            self.index = self._index(storage_ctx)
        else:
            # Build index with explicit storage_context so it uses the persistent docstore.
            self.index = VectorStoreIndex(
                nodes=[],
                embed_model=self.embed_model,
                storage_context=storage_ctx,
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

        sources: list[dict[str, Any]] = []
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

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection by name from Qdrant and clean up source files.

        Args:
            name (str): Name of the collection to delete.

        Raises:
            ValueError: If the name is empty.
        """
        if not name or not name.strip():
            raise ValueError("Collection name cannot be empty")

        # 1. Delete from Qdrant API
        try:
            self.qdrant_client.delete_collection(name)
            logger.info("Deleted collection '{}' from Qdrant.", name)
        except Exception as e:
            logger.error("Failed to delete collection '{}' via API: {}", name, e)

        # 2. Cleanup source files
        try:
            src_path = self.qdrant_src_dir / name
            if src_path.exists():

                def on_error(func: Callable, path: str, _exc_info: Any) -> None:
                    """
                    Error handler for shutil.rmtree.

                    Attempts to fix permissions/flags and retry operation.

                    Args:
                        func (Callable): The function that raised the exception.
                        path (str): The path name passed to function.
                        _exc_info (Any): The exception information returned by sys.exc_info().
                    """
                    try:
                        # 1. Try adding write permission
                        os.chmod(path, stat.S_IWUSR | stat.S_IREAD)

                        # 2. Try clearing flags (macOS/BSD specific)
                        if sys.platform == "darwin":
                            try:
                                # Clear all file flags (uchg, etc.)
                                os.chflags(path, 0)
                            except (AttributeError, OSError):
                                pass

                        # 3. Retry the failed operation
                        func(path)
                    except Exception as e:
                        logger.warning("Failed to force delete {}: {}", path, e)

                shutil.rmtree(path=src_path, onerror=on_error)
                logger.info("Deleted source directory for collection '{}'.", name)
        except Exception as e:
            logger.error(
                f"Failed to delete source directory for collection '{name}': {e}"
            )

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
        storage_ctx = self._storage_context(vector_store)

        # Build index with explicit storage_context so it uses the persistent docstore.
        self.index = VectorStoreIndex(
            nodes=[],
            embed_model=self.embed_model,
            storage_context=storage_ctx,
        )

        # Process batches from the pipeline generator
        for docs, nodes in pipeline.build(self._get_existing_file_hashes()):
            if nodes:
                # Filter nodes for vector indexing (exclude coarse chunks from vector store)
                is_hierarchical = any("docint_hier_type" in n.metadata for n in nodes)
                vector_nodes = (
                    [n for n in nodes if n.metadata.get("docint_hier_type") != "coarse"]
                    if is_hierarchical
                    else nodes
                )

                # Explicitly persist to docstore first to ensure data safety
                logger.debug("Persisting {} nodes to DocStore...", len(nodes))
                self.index.docstore.add_documents(nodes, allow_update=True)
                if vector_nodes:
                    self.index.insert_nodes(vector_nodes)

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
    def init_session_store(self, db_url: str) -> None:
        """
        Initialize the relational session store via SessionManager.

        Args:
            db_url (str): The database URL for the session store.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        self.sessions.init_session_store(db_url=db_url)

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

    def list_documents(self) -> list[dict[str, Any]]:
        """
        List all documents in the current collection by scanning all points.

        Returns:
            list[dict[str, Any]]: A list of document metadata dictionaries.
        """
        if not self.qdrant_collection:
            return []

        docs_map: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=256,
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
                payload = getattr(point, "payload", {}) or {}
                origin = payload.get("origin") or {}
                filename = (
                    origin.get("filename")
                    or payload.get("file_name")
                    or payload.get("filename")
                    or payload.get("file_path")
                )
                if not filename:
                    continue

                if filename not in docs_map:
                    docs_map[filename] = {
                        "filename": filename,
                        "mimetype": (
                            payload.get("filetype")
                            or payload.get("mimetype")
                            or payload.get("file_type")
                            or origin.get("filetype")
                            or payload.get("file_format")
                            or origin.get("mimetype")
                        ),
                        "file_hash": payload.get("file_hash")
                        or origin.get("file_hash"),
                        "node_count": 0,
                        "pages": set(),
                        "max_rows": 0,
                        "max_duration": 0.0,
                        "entity_types": set(),
                    }

                entry = docs_map[filename]
                entry["node_count"] += 1

                # Extract entities from payload
                ents = payload.get("entities") or []
                if isinstance(ents, list):
                    for e in ents:
                        if isinstance(e, dict):
                            t = e.get("type", e.get("label"))
                            if t:
                                entry["entity_types"].add(t)

                page = (
                    payload.get("page")
                    or payload.get("page_number")
                    or origin.get("page_no")
                )

                # Try getting page from doc_items (Docling structure)
                if page is None:
                    doc_items = payload.get("doc_items")
                    if isinstance(doc_items, list):
                        for item in doc_items:
                            if isinstance(item, dict):
                                prov = item.get("prov")
                                if isinstance(prov, list):
                                    for p in prov:
                                        if isinstance(p, dict) and "page_no" in p:
                                            page = p["page_no"]
                                            break
                            if page is not None:
                                break

                if page is not None:
                    try:
                        entry["pages"].add(int(page))
                    except (ValueError, TypeError):
                        entry["pages"].add(page)

                # Table rows logic
                table_info = payload.get("table")
                if isinstance(table_info, dict):
                    rows = table_info.get("n_rows")
                    if isinstance(rows, (int, float)):
                        entry["max_rows"] = max(entry["max_rows"], int(rows))

                # Audio duration logic
                end_sec = payload.get("end_seconds") or (
                    payload.get("extra_metadata") or {}
                ).get("end_seconds")
                if isinstance(end_sec, (int, float)):
                    entry["max_duration"] = max(entry["max_duration"], float(end_sec))

            if offset is None:
                break

        results = []
        for _, data in docs_map.items():
            data["page_count"] = len(data.pop("pages"))
            data["entity_types"] = sorted(list(data.get("entity_types", set())))
            if "entity_types" in data and isinstance(data["entity_types"], set):
                # Fallback if get didn't return set but pop of set or something (redundant with line above but safer)
                pass

            if data["max_rows"] == 0:
                del data["max_rows"]
            if data["max_duration"] == 0.0:
                del data["max_duration"]
            results.append(data)

        return sorted(results, key=lambda x: str(x["filename"]))

    def get_collection_ie(self, refresh: bool = False) -> list[dict[str, Any]]:
        """
        Fetch all nodes from the current collection and return their IE metadata.

        Returns:
            list[dict[str, Any]]: A list of source metadata dictionaries containing IE data.
        """
        if not self.qdrant_collection:
            return []

        if self.ie_sources and not refresh:
            return self.ie_sources

        if self.ie_sources:
            return self.ie_sources

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

        self.ie_sources = sources
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
