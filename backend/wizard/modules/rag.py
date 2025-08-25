from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from fastembed import SparseTextEmbedding
from helpers.unescape_unicode import unescape_unicode
from llama_index.core import (
    Response,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import BaseNode, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from modules.readers.table_reader import TableReader
from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    VectorParams,
)
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")

# --- Session persistence (ORM) ---
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True)  # external session id
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    rolling_summary = Column(Text, default="", nullable=False)
    turns = relationship(
        "Turn",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Turn.idx",
    )


class Turn(Base):
    __tablename__ = "turns"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True)
    idx = Column(Integer, nullable=False)  # 0..N
    user_text = Column(Text, nullable=False)
    rewritten_query = Column(Text, nullable=True)
    model_response = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    conversation = relationship("Conversation", back_populates="turns")
    citations = relationship(
        "Citation", back_populates="turn", cascade="all, delete-orphan"
    )


class Citation(Base):
    __tablename__ = "citations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    turn_id = Column(Integer, ForeignKey("turns.id"), index=True)
    node_id = Column(String, nullable=True)  # LlamaIndex node id or Qdrant point id
    score = Column(Float, nullable=True)
    filename = Column(String, nullable=True)
    filetype = Column(String, nullable=True)
    source = Column(String, nullable=True)  # "table" or ""
    page = Column(Integer, nullable=True)
    row = Column(Integer, nullable=True)
    turn = relationship("Turn", back_populates="citations")


def _make_session_maker(db_url: str = "sqlite:///rag_sessions.db"):
    engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@dataclass(slots=True)
class RAG:
    # --- Path setup ---
    data_dir: Path | None = None
    persist_dir: Path | None = None
    # --- Host path (optional; used only for FS fallback on local Docker)
    qdrant_host_dir: str | None = None

    # --- Models ---
    embed_model_id: str = "BAAI/bge-m3"
    rerank_model_id: str = "BAAI/bge-reranker-v2-m3"
    gen_model_id: str = "qwen3:8b"
    sparse_model_id: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # --- Backend controls ---
    qdrant_host: str = QDRANT_HOST
    qdrant_port: int = 6333
    qdrant_collection: str = "default"
    qdrant_index_dims: int = 768

    # --- Performance / device controls ---
    embed_device: str = "cpu"  # run embeddings on CPU to avoid GPU contention
    rerank_device: str = "cpu"  # run cross-encoder reranker on CPU

    # --- Chunking controls ---
    chunk_size: int = 1024
    chunk_overlap: int = 160

    # --- Ollama Parameters ---
    base_url: str = OLLAMA_HOST
    context_window: int = -1
    temperature: float = 0.2
    request_timeout: int = 240
    thinking: bool = True
    # Forwarded to the underlying Ollama client as extra model options
    ollama_options: dict[str, Any] | None = None

    # --- Reranking / retrieval ---
    enable_hybrid: bool = True
    embed_batch_size: int = 64
    rerank_top_n: int = 5
    retrieve_similarity_top_k: int = rerank_top_n * 5

    # --- TableReader config ---
    table_text_cols: list[str] | None = None
    table_metadata_cols: list[str] | None = None
    table_id_col: str | None = None
    table_excel_sheet: str | int | None = None
    table_row_limit: int | None = None

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

    reader: DoclingReader | None = field(default=None, init=False)
    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    pdf_node_parser: DoclingNodeParser | None = field(default=None, init=False)
    table_node_parser: SentenceSplitter | None = field(default=None, init=False)

    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)

    index: VectorStoreIndex | None = field(default=None, init=False)
    query_engine: RetrieverQueryEngine | None = field(default=None, init=False)

    # Chat/session runtime
    chat_engine: RetrieverQueryEngine | None = field(default=None, init=False)
    chat_memory: Any | None = field(default=None, init=False)
    _SessionMaker: Any | None = field(default=None, init=False, repr=False)
    session_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Cap top_n and similarity_top_k to keep reranking/lightweight
        if self.rerank_top_n < 1:
            self.rerank_top_n = 5
        # Keep similarity_top_k in a sane range; actual k is recalculated when creating the retriever
        if self.retrieve_similarity_top_k < self.rerank_top_n:
            self.retrieve_similarity_top_k = self.rerank_top_n * 5
        # Bound chunk params
        self.chunk_size = max(256, min(self.chunk_size, 1024))
        self.chunk_overlap = max(
            0, min(self.chunk_overlap, int(self.chunk_size * 0.25))
        )

    # --- Static methods ---
    @staticmethod
    def _list_supported_sparse_models() -> list[str]:
        try:
            return [m["model"] for m in SparseTextEmbedding.list_supported_models()]
        except ImportError:
            return []

    # --- Properties (lazy loading) ---
    @property
    def device(self) -> str:
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
        Lazily initializes and returns the embedding model (HF or Ollama),
        depending on `self.embed_backend`.
        """
        self._embed_model = HuggingFaceEmbedding(
            model_name=self.embed_model_id,
            device=self.embed_device,
            normalize=True,
        )
        if self._embed_model is None:
            raise RuntimeError("Embedding model could not be initialized.")
        logger.info("Embedding model (HF) initialized: %s", self.embed_model_id)
        return self._embed_model

    @property
    def gen_model(self) -> Ollama:
        if self._gen_model is None:
            self._gen_model = Ollama(
                base_url=self.base_url,
                model=self.gen_model_id,
                temperature=self.temperature,
                context_window=self.context_window,
                request_timeout=self.request_timeout,
                thinking=self.thinking,
                additional_kwargs=self.ollama_options,
            )
            logger.info("Gen model (Ollama) initialized: %s", self.gen_model_id)
        return self._gen_model

    @property
    def sparse_model(self) -> str | None:
        """Return the configured sparse model id for hybrid retrieval."""
        if not self.enable_hybrid:
            return None
        if self.sparse_model_id not in self._list_supported_sparse_models():
            raise ValueError(
                f"Sparse model {self.sparse_model_id!r} not supported. "
                f"Supported: {self._list_supported_sparse_models()}"
            )
        return self.sparse_model_id

    @property
    def reranker(self) -> SentenceTransformerRerank:
        if self._reranker is None:
            self._reranker = SentenceTransformerRerank(
                top_n=self.rerank_top_n,
                model=self.rerank_model_id,
                device=self.rerank_device,
            )
            logger.info("Reranker initialized: %s", self.rerank_model_id)
        return self._reranker

    @property
    def qdrant_client(self) -> QdrantClient:
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                host=self.qdrant_host, port=self.qdrant_port
            )
            logger.info(
                f"Qdrant client initialized: {self.qdrant_host}:{self.qdrant_port}"
            )
        return self._qdrant_client

    @property
    def qdrant_aclient(self) -> AsyncQdrantClient:
        if self._qdrant_aclient is None:
            self._qdrant_aclient = AsyncQdrantClient(
                host=self.qdrant_host, port=self.qdrant_port
            )
            logger.info(
                "Qdrant async client initialized: %s: %s",
                self.qdrant_host,
                self.qdrant_port,
            )
        return self._qdrant_aclient

    # --- Build pieces ---
    def _create_doc_loader(self) -> None:
        # PDF reader (Docling) as before
        pdf_reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)

        # Table reader for CSV/TSV/XLSX/Parquet
        table_reader = TableReader(
            text_cols=self.table_text_cols,
            metadata_cols=self.table_metadata_cols,
            id_col=self.table_id_col,
            excel_sheet=self.table_excel_sheet,
            limit=self.table_row_limit,
        )

        self.dir_reader = SimpleDirectoryReader(
            input_dir=self.data_dir,
            file_extractor={
                ".pdf": pdf_reader,
                ".csv": table_reader,
                ".tsv": TableReader(
                    csv_sep="\t",  # allow explicit TSV sep
                    text_cols=self.table_text_cols,
                    metadata_cols=self.table_metadata_cols,
                    id_col=self.table_id_col,
                    limit=self.table_row_limit,
                ),
                ".xlsx": table_reader,
                ".xls": table_reader,
                ".parquet": TableReader(
                    text_cols=self.table_text_cols or ["text"],
                    metadata_cols=self.table_metadata_cols,
                    id_col=self.table_id_col,
                    limit=self.table_row_limit,
                ),
            },
        )
        self.pdf_node_parser = DoclingNodeParser()
        self.table_node_parser = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _create_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.qdrant_client,
            aclient=self.qdrant_aclient,
            collection_name=self.qdrant_collection,
            enable_hybrid=self.enable_hybrid,
            fastembed_sparse_model=self.sparse_model,
        )

    def _create_storage_context(
        self, vector_store: QdrantVectorStore
    ) -> StorageContext:
        return StorageContext.from_defaults(vector_store=vector_store)

    def _create_index(self, storage_ctx: StorageContext) -> VectorStoreIndex:
        return VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_ctx,
            embed_model=self.embed_model,
        )

    def _docs_to_nodes(self) -> None:
        """Convert loaded documents into nodes using the appropriate parsers."""
        if self.dir_reader is None:
            raise RuntimeError("Directory reader is not initialized.")
        self.docs = self.dir_reader.load_data()
        if self.pdf_node_parser is None or self.table_node_parser is None:
            raise RuntimeError("Node parsers are not initialized.")

        pdf_docs, table_docs = [], []
        for d in self.docs:
            meta = getattr(d, "metadata", {}) or {}
            file_type = (meta.get("file_type") or "").lower()
            source_kind = meta.get("source", "") or ""

            if source_kind == "table":
                table_docs.append(d)
            elif file_type.startswith("application/pdf") or str(
                meta.get("file_path", "")
            ).lower().endswith(".pdf"):
                pdf_docs.append(d)
            else:
                # default to pdf parser when in doubt (since DoclingReader produced these)
                pdf_docs.append(d)

        nodes: list[BaseNode] = []
        if pdf_docs:
            nodes.extend(self.pdf_node_parser.get_nodes_from_documents(pdf_docs))
        if table_docs:
            nodes.extend(self.table_node_parser.get_nodes_from_documents(table_docs))

        self.nodes = nodes

    def _probe_embedding_dimension(self) -> int:
        """
        Probe the embedding model with a test string to determine the output dimension.

        Returns:
            int: The dimensionality of the embedding vectors.
        """        
        try:
            probe_vec = self.embed_model.get_text_embedding("dimension probe")
            dim = (
                len(probe_vec)
                if isinstance(probe_vec, (list, tuple))
                else int(getattr(probe_vec, "shape", [0])[0])
            )
            if not isinstance(dim, int) or dim <= 0:
                dim = self.qdrant_index_dims  # fallback
        except Exception:
            dim = self.qdrant_index_dims
        return dim
    
    def _ensure_qdrant_collection(self, dim: int) -> None:
        """
        Create or gently tune the Qdrant collection for better runtime performance.
        - Stores vectors and HNSW graph on disk to reduce RAM pressure
        - Uses lighter HNSW build params
        - Enables INT8 scalar quantization on vectors (lossy but fast & memory-light)

        This will **not** drop an existing collection. It creates the collection if missing
        and attempts safe, idempotent updates otherwise.

        Args:
            dim (int): Dimensionality of the vectors to be stored.
        """
        name = self.qdrant_collection

        # First attempt to update an existing collection. If this fails (e.g. the
        # collection does not exist yet or the server rejects the config), fall back
        # to creating the collection with our preferred settings. Any errors during
        # creation are logged but otherwise ignored so that higher-level code can
        # decide how to proceed.
        try:
            self.qdrant_client.update_collection(
                collection_name=name,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=64, on_disk=True),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    indexing_threshold=20000,
                    memmap_threshold=20000,
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type="int8",
                        quantile=0.99,
                        always_ram=False,
                    )
                ),
            )
            logger.info(
                "Qdrant collection '%s' updated with on-disk HNSW & INT8 quantization.",
                name,
            )
            return
        except Exception as e:
            logger.warning(
                "Could not update existing Qdrant collection '%s': %s", name, e
            )

        try:
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dim, distance=Distance.COSINE, on_disk=True
                ),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=64, on_disk=True),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    indexing_threshold=20000,
                    memmap_threshold=20000,
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type="int8",
                        quantile=0.99,
                        always_ram=False,
                    )
                ),
            )
            logger.info(
                "Qdrant collection '%s' created (dim=%d, on-disk vectors+HNSW, INT8 quantization).",
                name,
                dim,
            )
        except Exception as e:
            logger.warning(
                "Failed to create Qdrant collection '%s' with tuned settings: %s",
                name,
                e,
            )

    def create_index(self) -> None:
        """
        Create the full index by loading documents, converting to nodes, and
        setting up the Qdrant collection and vector store.
        """        
        dim = self._probe_embedding_dimension()
        self._ensure_qdrant_collection(dim)

        vector_store = self._create_vector_store()
        storage_ctx = self._create_storage_context(vector_store)
        self.index = self._create_index(storage_ctx)

    def create_query_engine(self) -> None:
        """
        Create the query engine with a retriever and reranker.

        Raises:
            RuntimeError: If the index is not initialized.
        """        
        if self.index is None:
            raise RuntimeError("Index is not initialized. Cannot create query engine.")
        k = min(max(self.retrieve_similarity_top_k, self.rerank_top_n * 8), 64)
        self.query_engine = RetrieverQueryEngine.from_args(
            self.index.as_retriever(similarity_top_k=k),
            llm=self.gen_model,
            node_postprocessors=[self.reranker],
        )

    def _extract_relevant_data(
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
            import re

            m = re.search(r"<think>(.*?)</think>", resp_text, flags=re.DOTALL)
            reason = (m.group(1).strip() if m else None) or reason
            resp_text = re.sub(
                r"<think>.*?</think>", "", resp_text, flags=re.DOTALL
            ).strip()
        # normalize unicode escapes in response text
        resp_text = unescape_unicode(resp_text)

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
            source_kind = meta.get(
                "source", "document" if filetype.startswith("application/pdf") else ""
            )

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

            src = {
                "text": unescape_unicode(getattr(node, "text", "") or ""),
                "filename": filename,
                "filetype": filetype,
                "source": source_kind,
            }
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
    def _resolve_qdrant_host_dir(self) -> Path | None:
        """Best-effort resolution of the host directory where Qdrant stores data.
        Used only as a *fallback* when we cannot reach the Qdrant API.
        Priority: explicit field -> env var -> platform default under home.
        Returns a Path to the directory that contains the `collections/` subfolder.
        """
        if self.qdrant_host_dir:
            return Path(self.qdrant_host_dir)
        env = os.getenv("QDRANT_HOST_DIR") or os.getenv("QDRANT_PATH")
        if env:
            return Path(env)
        home = os.getenv("HOME") or os.getenv("USERPROFILE")
        if home:
            return Path(home) / ".qdrant" / "storage"
        return None

    def list_collections(self, prefer_api: bool = True) -> list[str]:
        """Return a list of collection names. Uses Qdrant API when available; falls back to listing the host storage path."""
        if prefer_api:
            try:
                resp = self.qdrant_client.get_collections()
                names = [c.name for c in getattr(resp, "collections", []) or []]
                if names:
                    return sorted(names)
            except Exception as e:
                logger.warning(
                    "Qdrant API list_collections failed, will try FS fallback: %s", e
                )
        base = self._resolve_qdrant_host_dir()
        if base is None:
            return []
        collections_dir = base / "collections"
        try:
            if not collections_dir.exists():
                return []
            return sorted([p.name for p in collections_dir.iterdir() if p.is_dir()])
        except Exception as e:
            logger.warning("FS fallback list_collections failed: %s", e)
            return []

    # def get_collection_info(self, name: str) -> dict[str, Any]:
    #     """
    #     Return lightweight info about a collection (status, vectors, on-disk flags). Uses API if possible; otherwise returns an FS-only stub.
    #     """
    #     info: dict[str, Any] = {"name": name}
    #     try:
    #         ci = self.qdrant_client.get_collection(collection_name=name)
    #         info.update(
    #             {
    #                 "status": getattr(ci, "status", None),
    #                 "vectors_count": getattr(
    #                     getattr(ci, "vectors_count", None), "total", None
    #                 )
    #                 or getattr(ci, "vectors_count", None),
    #                 "on_disk_payload": bool(
    #                     getattr(
    #                         getattr(ci, "payload_storage_type", None), "on_disk", False
    #                     )
    #                 )
    #                 if hasattr(getattr(ci, "payload_storage_type", None), "on_disk")
    #                 else None,
    #             }
    #         )
    #     except Exception as e:
    #         logger.warning("Qdrant API get_collection failed for '%s': %s", name, e)
    #         base = self._resolve_qdrant_host_dir()
    #         if base:
    #             path = base / "collections" / name
    #             info.update({"path": str(path), "exists_on_disk": path.exists()})
    #     return info

    def select_or_create_collection(self, name: str, dim: int | None = None) -> None:
        """
        Switch active collection; create/tune it if missing. If `dim` is not provided, probe via 
        the embedding model once.
        
        Args:
            name (str): The name of the collection to select or create.
            dim (int | None): The dimensionality of the vectors. If None, will be probed.
        
        Raises:
            ValueError: If the collection name is empty or invalid.
        """
        if not name or not name.strip():
            raise ValueError("Collection name cannot be empty.")
        self.qdrant_collection = name.strip()

        # Reset any state tied to the previously selected collection so that
        # future queries do not use stale indexes or conversations.
        self.docs.clear()
        self.nodes.clear()
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.chat_memory = None
        self.session_id = None

        if dim is None:
            dim = self._probe_embedding_dimension()
        self._ensure_qdrant_collection(dim)

    # --- Public API ---
    def ingest_docs(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self._create_doc_loader()
        self._docs_to_nodes()
        self.create_index()
        self.create_query_engine()
        try:
            eff_k = getattr(self.query_engine.retriever, "similarity_top_k", None)
        except Exception:
            eff_k = None
        logger.info(
            "Effective retrieval k=%s | top_n=%s | chunk_size=%s | overlap=%s | embed_device=%s | rerank_device=%s",
            eff_k,
            self.rerank_top_n,
            self.chunk_size,
            self.chunk_overlap,
            self.embed_device,
            self.rerank_device,
        )
        logger.info("Documents ingested successfully.")

    async def asingest_docs(self, data_dir: str | Path) -> None:
        """
        Async ingestion path: parses docs, builds an empty index, then inserts nodes concurrently.
        Requires an event loop (e.g., FastAPI startup task or any asyncio context).
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self._create_doc_loader()
        self._docs_to_nodes()
        self.create_empty_index()
        if self.index is None:
            raise RuntimeError("Index is not initialized for async ingestion.")
        # Concurrent, non-blocking upsert into Qdrant via aclient
        await self.index.ainsert_nodes(self.nodes)
        self.create_query_engine()
        try:
            eff_k = getattr(self.query_engine.retriever, "similarity_top_k", None)
        except Exception:
            eff_k = None
        logger.info(
            "Effective retrieval k=%s | top_n=%s | chunk_size=%s | overlap=%s | embed_device=%s | rerank_device=%s",
            eff_k,
            self.rerank_top_n,
            self.chunk_size,
            self.chunk_overlap,
            self.embed_device,
            self.rerank_device,
        )
        logger.info("Documents ingested successfully (async path).")

    def run_query(self, prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("Query prompt cannot be empty.")
        if self.query_engine is None:
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs() first."
            )
        result = self.query_engine.query(prompt)
        if not isinstance(result, Response):
            raise TypeError(f"Expected Response, got {type(result).__name__}")
        return self._extract_relevant_data(prompt, result)

    async def run_query_async(self, prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("Query prompt cannot be empty.")
        if self.query_engine is None:
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs()/asingest_docs() first."
            )
        result = await self.query_engine.aquery(prompt)
        if not isinstance(result, Response):
            raise TypeError(f"Expected Response, got {type(result).__name__}")
        return self._extract_relevant_data(prompt, result)

    def save_index(self, persist_dir: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Index is not initialized. Nothing to save.")
        self.persist_dir = Path(persist_dir)
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info("Index saved to %s", str(self.persist_dir))

    # def load_index(self) -> None:
    #     """
    #     Attach to Qdrant collection and build the index from the vector store only.
    #     Persist dirs are ignored in this production-oriented path.
    #     """
    #     # Vector store backed by the existing Qdrant collection
    #     vector_store = QdrantVectorStore(
    #         client=self.qdrant_client,
    #         aclient=self.qdrant_aclient,
    #         collection_name=self.qdrant_collection,
    #         enable_hybrid=self.enable_hybrid,
    #         fastembed_sparse_model=self.sparse_model,
    #     )

    #     storage_context = StorageContext.from_defaults(vector_store=vector_store)
    #     try:
    #         self.index = VectorStoreIndex.from_vector_store(
    #             vector_store=vector_store,
    #             storage_context=storage_context,
    #             embed_model=self.embed_model,
    #         )
    #     except Exception as e:
    #         raise RuntimeError(
    #             f"Failed to attach to Qdrant collection '{self.qdrant_collection}': {e}"
    #         ) from e

    #     if not isinstance(self.index, VectorStoreIndex):
    #         raise TypeError("Loaded index is not a VectorStoreIndex")

    #     # Create query engine
    #     self.create_query_engine()
    #     logger.info(
    #         "Index attached to Qdrant collection '%s' (host=%s, port=%s)",
    #         self.qdrant_collection,
    #         self.qdrant_host,
    #         self.qdrant_port,
    #     )

    # --- Session store wiring ---
    def init_session_store(self, db_url: str = "sqlite:///rag_sessions.db") -> None:
        """Initialize (or reinitialize) the SQLAlchemy session factory."""
        self._SessionMaker = _make_session_maker(db_url)

    def _ensure_store(self) -> None:
        if self._SessionMaker is None:
            self.init_session_store()

    def _load_or_create_convo(self, session_id: str):
        self._ensure_store()
        s = self._SessionMaker()
        conv = s.get(Conversation, session_id)
        if conv is None:
            conv = Conversation(id=session_id)
            s.add(conv)
            s.commit()
        return s, conv

    def _get_rolling_summary(self, session_id: str) -> str:
        self._ensure_store()
        s = self._SessionMaker()
        conv = s.get(Conversation, session_id)
        return (conv.rolling_summary or "") if conv else ""

    # --- Chat session lifecycle ---
    def _persist_turn(
        self, session_id: str, user_msg: str, resp: Any, data: dict
    ) -> None:
        """Persist conversation turn + citations in the relational store."""
        self._ensure_store()
        s, conv = self._load_or_create_convo(session_id)

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
                node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)

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
        """Maintain a compact rolling summary to bound future context windows."""
        self._ensure_store()
        s = self._SessionMaker()
        conv = s.get(Conversation, session_id)
        if not conv or len(conv.turns) == 0 or (len(conv.turns) % every_n_turns) != 0:
            return

        # Build a concise slice of the last N turns
        slice_text = []
        for turn in conv.turns[-every_n_turns:]:
            slice_text.append(
                f"User: {turn.user_text}\nAssistant: {turn.model_response}"
            )
        prompt = (
            "Summarize the key facts and user intent from the following chat turns. "
            "Keep it under 10 sentences and avoid speculation.\n\n"
            + "\n\n".join(slice_text)
        )
        # Use the same LLM to summarize
        summary_resp = self.gen_model.complete(prompt)
        new_summary = (conv.rolling_summary + "\n" + summary_resp.text).strip()
        conv.rolling_summary = new_summary
        s.commit()

    def _get_node_text_by_id(self, node_id: str) -> str | None:
        """Best-effort fetch of a node's text from the index docstore given its id.
        Tries several access patterns to be robust across LlamaIndex versions.
        Returns None if not found or if index/docstore are unavailable.
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
                        if text:
                            return unescape_unicode(text)
                        # Some versions store content on .get_content(), or .get_text()
                        if hasattr(node, "get_content") and callable(node.get_content):
                            t = node.get_content()
                            if isinstance(t, str) and t:
                                return unescape_unicode(t)
                        if hasattr(node, "get_text") and callable(node.get_text):
                            t = node.get_text()
                            if isinstance(t, str) and t:
                                return unescape_unicode(t)
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
                        return unescape_unicode(txt.strip())
        except Exception:
            pass
        return None

    # Export session data as JSON
    def export_session(
        self, session_id: str | None = None, out_dir: str | Path = "session"
    ) -> Path:
        """
        Export a session as a portable bundle:
        - session.json
        - messages.jsonl
        - citations.parquet  (if pandas/pyarrow available)
        - transcript.md
        - manifest.json (sha256 of each file)
        """
        self._ensure_store()
        s = self._SessionMaker()

        if not session_id and self.session_id is not None:
            session_id: str = self.session_id

        conv = s.get(Conversation, session_id)
        if conv is None:
            raise ValueError(f"No conversation found for session_id={session_id}")

        out_dir = Path(out_dir) / session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) session.json
        session_meta = {
            "schema_version": "1.0.0",
            "session_id": conv.id,
            "created_at": conv.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "turn_count": len(conv.turns),
            "rolling_summary": conv.rolling_summary or "",
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
                "host": self.qdrant_host,
                "port": self.qdrant_port,
                "collection": self.qdrant_collection,
                "host_dir": str(self._resolve_qdrant_host_dir() or ""),
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
                    "created_at": t.created_at.replace(tzinfo=timezone.utc).isoformat(),
                    "user_text": t.user_text,
                    "rewritten_query": t.rewritten_query,
                    "assistant_text": t.model_response,
                    "reasoning": t.reasoning,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # 3) citations.parquet (optional if pandas/pyarrow present)
        try:
            rows = []
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
                pd.DataFrame(
                    [],
                    columns=[
                        "turn_idx",
                        "node_id",
                        "score",
                        "filename",
                        "filetype",
                        "source",
                        "page",
                        "row",
                    ],
                ).to_parquet(out_dir / "citations.parquet", index=False)
        except Exception as e:
            logger.warning(
                "Skipping citations.parquet export (pandas/pyarrow not available?): %s",
                e,
            )

        # 4) transcript.md (human-readable)
        lines = ["# Transcript", f"Session: `{conv.id}`", ""]
        if conv.rolling_summary:
            lines += ["## Rolling Summary", "", conv.rolling_summary, ""]
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
                    # Try to fetch node text via docstore
                    excerpt = None
                    if c.node_id:
                        excerpt = self._get_node_text_by_id(c.node_id)
                    if excerpt is None:
                        excerpt = "[source text unavailable]"
                    else:
                        excerpt = excerpt.strip()
                        # avoid mile-long transcript entries
                        max_chars = 800
                        if len(excerpt) > max_chars:
                            excerpt = excerpt[:max_chars].rstrip() + " …"
                    # Render as a nested blockquote in markdown for readability
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
                manifest[name] = {"sha256": sha256_file(fp), "bytes": fp.stat().st_size}
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

    def start_session(self, session_id: str | None = None) -> str:
        """
        Create/attach a chat engine with history-aware question condensation.
        If no session_id is provided, a new UUIDv4 is generated and persisted.
        Returns the active session_id.
        """
        # 1) ensure a valid session id
        if not session_id:
            session_id = str(uuid.uuid4())
        self.session_id = session_id

        # 2) ensure a Conversation row exists (idempotent)
        self._load_or_create_convo(session_id)

        # 3) seed memory from rolling summary
        rolling = self._get_rolling_summary(session_id)
        self.chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=2000, chat_history=[]
        )
        if rolling:
            self.chat_memory.put("system", f"Conversation summary so far:\n{rolling}")

        if self.query_engine is None:
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs() first."
            )

        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=self.query_engine,  # reuse retriever + reranker
            memory=self.chat_memory,
            llm=self.gen_model,
        )
        return session_id

    def chat(self, user_msg: str) -> dict[str, Any]:
        """Run one conversational turn, persist it, and return your normalized payload."""
        if not user_msg.strip():
            raise ValueError("Chat prompt cannot be empty.")
        if self.query_engine is None:
            raise RuntimeError(
                "Query engine has not been initialized. Call ingest_docs() first."
            )

        # (Re)create engine if session changed or not started
        session_id = self.session_id
        if self.chat_engine is None or getattr(self, "_session_id", None) != session_id:
            self.start_session(session_id)

        resp = self.chat_engine.chat(user_msg)
        data = self._extract_relevant_data(user_msg, resp)
        self._persist_turn(session_id, user_msg, resp, data)
        self._maybe_update_summary(session_id)
        return data, resp
