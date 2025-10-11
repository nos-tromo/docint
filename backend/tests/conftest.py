import sys
import types


def _install_torch_stub() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class _MPS:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def is_built() -> bool:
                return False

        torch.cuda = _Cuda()
        torch.backends = types.SimpleNamespace(mps=_MPS())
        sys.modules["torch"] = torch


def _install_fastembed_stub() -> None:
    try:
        import fastembed  # noqa: F401
    except ModuleNotFoundError:
        fastembed = types.ModuleType("fastembed")

        class SparseTextEmbedding:
            @staticmethod
            def list_supported_models() -> list[dict[str, str]]:
                return [{"model": "stub-model"}]

        fastembed.SparseTextEmbedding = SparseTextEmbedding
        sys.modules["fastembed"] = fastembed


def _install_llama_index_stub() -> None:
    try:
        import llama_index  # noqa: F401
    except ModuleNotFoundError:
        llama_index = types.ModuleType("llama_index")

        # --- core schema -------------------------------------------------
        core_module = types.ModuleType("llama_index.core")

        class BaseNode:
            def __init__(self, text: str = "", metadata: dict | None = None):
                self.text = text
                self.metadata = metadata or {}

        class Document(BaseNode):
            def __init__(
                self,
                text: str,
                metadata: dict | None = None,
                doc_id: str | None = None,
            ):
                super().__init__(text=text, metadata=metadata)
                self.doc_id = doc_id

            def to_dict(self) -> dict:
                return {
                    "text": self.text,
                    "metadata": self.metadata,
                    "doc_id": self.doc_id,
                }

        class Response:
            def __init__(
                self,
                response: str | None = None,
                source_nodes: list | None = None,
                metadata: dict | None = None,
                text: str | None = None,
            ):
                self.response = response
                self.source_nodes = source_nodes or []
                self.metadata = metadata or {}
                self.text = text

        class SimpleDirectoryReader:
            def __init__(self, input_dir=None, **kwargs):
                self.input_dir = input_dir
                self.kwargs = kwargs

            def load_data(self):
                return []

        class StorageContext:
            def __init__(self, vector_store=None):
                self.vector_store = vector_store
                self.docstore = {}

            @classmethod
            def from_defaults(cls, vector_store=None):
                return cls(vector_store=vector_store)

        class _StubRetriever:
            def __init__(self, nodes=None):
                self.nodes = nodes or []
                self.similarity_top_k = None

            def query(self, prompt: str):
                return Response(
                    response=f"stub response for: {prompt}",
                    source_nodes=[],
                )

        class VectorStoreIndex:
            def __init__(self, nodes=None, storage_context=None, embed_model=None):
                self.nodes = list(nodes or [])
                self.storage_context = storage_context or StorageContext()
                self._retriever = _StubRetriever(self.nodes)

            def as_retriever(self, similarity_top_k: int = 1):
                self._retriever.similarity_top_k = similarity_top_k
                return self._retriever

            async def ainsert_nodes(self, nodes):
                self.nodes.extend(nodes)

        core_module.BaseNode = BaseNode
        core_module.Document = Document
        core_module.Response = Response
        core_module.SimpleDirectoryReader = SimpleDirectoryReader
        core_module.StorageContext = StorageContext
        core_module.VectorStoreIndex = VectorStoreIndex

        # --- Chat engine -------------------------------------------------
        chat_engine_module = types.ModuleType("llama_index.core.chat_engine")

        class CondenseQuestionChatEngine:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        chat_engine_module.CondenseQuestionChatEngine = CondenseQuestionChatEngine

        # --- Embeddings --------------------------------------------------
        embeddings_module = types.ModuleType("llama_index.core.embeddings")

        class BaseEmbedding:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        embeddings_module.BaseEmbedding = BaseEmbedding

        # --- LLMs --------------------------------------------------------
        llms_module = types.ModuleType("llama_index.core.llms")

        class ChatMessage:
            def __init__(self, role: str, content: str):
                self.role = role
                self.content = content

        class MessageRole:
            USER = "user"
            ASSISTANT = "assistant"

        llms_module.ChatMessage = ChatMessage
        llms_module.MessageRole = MessageRole

        # --- Memory ------------------------------------------------------
        memory_module = types.ModuleType("llama_index.core.memory")

        class ChatMemoryBuffer:
            def __init__(self, *args, **kwargs):
                self.buffer = []

            def put(self, message):
                self.buffer.append(message)

        memory_module.ChatMemoryBuffer = ChatMemoryBuffer

        # --- Node parser -------------------------------------------------
        node_parser_module = types.ModuleType("llama_index.core.node_parser")

        class SentenceSplitter:
            def __init__(self, chunk_size=1024, chunk_overlap=0):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def get_nodes_from_documents(self, documents):
                return [
                    BaseNode(
                        text=getattr(doc, "text", ""),
                        metadata=getattr(doc, "metadata", {}),
                    )
                    for doc in documents
                ]

        node_parser_module.SentenceSplitter = SentenceSplitter

        # --- Postprocessor -----------------------------------------------
        postprocessor_module = types.ModuleType("llama_index.core.postprocessor")

        class SentenceTransformerRerank:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __call__(self, nodes):
                return nodes

        postprocessor_module.SentenceTransformerRerank = SentenceTransformerRerank

        # --- Query engine ------------------------------------------------
        query_engine_module = types.ModuleType("llama_index.core.query_engine")

        class RetrieverQueryEngine:
            def __init__(self, retriever, llm=None, node_postprocessors=None):
                self.retriever = retriever
                self.llm = llm
                self.node_postprocessors = node_postprocessors or []

            @classmethod
            def from_args(cls, retriever, llm=None, node_postprocessors=None):
                return cls(retriever, llm=llm, node_postprocessors=node_postprocessors)

            def query(self, prompt: str):
                if hasattr(self.retriever, "query"):
                    return self.retriever.query(prompt)
                return Response(response="")

            async def aquery(self, prompt: str):
                return self.query(prompt)

        query_engine_module.RetrieverQueryEngine = RetrieverQueryEngine

        # --- HuggingFace embedding ---------------------------------------
        embeddings_hf_module = types.ModuleType(
            "llama_index.embeddings.huggingface"
        )

        class HuggingFaceEmbedding:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        embeddings_hf_module.HuggingFaceEmbedding = HuggingFaceEmbedding

        # --- Ollama ------------------------------------------------------
        ollama_module = types.ModuleType("llama_index.llms.ollama")

        class Ollama:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def chat(self, prompt: str) -> str:
                return f"ollama: {prompt}"

        ollama_module.Ollama = Ollama

        # --- Docling parser/reader ---------------------------------------
        node_parser_docling_module = types.ModuleType(
            "llama_index.node_parser.docling"
        )

        class DoclingNodeParser:
            def get_nodes_from_documents(self, documents):
                return [
                    BaseNode(
                        text=getattr(doc, "text", ""),
                        metadata=getattr(doc, "metadata", {}),
                    )
                    for doc in documents
                ]

        node_parser_docling_module.DoclingNodeParser = DoclingNodeParser

        readers_docling_module = types.ModuleType("llama_index.readers.docling")

        class DoclingReader:
            class ExportType:
                JSON = "json"

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def load_data(self, *args, **kwargs):
                return []

        readers_docling_module.DoclingReader = DoclingReader

        # --- Vector stores -----------------------------------------------
        vector_store_qdrant_module = types.ModuleType(
            "llama_index.vector_stores.qdrant"
        )

        class QdrantVectorStore:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        vector_store_qdrant_module.QdrantVectorStore = QdrantVectorStore

        # Attach submodules
        llama_index.core = core_module
        llama_index.embeddings = types.SimpleNamespace(huggingface=embeddings_hf_module)
        llama_index.llms = types.SimpleNamespace(ollama=ollama_module)
        llama_index.node_parser = types.SimpleNamespace(docling=node_parser_docling_module)
        llama_index.readers = types.SimpleNamespace(docling=readers_docling_module)
        llama_index.vector_stores = types.SimpleNamespace(qdrant=vector_store_qdrant_module)

        sys.modules["llama_index"] = llama_index
        sys.modules["llama_index.core"] = core_module
        sys.modules["llama_index.core.chat_engine"] = chat_engine_module
        sys.modules["llama_index.core.embeddings"] = embeddings_module
        sys.modules["llama_index.core.llms"] = llms_module
        sys.modules["llama_index.core.memory"] = memory_module
        sys.modules["llama_index.core.node_parser"] = node_parser_module
        sys.modules["llama_index.core.postprocessor"] = postprocessor_module
        sys.modules["llama_index.core.query_engine"] = query_engine_module
        sys.modules["llama_index.core.schema"] = types.SimpleNamespace(
            BaseNode=core_module.BaseNode,
            Document=core_module.Document,
        )
        sys.modules["llama_index.embeddings.huggingface"] = embeddings_hf_module
        sys.modules["llama_index.llms.ollama"] = ollama_module
        sys.modules["llama_index.node_parser.docling"] = node_parser_docling_module
        sys.modules["llama_index.readers.docling"] = readers_docling_module
        sys.modules["llama_index.vector_stores.qdrant"] = vector_store_qdrant_module


def _install_qdrant_stub() -> None:
    try:
        import qdrant_client  # noqa: F401
    except ModuleNotFoundError:
        qdrant_client = types.ModuleType("qdrant_client")

        class QdrantClient:
            def __init__(self, url=None):
                self.url = url
                self._collections = []
                self._payloads = {}

            def get_collections(self):
                collections = [types.SimpleNamespace(name=name) for name in self._collections]
                return types.SimpleNamespace(collections=collections)

            def retrieve(self, collection_name=None, ids=None):
                results = []
                for _id in ids or []:
                    payload = self._payloads.get(_id, {})
                    results.append(types.SimpleNamespace(payload=payload))
                return results

        qdrant_client.QdrantClient = QdrantClient

        async_module = types.ModuleType("qdrant_client.async_qdrant_client")

        class AsyncQdrantClient:
            def __init__(self, url=None):
                self.url = url

        async_module.AsyncQdrantClient = AsyncQdrantClient

        sys.modules["qdrant_client"] = qdrant_client
        sys.modules["qdrant_client.async_qdrant_client"] = async_module


def _install_sqlalchemy_stub() -> None:
    try:
        import sqlalchemy  # noqa: F401
        import sqlalchemy.orm  # noqa: F401
    except ModuleNotFoundError:
        sqlalchemy = types.ModuleType("sqlalchemy")

        class Column:
            def __init__(self, column_type=None, *args, **kwargs):
                self.column_type = column_type
                self.args = args
                self.kwargs = kwargs

        def ForeignKey(target, **kwargs):
            return target

        def create_engine(url, future=True):
            return types.SimpleNamespace(url=url)

        sqlalchemy.Column = Column
        sqlalchemy.DateTime = type("DateTime", (), {})
        sqlalchemy.Float = type("Float", (), {})
        sqlalchemy.ForeignKey = ForeignKey
        sqlalchemy.Integer = type("Integer", (), {})
        sqlalchemy.String = type("String", (), {})
        sqlalchemy.Text = type("Text", (), {})
        sqlalchemy.create_engine = create_engine

        orm_module = types.ModuleType("sqlalchemy.orm")

        def declarative_base():
            class Base:
                metadata = types.SimpleNamespace(create_all=lambda engine: None)

            return Base

        def relationship(*args, **kwargs):
            return types.SimpleNamespace(args=args, kwargs=kwargs)

        class _Session:
            def __init__(self):
                self._store = {}

            def add(self, obj):
                key = getattr(obj, "id", None)
                if key is not None:
                    self._store[key] = obj

            def commit(self):
                return None

            def close(self):
                return None

            def get(self, model, key):
                return self._store.get(key)

        class _SessionFactory:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self):
                return _Session()

        def sessionmaker(**kwargs):
            return _SessionFactory(**kwargs)

        orm_module.declarative_base = declarative_base
        orm_module.relationship = relationship
        orm_module.sessionmaker = sessionmaker

        sqlalchemy.orm = orm_module

        sys.modules["sqlalchemy"] = sqlalchemy
        sys.modules["sqlalchemy.orm"] = orm_module


_install_torch_stub()
_install_fastembed_stub()
_install_llama_index_stub()
_install_qdrant_stub()
_install_sqlalchemy_stub()
