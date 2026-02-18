import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def set_offline_env() -> None:
    """
    Log the current offline mode status.

    The actual env vars (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, etc.) are set at
    module level immediately after ``load_dotenv()`` so they are available before
    ``huggingface_hub`` / ``transformers`` cache their values at import time.
    This function re-applies them (idempotent) and emits a log message.
    """
    if os.getenv("DOCINT_OFFLINE", "1").lower() in {"1", "true", "yes"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Point fastembed at the HF hub cache so it reuses models downloaded
        # by ``model_cfg.py`` instead of trying to fetch them.
        if not os.getenv("FASTEMBED_CACHE_PATH"):
            default_hf_cache = str(Path.home() / ".cache" / "huggingface" / "hub")
            os.environ["FASTEMBED_CACHE_PATH"] = os.getenv(
                "HF_HUB_CACHE", default_hf_cache
            )

        logger.info("Set Hugging Face libraries to offline mode.")
    else:
        logger.info("Hugging Face libraries are in online mode.")


set_offline_env()  # Apply offline settings at module load time


def resolve_hf_cache_path(
    cache_dir: Path, repo_id: str, filename: str | None = None
) -> Path | None:
    """
    Resolve a HuggingFace model or file path from the local HF cache.

    HF hub stores downloads under:
        {cache_dir}/models--{org}--{repo}/snapshots/{commit_hash}/

    Args:
        cache_dir: The HF hub cache directory (e.g. ~/.cache/huggingface/hub).
        repo_id: The HuggingFace repository ID (e.g. "BAAI/bge-m3").
        filename: Optional file name within the snapshot directory. If provided,
                  return the path to that specific file; otherwise return the
                  snapshot directory itself.

    Returns:
        The resolved Path if found, otherwise None.
    """
    model_dir_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_dir = cache_dir / model_dir_name

    if not model_cache_dir.exists():
        return None

    ref_path = model_cache_dir / "refs" / "main"
    if not ref_path.exists():
        return None

    commit_hash = ref_path.read_text().strip()
    snapshot_path = model_cache_dir / "snapshots" / commit_hash

    if filename:
        file_path = snapshot_path / filename
        return file_path if file_path.exists() else None

    return snapshot_path if snapshot_path.exists() else None


@dataclass(frozen=True)
class HostConfig:
    """
    Dataclass for host configuration.
    """

    backend_host: str
    backend_public_host: str
    qdrant_host: str
    cors_allowed_origins: str


@dataclass(frozen=True)
class IngestionConfig:
    coarse_chunk_size: int
    docling_accelerator_num_threads: int
    docstore_batch_size: int
    fine_chunk_overlap: int
    fine_chunk_size: int
    hierarchical_chunking_enabled: bool
    ingestion_batch_size: int
    sentence_splitter_chunk_overlap: int
    sentence_splitter_chunk_size: int
    supported_filetypes: list[str]


@dataclass(frozen=True)
class ModelConfig:
    """
    Dataclass for model configuration.
    """

    embed_model_file: str
    embed_model_repo: str
    ner_model: str
    rerank_model: str
    sparse_model: str
    text_model_file: str
    text_model_repo: str
    vision_model_file: str
    vision_model_repo: str
    whisper_model: str


@dataclass(frozen=True)
class NERConfig:
    """
    Dataclass for information extraction configuration.
    """

    enabled: bool
    max_chars: int
    max_workers: int
    engine: str


@dataclass(frozen=True)
class OpenAIConfig:
    """
    Dataclass for OpenAI-compatible API configuration.
    """

    temperature: float
    max_retries: int
    timeout: float
    reuse_client: bool
    ctx_window: int
    api_key: str
    api_base: str
    inference_server: str


@dataclass(frozen=True)
class PathConfig:
    """
    Dataclass for path configuration.
    """

    data: Path
    logs: Path
    queries: Path
    results: Path
    prompts: Path
    qdrant_collections: Path
    qdrant_sources: Path
    hf_hub_cache: Path
    llama_cpp_cache: Path


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Dataclass for RAG (Retrieval-Augmented Generation) configuration.
    """

    rerank_use_fp16: bool
    retrieve_top_k: int


@dataclass(frozen=True)
class SessionConfig:
    """
    Dataclass for session configuration.
    """

    session_store: str


def load_host_env(
    default_backend_host: str = "http://localhost:8000",
    default_qdrant_host: str = "http://localhost:6333",
    default_cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501",
) -> HostConfig:
    """
    Loads host configuration from environment variables or defaults.

    Args:
        default_backend_host (str): Default backend host URL.
        default_qdrant_host (str): Default Qdrant host URL.
        default_cors_origins (str): Default CORS allowed origins.

    Returns:
        HostConfig: Dataclass containing host configuration.
        - backend_host (str): The backend host URL.
        - backend_public_host (str): The public backend host URL. Required to enable document preview features
            in the Docker environment.
        - qdrant_host (str): The Qdrant host URL.
        - cors_allowed_origins (str): Comma-separated list of allowed CORS origins.
    """
    return HostConfig(
        backend_host=os.getenv("BACKEND_HOST", default_backend_host),
        backend_public_host=os.getenv("BACKEND_PUBLIC_HOST", default_backend_host),
        qdrant_host=os.getenv("QDRANT_HOST", default_qdrant_host),
        cors_allowed_origins=os.getenv("CORS_ALLOWED_ORIGINS", default_cors_origins),
    )


def load_ingestion_env(
    default_coarse_chunk_size: int = 8192,
    default_docling_accelerator_num_threads: int = 4,
    default_docstore_batch_size: int = 100,
    default_fine_chunk_overlap: int = 0,
    default_fine_chunk_size: int = 8192,
    default_hierarchical_chunking_enabled: bool = True,
    default_ingestion_batch_size: int = 5,
    default_sentence_splitter_chunk_overlap: int = 64,
    default_sentence_splitter_chunk_size: int = 1024,
    default_supported_filetypes: list[str] = [
        ".avi",
        ".csv",
        ".docx",
        ".flv",
        ".gif",
        ".jpeg",
        ".jpg",
        ".jsonl",
        ".md",
        ".mkv",
        ".mov",
        ".mpeg",
        ".mpg",
        ".mp3",
        ".mp4",
        ".m4a",
        ".m4v",
        ".ogg",
        ".parquet",
        ".pdf",
        ".png",
        ".tsv",
        ".txt",
        ".wav",
        ".webm",
        ".wmv",
        ".xls",
        ".xlsx",
    ],
) -> IngestionConfig:
    """
    Loads ingestion configuration from environment variables or defaults.

    Returns:
        IngestionConfig: Dataclass containing ingestion configuration.
        - coarse_chunk_size (int): The coarse chunk size for hierarchical chunking.
        - docling_accelerator_num_threads (int): The default number of threads for Docling accelerator.
        - docstore_batch_size (int): The batch size for document store operations.
        - fine_chunk_overlap (int): The fine chunk overlap size for hierarchical chunking.
        - fine_chunk_size (int): The fine chunk size for hierarchical chunking.
        - hierarchical_chunking_enabled (bool): Whether hierarchical chunking is enabled.
        - ingestion_batch_size (int): The batch size for ingestion.
        - sentence_splitter_chunk_overlap (int): The chunk overlap size for sentence splitting.
        - sentence_splitter_chunk_size (int): The chunk size for sentence splitting.
        - supported_filetypes (list[str]): List of supported file extensions for ingestion.
    """
    return IngestionConfig(
        coarse_chunk_size=int(
            os.getenv("COARSE_CHUNK_SIZE", default_coarse_chunk_size)
        ),
        docling_accelerator_num_threads=int(
            os.getenv(
                "DOCLING_ACCELERATOR_NUM_THREADS",
                default_docling_accelerator_num_threads,
            )
        ),
        docstore_batch_size=int(
            os.getenv("DOCSTORE_BATCH_SIZE", default_docstore_batch_size)
        ),
        fine_chunk_overlap=int(
            os.getenv("FINE_CHUNK_OVERLAP", default_fine_chunk_overlap)
        ),
        fine_chunk_size=int(os.getenv("FINE_CHUNK_SIZE", default_fine_chunk_size)),
        hierarchical_chunking_enabled=str(
            os.getenv("HIERARCHICAL_CHUNKING_ENABLED", "true")
        ).lower()
        in {"true", "1", "yes"},
        ingestion_batch_size=int(
            os.getenv("INGESTION_BATCH_SIZE", default_ingestion_batch_size)
        ),
        sentence_splitter_chunk_overlap=int(
            os.getenv(
                "SENTENCE_SPLITTER_CHUNK_OVERLAP",
                default_sentence_splitter_chunk_overlap,
            )
        ),
        sentence_splitter_chunk_size=int(
            os.getenv(
                "SENTENCE_SPLITTER_CHUNK_SIZE", default_sentence_splitter_chunk_size
            )
        ),
        supported_filetypes=default_supported_filetypes,
    )


def load_model_env(
    default_embed_model_str: str = "ggml-org/bge-m3-Q8_0-GGUF;bge-m3-q8_0.gguf",
    default_ner_model: str = "gliner-community/gliner_large-v2.5",
    default_rerank_model: str = "BAAI/bge-reranker-v2-m3",
    default_sparse_model: str = "Qdrant/all_miniLM_L6_v2_with_attentions",
    default_text_model_str: str = "unsloth/Qwen3-1.7B-GGUF;Qwen3-1.7B-Q4_K_M.gguf",
    default_vision_model_str: str = "Qwen/Qwen3-VL-8B-Instruct-GGUF;Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    default_whisper_model: str = "turbo",
) -> ModelConfig:
    """
    Loads model configuration from environment variables or defaults.

    Args:
        default_embed_model_str (str): Default embedding model identifier.
        default_ner_model (str): Default NER model identifier.
        default_rerank_model (str): Default reranker model identifier.
        default_sparse_model (str): Default sparse model identifier.
        default_text_model_str (str): Default text model identifier.
        default_vision_model_str (str): Default vision model identifier.
        default_whisper_model (str): Default Whisper model identifier.

    Returns:
        ModelConfig: Dataclass containing model configuration.
        - embed_model_file (str): The embedding model file name.
        - embed_model_repo (str): The embedding model HuggingFace repo ID for cache resolution
        - ner_model (str): The NER model identifier.
        - rerank_model (str): The reranker model identifier.
        - sparse_model (str): The sparse model identifier.
        - text_model_file (str): The text model file name.
        - text_model_repo (str): The text model HuggingFace repo ID for cache resolution
        - vision_model_file (str): The vision model file name.
        - vision_model_repo (str): The vision model HuggingFace repo ID for cache resolution
        - whisper_model (str): The Whisper model identifier.
    """

    def resolve_model_name(model_str: str) -> tuple[str, str]:
        """
        Resolve a model string into its repo ID and file name components.

        The model string can be in the format "repo_id;file_name" (required for llama.cpp) or just "model_name".
        If only "model_name" is provided, it is treated as both the repo ID and file name.

        Args:
            model_str (str): The model string to resolve.

        Returns:
            tuple[str, str] | str: A tuple of (repo_id, file_name) if the input contains a semicolon.
        """
        if ";" in model_str:
            repo_id, file_name = model_str.split(";", 1)
            return repo_id.strip(), file_name.strip()
        else:
            return model_str.strip(), model_str.strip()

    embed_model_repo, embed_model_file = resolve_model_name(
        os.getenv("EMBED_MODEL", default_embed_model_str)
    )
    text_model_repo, text_model_file = resolve_model_name(
        os.getenv("LLM", default_text_model_str)
    )
    vision_model_repo, vision_model_file = resolve_model_name(
        os.getenv("VLM", default_vision_model_str)
    )

    return ModelConfig(
        embed_model_file=embed_model_file,
        embed_model_repo=embed_model_repo,
        ner_model=os.getenv("NER_MODEL", default_ner_model),
        rerank_model=os.getenv("RERANK_MODEL", default_rerank_model),
        sparse_model=os.getenv("SPARSE_MODEL", default_sparse_model),
        text_model_file=text_model_file,
        text_model_repo=text_model_repo,
        vision_model_file=vision_model_file,
        vision_model_repo=vision_model_repo,
        whisper_model=os.getenv("WHISPER_MODEL", default_whisper_model),
    )


def load_ner_env(
    default_enabled: bool = True,
    default_max_chars: int = 1024,
    default_max_workers: int = 4,
    default_engine: str = "gliner",
) -> NERConfig:
    """
    Loads information extraction configuration from environment variables or defaults.

    Args:
        default_enabled (bool): Default value to enable NER extraction. Set to True to enable by default.
        default_max_chars (int): Default maximum characters for NER extraction.
        default_max_workers (int): Default maximum worker threads for NER extraction.
        default_engine (str): Default NER engine to use. Options: gliner, llm.

    Returns:
        NERConfig: Dataclass containing NER configuration.
        - enabled (bool): Whether to run entity/relation extraction during ingestion.
        - max_chars (int): Maximum characters from each node to send to the extractor.
        - max_workers (int): Maximum number of worker threads for NER extraction.
        - engine (str): The NER engine to use. Options: gliner, llm.

    Raises:
        ValueError: If an unsupported NER engine is specified.
    """
    engine = os.getenv("NER_ENGINE", default_engine).lower()
    if engine not in {"gliner", "llm"}:
        raise ValueError(
            f"Unsupported NER engine: {engine}. Supported options are: 'gliner', 'llm'."
        )

    return NERConfig(
        enabled=str(os.getenv("ENABLE_NER", default_enabled)).lower()
        in {"true", "1", "yes"},
        max_chars=int(os.getenv("NER_MAX_CHARS", default_max_chars)),
        max_workers=int(os.getenv("NER_MAX_WORKERS", default_max_workers)),
        engine=engine,
    )


def load_openai_env(
    default_temperature: float = 0.1,
    default_max_retries: int = 2,
    default_timeout: float = 60.0,
    default_reuse_client: bool = False,
    default_ctx_window: int = 32768,
    default_api_key: str = "sk-no-key-required",
    default_api_base: str = "http://localhost:8080/v1",
    default_inference_server: str = "llama.cpp",
) -> OpenAIConfig:
    """
    Loads OpenAI configuration from environment variables or defaults.

    Args:
        default_temperature (float): Default temperature for text generation.
        default_max_retries (int): Default number of retries.
        default_timeout (float): Default timeout in seconds.
        default_reuse_client (bool): Whether to reuse the OpenAI client across calls. Default is False.
        default_ctx_window (int): Default context window size for models that support it.
        default_api_key (str): Default OpenAI API key.
        default_api_base (str): Default OpenAI API base URL.
        default_inference_server (str): Default inference server type (e.g. "llama.cpp", "ollama", "openai", "vllm"). Default is "llama.cpp".

    Returns:
        OpenAIConfig: Dataclass containing OpenAI configuration.

    Raises:
        ValueError: If an unsupported inference server is specified.
    """
    inference_server = os.getenv("INFERENCE_SERVER", default_inference_server).lower()
    if inference_server not in {
        "llama.cpp",
        "llama_cpp",
        "llamacpp",
        "ollama",
        "openai",
        "vllm",
    }:
        raise ValueError(
            f"Unsupported inference server: {inference_server}. "
            f"Supported options are: 'ollama', 'llama.cpp', 'openai', 'vllm'."
        )

    return OpenAIConfig(
        temperature=float(os.getenv("OPENAI_TEMPERATURE", default_temperature)),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", default_max_retries)),
        timeout=float(os.getenv("OPENAI_TIMEOUT", default_timeout)),
        reuse_client=str(os.getenv("OPENAI_REUSE_CLIENT", default_reuse_client)).lower()
        in {"true", "1", "yes"},
        ctx_window=int(os.getenv("OPENAI_CTX_WINDOW", default_ctx_window)),
        api_key=os.getenv("OPENAI_API_KEY", default_api_key),
        api_base=os.getenv("OPENAI_API_BASE", default_api_base),
        inference_server=inference_server,
    )


def load_path_env() -> PathConfig:
    """
    Loads path configuration from environment variables or defaults.

    Returns:
        PathConfig: Dataclass containing path configuration.
        - data (Path): Path to the data directory.
        - logs (Path): Path to the logs file.
        - queries (Path): Path to the queries file.
        - results (Path): Path to the results directory.
        - prompts (Path): Path to the prompts directory.
        - qdrant_collections (Path): Path to the Qdrant collections directory.
        - qdrant_sources (Path): Path to the Qdrant sources directory.
        - hf_hub_cache (Path): Path to the Hugging Face Hub cache directory.
        - llama_cpp_cache (Path): Path to the llama.cpp cache directory.
    """
    home_dir: Path = Path.home()
    docint_home_dir: Path = home_dir / "docint"
    default_data_dir: Path = docint_home_dir / "data"
    default_query_dir: Path = docint_home_dir / "queries.txt"
    default_results_dir: Path = docint_home_dir / "results"
    default_model_cache: Path = home_dir / ".cache"
    default_hf_hub_cache: Path = default_model_cache / "huggingface" / "hub"
    default_llama_cpp_cache: Path = default_model_cache / "llama.cpp"

    utils_dir: Path = Path(__file__).parent.resolve()
    default_prompts_dir: Path = utils_dir / "prompts"
    project_root: Path = utils_dir.parents[1]
    default_log_dir = project_root / ".logs" / "docint.log"

    default_qdrant_collections = Path(
        os.getenv("QDRANT_COL_DIR", "qdrant_storage")
    ).expanduser()
    qdrant_sources_env = os.getenv("QDRANT_SRC_DIR")

    # Default sources root alongside Qdrant storage; fall back to a "sources" sibling.
    if qdrant_sources_env:
        default_qdrant_sources = Path(qdrant_sources_env).expanduser()
    else:
        default_sources_base = (
            default_qdrant_collections.parent
            if default_qdrant_collections.parent != Path(".")
            else default_qdrant_collections
        )
        default_qdrant_sources = (default_sources_base / "sources").expanduser()

    return PathConfig(
        data=Path(os.getenv("DATA_PATH", default_data_dir)).expanduser(),
        logs=Path(os.getenv("LOG_PATH", default_log_dir)).expanduser(),
        queries=Path(os.getenv("QUERIES_PATH", default_query_dir)).expanduser(),
        results=Path(os.getenv("RESULTS_PATH", default_results_dir)).expanduser(),
        prompts=default_prompts_dir,
        qdrant_collections=default_qdrant_collections,
        qdrant_sources=default_qdrant_sources,
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", default_hf_hub_cache)).expanduser(),
        llama_cpp_cache=Path(
            os.getenv("LLAMA_CPP_CACHE", default_llama_cpp_cache)
        ).expanduser(),
    )


def load_retrieval_env(
    default_rerank_use_fp16: bool = False,
    default_retrieve_top_k: int = 20,
) -> RetrievalConfig:
    """
    Loads retrieval configuration from environment variables or defaults.

    Args:
        default_rerank_use_fp16 (bool): Default flag to use FP16 for reranker model. Default is False.
        default_retrieve_top_k (int): Default number of top documents to retrieve.

    Returns:
        RetrievalConfig: Dataclass containing retrieval configuration.
        - rerank_use_fp16 (bool): Whether to use FP16 for the reranker model.
        - retrieve_top_k (int): The number of top documents to retrieve for RAG
    """
    return RetrievalConfig(
        rerank_use_fp16=str(
            os.getenv("RERANK_USE_FP16", default_rerank_use_fp16)
        ).lower()
        in {"true", "1", "yes"},
        retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", default_retrieve_top_k)),
    )


def load_session_env(
    default_session_store: str = "sqlite:///sessions.db",
) -> SessionConfig:
    """
    Loads session configuration from environment variables or defaults.

    Args:
        default_session_store (str): Default session store configuration (e.g. database URL or file path).
            Default is "sqlite:///sessions.db".

    Returns:
        SessionConfig: Dataclass containing session configuration.
        - session_store (str): The session store configuration. Default is "sqlite:///sessions.db".
    """
    return SessionConfig(
        session_store=os.getenv("SESSION_STORE", default_session_store)
    )
