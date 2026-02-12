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
class InformationExtractionConfig:
    """
    Dataclass for information extraction configuration.
    """

    ie_enabled: bool
    ie_max_chars: int
    ie_max_workers: int
    ie_engine: str


@dataclass(frozen=True)
class LlamaCppConfig:
    """
    Dataclass for Llama.cpp configuration.
    """

    ctx_window: int
    max_new_tokens: int
    request_timeout: int
    seed: int
    temperature: float
    n_gpu_layers: int
    top_k: int
    top_p: float
    repeat_penalty: float


@dataclass(frozen=True)
class ModelConfig:
    """
    Dataclass for model configuration.
    """

    embed_model: str
    sparse_model: str
    ner_model: str
    rerank_model: str
    llm: str
    llm_file: str
    llm_tokenizer: str
    vlm: str
    vlm_file: str
    whisper_model: str


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
    required_exts: Path
    hf_hub_cache: Path
    llama_cpp_cache: Path


@dataclass(frozen=True)
class RAGConfig:
    """
    Dataclass for RAG (Retrieval-Augmented Generation) configuration.
    """

    docstore_batch_size: int
    ingestion_batch_size: int
    docling_accelerator_num_threads: int
    retrieve_top_k: int
    sentence_splitter_chunk_overlap: int
    sentence_splitter_chunk_size: int
    hierarchical_chunking_enabled: bool
    coarse_chunk_size: int
    fine_chunk_size: int
    fine_chunk_overlap: int
    rerank_use_fp16: bool


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


def load_ie_env(
    default_ie_enabled: bool = True,
    default_ie_max_chars: int = 800,
    default_ie_max_workers: int = 4,
    default_ie_engine: str = "gliner",
) -> InformationExtractionConfig:
    """
    Loads information extraction configuration from environment variables or defaults.

    Args:
        default_ie_enabled (bool): Default value to enable IE extraction. Set to True to enable by default.
        default_ie_max_chars (int): Default maximum characters for IE extraction.
        default_ie_max_workers (int): Default maximum worker threads for IE extraction.
        default_ie_engine (str): Default IE engine to use. Options: gliner, llama_cpp.

    Returns:
        IEConfig: Dataclass containing IE configuration.
        - ie_enabled (bool): Whether to run entity/relation extraction during ingestion.
        - ie_max_chars (int): Maximum characters from each node to send to the extractor.
        - ie_max_workers (int): Maximum number of worker threads for IE extraction.
    """
    return InformationExtractionConfig(
        ie_enabled=str(os.getenv("ENABLE_IE", default_ie_enabled)).lower()
        in {"true", "1", "yes"},
        ie_max_chars=int(os.getenv("IE_MAX_CHARS", default_ie_max_chars)),
        ie_max_workers=int(os.getenv("IE_MAX_WORKERS", default_ie_max_workers)),
        ie_engine=os.getenv("IE_ENGINE", default_ie_engine).lower(),
    )


def load_llama_cpp_env(
    default_ctx_window: int = 8192,
    default_max_new_tokens: int = 1024,
    default_request_timeout: int = 1200,
    default_seed: int = 42,
    default_temperature: float = 0.1,
    default_n_gpu_layers: int = -1,
    default_top_k: int = 40,
    default_top_p: float = 0.95,
    default_repeat_penalty: float = 1.1,
) -> LlamaCppConfig:
    """
    Loads Llama.cpp configuration from environment variables or defaults.

    Args:
        default_ctx_window (int): Default context window size.
        default_max_new_tokens (int): Default maximum new tokens per completion.
        default_request_timeout (int): Default request timeout in seconds.
        default_seed (int): Default random seed for generation.
        default_temperature (float): Default temperature setting for generation.
        default_n_gpu_layers (int): Default number of layers to offload to GPU (-1 = all).
        default_top_k (int): Default top_k setting for generation.
        default_top_p (float): Default top_p setting for generation.
        default_repeat_penalty (float): Default repetition penalty for generation.

    Returns:
        LlamaCppConfig: Dataclass containing Llama.cpp configuration.
        - ctx_window (int): The context window size.
        - max_new_tokens (int): Maximum number of tokens to generate per completion.
        - request_timeout (int): The request timeout in seconds.
        - seed (int): The random seed for generation.
        - temperature (float): The temperature setting for generation.
        - n_gpu_layers (int): Number of layers to offload to GPU (-1 = all).
        - top_k (int): The top_k setting for generation.
        - top_p (float): The top_p setting for generation.
        - repeat_penalty (float): The repetition penalty for generation.
    """
    return LlamaCppConfig(
        ctx_window=int(os.getenv("LLAMA_CPP_CTX_WINDOW", default_ctx_window)),
        max_new_tokens=int(
            os.getenv("LLAMA_CPP_MAX_NEW_TOKENS", default_max_new_tokens)
        ),
        request_timeout=int(
            os.getenv("LLAMA_CPP_REQUEST_TIMEOUT", default_request_timeout)
        ),
        seed=int(os.getenv("LLAMA_CPP_SEED", default_seed)),
        temperature=float(os.getenv("LLAMA_CPP_TEMPERATURE", default_temperature)),
        n_gpu_layers=int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", default_n_gpu_layers)),
        top_k=int(os.getenv("LLAMA_CPP_TOP_K", default_top_k)),
        top_p=float(os.getenv("LLAMA_CPP_TOP_P", default_top_p)),
        repeat_penalty=float(
            os.getenv("LLAMA_CPP_REPEAT_PENALTY", default_repeat_penalty)
        ),
    )


def load_model_env(
    default_embed_model: str = "BAAI/bge-m3",
    default_sparse_model: str = "Qdrant/all_miniLM_L6_v2_with_attentions",
    default_ner_model: str = "gliner-community/gliner_large-v2.5",
    default_rerank_model: str = "BAAI/bge-reranker-v2-m3",
    default_llm: str = "unsloth/Qwen3-1.7B-GGUF",
    default_llm_file: str = "Qwen3-1.7B-Q4_K_M.gguf",
    default_llm_tokenizer: str = "Qwen/Qwen3-1.7B",
    default_vlm: str = "Qwen/Qwen3-VL-8B-Instruct-GGUF",
    default_vlm_file: str = "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    default_whisper_model: str = "turbo",
) -> ModelConfig:
    """
    Loads model configuration from environment variables or defaults.

    Args:
        default_embed_model (str): Default embedding model identifier.
        default_sparse_model (str): Default sparse model identifier.
        default_ner_model (str): Default NER model identifier.
        default_rerank_model (str): Default reranker model identifier.
        default_llm (str): Default LLM (Language Model) identifier for generation.
        default_llm_file (str): Default local file name for the LLM model (GGUF format).
        default_llm_tokenizer (str): Default HuggingFace repo for the LLM tokenizer.
            Used by apply_chat_template() to format prompts. Leave empty to auto-detect.
        default_vlm (str): Default VLM (Vision-Language Model) identifier for generation.
        default_vlm_file (str): Default local file name for the VLM model (GGUF format).
        default_whisper_model (str): Default Whisper model identifier.

    Returns:
        ModelConfig: Dataclass containing model configuration.
        - embed_model (str): The embedding model identifier.
        - sparse_model (str): The sparse model identifier.
        - ner_model (str): The NER model identifier.
        - rerank_model (str): The reranker model identifier.
        - llm (str): The LLM (Language Model) identifier for generation.
        - llm_file (str): The local file name for the LLM model (GGUF format).
        - llm_tokenizer (str): HuggingFace repo for the LLM tokenizer.
        - vlm (str): The VLM (Vision-Language Model) identifier for generation.
        - vlm_file (str): The local file name for the VLM model (GGUF format).
        - whisper_model (str): The Whisper model identifier.
    """
    return ModelConfig(
        embed_model=os.getenv("EMBED_MODEL", default_embed_model),
        sparse_model=os.getenv("SPARSE_MODEL", default_sparse_model),
        ner_model=os.getenv("NER_MODEL", default_ner_model),
        rerank_model=os.getenv("RERANK_MODEL", default_rerank_model),
        llm=os.getenv("LLM", default_llm),
        llm_file=os.getenv("LLM_FILE", default_llm_file),
        llm_tokenizer=os.getenv("LLM_TOKENIZER", default_llm_tokenizer),
        vlm=os.getenv("VLM", default_vlm),
        vlm_file=os.getenv("VLM_FILE", default_vlm_file),
        whisper_model=os.getenv("WHISPER_MODEL", default_whisper_model),
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
        - required_exts (Path): Path to the required extensions file.
        - hf_hub_cache (Path): Path to the Hugging Face Hub cache directory.
        - llama_cpp_cache (Path): Path to the Llama.cpp cache directory.
    """
    home_dir: Path = Path.home()
    docint_home_dir: Path = home_dir / "docint"
    default_data_dir: Path = docint_home_dir / "data"
    default_query_dir: Path = docint_home_dir / "queries.txt"
    default_results_dir: Path = docint_home_dir / "results"
    default_cache_dir: Path = home_dir / ".cache"
    default_hf_hub_cache: Path = default_cache_dir / "huggingface" / "hub"
    default_llama_cpp_cache: Path = default_cache_dir / "llama.cpp"

    project_root: Path = Path(__file__).parents[2].resolve()
    default_log_dir = project_root / ".logs" / "docint.log"
    utils_dir: Path = project_root / "docint" / "utils"
    default_prompts_dir: Path = utils_dir / "prompts"
    default_exts_dir: Path = utils_dir / "required_exts.txt"

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
        required_exts=default_exts_dir,
        qdrant_collections=default_qdrant_collections,
        qdrant_sources=default_qdrant_sources,
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", default_hf_hub_cache)).expanduser(),
        llama_cpp_cache=Path(
            os.getenv("LLAMA_CPP_CACHE", default_llama_cpp_cache)
        ).expanduser(),
    )


def load_rag_env(
    default_docstore_batch_size: int = 100,
    default_ingestion_batch_size: int = 5,
    default_docling_accelerator_num_threads: int = 4,
    default_retrieve_top_k: int = 20,
    default_sentence_splitter_chunk_overlap: int = 64,
    default_sentence_splitter_chunk_size: int = 1024,
    default_hierarchical_chunking_enabled: bool = True,
    default_coarse_chunk_size: int = 8192,
    default_fine_chunk_size: int = 8192,
    default_fine_chunk_overlap: int = 0,
    default_rerank_use_fp16: bool = False,
) -> RAGConfig:
    """
    Loads RAG (Retrieval-Augmented Generation) configuration from environment variables or defaults.

    Args:
        default_docstore_batch_size (int): Default batch size for document store operations.
        default_ingestion_batch_size (int): Default batch size for ingestion.
        default_docling_accelerator_num_threads (int): Default number of threads for Docling accelerator.
        default_retrieve_top_k (int): Default number of top documents to retrieve.
        default_sentence_splitter_chunk_overlap (int): Default chunk overlap size for sentence splitting.
        default_sentence_splitter_chunk_size (int): Default chunk size for sentence splitting.
        default_hierarchical_chunking_enabled (bool): Default flag to enable hierarchical chunking.
        default_coarse_chunk_size (int): Default coarse chunk size for hierarchical chunking.
        default_fine_chunk_size (int): Default fine chunk size for hierarchical chunking.
        default_fine_chunk_overlap (int): Default fine chunk overlap size for hierarchical chunking.
        default_rerank_use_fp16 (bool): Default flag to use FP16 for reranker model. Default is False.

    Returns:
        RAGConfig: Dataclass containing RAG configuration.
        - docstore_batch_size (int): The batch size for document store operations.
        - ingestion_batch_size (int): The batch size for ingestion.
        - docling_accelerator_num_threads (int): The default number of threads for Docling accelerator.
        - retrieve_top_k (int): The number of top documents to retrieve.
        - sentence_splitter_chunk_overlap (int): The chunk overlap size for sentence splitting.
        - sentence_splitter_chunk_size (int): The chunk size for sentence splitting.
        - hierarchical_chunking_enabled (bool): Whether hierarchical chunking is enabled.
        - coarse_chunk_size (int): The coarse chunk size for hierarchical chunking.
        - fine_chunk_size (int): The fine chunk size for hierarchical chunking.
        - fine_chunk_overlap (bool): The fine chunk overlap size for hierarchical chunking.
    """
    return RAGConfig(
        docstore_batch_size=int(
            os.getenv("DOCSTORE_BATCH_SIZE", default_docstore_batch_size)
        ),
        ingestion_batch_size=int(
            os.getenv("INGESTION_BATCH_SIZE", default_ingestion_batch_size)
        ),
        docling_accelerator_num_threads=int(
            os.getenv(
                "DOCLING_ACCELERATOR_NUM_THREADS",
                default_docling_accelerator_num_threads,
            )
        ),
        retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", default_retrieve_top_k)),
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
        hierarchical_chunking_enabled=str(
            os.getenv(
                "HIERARCHICAL_CHUNKING_ENABLED", default_hierarchical_chunking_enabled
            )
        ).lower()
        in {"true", "1", "yes"},
        coarse_chunk_size=int(
            os.getenv("COARSE_CHUNK_SIZE", default_coarse_chunk_size)
        ),
        fine_chunk_size=int(os.getenv("FINE_CHUNK_SIZE", default_fine_chunk_size)),
        fine_chunk_overlap=int(
            os.getenv("FINE_CHUNK_OVERLAP", default_fine_chunk_overlap)
        ),
        rerank_use_fp16=str(
            os.getenv("RERANK_USE_FP16", default_rerank_use_fp16)
        ).lower()
        in {"true", "1", "yes"},
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
