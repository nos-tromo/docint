import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@dataclass(frozen=True)
class HostConfig:
    """
    Dataclass for host configuration.
    """

    backend_host: str
    backend_public_host: str
    ollama_host: str
    qdrant_host: str
    cors_allowed_origins: str


@dataclass(frozen=True)
class InformationExtractionConfig:
    """
    Dataclass for information extraction configuration.
    """

    ie_enabled: bool
    ie_max_chars: int


@dataclass(frozen=True)
class ModelConfig:
    """
    Dataclass for model configuration.
    """

    embed_model: str
    sparse_model: str
    gen_model: str
    vision_model: str
    whisper_model: str


@dataclass(frozen=True)
class OllamaConfig:
    """
    Dataclass for Ollama configuration.
    """

    ctx_window: int
    request_timeout: int
    seed: int
    temperature: float
    thinking: bool
    top_k: int
    top_p: float


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


@dataclass(frozen=True)
class RAGConfig:
    """
    Dataclass for RAG (Retrieval-Augmented Generation) configuration.
    """

    ingestion_batch_size: int
    retrieve_top_k: int
    sentence_splitter_chunk_overlap: int
    sentence_splitter_chunk_size: int


def load_host_env() -> HostConfig:
    """
    Loads host configuration from environment variables or defaults.

    Returns:
        HostConfig: Dataclass containing host configuration.
        - backend_host (str): The backend host URL.
        - backend_public_host (str): The public backend host URL.
        - ollama_host (str): The Ollama host URL.
        - qdrant_host (str): The Qdrant host URL.
        - cors_allowed_origins (str): Comma-separated list of allowed CORS origins.
    """
    default_backend_host = "http://localhost:8000"
    default_ollama_host = "http://localhost:11434"
    default_qdrant_host = "http://localhost:6333"
    default_cors_origins = "http://localhost:8501,http://127.0.0.1:8501"

    return HostConfig(
        backend_host=os.getenv("BACKEND_HOST", default_backend_host),
        backend_public_host=os.getenv("BACKEND_PUBLIC_HOST", default_backend_host),
        ollama_host=os.getenv("OLLAMA_HOST", default_ollama_host),
        qdrant_host=os.getenv("QDRANT_HOST", default_qdrant_host),
        cors_allowed_origins=os.getenv("CORS_ALLOWED_ORIGINS", default_cors_origins),
    )


def load_ie_env() -> InformationExtractionConfig:
    """
    Loads information extraction configuration from environment variables or defaults.

    Returns:
        IEConfig: Dataclass containing IE configuration.
        - ie_enabled (bool): Whether to run entity/relation extraction during ingestion.
        - ie_max_chars (int): Maximum characters from each node to send to the extractor.
    """
    default_ie_enabled = "1"
    default_ie_max_chars = "800"

    return InformationExtractionConfig(
        ie_enabled=os.getenv("ENABLE_IE", default_ie_enabled).lower()
        in {"1", "true", "yes"},
        ie_max_chars=int(os.getenv("IE_MAX_CHARS", default_ie_max_chars)),
    )


def load_model_env() -> ModelConfig:
    """
    Loads model configuration from environment variables or defaults.

    Returns:
        ModelConfig: Dataclass containing model configuration.
        - embed_model (str): The embedding model identifier.
        - sparse_model (str): The sparse model identifier.
        - gen_model (str): The generation model identifier.
        - vision_model (str): The vision model identifier.
        - whisper_model (str): The Whisper model identifier.
    """
    default_embed_model = "BAAI/bge-m3"
    default_sparse_model = "Qdrant/all_miniLM_L6_v2_with_attentions"
    default_gen_model = "gpt-oss:20b"
    default_vision_model = "qwen3-vl:8b"
    default_whisper_model = "turbo"

    return ModelConfig(
        embed_model=os.getenv("EMBED_MODEL", default_embed_model),
        sparse_model=os.getenv("SPARSE_MODEL", default_sparse_model),
        gen_model=os.getenv("LLM", default_gen_model),
        vision_model=os.getenv("VLM", default_vision_model),
        whisper_model=os.getenv("WHISPER_MODEL", default_whisper_model),
    )


def load_ollama_env() -> OllamaConfig:
    """
    Loads Ollama configuration from environment variables or defaults.

    Returns:
        OllamaConfig: Dataclass containing Ollama configuration.
        - ctx_window (int): The context window size.
        - request_timeout (int): The request timeout in seconds.
        - seed (int): The random seed for generation.
        - temperature (float): The temperature setting for generation.
        - thinking (bool): Whether to enable thinking mode.
        - top_k (int): The top_k setting for generation.
        - top_p (float): The top_p setting for generation.
    """
    default_ctx_window = "8192"
    default_request_timeout = "1200"
    default_seed = "42"
    default_temperature = "0.0"
    default_thinking = "true"
    default_top_k = "1"
    default_top_p = "0"

    return OllamaConfig(
        ctx_window=int(os.getenv("OLLAMA_CTX_WINDOW", default_ctx_window)),
        request_timeout=int(
            os.getenv("OLLAMA_REQUEST_TIMEOUT", default_request_timeout)
        ),
        seed=int(os.getenv("OLLAMA_SEED", default_seed)),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", default_temperature)),
        thinking=os.getenv("OLLAMA_THINKING", default_thinking).lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        },
        top_k=int(os.getenv("OLLAMA_TOP_K", default_top_k)),
        top_p=float(os.getenv("OLLAMA_TOP_P", default_top_p)),
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
    """
    home_dir = Path.home()
    docint_home_dir: Path = home_dir / "docint"
    default_data_dir: Path = docint_home_dir / "data"
    default_query_dir: Path = docint_home_dir / "queries.txt"
    default_results_dir: Path = docint_home_dir / "results"
    default_hf_hub_cache: Path = home_dir / ".cache" / "huggingface" / "hub"

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
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", default_hf_hub_cache)).expanduser(),
        logs=Path(os.getenv("LOG_PATH", default_log_dir)).expanduser(),
        queries=Path(os.getenv("QUERIES_PATH", default_query_dir)).expanduser(),
        results=Path(os.getenv("RESULTS_PATH", default_results_dir)).expanduser(),
        prompts=default_prompts_dir,
        required_exts=default_exts_dir,
        qdrant_collections=default_qdrant_collections,
        qdrant_sources=default_qdrant_sources,
    )


def load_rag_env() -> RAGConfig:
    """
    Loads RAG (Retrieval-Augmented Generation) configuration from environment variables or defaults.

    Returns:
        RAGConfig: Dataclass containing RAG configuration.
        - ingestion_batch_size (int): The batch size for ingestion.
        - retrieve_top_k (int): The number of top documents to retrieve.
        - sentence_splitter_chunk_overlap (int): The chunk overlap size for sentence splitting.
        - sentence_splitter_chunk_size (int): The chunk size for sentence splitting.
    """
    default_ingestion_batch_size = "5"
    default_retrieve_top_k = "20"
    default_sentence_splitter_chunk_overlap = "64"
    default_sentence_splitter_chunk_size = "1024"

    return RAGConfig(
        ingestion_batch_size=int(
            os.getenv("INGESTION_BATCH_SIZE", default_ingestion_batch_size)
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
    )


def set_offline_env() -> None:
    """
    Sets environment variables to configure Hugging Face libraries for offline mode.

    This function ensures that Hugging Face libraries such as `transformers` and
    `llama_index` operate in offline mode by setting the appropriate environment
    variables. It should be invoked before importing these libraries to avoid
    unexpected behavior.

    Environment Variables Set:
    - `HF_HUB_OFFLINE`: Forces the Hugging Face Hub to operate in offline mode.
    - `TRANSFORMERS_OFFLINE`: Disables online access for the `transformers` library.
    - `HF_HUB_DISABLE_TELEMETRY`: Disables telemetry data collection by Hugging Face.
    - `HF_HUB_DISABLE_PROGRESS_BARS`: Suppresses progress bars in Hugging Face operations.
    - `HF_HUB_DISABLE_SYMLINKS_WARNING`: Disables symlink-related warnings.
    - `KMP_DUPLICATE_LIB_OK`: Resolves potential library duplication issues.

    Note:
    Call this function before importing `transformers` or `llama_index` to ensure
    the offline mode is applied correctly.
    """
    docint_offline = os.getenv("DOCINT_OFFLINE", "1").lower() in {"1", "true", "yes"}

    if docint_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        logger.info("Set Hugging Face libraries to offline mode.")
    else:
        logger.info("Hugging Face libraries are in online mode.")
