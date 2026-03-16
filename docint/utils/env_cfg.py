import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def set_offline_env() -> None:
    """Log the current offline mode status.

    The actual env vars (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, etc.) are set at
    module level immediately after ``load_dotenv()`` so they are available before
    ``huggingface_hub`` / ``transformers`` cache their values at import time.
    This function re-applies them (idempotent) and emits a log message.
    """
    if str(os.getenv("DOCINT_OFFLINE", "1")).lower() in {"1", "true", "yes"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
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
    """Resolve a HuggingFace model or file path from the local HF cache.

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
class GraphRAGConfig:
    """Dataclass for graph-assisted retrieval configuration."""

    enabled: bool
    neighbor_hops: int
    top_k_nodes: int
    min_edge_weight: int
    max_neighbors: int


def load_graphrag_env(
    default_enabled: bool = True,
    default_neighbor_hops: int = 1,
    default_top_k_nodes: int = 100,
    default_min_edge_weight: int = 1,
    default_max_neighbors: int = 6,
) -> GraphRAGConfig:
    """Load GraphRAG configuration from environment variables.

    Args:
        default_enabled: Whether graph-assisted retrieval is enabled by default.
        default_neighbor_hops: Number of graph hops used for query expansion.
        default_top_k_nodes: Maximum number of graph nodes kept in memory.
        default_min_edge_weight: Minimum edge weight used for graph filtering.
        default_max_neighbors: Maximum number of neighbor entities appended to a query.

    Returns:
        GraphRAGConfig: Parsed graph retrieval settings.
    """
    return GraphRAGConfig(
        enabled=str(os.getenv("GRAPHRAG_ENABLED", default_enabled)).lower()
        in {"true", "1", "yes"},
        neighbor_hops=max(
            1, int(os.getenv("GRAPHRAG_NEIGHBOR_HOPS", default_neighbor_hops))
        ),
        top_k_nodes=max(1, int(os.getenv("GRAPHRAG_TOP_K_NODES", default_top_k_nodes))),
        min_edge_weight=max(
            1, int(os.getenv("GRAPHRAG_MIN_EDGE_WEIGHT", default_min_edge_weight))
        ),
        max_neighbors=max(
            1, int(os.getenv("GRAPHRAG_MAX_NEIGHBORS", default_max_neighbors))
        ),
    )


@dataclass(frozen=True)
class HateSpeechConfig:
    """Dataclass for hate-speech detection configuration."""

    enabled: bool
    max_chars: int
    max_workers: int


def load_hate_speech_env(
    default_enabled: bool = False,
    default_max_chars: int = 2048,
    default_max_workers: int = 1,
) -> HateSpeechConfig:
    """Load hate-speech detection settings from environment variables.

    Args:
        default_enabled: Whether hate-speech detection runs during ingestion.
        default_max_chars: Maximum characters from each chunk sent to the detector.
        default_max_workers: Maximum worker threads for parallel hate-speech detection.

    Returns:
        HateSpeechConfig: Parsed hate-speech detection configuration.
    """
    return HateSpeechConfig(
        enabled=str(os.getenv("ENABLE_HATE_SPEECH_DETECTION", default_enabled)).lower()
        in {"true", "1", "yes"},
        max_chars=max(256, int(os.getenv("HATE_SPEECH_MAX_CHARS", default_max_chars))),
        max_workers=max(
            1, int(os.getenv("HATE_SPEECH_MAX_WORKERS", default_max_workers))
        ),
    )


@dataclass(frozen=True)
class HostConfig:
    """Dataclass for host configuration."""

    backend_host: str
    backend_public_host: str
    qdrant_host: str
    cors_allowed_origins: str


def load_host_env(
    default_backend_host: str = "http://localhost:8000",
    default_qdrant_host: str = "http://localhost:6333",
    default_cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501",
) -> HostConfig:
    """Loads host configuration from environment variables or defaults.

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


@dataclass(frozen=True)
class ImageIngestionConfig:
    """Configuration for image ingestion and image-vector indexing."""

    enabled: bool
    embedding_enabled: bool
    tagging_enabled: bool
    collection_name: str
    vector_name: str
    cache_by_hash: bool
    fail_on_embedding_error: bool
    fail_on_tagging_error: bool
    retrieve_top_k: int
    tagging_max_image_dimension: int = 1024


def load_image_ingestion_config(
    default_image_ingestion_enabled: bool = True,
    default_image_embedding_enabled: bool = True,
    default_image_tagging_enabled: bool = True,
    default_image_qdrant_collection: str = "{collection}_images",
    default_image_qdrant_vector_name: str = "image-dense",
    default_image_cache_by_hash: bool = True,
    default_fail_on_embedding_error: bool = False,
    default_fail_on_tagging_error: bool = False,
    default_retrieve_top_k: int = 5,
    default_tagging_max_image_dimension: int = 1024,
) -> ImageIngestionConfig:
    """Load image ingestion settings from environment variables.

    Args:
        default_image_ingestion_enabled (bool): Whether image ingestion is enabled by default.
        default_image_embedding_enabled (bool): Whether to generate embeddings for images by default.
        default_image_tagging_enabled (bool): Whether to generate tags for images by default.
        default_image_qdrant_collection (str): Default Qdrant collection name for storing image vectors.
        default_image_qdrant_vector_name (str): Default name of the vector field in Qdrant for image vectors.
        default_image_cache_by_hash (bool): Whether to cache image embeddings by hash to avoid redundant computation. Default is True.
        default_fail_on_embedding_error (bool): Whether to fail the entire ingestion if image embedding fails. Default is False.
        default_fail_on_tagging_error (bool): Whether to fail the entire ingestion if image tagging fails. Default is False.
        default_retrieve_top_k (int): The number of top image matches to retrieve for a text query. Default is 5.
        default_tagging_max_image_dimension (int): Maximum pixel dimension (width or height) for
            images sent to the vision tagging endpoint. Larger images are down-scaled. Default is 1024.

    Returns:
        ImageIngestionConfig: Dataclass containing image ingestion configuration.
        - enabled (bool): Whether image ingestion is enabled.
        - embedding_enabled (bool): Whether to generate embeddings for images.
        - tagging_enabled (bool): Whether to generate tags for images.
        - collection_name (str): The Qdrant collection name for storing image vectors.
        - vector_name (str): The name of the vector field in Qdrant.
        - cache_by_hash (bool): Whether to cache image embeddings by hash to avoid redundant computation.
        - fail_on_embedding_error (bool): Whether to fail the entire ingestion if image embedding fails.
        - fail_on_tagging_error (bool): Whether to fail the entire ingestion if image tagging fails.
        - retrieve_top_k (int): The number of top image matches to retrieve for a text query.
        - tagging_max_image_dimension (int): Maximum pixel dimension (width or height) for
            images sent to the vision tagging endpoint. Larger images are down-scaled.
    """
    return ImageIngestionConfig(
        enabled=str(
            os.getenv("IMAGE_INGESTION_ENABLED", default_image_ingestion_enabled)
        ).lower()
        in {"1", "true", "yes"},
        embedding_enabled=str(
            os.getenv("IMAGE_EMBEDDING_ENABLED", default_image_embedding_enabled)
        ).lower()
        in {"1", "true", "yes"},
        tagging_enabled=str(
            os.getenv("IMAGE_TAGGING_ENABLED", default_image_tagging_enabled)
        ).lower()
        in {"1", "true", "yes"},
        collection_name=os.getenv(
            "IMAGE_QDRANT_COLLECTION", default_image_qdrant_collection
        ),
        vector_name=os.getenv(
            "IMAGE_QDRANT_VECTOR_NAME", default_image_qdrant_vector_name
        ),
        cache_by_hash=str(
            os.getenv("IMAGE_CACHE_BY_HASH", default_image_cache_by_hash)
        ).lower()
        in {"1", "true", "yes"},
        fail_on_embedding_error=str(
            os.getenv("IMAGE_FAIL_ON_EMBED_ERROR", default_fail_on_embedding_error)
        ).lower()
        in {"1", "true", "yes"},
        fail_on_tagging_error=str(
            os.getenv("IMAGE_FAIL_ON_TAG_ERROR", default_fail_on_tagging_error)
        ).lower()
        in {"1", "true", "yes"},
        retrieve_top_k=int(os.getenv("IMAGE_RETRIEVE_TOP_K", default_retrieve_top_k)),
        tagging_max_image_dimension=int(
            os.getenv(
                "IMAGE_TAGGING_MAX_IMAGE_DIM", default_tagging_max_image_dimension
            )
        ),
    )


@dataclass(frozen=True)
class IngestionConfig:
    """Dataclass for ingestion configuration."""

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
    """Loads ingestion configuration from environment variables or defaults.

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


@dataclass(frozen=True)
class ModelConfig:
    """Dataclass for model configuration."""

    embed_model_file: str
    embed_model_repo: str
    image_embed_model: str
    ner_model: str
    rerank_model: str
    sparse_model: str
    text_model_file: str
    text_model_repo: str
    vision_model_file: str
    vision_model_mmproj_file: str
    vision_model_repo: str
    whisper_model: str


def load_model_env(
    default_embed_model: str = "bge-m3",
    default_image_embed_model: str = "openai/clip-vit-base-patch32",
    default_ner_model: str = "gliner-community/gliner_large-v2.5",
    default_rerank_model: str = "BAAI/bge-reranker-v2-m3",
    default_sparse_model: str = "Qdrant/all_miniLM_L6_v2_with_attentions",
    default_text_model_str: str = "gpt-oss:20b",
    default_vision_model_str: str = "qwen3.5:9b",
    default_whisper_model: str = "turbo",
) -> ModelConfig:
    """Loads model configuration from environment variables or defaults.

    Args:
        default_embed_model(str): Default embedding model identifier for Ollama-compatible embeddings.
        default_image_embed_model (str): Default image embedding model identifier.
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
        - image_embed_model (str): The image embedding model identifier.
        - ner_model (str): The NER model identifier.
        - rerank_model (str): The reranker model identifier.
        - sparse_model (str): The sparse model identifier.
        - text_model_file (str): The text model file name.
        - text_model_repo (str): The text model HuggingFace repo ID for cache resolution
        - vision_model_file (str): The vision model file name.
        - vision_model_mmproj_file (str): The vision model MMProj file name.
        - vision_model_repo (str): The vision model HuggingFace repo ID for cache resolution
        - whisper_model (str): The Whisper model identifier.
    """

    def resolve_model_name(model_str: str) -> tuple[str, str, str]:
        """Resolves a model string into its components: repo, model, and mmproj.

        Args:
            model_str (str): The model string in the format "repo;model;mmproj".

        Raises:
            ValueError: If the model string is not in the expected format.

        Returns:
            tuple[str, str, str]: A tuple containing the repo, model, and mmproj identifiers.
        """
        parts = [p.strip() for p in model_str.split(";")]
        if not 1 <= len(parts) <= 3:
            raise ValueError(f"Invalid vision model string: {model_str}")

        repo = parts[0]
        model = parts[1] if len(parts) >= 2 else repo
        mmproj = parts[2] if len(parts) == 3 else model
        return repo, model, mmproj

    embed_model_repo, embed_model_file, _ = resolve_model_name(
        os.getenv("EMBED_MODEL", default_embed_model)
    )
    text_model_repo, text_model_file, _ = resolve_model_name(
        os.getenv("LLM", default_text_model_str)
    )
    vision_model_repo, vision_model_file, vision_model_mmproj_file = resolve_model_name(
        os.getenv("VLM", default_vision_model_str)
    )

    return ModelConfig(
        embed_model_file=embed_model_file,
        embed_model_repo=embed_model_repo,
        image_embed_model=os.getenv("IMAGE_EMBED_MODEL", default_image_embed_model),
        ner_model=os.getenv("NER_MODEL", default_ner_model),
        rerank_model=os.getenv("RERANK_MODEL", default_rerank_model),
        sparse_model=os.getenv("SPARSE_MODEL", default_sparse_model),
        text_model_file=text_model_file,
        text_model_repo=text_model_repo,
        vision_model_file=vision_model_file,
        vision_model_mmproj_file=vision_model_mmproj_file,
        vision_model_repo=vision_model_repo,
        whisper_model=os.getenv("WHISPER_MODEL", default_whisper_model),
    )


@dataclass(frozen=True)
class NERConfig:
    """Dataclass for information extraction configuration."""

    enabled: bool
    max_chars: int
    max_workers: int


def load_ner_env(
    default_enabled: bool = True,
    default_max_chars: int = 1024,
    default_max_workers: int = 4,
) -> NERConfig:
    """Loads information extraction configuration from environment variables or defaults.

    Args:
        default_enabled (bool): Default value to enable NER extraction. Set to True to enable by default.
        default_max_chars (int): Default maximum characters for a processed chunk for NER extraction.
        default_max_workers (int): Default maximum worker threads for NER extraction.

    Returns:
        NERConfig: Dataclass containing NER configuration.
        - enabled (bool): Whether to run entity/relation extraction during ingestion.
        - max_chars (int): Maximum characters from each node to send to the extractor.
        - max_workers (int): Maximum number of worker threads for NER extraction.

    Raises:
        ValueError: If an unsupported NER engine is specified.
    """
    return NERConfig(
        enabled=str(os.getenv("NER_ENABLED", default_enabled)).lower()
        in {"true", "1", "yes"},
        max_chars=int(os.getenv("NER_MAX_CHARS", default_max_chars)),
        max_workers=int(os.getenv("NER_MAX_WORKERS", default_max_workers)),
    )


@dataclass(frozen=True)
class OpenAIConfig:
    """Dataclass for OpenAI-compatible API configuration."""

    api_base: str
    api_key: str
    ctx_window: int
    dimensions: int
    max_retries: int
    model_provider: str
    reuse_client: bool
    seed: int
    temperature: float
    timeout: float
    top_p: float


def load_openai_env(
    default_api_base: str = "http://localhost:11434/v1",
    default_api_key: str = "sk-no-key-required",
    default_ctx_window: int = 4096,
    default_dimensions: int = 1024,
    default_max_retries: int = 2,
    default_model_provider: str = "ollama",
    default_reuse_client: bool = False,
    default_seed: int = 42,
    default_temperature: float = 0.0,
    default_timeout: float = 300.0,
    default_top_p: float = 0.0,
) -> OpenAIConfig:
    """Loads OpenAI configuration from environment variables or defaults.

    Args:
        default_api_base (str): Default OpenAI API base URL.
        default_api_key (str): Default OpenAI API key.
        default_ctx_window (int): Default context window size for models that support it.
        default_dimensions (int): Default embedding dimensions for embedding models.
        default_max_retries (int): Default number of retries.
        default_model_provider (str): Default inference server type (e.g. "llama.cpp", "ollama", "openai", "vllm"). Default is "ollama".
        default_reuse_client (bool): Whether to reuse the OpenAI client across calls. Default is False.
        default_seed (int): Default random seed for reproducibility.
        default_temperature (float): Default temperature for text generation.
        default_timeout (float): Default timeout in seconds.
        default_top_p (float): Default top_p for nucleus sampling.

    Returns:
        OpenAIConfig: Dataclass containing OpenAI configuration.
        - api_base (str): The OpenAI API base URL.
        - api_key (str): The OpenAI API key.
        - ctx_window (int): The context window size for models that support it.
        - dimensions (int): The embedding dimensions for embedding models.
        - max_retries (int): The number of retries for API calls.
        - model_provider (str): The inference server type (e.g. "llama.cpp", "ollama", "openai").
        - reuse_client (bool): Whether to reuse the OpenAI client across calls.
        - seed (int): Random seed for reproducibility.
        - temperature (float): Temperature for text generation.
        - timeout (float): Timeout in seconds for API calls.
        - top_p (float): Top_p for nucleus sampling.

    Raises:
        ValueError: If an unsupported inference server is specified.
    """
    model_provider = os.getenv("MODEL_PROVIDER", default_model_provider).lower()
    if model_provider not in {
        "llama.cpp",
        "llama_cpp",
        "llamacpp",
        "ollama",
        "openai",
    }:
        raise ValueError(
            f"Unsupported inference server: {model_provider}. "
            f"Supported options are: 'ollama', 'llama.cpp', 'openai'."
        )

    return OpenAIConfig(
        api_base=os.getenv("OPENAI_API_BASE", default_api_base),
        api_key=os.getenv("OPENAI_API_KEY", default_api_key),
        ctx_window=int(os.getenv("OPENAI_CTX_WINDOW", default_ctx_window)),
        dimensions=int(os.getenv("OPENAI_DIMENSIONS", default_dimensions)),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", default_max_retries)),
        model_provider=model_provider,
        reuse_client=str(os.getenv("OPENAI_REUSE_CLIENT", default_reuse_client)).lower()
        in {"true", "1", "yes"},
        seed=int(os.getenv("OPENAI_SEED", default_seed)),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", default_temperature)),
        timeout=float(os.getenv("OPENAI_TIMEOUT", default_timeout)),
        top_p=float(os.getenv("OPENAI_TOP_P", default_top_p)),
    )


@dataclass(frozen=True)
class PathConfig:
    """Dataclass for path configuration."""

    artifacts: Path
    data: Path
    logs: Path
    queries: Path
    results: Path
    prompts: Path
    qdrant_sources: Path
    hf_hub_cache: Path
    llama_cpp_cache: Path


def load_path_env() -> PathConfig:
    """Loads path configuration from environment variables or defaults.

    Returns:
        PathConfig: Dataclass containing path configuration.
        - artifacts (Path): Root directory for pipeline processing artifacts.
        - data (Path): Path to the data directory.
        - logs (Path): Path to the logs file.
        - queries (Path): Path to the queries file.
        - results (Path): Path to the results directory.
        - prompts (Path): Path to the prompts directory.
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

    default_qdrant_sources: Path = docint_home_dir / "qdrant_sources"
    default_artifacts_dir: Path = docint_home_dir / "artifacts"

    return PathConfig(
        artifacts=Path(
            os.getenv("PIPELINE_ARTIFACTS_DIR", default_artifacts_dir)
        ).expanduser(),
        data=Path(os.getenv("DATA_PATH", default_data_dir)).expanduser(),
        logs=Path(os.getenv("LOG_PATH", default_log_dir)).expanduser(),
        queries=Path(os.getenv("QUERIES_PATH", default_query_dir)).expanduser(),
        results=Path(os.getenv("RESULTS_PATH", default_results_dir)).expanduser(),
        prompts=default_prompts_dir,
        qdrant_sources=Path(
            os.getenv("QDRANT_SRC_DIR", default_qdrant_sources)
        ).expanduser(),
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", default_hf_hub_cache)).expanduser(),
        llama_cpp_cache=Path(
            os.getenv("LLAMA_CPP_CACHE", default_llama_cpp_cache)
        ).expanduser(),
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the document processing pipeline."""

    text_coverage_threshold: float
    pipeline_version: str
    artifacts_dir: str
    max_retries: int
    force_reprocess: bool
    max_workers: int
    enable_vision_ocr: bool
    vision_ocr_timeout: float
    vision_ocr_max_retries: int
    vision_ocr_max_image_dimension: int
    vision_ocr_max_tokens: int


def load_pipeline_config(
    default_text_coverage_threshold: float = 0.01,
    default_pipeline_version: str = "1.0.0",
    default_artifacts_dir: str | None = None,
    default_max_retries: int = 2,
    default_force_reprocess: bool = False,
    default_max_workers: int = 4,
    default_enable_vision_ocr: bool = True,
    default_vision_ocr_timeout: float = 60.0,
    default_vision_ocr_max_retries: int = 1,
    default_vision_ocr_max_image_dimension: int = 1024,
    default_vision_ocr_max_tokens: int = 4096,
) -> PipelineConfig:
    """Build a ``PipelineConfig`` from environment variables with sensible defaults.

    Args:
        default_text_coverage_threshold: Default characters-per-area threshold for OCR classification.
        default_pipeline_version: Default pipeline version string.
        default_artifacts_dir: Default root directory for artifacts. If None, uses the value from ``load_path_env().artifacts``.
        default_max_retries: Default maximum retry attempts per page stage.
        default_force_reprocess: Default flag to force reprocessing of pages.
        default_max_workers: Default maximum number of parallel workers for document processing.
        default_enable_vision_ocr: Default flag to enable vision OCR fallback for scanned pages.
        default_vision_ocr_timeout: Default per-request timeout in seconds for vision OCR API calls.
        default_vision_ocr_max_retries: Default maximum retries for a single vision OCR API call.
        default_vision_ocr_max_image_dimension: Default maximum pixel dimension for images sent to the vision OCR endpoint.
        default_vision_ocr_max_tokens: Default maximum number of tokens the vision LLM may generate per OCR request.

    Returns:
        A fully-initialised ``PipelineConfig``.
        - text_coverage_threshold (float): Characters-per-area threshold for OCR classification.
        - pipeline_version (str): Semver string identifying the pipeline logic version.
        - artifacts_dir (str): Root directory for artifact output.
        - max_retries (int): Maximum retry attempts per stage on a given page.
        - force_reprocess (bool): When True, ignore existing artifacts and reprocess.
        - max_workers (int): Maximum parallel workers for document-level processing.
        - enable_vision_ocr (bool): When True, use the vision LLM as a fallback OCR engine for scanned pages that have no extractable text layer.
        - vision_ocr_timeout (float): Per-request timeout in seconds for vision OCR API calls (separate from the global ``OPENAI_TIMEOUT``).
        - vision_ocr_max_retries (int): Maximum retries for a single vision OCR API call.
        - vision_ocr_max_image_dimension (int): Maximum pixel dimension (width or height) for images sent to the vision OCR endpoint.  Larger renders are down-scaled proportionally before encoding.
        - vision_ocr_max_tokens (int): Maximum number of tokens the vision LLM may generate per OCR request.  Keeps response time bounded.
    """
    pipeline_version = os.getenv("PIPELINE_VERSION", default_pipeline_version).strip()
    if not pipeline_version:
        pipeline_version = default_pipeline_version

    return PipelineConfig(
        text_coverage_threshold=float(
            os.getenv(
                "PIPELINE_TEXT_COVERAGE_THRESHOLD", default_text_coverage_threshold
            )
        ),
        pipeline_version=pipeline_version,
        artifacts_dir=str(load_path_env().artifacts),
        max_retries=int(os.getenv("PIPELINE_MAX_RETRIES", default_max_retries)),
        force_reprocess=os.getenv("PIPELINE_FORCE_REPROCESS", "false").lower()
        in {"true", "1", "yes"},
        max_workers=int(os.getenv("PIPELINE_MAX_WORKERS", default_max_workers)),
        enable_vision_ocr=os.getenv("PIPELINE_ENABLE_VISION_OCR", "true").lower()
        in {"true", "1", "yes"},
        vision_ocr_timeout=float(
            os.getenv("PIPELINE_VISION_OCR_TIMEOUT", default_vision_ocr_timeout)
        ),
        vision_ocr_max_retries=int(
            os.getenv("PIPELINE_VISION_OCR_MAX_RETRIES", default_vision_ocr_max_retries)
        ),
        vision_ocr_max_image_dimension=int(
            os.getenv(
                "PIPELINE_VISION_OCR_MAX_IMAGE_DIM",
                default_vision_ocr_max_image_dimension,
            )
        ),
        vision_ocr_max_tokens=int(
            os.getenv("PIPELINE_VISION_OCR_MAX_TOKENS", default_vision_ocr_max_tokens)
        ),
    )


@dataclass(frozen=True)
class ResponseValidationConfig:
    """Dataclass for response validation configuration."""

    enabled: bool


def load_response_validation_env(
    default_enabled: bool = True,
) -> ResponseValidationConfig:
    """Load response-validation configuration from environment variables.

    Args:
        default_enabled (bool): Whether response validation is enabled by default.

    Returns:
        ResponseValidationConfig: Parsed response-validation settings.
        - enabled (bool): Whether to run response validation on generated answers.
    """
    return ResponseValidationConfig(
        enabled=str(os.getenv("RESPONSE_VALIDATION_ENABLED", default_enabled)).lower()
        in {"true", "1", "yes"}
    )


@dataclass(frozen=True)
class RetrievalConfig:
    """Dataclass for RAG (Retrieval-Augmented Generation) configuration."""

    rerank_use_fp16: bool
    retrieve_top_k: int


def load_retrieval_env(
    default_rerank_use_fp16: bool = False,
    default_retrieve_top_k: int = 20,
) -> RetrievalConfig:
    """Loads retrieval configuration from environment variables or defaults.

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


@dataclass(frozen=True)
class SessionConfig:
    """Dataclass for session configuration."""

    session_store: str


def load_session_env(
    default_session_store: str = "sqlite:///sessions.db",
) -> SessionConfig:
    """Loads session configuration from environment variables or defaults.

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


@dataclass(frozen=True)
class SummaryConfig:
    """Dataclass for collection summarization precision settings."""

    coverage_target: float
    max_docs: int
    per_doc_top_k: int
    final_source_cap: int


def load_summary_env(
    default_coverage_target: float = 0.70,
    default_max_docs: int = 30,
    default_per_doc_top_k: int = 4,
    default_final_source_cap: int = 24,
) -> SummaryConfig:
    """Load collection summary precision settings from environment variables.

    Args:
        default_coverage_target (float): Target minimum document coverage ratio.
        default_max_docs (int): Maximum number of documents sampled for summary.
        default_per_doc_top_k (int): Maximum evidence chunks retrieved per document.
        default_final_source_cap (int): Maximum number of merged summary sources.

    Returns:
        SummaryConfig: Parsed summary precision settings.
        - coverage_target (float): Target minimum document coverage ratio for summaries. Value is clamped to [0.0, 1.0].
        - max_docs (int): Maximum number of documents to sample for summarization.
        - per_doc_top_k (int): Maximum number of evidence chunks to retrieve per document.
        - final_source_cap (int): Maximum number of merged sources to include in the final summary answer to keep it concise and focused.
    """
    raw_target = float(os.getenv("SUMMARY_COVERAGE_TARGET", default_coverage_target))
    target = min(1.0, max(0.0, raw_target))
    return SummaryConfig(
        coverage_target=target,
        max_docs=max(1, int(os.getenv("SUMMARY_MAX_DOCS", default_max_docs))),
        per_doc_top_k=max(
            1, int(os.getenv("SUMMARY_PER_DOC_TOP_K", default_per_doc_top_k))
        ),
        final_source_cap=max(
            1, int(os.getenv("SUMMARY_FINAL_SOURCE_CAP", default_final_source_cap))
        ),
    )


@dataclass(frozen=True)
class WhisperConfig:
    """Dataclass for Whisper runtime configuration."""

    max_workers: int
    task: Literal["transcribe", "translate"]


def load_whisper_env(
    default_max_workers: int = 1,
    default_task: Literal["transcribe", "translate"] = "transcribe",
) -> WhisperConfig:
    """Load Whisper runtime settings from environment variables.

    Args:
        default_max_workers: Default number of file-level Whisper workers.
        default_task: Default Whisper task selector.

    Returns:
        WhisperConfig: Parsed Whisper runtime configuration.

    Raises:
        ValueError: If an invalid value is provided for WHISPER_TASK or WHISPER_MAX_WORKERS.
    """

    raw_max_workers = os.getenv("WHISPER_MAX_WORKERS")
    if raw_max_workers is None:
        max_workers = default_max_workers
    else:
        try:
            parsed_max_workers = int(raw_max_workers)
        except ValueError:
            logger.warning(
                "Invalid WHISPER_MAX_WORKERS value '{}'; falling back to {}.",
                raw_max_workers,
                default_max_workers,
            )
            parsed_max_workers = default_max_workers

        if parsed_max_workers < 1:
            logger.warning(
                "WHISPER_MAX_WORKERS must be >= 1; received '{}'. Falling back to {}.",
                raw_max_workers,
                default_max_workers,
            )
            parsed_max_workers = default_max_workers
        max_workers = parsed_max_workers

    raw_task = os.getenv("WHISPER_TASK", default_task)
    task = raw_task.strip().lower()
    if task not in {"transcribe", "translate"}:
        logger.warning(
            "Invalid WHISPER_TASK value '{}'; falling back to '{}'.",
            raw_task,
            default_task,
        )
        task = default_task

    return WhisperConfig(
        max_workers=max_workers,
        task=cast(Literal["transcribe", "translate"], task),
    )
