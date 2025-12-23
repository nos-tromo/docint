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

    enabled: bool
    max_chars: int


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
    xdg_cache_home: Path
    hf_hub_cache: Path


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
    return HostConfig(
        backend_host=os.getenv("BACKEND_HOST", "http://localhost:8000"),
        backend_public_host=os.getenv("BACKEND_PUBLIC_HOST", "http://localhost:8000"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        qdrant_host=os.getenv("QDRANT_HOST", "http://localhost:6333"),
        cors_allowed_origins=os.getenv(
            "CORS_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
        ),
    )


def load_ie_env() -> InformationExtractionConfig:
    """
    Loads information extraction configuration from environment variables or defaults.

    Returns:
        IEConfig: Dataclass containing IE configuration.
        - enabled (bool): Whether to run entity/relation extraction during ingestion.
        - max_chars (int): Maximum characters from each node to send to the extractor.
    """

    def _as_bool(val: str | None, default: bool = False) -> bool:
        if val is None:
            return default
        return val.lower() in {"1", "true", "yes", "on"}

    return InformationExtractionConfig(
        enabled=_as_bool(os.getenv("ENABLE_IE"), False),
        max_chars=int(os.getenv("IE_MAX_CHARS", "800")),
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
    return ModelConfig(
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        sparse_model=os.getenv(
            "SPARSE_MODEL", "Qdrant/all_miniLM_L6_v2_with_attentions"
        ),
        gen_model=os.getenv("LLM", "qwen3:14b"),
        vision_model=os.getenv("VLM", "qwen3-vl:8b"),
        whisper_model=os.getenv("WHISPER_MODEL", "turbo"),
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
    return OllamaConfig(
        ctx_window=int(os.getenv("OLLAMA_CTX_WINDOW", "8192")),
        request_timeout=int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "1200")),
        seed=int(os.getenv("OLLAMA_SEED", "42")),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.0")),
        thinking=os.getenv("OLLAMA_THINKING", "true").lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        },
        top_k=int(os.getenv("OLLAMA_TOP_K", "1")),
        top_p=float(os.getenv("OLLAMA_TOP_P", "0")),
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
        - xdg_cache_home (Path): Path to the XDG cache home directory.
        - hf_hub_cache (Path): Path to the Hugging Face Hub cache directory.
    """
    home_dir = Path.home()
    xdg_cache_home_dir: Path = home_dir / ".cache"
    hf_hub_cache_dir: Path = xdg_cache_home_dir / "huggingface" / "hub"
    data_dir: Path = home_dir / "docint"
    project_root: Path = Path(__file__).parents[2].resolve()
    utils_dir: Path = project_root / "docint" / "utils"

    qdrant_collections = Path(
        os.getenv("QDRANT_COL_DIR", "qdrant_storage")
    ).expanduser()
    qdrant_sources_env = os.getenv("QDRANT_SRC_DIR")

    # Default sources root alongside Qdrant storage; fall back to a "sources" sibling.
    if qdrant_sources_env:
        qdrant_sources = Path(qdrant_sources_env).expanduser()
    else:
        default_sources_base = (
            qdrant_collections.parent
            if qdrant_collections.parent != Path(".")
            else qdrant_collections
        )
        qdrant_sources = (default_sources_base / "sources").expanduser()

    return PathConfig(
        data=Path(os.getenv("DATA_PATH", data_dir / "data")).expanduser(),
        logs=Path(
            os.getenv("LOG_PATH", project_root / ".logs" / "docint.log")
        ).expanduser(),
        queries=Path(os.getenv("QUERIES_PATH", data_dir / "queries.txt")).expanduser(),
        results=Path(os.getenv("RESULTS_PATH", data_dir / "results")).expanduser(),
        prompts=utils_dir / "prompts",
        qdrant_collections=qdrant_collections,
        qdrant_sources=qdrant_sources,
        required_exts=utils_dir / "required_exts.txt",
        xdg_cache_home=Path(
            os.getenv("XDG_CACHE_HOME", xdg_cache_home_dir)
        ).expanduser(),
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", hf_hub_cache_dir)).expanduser(),
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
