import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from docint.utils.logging_cfg import setup_logging

load_dotenv()
setup_logging()


@dataclass(frozen=True)
class ModelConfig:
    embed_model: str
    sparse_model: str
    gen_model: str
    vision_model: str
    whisper_model: str


@dataclass(frozen=True)
class HostConfig:
    ollama: str
    qdrant: str


@dataclass(frozen=True)
class PathConfig:
    data: Path
    queries: Path
    results: Path
    prompts: Path
    qdrant_collections: Path
    required_exts: Path
    xdg_cache_home: Path
    hf_hub_cache: Path


def load_host_env() -> HostConfig:
    """
    Loads host configuration from environment variables or defaults.

    Returns:
        HostConfig: Dataclass containing host configuration.
    """
    return HostConfig(
        ollama=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        qdrant=os.getenv("QDRANT_HOST", "http://localhost:6333"),
    )


def load_model_env() -> ModelConfig:
    """
    Loads model configuration from environment variables or defaults.

    Returns:
        ModelConfig: Dataclass containing model configuration.
    """
    return ModelConfig(
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        sparse_model=os.getenv(
            "SPARSE_MODEL", "Qdrant/all_miniLM_L6_v2_with_attentions"
        ),
        gen_model=os.getenv("LLM", "granite4:7b-a1b-h"),
        vision_model=os.getenv("VLM", "qwen3-vl:8b"),
        whisper_model=os.getenv("WHISPER_MODEL", "turbo"),
    )


def load_path_env() -> PathConfig:
    """
    Loads path configuration from environment variables or defaults.

    Returns:
        PathConfig: Dataclass containing path configuration.
    """
    home_dir = Path.home()
    docint_data_dir: Path = home_dir / "docint"
    package_dir: Path = Path(__file__).parents[1].resolve()
    xdg_cache_home_dir: Path = home_dir / ".cache"
    hf_hub_cache_dir: Path = xdg_cache_home_dir / "huggingface" / "hub"

    return PathConfig(
        data=Path(os.getenv("DATA_PATH", docint_data_dir / "data")).expanduser(),
        queries=Path(
            os.getenv("QUERIES_PATH", docint_data_dir / "queries.txt")
        ).expanduser(),
        results=Path(
            os.getenv("RESULTS_PATH", docint_data_dir / "results")
        ).expanduser(),
        prompts=package_dir / "utils" / "prompts",
        qdrant_collections=Path(
            os.getenv("QDRANT_COL_DIR", "qdrant_collections")
        ).expanduser(),
        required_exts=package_dir / "core" / "readers" / "required_exts.txt",
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

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logger.info("Set Hugging Face libraries to offline mode.")
