import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure backend is in path if running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))

from dotenv import load_dotenv
from loguru import logger

from docint.utils.logging_cfg import setup_logging

load_dotenv()
setup_logging()


@dataclass(frozen=True)
class PathConfig:
    xdg_cache_home: Path
    hf_hub_cache: Path


@dataclass(frozen=True)
class ModelConfig:
    embed_model: str
    sparse_model: str
    gen_model: str
    vision_model: str
    whisper_model: str


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


def load_path_env() -> PathConfig:
    """
    Loads path configuration from environment variables or defaults.

    Returns:
        PathConfig: Dataclass containing path configuration.
    """
    xdg_cache_home_dir: Path = Path.home() / ".cache"
    hf_hub_cache_dir: Path = xdg_cache_home_dir / "huggingface" / "hub"
    return PathConfig(
        xdg_cache_home=Path(
            os.getenv("XDG_CACHE_HOME", xdg_cache_home_dir)
        ).expanduser(),
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", hf_hub_cache_dir)).expanduser(),
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
