"""Model asset download and cache-residency orchestration for offline deployments."""

import os
import sys
from pathlib import Path

# load-models is an online operation — override offline mode before env_cfg
# sets HF_HUB_OFFLINE at import time.  load_dotenv() in env_cfg honours
# existing env vars (override=False), so this takes precedence over .env.
os.environ["DOCINT_OFFLINE"] = "0"

# isort: off
# Import env_cfg BEFORE any third-party libraries so that HF_HUB_OFFLINE and
# TRANSFORMERS_OFFLINE env vars are set before huggingface_hub caches them.
from docint.utils.env_cfg import (
    load_model_env,
    load_openai_env,
    load_path_env,
    resolve_hf_cache_path,
)
# isort: on

import ollama
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger

from docint.utils.logger_cfg import init_logger

load_dotenv()
init_logger()


def load_hf_model(model_id: str, cache_folder: Path, kw: str, trust_remote_code: bool = False) -> None:
    """Ensure a Hugging Face model snapshot is available in the local cache.

    Args:
        model_id (str): The name of the model to load.
        cache_folder (Path): The path to the cache folder.
        kw (str): The keyword for the model type (e.g., "embedding").
        trust_remote_code (bool): Whether to trust remote code. Defaults to False.
    """
    resolved = resolve_hf_cache_path(cache_dir=cache_folder, repo_id=model_id)

    if resolved:
        logger.info("Found local cache for {} at {}", model_id, resolved)
        model_id = str(resolved)
    else:
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_folder,
        )

    logger.info("Loaded {} model: {}", kw, model_id)


def load_ollama_model(model_id: str, kw: str, host: str = "http://localhost:11434") -> None:
    """Loads the specified model using Ollama.

    Args:
        model_id (str): The ID of the model to load.
        kw (str): The keyword for the model type (e.g., "text" or "vision").
        host (str): The host URL for the Ollama server. Defaults to "http://localhost:11434".
    """
    # Remove the trailing slash and /v1 if present to get the base URL for the Ollama client
    clean_host = host.rstrip("/").removesuffix("/v1")

    client = ollama.Client(host=clean_host)
    models_response = client.list()
    existing_models = [m["model"] for m in models_response.get("models", [])]

    # Check if model exists
    if model_id in existing_models or f"{model_id}:latest" in existing_models:
        logger.info("Model '{}' is already available.", model_id)
        return

    logger.info("Model '{}' not found. Pulling...", model_id)

    # Stream the pull progress
    for progress in client.pull(model_id, stream=True):
        if "completed" in progress and "total" in progress:
            percent = (progress["completed"] / progress["total"]) * 100
            if int(percent) % 10 == 0:  # Log every 10%
                logger.debug("Pulling {}: {:.1f}%", model_id, percent)

    logger.info("Successfully pulled model '{}'.", model_id)


def main() -> None:
    """Main function to verify configuration loading."""
    # Load configurations
    path_config = load_path_env()
    model_config = load_model_env()
    openai_config = load_openai_env()

    # Log the loaded configurations
    for path in path_config.__dataclass_fields__.keys():
        logger.info("{} path: {}", path, getattr(path_config, path))
    for model_id in model_config.__dataclass_fields__.keys():
        logger.info("{}: {}", model_id, getattr(model_config, model_id))

    # NER and CLIP are no longer loaded here: docint now calls the
    # remote GLiNER and CLIP services hosted by vllm-service over HTTP.
    # Model preloading happens in the vllm-service `ner` / `clip` (or
    # `gliner-ner` / `clip-embed`) containers, not here.

    # Hugging Face
    hf_assets: list[tuple[str, str]] = [
        (model_config.sparse_model, "sparse"),
        (model_config.rerank_model, "rerank"),
    ]
    # Worker-side embedding tokenizer (ships with docint regardless of
    # inference provider — used for offline token counting before the
    # provider sees the request). Skipped when the provider tokenizes
    # server-side (e.g. openai) and the repo default is therefore empty.
    embed_tokenizer_repo = getattr(model_config, "embed_tokenizer_repo", "")
    if embed_tokenizer_repo:
        hf_assets.append((embed_tokenizer_repo, "embed-tokenizer"))

    for model_id, kw in hf_assets:
        load_hf_model(
            model_id=model_id,
            cache_folder=path_config.hf_hub_cache,
            kw=kw,
        )

    # Ollama / vLLM
    for model_id, kw in [
        (model_config.embed_model, "embedding"),
        (model_config.text_model, "text"),
        (model_config.vision_model, "vision"),
    ]:
        if openai_config.inference_provider == "ollama":
            load_ollama_model(model_id=model_id, kw=kw, host=openai_config.api_base)

        elif openai_config.inference_provider == "vllm":
            load_hf_model(
                model_id=model_id,
                cache_folder=path_config.hf_hub_cache,
                kw=kw,
            )

    # Remote OpenAI-compatible APIs provision text/embedding/vision
    # endpoints themselves; only the app-local auxiliary assets above are
    # prepared here.
    if openai_config.inference_provider == "openai":
        logger.info(
            "Using {} as inference server. No local text/embedding/vision model loading required.",
            openai_config.inference_provider,
        )


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
