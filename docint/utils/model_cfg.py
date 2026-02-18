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
import whisper
from docling.models.stages.code_formula.code_formula_model import CodeFormulaModel
from docling.models.stages.layout.layout_model import LayoutModel
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.models.stages.picture_classifier.document_picture_classifier import (
    DocumentPictureClassifier,
    DocumentPictureClassifierOptions,
)
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from dotenv import load_dotenv
from gliner import GLiNER
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger

from docint.utils.logging_cfg import setup_logging

load_dotenv()
setup_logging()


def load_docling_models() -> None:
    """
    Preloads Docling models to the HuggingFace cache.

    We invoke the `download_models` static method of each model class directly.
    This ensures that we use the exact same logic (repo_id, revision, local_dir)
    that the runtime uses when initializing these models.

    Note: RapidOCR uses a custom cache location, while others default to the
    standard HF cache when no local_dir is provided.
    """
    try:
        # 1. RapidOCR (Custom cache location)
        # Note: RapidOCR requires a backend argument. We use "onnxruntime" as it's the default.
        RapidOcrModel.download_models(backend="onnxruntime", progress=True)
        logger.info("Loaded RapidOCR model")

        # 2. Layout Model (Standard HF cache)
        LayoutModel.download_models(progress=True)
        logger.info("Loaded Layout model")

        # 3. Table Structure Model (Standard HF cache)
        TableStructureModel.download_models(progress=True)
        logger.info("Loaded Table Structure model")

        # 4. Code/Formula Model (Standard HF cache)
        CodeFormulaModel.download_models(progress=True)
        logger.info("Loaded Code/Formula model")

        # 5. Picture Classifier (Standard HF cache)
        DocumentPictureClassifier.download_models(
            repo_id=DocumentPictureClassifierOptions().repo_id, progress=True
        )
        logger.info("Loaded Picture Classifier model")

    except Exception as e:
        logger.warning("Failed to download Docling models: {}", e)


def load_hf_model(
    model_id: str, cache_folder: Path, kw: str, trust_remote_code: bool = False
) -> None:
    """
    Loads and returns the HuggingFace embedding model.

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


def load_llama_cpp_model(cache_dir: Path, model_id: str, repo_id: str, kw: str) -> None:
    """
    Loads the llama.cpp model.

    Args:
        cache_dir (Path): The path to the cache directory.
        model_id (str): The name of the model to load.
        repo_id (str): The repository ID for the model.
        kw (str): The keyword for the model type (e.g., "text" or "vision").
    """

    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / model_id

    # Check if model already exists (direct path)
    if model_path.exists():
        logger.info(
            "{} model '{}' is already available at {}",
            kw.capitalize(),
            model_id,
            model_path,
        )
        return

    # Check HF cache structure
    if repo_id:
        resolved = resolve_hf_cache_path(
            cache_dir=cache_dir, repo_id=repo_id, filename=model_id
        )
        if resolved:
            logger.info(
                "{} model '{}' is already available at {}",
                kw.capitalize(),
                model_id,
                resolved,
            )
            return

    if repo_id is None:
        logger.warning(
            "{} model '{}' not found locally and no repo_id provided for download",
            kw.capitalize(),
            model_id,
        )
        return

    logger.info(
        "{} model '{}' not found. Downloading from {}...",
        kw.capitalize(),
        model_id,
        repo_id,
    )

    # Download from Hugging Face
    # We use hf_hub_download to fetch the GGUF file directly.
    # We catch EntryNotFoundError to handle cases where the filename config might be slightly off
    # or if the user provided a full path instead of just the filename.
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_id,
            cache_dir=cache_dir,
            local_dir=cache_dir,  # Download directly to the folder where llama-server looks
            local_dir_use_symlinks=False,
        )
        logger.info("Loaded {} model '{}' to {}", kw, model_id, downloaded_path)
    except Exception as e:
        logger.error("Failed to download {} model '{}': {}", kw, model_id, e)
        # Verify if the file might actually be there under a different name or if the config is wrong
        # But we continue to let the caller handle the missing model later.
        pass

    # Fallback/Check: We don't verify file presence here because it's remote (in shared volume).
    # But since we share the volume, we technically COULD check.
    # For now, we assume the server handles it.


def load_ollama_model(
    model_id: str, kw: str, host: str = "http://localhost:11434"
) -> None:
    """
    Loads the specified model using Ollama.

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


def load_whisper_model(model_id: str) -> None:
    """
    Loads and returns the Whisper model.

    Args:
        model_id (str): The name of the model to load.
    """
    whisper.load_model(name=model_id)
    logger.info("Loaded whisper model: {}", model_id)


def main() -> None:
    """
    Main function to verify configuration loading.
    """
    # Load configurations
    path_config = load_path_env()
    model_config = load_model_env()
    openai_config = load_openai_env()

    # Log the loaded configurations
    for path in paths.__dataclass_fields__.keys():
        logger.info("{} path: {}", path, getattr(paths, path))
    for model_id in models.__dataclass_fields__.keys():
        logger.info("{}: {}", model_id, getattr(models, model_id))

    # Load the app's models
    # Docling
    load_docling_models()

    # Hugging Face
    for model_id, kw in [
        (models.embed_model, "embedding"),
        (models.sparse_model, "sparse"),
        (models.rerank_model, "rerank"),
    ]:
        load_hf_model(
            model_id=model_id,
            cache_folder=paths.hf_hub_cache,
            kw=kw,
        )

    # LLaMA.cpp
    if openai_config.inference_server in {"llama.cpp", "llama_cpp", "llamacpp"}:
        for model_id, repo_id, kw in [
            # model_id refers to the GGUF filename, repo_id is the HuggingFace repo where it lives.
            # We need both to resolve cache correctly.
            (model_config.embed_model_file, model_config.embed_model_repo, "embedding"),
            (model_config.text_model_file, model_config.text_model_repo, "text"),
            (model_config.vision_model_file, model_config.vision_model_repo, "vision"),
        ]:
            load_llama_cpp_model(
                cache_dir=path_config.llama_cpp_cache,
                model_id=model_id,
                repo_id=repo_id,
                kw=kw,
            )
    # Ollama
    if openai_config.inference_server in {"ollama"}:
        for model_id, kw in [
            (model_config.embed_model_file, "embedding"),
            (model_config.text_model_file, "text"),
            (model_config.vision_model_file, "vision"),
        ]:
            load_ollama_model(model_id=model_id, kw=kw, host=openai_config.api_base)

    # vLLM
    if openai_config.inference_server in {"vllm"}:
        # TODO: Add vLLM loading logic here when we support vLLM as an inference server option.
        logger.warning("vLLM inference server support is not yet implemented.")

    # OpenAI API
    if openai_config.inference_server in {"openai"}:
        # For OpenAI API, we don't have local model loading.
        logger.info(
            "Using OpenAI API as inference server. No local model loading required."
        )

    # Whisper
    load_whisper_model(model_id=models.whisper_model)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
