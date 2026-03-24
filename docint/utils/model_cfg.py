# ruff: noqa: E402

import shutil
import sys
from pathlib import Path

# isort: off
from docint.utils.env_cfg import (
    bootstrap_config,
    load_model_env,
    load_openai_env,
    load_path_env,
    resolve_hf_cache_path,
    set_online_env,
)

bootstrap_config(role="worker")
set_online_env()
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
from docling.models.inference_engines.image_classification.transformers_engine import (
    TransformersImageClassificationEngineOptions,
)
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from gliner import GLiNER
from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoProcessor, CLIPModel

from docint.utils.logging_cfg import setup_logging

setup_logging()


def load_clip_model(model_id: str, cache_folder: Path) -> None:
    """Preloads the CLIP model to the HuggingFace cache.

    Args:
        model_id (str): The name of the CLIP model to load.
        cache_folder (Path): The path to the HuggingFace cache folder.
    """
    resolved = resolve_hf_cache_path(cache_dir=cache_folder, repo_id=model_id)
    if resolved:
        logger.info("Found local cache for CLIP at {}", resolved)
        try:
            CLIPModel.from_pretrained(
                pretrained_model_name_or_path=str(resolved),
                local_files_only=True,
            )
            AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=str(resolved),
                local_files_only=True,
            )
        except Exception as e:
            logger.warning(
                "Failed to load CLIP from local cache: {}. Retrying with download...", e
            )
            CLIPModel.from_pretrained(pretrained_model_name_or_path=model_id)
            AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_id)
    else:
        CLIPModel.from_pretrained(pretrained_model_name_or_path=model_id)
        AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_id)
    logger.info("Loaded CLIP model: {}", model_id)


def load_docling_models() -> None:
    """Preloads Docling models to the HuggingFace cache.

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
        opts = DocumentPictureClassifierOptions(
            engine_options=TransformersImageClassificationEngineOptions()
        )
        DocumentPictureClassifier.download_models(repo_id=opts.repo_id, progress=True)
        logger.info("Loaded Picture Classifier model")

    except Exception as e:
        logger.warning("Failed to download Docling models: {}", e)


def load_gliner_model(model_id: str, cache_folder: Path) -> None:
    """Loads the GLiNER model.

    Args:
        model_id (str): The name of the GLiNER model to load.
        cache_folder (Path): The path to the HuggingFace cache folder.
    """
    resolved = resolve_hf_cache_path(cache_dir=cache_folder, repo_id=model_id)
    if resolved:
        logger.info("Found local cache for GLiNER at {}", resolved)
        try:
            GLiNER.from_pretrained(model_id=str(resolved), local_files_only=True)
        except Exception as e:
            logger.warning(
                "Failed to load GLiNER from local cache: {}. Retrying with download...",
                e,
            )
            GLiNER.from_pretrained(model_id=model_id)
    else:
        GLiNER.from_pretrained(model_id=model_id)
    logger.info("Loaded GLiNER model: {}", model_id)


def load_hf_model(
    model_id: str, cache_folder: Path, kw: str, trust_remote_code: bool = False
) -> None:
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


def _link_or_copy_model_file(source: Path, destination: Path) -> None:
    """Materialize a model file at ``destination`` from ``source``.

    Args:
        source: Existing source file path.
        destination: Target file path to create.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        destination.symlink_to(source)
    except Exception:
        shutil.copy2(source, destination)


def load_ollama_model(
    model_id: str, kw: str, host: str = "http://localhost:11434"
) -> None:
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


def load_whisper_model(model_id: str) -> None:
    """Loads and returns the Whisper model.

    Args:
        model_id (str): The name of the model to load.
    """
    whisper.load_model(name=model_id)
    logger.info("Loaded whisper model: {}", model_id)


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

    # Load the app's models
    # CLIP (used by Picture Classifier and Layout Model)
    load_clip_model(
        model_id=model_config.image_embed_model, cache_folder=path_config.hf_hub_cache
    )

    # Docling
    load_docling_models()

    # GLiNER
    load_gliner_model(
        model_id=model_config.ner_model, cache_folder=path_config.hf_hub_cache
    )

    # Hugging Face
    for model_id, kw in [
        (model_config.sparse_model, "sparse"),
        (model_config.rerank_model, "rerank"),
    ]:
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

    # Remote OpenAI-compatible APIs
    if openai_config.inference_provider in {"openai"}:
        # For remote OpenAI-compatible APIs, the serving stack provisions the
        # text, embedding, and vision endpoints. Only the app-local auxiliary
        # models above are prepared here.
        logger.info(
            "Using {} as inference server. No local text/embedding/vision model loading required.",
            openai_config.inference_provider,
        )

    # Whisper
    load_whisper_model(model_id=model_config.whisper_model)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
