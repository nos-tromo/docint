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
from docint.utils.env_cfg import load_model_env, load_path_env, resolve_hf_cache_path
# isort: on

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
from huggingface_hub import snapshot_download
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from loguru import logger
from transformers import AutoTokenizer

from docint.utils.llama_cpp_cfg import LlamaCppPipeline
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
    resolved = resolve_hf_cache_path(cache_folder, model_id)

    if resolved:
        logger.info("Found local cache for {} at {}", model_id, resolved)
        model_id = str(resolved)
    else:
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_folder,
        )

    if kw == "embedding":
        HuggingFaceEmbedding(
            model_name=model_id,
            trust_remote_code=trust_remote_code,
        )
    logger.info("Loaded {} model: {}", kw, model_id)


def load_gliner_model(model_id: str, cache_folder: Path) -> None:
    """
    Loads the GLiNER model.

    Args:
        model_id (str): The name of the GLiNER model to load.
        cache_folder (Path): The path to the HuggingFace cache folder.
    """
    resolved = resolve_hf_cache_path(cache_dir=cache_folder, repo_id=model_id)
    if resolved:
        logger.info("Found local cache for GLiNER at {}", resolved)
        GLiNER.from_pretrained(model_id=str(resolved), local_files_only=True)
    else:
        GLiNER.from_pretrained(model_id=model_id)
    logger.info("Loaded GLiNER model: {}", model_id)


def load_llama_cpp_model(model_id: str, repo_id: str, kw: str) -> None:
    """
    Loads a Llama.cpp GGUF model.

    Args:
        model_id (str): The name of the model file or repo/file (e.g., "model.gguf" or "user/repo/model.gguf").
        repo_id (str): The Hugging Face repository ID for the model (e.g., "user/repo").
        kw (str): The keyword for the model type (e.g., "generator").
    """
    LlamaCppPipeline.ensure_model(model_id, repo_id=repo_id)
    logger.info("Loaded {}: {}", kw, model_id)


def load_tokenizer(tokenizer_id: str, cache_folder: Path) -> None:
    """
    Downloads and caches a HuggingFace tokenizer for chat template formatting.

    This ensures the tokenizer files are available locally for offline use.
    Only the tokenizer files are downloaded (tokenizer.json, tokenizer_config.json,
    etc.), not the full model weights.

    Args:
        tokenizer_id: HuggingFace repo containing the tokenizer
            (e.g. ``Qwen/Qwen3-1.7B``).
        cache_folder: HuggingFace hub cache directory.
    """
    resolved = resolve_hf_cache_path(cache_folder, tokenizer_id)
    if resolved:
        logger.info("Tokenizer '{}' already cached at {}", tokenizer_id, resolved)
        return

    logger.info("Downloading tokenizer '{}'...", tokenizer_id)
    AutoTokenizer.from_pretrained(
        tokenizer_id,
        cache_dir=cache_folder,
        trust_remote_code=True,
    )
    logger.info("Loaded tokenizer: {}", tokenizer_id)


def load_whisper_model(model_id: str) -> None:
    """
    Loads and returns the Whisper model.

    Args:
        model_id (str): The name of the model to load.
    """
    whisper.load_model(model_id)
    logger.info("Loaded whisper model: {}", model_id)


def main() -> None:
    """
    Main function to verify configuration loading.
    """
    # Load configurations
    paths = load_path_env()
    models = load_model_env()

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
        (models.ner_model, "NER"),
        (models.rerank_model, "rerank"),
    ]:
        load_hf_model(
            model_id=model_id,
            cache_folder=paths.hf_hub_cache,
            kw=kw,
        )

    # GLiNER
    load_gliner_model(model_id=models.ner_model, cache_folder=paths.hf_hub_cache)

    # Llama.cpp models
    for model_id, repo_id, kw in [
        (models.llm_file, models.llm, "LLM"),
        (models.vlm_file, models.vlm, "VLM"),
    ]:
        load_llama_cpp_model(model_id, repo_id, kw)

    # LLM tokenizer (for chat template formatting)
    if models.llm_tokenizer:
        load_tokenizer(models.llm_tokenizer, paths.hf_hub_cache)

    # Whisper
    load_whisper_model(models.whisper_model)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
