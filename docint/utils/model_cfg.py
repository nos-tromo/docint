import sys
from pathlib import Path

import whisper
from docling.models.code_formula_model import CodeFormulaModel
from docling.models.document_picture_classifier import DocumentPictureClassifier
from docling.models.layout_model import LayoutModel
from docling.models.rapid_ocr_model import RapidOcrModel
from docling.models.table_structure_model import TableStructureModel
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from loguru import logger

from docint.utils.env_cfg import load_model_env, load_path_env
from docint.utils.logging_cfg import setup_logging
from docint.utils.ollama_cfg import OllamaPipeline

load_dotenv()
setup_logging()


def resolve_model_path(model_name: str, cache_folder: Path) -> str:
    """
    Resolves the model path to the local cache directory if available.
    This helps bypass online checks in the transformers library when running offline.

    Args:
        model_name (str): The name of the model (e.g., "bert-base-uncased").
        cache_folder (Path): The path to the Hugging Face cache directory.

    Returns:
        str: Local path to the model if cached, else the original model name.
    """
    if "/" in model_name and not Path(model_name).exists():
        repo_id = model_name
        model_dir_name = f"models--{repo_id.replace('/', '--')}"
        model_cache_dir = cache_folder / model_dir_name

        if model_cache_dir.exists():
            ref_path = model_cache_dir / "refs" / "main"
            if ref_path.exists():
                with open(ref_path, "r") as f:
                    commit_hash = f.read().strip()
                snapshot_path = model_cache_dir / "snapshots" / commit_hash
                if snapshot_path.exists():
                    return str(snapshot_path)
    return model_name


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
        DocumentPictureClassifier.download_models(progress=True)
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
    resolved_model_name = resolve_model_path(model_id, cache_folder)

    if resolved_model_name != model_id:
        logger.info("Found local cache for {} at {}", model_id, resolved_model_name)
        model_id = resolved_model_name
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


def load_ollama_model(model_id: str, kw: str) -> None:
    """
    Loads and returns the Ollama model.

    Args:
        model_id (str): The name of the model to load.
        kw (str): The keyword for the model type (e.g., "generator").

    Returns:
        Ollama: The initialized generation model.
    """
    OllamaPipeline.ensure_model(model_id)
    logger.info("Loaded {} model: {}", kw, model_id)


def load_whisper_model(model_id: str) -> None:
    """
    Loads and returns the Whisper model.

    Args:
        model_id (str): The name of the model to load.
    """
    whisper.load_model("turbo")
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
    for model in models.__dataclass_fields__.keys():
        logger.info("{}: {}", model, getattr(models, model))

    # Load the app's models
    # Docling
    load_docling_models()

    # Hugging Face
    for model_id, kw in [
        (models.embed_model, "embedding"),
        (models.sparse_model, "sparse"),
    ]:
        load_hf_model(
            model_id=model_id,
            cache_folder=paths.hf_hub_cache,
            kw=kw,
        )

    # Ollama
    for model, kw in [
        (models.gen_model, "generator"),
        (models.vision_model, "vision"),
    ]:
        load_ollama_model(model, kw)

    # Whisper
    load_whisper_model(models.whisper_model)

    logger.info("All models loaded successfully.")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
