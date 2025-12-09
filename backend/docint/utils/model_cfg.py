import sys
from pathlib import Path

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

    Returns:
        Ollama: The initialized generation model.
    """
    OllamaPipeline.ensure_model(model_id)
    logger.info("Loaded {} model: {}", kw, model_id)


def main() -> None:
    """
    Main function to verify configuration loading.
    """
    paths = load_path_env()
    models = load_model_env()

    logger.info("Paths: {}", paths)
    logger.info("Models: {}", models)

    # Load the RAG models
    load_hf_model(
        model_id=models.embed_model,
        cache_folder=paths.hf_hub_cache,
        kw="embedding",
    )
    load_hf_model(
        model_id=models.sparse_model,
        cache_folder=paths.hf_hub_cache,
        kw="sparse",
    )
    load_ollama_model(models.gen_model, "generator")
    load_ollama_model(models.vision_model, "vision")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
