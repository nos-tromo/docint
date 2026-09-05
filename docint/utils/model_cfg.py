"""Embedding-tokenizer cache preparation for offline token counting."""

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
    load_path_env,
    resolve_hf_cache_path,
)
# isort: on

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger

from docint.utils.logger_cfg import init_logger

load_dotenv()
init_logger()


# Tokenizer files only. The repo also carries the model weights, which docint
# never loads — chat, embedding, rerank, NER and CLIP are all remote calls.
TOKENIZER_PATTERNS: tuple[str, ...] = (
    "tokenizer*",
    "sentencepiece*",
    "special_tokens_map.json",
    "config.json",
)


def load_embed_tokenizer(repo_id: str, cache_folder: Path) -> None:
    """Ensure the embedding tokenizer's files are in the local HF cache.

    Args:
        repo_id (str): Hugging Face repository id of the tokenizer
            (``EMBED_TOKENIZER_REPO``, e.g. ``"BAAI/bge-m3"``).
        cache_folder (Path): Root of the Hugging Face hub cache to
            populate.
    """
    resolved = resolve_hf_cache_path(cache_dir=cache_folder, repo_id=repo_id)

    if resolved:
        logger.info("Embedding tokenizer '{}' already cached at {}", repo_id, resolved)
        return

    snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_folder,
        allow_patterns=list(TOKENIZER_PATTERNS),
    )
    logger.info("Cached embedding tokenizer '{}' in {}", repo_id, cache_folder)


def main() -> None:
    """Cache the embedding tokenizer used for offline token counting.

    The tokenizer is the only model asset docint holds locally: every
    model call (chat, embedding, rerank, NER, CLIP, sparse) is an HTTP
    request to vllm-service, which preloads its own weights. The Docker
    image runs this at build time, so this command is for ``uv run``
    development hosts.
    """
    path_config = load_path_env()
    model_config = load_model_env()

    repo_id = model_config.embed_tokenizer_repo
    if not repo_id:
        logger.info("EMBED_TOKENIZER_REPO is empty — the provider tokenizes server-side, nothing to cache.")
        return

    load_embed_tokenizer(repo_id=repo_id, cache_folder=path_config.hf_hub_cache)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
