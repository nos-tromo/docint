"""Offline-first embedding tokenizer loader.

The pre-embed re-splitter (:mod:`docint.utils.embed_chunking`) needs an
accurate token count to bound chunks at the provider's context window.
A character/token ratio heuristic is inherently wrong for multilingual
content (German compounds, CJK, URLs, numeric-dense transcripts), so
this module loads the embedding model's real tokenizer from the local
Hugging Face cache populated by ``uv run load-models``
(:mod:`docint.utils.model_cfg`) and exposes it as a callable.

The returned callable matches the llama_index
:class:`~llama_index.core.node_parser.SentenceSplitter` ``tokenizer``
contract — it takes a string and returns a list whose length equals
the token count — so the same counter is used for upstream chunking
and downstream fit checks.

Offline guarantee: the loader passes ``local_files_only=True`` to
:meth:`transformers.AutoTokenizer.from_pretrained`, so a run with
``DOCINT_OFFLINE=1`` (the default) never reaches the network. If the
snapshot is not in the cache the loader returns ``None`` and the caller
falls back to the character-ratio estimator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from loguru import logger
from transformers import AutoTokenizer

from docint.utils.env_cfg import resolve_hf_cache_path


def build_embedding_token_counter(
    repo_id: str,
    cache_dir: Path,
) -> Callable[[str], list[int]] | None:
    """Build a bge-m3-accurate token counter from the local HF cache.

    Loads the tokenizer for *repo_id* from the cache at *cache_dir*
    using ``local_files_only=True``; the call never reaches the
    network. The returned callable encodes text with
    ``add_special_tokens=True`` and returns the full list of input
    ids, so ``len(counter(text))`` is the token count the embedding
    provider will see (BOS/EOS included) and the callable can be
    passed directly as ``SentenceSplitter(tokenizer=...)``.

    Args:
        repo_id: Hugging Face repository id (e.g. ``"BAAI/bge-m3"``).
            When an empty string is supplied the function short-circuits
            to ``None`` so callers can treat "no tokenizer configured"
            the same as "tokenizer not cached".
        cache_dir: Root of the Hugging Face hub cache populated by
            ``uv run load-models``.

    Returns:
        A callable ``(str) -> list[int]`` suitable for the llama_index
        tokenizer contract, or ``None`` when the snapshot is missing or
        the tokenizer fails to initialise. Callers should log the
        degraded state and fall back to the char-ratio estimator.
    """
    if not repo_id:
        return None

    resolved = resolve_hf_cache_path(cache_dir, repo_id)
    if resolved is None:
        logger.warning(
            "Embedding tokenizer cache not found for {} at {} — "
            "run `uv run load-models` to populate it.",
            repo_id,
            cache_dir,
        )
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(resolved),
            local_files_only=True,
            use_fast=True,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load embedding tokenizer for {} from {}: {}",
            repo_id,
            resolved,
            exc,
        )
        return None

    def _counter(text: str) -> list[int]:
        """Encode *text* with special tokens and return the id list.

        Args:
            text: Input text to tokenize.

        Returns:
            The full list of input ids including BOS/EOS.
        """
        return list(tokenizer.encode(text, add_special_tokens=True))

    return _counter
