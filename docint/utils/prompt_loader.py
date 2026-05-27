"""Load a locale-aware prompt template from ``docint/utils/prompts/``.

This is a thin helper for callsites outside of :class:`docint.core.rag.RAG`
that need to read a prompt template from the active language directory.
``RAG`` has its own ``_load_prompt_text`` static method that operates on a
:class:`pathlib.Path`; this helper resolves the path from a ``name`` and
the currently configured ``RESPONSE_LANGUAGE``.
"""

from __future__ import annotations

from loguru import logger

from docint.utils.env_cfg import load_language_env, load_path_env


def load_localized_prompt(name: str, *, default: str, required: bool = False) -> str:
    """Read ``{name}.txt`` from the active locale's prompt directory.

    Args:
        name: Prompt basename (without ``.txt``).
        default: Fallback text returned if the file is missing and
            ``required`` is False. Typically an English prompt baked into
            the calling module so the deploy keeps working even if the
            language pack is broken.
        required: When True, a missing file raises :class:`FileNotFoundError`
            instead of returning ``default``.

    Returns:
        The prompt text (stripped of trailing whitespace).

    Raises:
        FileNotFoundError: If the file is missing and ``required`` is True.
    """
    prompt_path = load_path_env().prompts / load_language_env().code / f"{name}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        if required:
            raise
        logger.warning(
            "Localized prompt '{}' not found at {}; falling back to bundled default.",
            name,
            prompt_path,
        )
        return default
