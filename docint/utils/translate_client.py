"""Display-time translation of a source snippet via the shared chat model.

Delegates to the OpenAI-compatible chat model behind the LiteLLM router — the
same endpoint docint uses for chat — so there is no dedicated translation
runtime and no app-to-app dependency. Fail-soft: any transport or model error
returns ``TranslateResult(ok=False)`` instead of raising, so a chat, an entities
view, or a report export never crashes on a bad translate call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from loguru import logger

from docint.utils.env_cfg import load_language_env, load_model_env
from docint.utils.openai_cfg import OpenAIPipeline
from docint.utils.prompt_loader import load_localized_prompt

_DEFAULT_TRANSLATE_PROMPT = (
    "Translate the user's message into the target language. Output only the "
    "translation — no preamble, no explanation, no quotation marks. Preserve "
    "names, numbers, URLs, and line breaks."
)


@dataclass(frozen=True)
class TranslateResult:
    """Outcome of a translate call.

    Attributes:
        ok: True on success (including a blank-input no-op).
        translation: Translated text, or None on failure.
        model: The model that produced (or would have produced) the translation.
        target_lang: The destination locale code (the operator's language).
        error: Short machine-readable failure token, or None on success.
    """

    ok: bool
    translation: str | None
    model: str
    target_lang: str
    error: str | None = None


@lru_cache(maxsize=1)
def _pipeline() -> OpenAIPipeline:
    """Process-wide chat pipeline, built once (thread-safe; call_chat is stateless)."""
    return OpenAIPipeline()


@lru_cache(maxsize=512)
def _translate_cached(text: str, target_lang: str, model: str) -> str:
    """Translate ``text`` (cached).

    Raises on transport/model failure so failures are *not* cached (only
    successful translations are).
    """
    system_prompt = load_localized_prompt("translate", default=_DEFAULT_TRANSLATE_PROMPT)
    return _pipeline().call_chat(text, system_prompt=system_prompt, model=model).strip()


def translate(text: str, *, target: str | None = None) -> TranslateResult:
    """Translate a snippet into the operator's locale (or ``target`` if given).

    Args:
        text: The source snippet (already held by the caller).
        target: Optional destination locale code; defaults to the active locale.

    Returns:
        TranslateResult: Fail-soft — never raises.
    """
    target_lang = target or load_language_env().code
    model = load_model_env().translate_model
    if not text or not text.strip():
        return TranslateResult(ok=True, translation="", model=model, target_lang=target_lang)
    try:
        translated = _translate_cached(text, target_lang, model)
        return TranslateResult(ok=True, translation=translated, model=model, target_lang=target_lang)
    except Exception as exc:  # fail-soft: degrade, never crash the caller
        logger.warning("Translation unavailable: {}", exc)
        return TranslateResult(ok=False, translation=None, model=model, target_lang=target_lang, error="unavailable")
