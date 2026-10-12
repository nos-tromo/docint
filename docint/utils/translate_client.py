"""Display-time translation of a source snippet via the shared chat model.

Delegates to the OpenAI-compatible chat model behind the LiteLLM router — the
same endpoint docint uses for chat — so there is no dedicated translation
runtime and no app-to-app dependency. Fail-soft: any transport or model error
returns ``TranslateResult(ok=False)`` instead of raising, so a chat, an entities
view, or a report export never crashes on a bad translate call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from loguru import logger

from docint.utils.env_cfg import load_language_env, load_model_env
from docint.utils.openai_cfg import OpenAIPipeline
from docint.utils.prompt_loader import load_localized_prompt

#: What the translate prompt has the model answer, instead of the text, when
#: there is nothing to translate. Protocol, so the same token in every locale.
_SAME_LANGUAGE = "[[SAME_LANGUAGE]]"

_DEFAULT_TRANSLATE_PROMPT = (
    "Translate the user's message into the target language. Output only the "
    "translation — no preamble, no explanation, no quotation marks. Preserve "
    "names, numbers, URLs, and line breaks. If the text is already in the target "
    "language, or contains nothing to translate (only names, numbers or URLs), "
    f"output exactly {_SAME_LANGUAGE} and nothing else."
)


@dataclass(frozen=True)
class TranslateResult:
    """Outcome of a translate call.

    Attributes:
        ok: True on success (including a blank-input no-op).
        translation: Translated text, or None on failure or when there is
            nothing to translate.
        model: The model that produced (or would have produced) the translation.
        target_lang: The destination locale code (the operator's language).
        error: Short machine-readable failure token, or None on success.
        same_language: True when the text needs no translation — it is
            already in ``target_lang``, or holds only names, numbers and URLs.
    """

    ok: bool
    translation: str | None
    model: str
    target_lang: str
    error: str | None = None
    same_language: bool = False


@lru_cache(maxsize=1)
def _pipeline() -> OpenAIPipeline:
    """Process-wide chat pipeline, built once (thread-safe; call_chat is stateless)."""
    return OpenAIPipeline()


@lru_cache(maxsize=4096)
def _translate_cached(text: str, target_lang: str, model: str) -> str:
    """Translate ``text`` (cached).

    Raises on transport/model failure so failures are *not* cached (only
    successful translations are). Sized for a whole Analysis section, since
    "Translate all" walks thousands of findings and its browser-side store is
    session-lifetime — a reload re-asks for every one.
    """
    system_prompt = load_localized_prompt("translate", default=_DEFAULT_TRANSLATE_PROMPT, lang=target_lang)
    return _pipeline().call_chat(text, system_prompt=system_prompt, model=model).strip()


def _words(text: str) -> list[str]:
    """Case-folded words of ``text``, for comparing a reply with its source.

    Quotes, punctuation, ``#`` and emoji drop out, so a reply that merely wraps
    or re-punctuates the source still compares equal to it.
    """
    return re.findall(r"\w+", text.casefold())


def translate(text: str, *, target_lang: str | None = None) -> TranslateResult:
    """Translate a snippet into the operator's active locale, or an explicit one.

    The destination defaults to the active locale (``RESPONSE_LANGUAGE``) — the
    locale translate prompt encodes it. Pass ``target_lang`` when the
    destination is dictated by a downstream model rather than by the operator:
    image retrieval translates to English because the CLIP text tower is
    English-only, whatever locale the UI runs in.

    Text needing no translation comes back with ``same_language`` set and no
    translation, so no caller can show a copy of the source as one. The model
    answers :data:`_SAME_LANGUAGE` for it, and a reply repeating the source
    word for word counts the same. The marker only counts leading the reply,
    and never when the source itself carries it, so no text can suppress its
    own translation.

    Args:
        text: The source snippet (already held by the caller).
        target_lang: Destination locale code, overriding ``RESPONSE_LANGUAGE``.

    Returns:
        TranslateResult: Fail-soft — never raises.
    """
    target_lang = target_lang or load_language_env().code
    model = load_model_env().translate_model
    if not text or not text.strip():
        return TranslateResult(ok=True, translation="", model=model, target_lang=target_lang)
    try:
        translated = _translate_cached(text, target_lang, model)
    except Exception as exc:  # fail-soft: degrade, never crash the caller
        logger.warning("Translation unavailable: {}", exc)
        return TranslateResult(ok=False, translation=None, model=model, target_lang=target_lang, error="unavailable")
    # Models re-bracket, quote or annotate the marker, so it counts as the
    # reply's first word — but only there, and never when the source holds it
    # too: matched anywhere, a hashtag or URL carrying it would let a text's
    # author suppress its faithful translation.
    marker = _SAME_LANGUAGE.strip("[]").casefold()
    reply, source = _words(translated), _words(text)
    if (reply[:1] == [marker] and marker not in source) or reply == source:
        return TranslateResult(ok=True, translation=None, model=model, target_lang=target_lang, same_language=True)
    return TranslateResult(ok=True, translation=translated, model=model, target_lang=target_lang)
