"""Locale-aware user-facing UI strings.

Holds the small set of non-prompt strings docint returns directly to the
user — currently just the clarification messages emitted when an agent
asks the user to refine a query. LLM-facing prompt templates live under
``docint/utils/prompts/{en,de}/`` and are not handled here.

The active language follows :func:`docint.utils.env_cfg.load_language_env`,
which is driven by the ``RESPONSE_LANGUAGE`` env var.
"""

from typing import Final

from docint.utils.env_cfg import SUPPORTED_LANGUAGES, load_language_env

UI_STRINGS: Final[dict[str, dict[str, str]]] = {
    "en": {
        "clarify_generic": "Could you clarify what you need?",
        "clarify_missing_label": "Missing details: {fields}",
        "clarify_missing_request": "Please provide: {fields}.",
    },
    "de": {
        "clarify_generic": "Können Sie präzisieren, was Sie benötigen?",
        "clarify_missing_label": "Fehlende Angaben: {fields}",
        "clarify_missing_request": "Bitte geben Sie an: {fields}.",
    },
}

# Sanity-check at import time so a missing translation surfaces as a clear
# startup error rather than a KeyError deep inside an agent path.
_EN_KEYS = set(UI_STRINGS["en"])
for _lang in SUPPORTED_LANGUAGES:
    if set(UI_STRINGS[_lang]) != _EN_KEYS:
        missing = _EN_KEYS.symmetric_difference(UI_STRINGS[_lang])
        raise RuntimeError(f"UI_STRINGS for '{_lang}' diverges from 'en' on: {sorted(missing)}")


def ui_string(key: str) -> str:
    """Return the UI string for ``key`` in the currently configured language.

    Args:
        key (str): Name of the UI string (see :data:`UI_STRINGS`).

    Returns:
        str: The localized string.

    Raises:
        KeyError: If ``key`` is not registered in :data:`UI_STRINGS`.
    """
    return UI_STRINGS[load_language_env().code][key]
