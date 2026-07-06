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
        "report_section_chat": "Chat answers",
        "report_section_entities": "Entity findings",
        "report_section_hate_speech": "Hate-speech findings",
        "report_section_summaries": "Summaries",
        "report_section_toc": "Contents",
        "report_label_collection": "Collection",
        "report_label_generated": "Generated",
        "report_label_operator": "Operator",
        "report_label_reference": "File reference",
        "report_label_reference_abbr": "File",
        "report_label_source": "Source",
        "report_label_page": "Page",
        "report_label_row": "Row",
        "report_label_score": "Score",
        "report_label_category": "Category",
        "report_label_confidence": "Confidence",
        "report_label_reason": "Reason",
        "report_label_entities": "Entities",
        "report_label_note": "Note",
        "report_label_question": "Question",
        "report_label_answer": "Answer",
        "report_label_sources": "Sources",
        "report_label_items": "Items",
        "report_label_reference_metadata": "Reference metadata",
        "report_label_machine_translation": "Machine translation",
        "report_disclaimer": "AI-generated report.",
        "report_empty": "This report has no items yet.",
    },
    "de": {
        "clarify_generic": "Können Sie präzisieren, was Sie benötigen?",
        "clarify_missing_label": "Fehlende Angaben: {fields}",
        "clarify_missing_request": "Bitte geben Sie an: {fields}.",
        "report_section_chat": "Chat-Antworten",
        "report_section_entities": "Entitäten",
        "report_section_hate_speech": "Gruppenbezogene Menschenfeindlichkeit (GMF)",
        "report_section_summaries": "Zusammenfassungen",
        "report_section_toc": "Inhaltsverzeichnis",
        "report_label_collection": "Sammlung",
        "report_label_generated": "Erstellt",
        "report_label_operator": "Bearbeiter/-in",
        "report_label_reference": "Aktenzeichen",
        "report_label_reference_abbr": "Az.",
        "report_label_source": "Quelle",
        "report_label_page": "Seite",
        "report_label_row": "Zeile",
        "report_label_score": "Bewertung",
        "report_label_category": "Kategorie",
        "report_label_confidence": "Konfidenz",
        "report_label_reason": "Begründung",
        "report_label_entities": "Entitäten",
        "report_label_note": "Notiz",
        "report_label_question": "Frage",
        "report_label_answer": "Antwort",
        "report_label_sources": "Quellen",
        "report_label_items": "Einträge",
        "report_label_reference_metadata": "Metadaten",
        "report_label_machine_translation": "Maschinelle Übersetzung",
        "report_disclaimer": ("Dieser Bericht wurde KI-gestützt erstellt."),
        "report_empty": "Dieser Bericht enthält noch keine Einträge.",
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
