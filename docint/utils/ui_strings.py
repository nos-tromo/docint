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
        "report_section_collection_overview": "Document overview",
        "report_overview_documents": "Documents",
        "report_overview_nodes": "Nodes",
        "report_overview_file_types": "File types",
        "report_overview_entity_types": "Entity types",
        "report_overview_col_document": "Document",
        "report_overview_col_type": "Type",
        "report_overview_col_units": "Pages / rows",
        "report_overview_col_hash": "Hash",
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
        "report_label_posting": "Posting",
        "report_label_account": "Account",
        "report_label_posting_text": "Posting text",
        "report_label_language": "Language",
        "report_label_speaker": "Speaker",
        "report_label_media_id": "Media ID",
        "report_label_machine_translation": "Machine translation",
        "report_label_image_evidence": "Image evidence",
        "report_label_video_keyframe": "Video keyframe",
        "extract_title": "Data extract",
        "extract_heading_documents": "Documents",
        "extract_heading_media": "Audio and video",
        "extract_heading_postings": "Postings",
        "extract_heading_images": "Images",
        "extract_label_contents": "Contents",
        "extract_label_transcript": "Transcript",
        "extract_label_keyframes": "Keyframes",
        "extract_label_figures": "Figures",
        "extract_label_text": "Text",
        "extract_label_description": "Description",
        "extract_label_tags": "Tags",
        "extract_label_ocr_text": "Text in image",
        "extract_label_document": "Document",
        "extract_label_clip": "Clip",
        "extract_label_segments": "Segments",
        "extract_note_order_approximate": (
            "This document records no page or character offsets, so the text below is in storage order, "
            "which may differ from the original reading order."
        ),
        "extract_note_pdf_skipped": (
            "The combined PDF was skipped: this extract is larger than the configured PDF limit. "
            "The per-source Markdown files and figures are complete."
        ),
        "extract_note_no_transcript": "No transcript was produced for this clip.",
        "extract_empty": "This collection holds nothing to extract.",
        "extract_disclaimer": "Transcripts, image descriptions and tags are machine-generated.",
        "report_disclaimer": "AI-generated report.",
        "report_empty": "This report has no items yet.",
    },
    "de": {
        "clarify_generic": "Können Sie präzisieren, was Sie benötigen?",
        "clarify_missing_label": "Fehlende Angaben: {fields}",
        "clarify_missing_request": "Bitte geben Sie an: {fields}.",
        "report_section_chat": "Chat-Antworten",
        "report_section_entities": "Entitäten",
        "report_section_hate_speech": "Hatespeech-Funde",
        "report_section_summaries": "Zusammenfassung",
        "report_section_toc": "Inhaltsverzeichnis",
        "report_section_collection_overview": "Dokumentenübersicht",
        "report_overview_documents": "Dokumente",
        "report_overview_nodes": "Knoten",
        "report_overview_file_types": "Dateitypen",
        "report_overview_entity_types": "Entitätstypen",
        "report_overview_col_document": "Dokument",
        "report_overview_col_type": "Typ",
        "report_overview_col_units": "Seiten / Zeilen",
        "report_overview_col_hash": "Hash",
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
        "report_label_posting": "Beitrag",
        "report_label_account": "Account",
        "report_label_posting_text": "Beitragstext",
        "report_label_language": "Sprache",
        "report_label_speaker": "Sprecher/-in",
        "report_label_media_id": "Medien-ID",
        "report_label_machine_translation": "Maschinelle Übersetzung",
        "report_label_image_evidence": "Bild",
        "report_label_video_keyframe": "Video-Frame",
        "extract_title": "Datenauszug",
        "extract_heading_documents": "Dokumente",
        "extract_heading_media": "Audio und Video",
        "extract_heading_postings": "Beiträge",
        "extract_heading_images": "Bilder",
        "extract_label_contents": "Inhalt",
        "extract_label_transcript": "Transkript",
        "extract_label_keyframes": "Video-Frames",
        "extract_label_figures": "Abbildungen",
        "extract_label_text": "Text",
        "extract_label_description": "Beschreibung",
        "extract_label_tags": "Schlagwörter",
        "extract_label_ocr_text": "Text im Bild",
        "extract_label_document": "Dokument",
        "extract_label_clip": "Clip",
        "extract_label_segments": "Segmente",
        "extract_note_order_approximate": (
            "Dieses Dokument enthält weder Seitenzahlen noch Zeichenpositionen. Der Text steht daher in "
            "Speicherreihenfolge, die von der ursprünglichen Lesereihenfolge abweichen kann."
        ),
        "extract_note_pdf_skipped": (
            "Das kombinierte PDF wurde übersprungen: Dieser Auszug überschreitet das konfigurierte PDF-Limit. "
            "Die Markdown-Dateien und Abbildungen je Quelle sind vollständig."
        ),
        "extract_note_no_transcript": "Für diesen Clip wurde kein Transkript erstellt.",
        "extract_empty": "Diese Collection enthält nichts, was ausgelesen werden könnte.",
        "extract_disclaimer": "Transkripte, Bildbeschreibungen und Schlagwörter sind maschinell erzeugt.",
        "report_disclaimer": ("KI-generierter Bericht."),
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
