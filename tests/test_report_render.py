"""Tests for report renderers (Markdown / HTML / PDF / JSON / CSV bundle)."""

import io
import json
import re
import zipfile
from typing import Any

import pytest

from docint.core.state import report_render as R
from docint.utils.ui_strings import ui_string


def _report() -> dict[str, Any]:
    """A report dict shaped like ReportManager.get_report output."""
    return {
        "id": 1,
        "title": "Case Alpha",
        "collection_name": "docs",
        "operator": "Jane Doe",
        "reference_number": "AZ-2026-42",
        "created_at": "2026-06-20T10:00:00+00:00",
        "updated_at": "2026-06-20T10:05:00+00:00",
        "item_count": 3,
        "items": [
            {
                "id": 1,
                "artifact_type": "chat_answer",
                "note": "key answer",
                "snapshot": {
                    "session_id": "s1",
                    "turn_idx": 0,
                    "user_text": "Who is Acme?",
                    "model_response": "Acme is an org.",
                    "sources": [{"filename": "a.pdf", "page": 2, "score": 0.91}],
                },
            },
            {
                "id": 2,
                "artifact_type": "entity_finding",
                "note": None,
                "snapshot": {
                    "chunk_id": "c1",
                    "entity_label": "Acme [ORG]",
                    "chunk_text": "Acme met Bob <script>alert(1)</script>",
                    "filename": "a.pdf",
                    "page": 2,
                    "entities": [{"text": "Acme", "type": "ORG"}, {"text": "Bob", "type": "PERSON"}],
                    "reference_metadata": {
                        "network": "Telegram",
                        "author": "alice",
                        "timestamp": "2026-01-02T00:00:00Z",
                        "uuid": "u-1",
                    },
                },
            },
            {
                "id": 3,
                "artifact_type": "hate_speech_finding",
                "note": None,
                "snapshot": {
                    "chunk_id": "c9",
                    "category": "slur",
                    "confidence": "high",
                    "reason": "contains slur",
                    "chunk_text": "bad text",
                    "filename": "b.json",
                    "reference_metadata": {"network": "X", "author": "bob", "timestamp": "2026-03-04"},
                },
            },
        ],
    }


def _empty() -> dict[str, Any]:
    return {
        "id": 2,
        "title": "Empty",
        "collection_name": None,
        "created_at": None,
        "updated_at": None,
        "item_count": 0,
        "items": [],
    }


def test_render_json_round_trips() -> None:
    """JSON export preserves the title and every item snapshot."""
    data = json.loads(R.render_json(_report()))
    assert data["title"] == "Case Alpha"
    assert len(data["items"]) == 3
    assert data["items"][1]["snapshot"]["entity_label"] == "Acme [ORG]"


def test_render_markdown_sections_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markdown sections render in Chat -> Entities -> Hate-speech order."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    md = R.render_markdown(_report())
    assert "# Case Alpha" in md
    i_chat = md.index(ui_string("report_section_chat"))
    i_ent = md.index(ui_string("report_section_entities"))
    i_hate = md.index(ui_string("report_section_hate_speech"))
    assert i_chat < i_ent < i_hate
    assert "Acme [ORG]" in md


def test_render_markdown_empty_uses_locale_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty report renders the localized 'no items' notice, not an error."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    assert R.render_markdown(_empty()).strip().endswith("no items yet.")


def test_render_de_locale_headings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Section headings are localized when the response language is German."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    md = R.render_markdown(_report())
    assert "Entitätsfunde" in md  # de translation of report_section_entities


def test_render_html_escapes_user_content_and_has_paged_media() -> None:
    """HTML escapes snapshot text and includes CSS paged-media rules."""
    htm = R.render_html(_report())
    assert "&lt;script&gt;" in htm  # snapshot text is HTML-escaped
    assert "<script>alert" not in htm
    assert "counter(page)" in htm
    # Case file rides the running top-right header; the report name is no longer
    # duplicated into the page header via a doctitle string.
    assert "position: running(refnum)" in htm
    assert "element(refnum)" in htm
    assert "string-set: doctitle" not in htm
    assert 'class="item"' in htm


def test_render_includes_case_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator and file reference appear in the Markdown and HTML headers."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    md = R.render_markdown(_report())
    assert "Jane Doe" in md and "AZ-2026-42" in md
    htm = R.render_html(_report())
    assert "Jane Doe" in htm and "AZ-2026-42" in htm


def test_summaries_render_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summaries lead the document, ahead of the chat-answers section."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = {
        "id": 9,
        "title": "Ordered",
        "collection_name": "c",
        "created_at": "2026-06-20T10:00:00+00:00",
        "items": [
            {
                "id": 1,
                "artifact_type": "chat_answer",
                "note": None,
                "snapshot": {"user_text": "q", "model_response": "a", "sources": []},
            },
            {
                "id": 2,
                "artifact_type": "summary",
                "note": None,
                "snapshot": {"collection": "c", "text": "the summary"},
            },
        ],
    }
    md = R.render_markdown(report)
    assert md.index(ui_string("report_section_summaries")) < md.index(ui_string("report_section_chat"))
    htm = R.render_html(report)
    assert htm.index(ui_string("report_section_summaries")) < htm.index(ui_string("report_section_chat"))


def test_findings_carry_reference_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entity and hate-speech findings surface their reference metadata (provenance)."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    label = ui_string("report_label_reference_metadata")
    for blob in (R.render_markdown(_report()), R.render_html(_report())):
        assert blob.count(label) >= 2  # one block for the entity finding, one for the hate finding
        assert "Telegram" in blob and "alice" in blob  # entity finding provenance
        assert "bob" in blob  # hate-speech finding provenance


def test_case_file_only_in_running_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """The case file rides the running header — not the subheader — and the date is date-only."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    htm = R.render_html(_report())
    meta = re.search(r'<div class="report-meta">(.*?)</div>', htm, re.S)
    assert meta is not None
    subheader = meta.group(1)
    assert "docs" in subheader  # collection
    assert "Jane Doe" in subheader  # operator
    assert "2026-06-20" in subheader  # creation date …
    assert "10:00" not in subheader  # … without the time component
    assert "AZ-2026-42" not in subheader  # the case file is kept out of the subheader
    # It rides the running header instead — shown bare, with no label prefix.
    refnum = re.search(r'<div class="running-refnum">(.*?)</div>', htm, re.S)
    assert refnum is not None
    assert refnum.group(1).strip() == "AZ-2026-42"
    assert ui_string("report_label_reference") not in htm  # no "File reference:" label leaks in

    # No case file set → no running-header marker at all (the header stays empty).
    assert 'class="running-refnum"' not in R.render_html(_empty())


@pytest.mark.parametrize("locale", ["en", "de"])
def test_running_header_bare_value_is_locale_agnostic(monkeypatch: pytest.MonkeyPatch, locale: str) -> None:
    """The case-file header shows the bare value with no label — identically in every locale.

    The label was dropped from the shared renderer (not a per-language string), so
    neither the English ``File reference`` nor the German ``Aktenzeichen`` prefix
    may leak into the header for either locale.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", locale)
    htm = R.render_html(_report())
    refnum = re.search(r'<div class="running-refnum">(.*?)</div>', htm, re.S)
    assert refnum is not None
    assert refnum.group(1).strip() == "AZ-2026-42"  # bare value, byte-for-byte the same en/de
    assert ui_string("report_label_reference") not in htm  # this locale's label never appears


def test_disclaimer_footer_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short AI-generation caveat is rendered for both Markdown and HTML/PDF."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    disclaimer = ui_string("report_disclaimer")
    assert disclaimer in R.render_markdown(_report())
    htm = R.render_html(_report())
    assert disclaimer in htm
    assert 'class="running-disclaimer"' in htm


def test_report_name_only_in_headline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The report name is the single H1 headline, not echoed elsewhere in Markdown."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    md = R.render_markdown(_report())
    assert md.count("# Case Alpha") == 1


def test_pdf_footer_layout() -> None:
    """Page numbers sit bottom-right, the AI disclaimer bottom-left, none centered."""
    htm = R.render_html(_report())
    assert "@bottom-right" in htm
    assert "@bottom-left" in htm  # AI-generated disclaimer footer
    assert "@bottom-center" not in htm


def test_csv_bundle_entries_and_canonical_schema() -> None:
    """The CSV bundle has per-type files reusing the canonical column schemas."""
    zf = zipfile.ZipFile(io.BytesIO(R.report_csv_bundle(_report())))
    assert set(zf.namelist()) == {"entity-findings.csv", "hate-speech.csv", "chat-answers.csv"}
    ent_header = zf.read("entity-findings.csv").decode("utf-8").splitlines()[0]
    assert "chunk_id" in ent_header  # reuses the canonical NER-source schema
    hate_header = zf.read("hate-speech.csv").decode("utf-8").splitlines()[0]
    assert "category" in hate_header


def test_csv_bundle_empty_has_readme() -> None:
    """An empty report's CSV bundle contains a README placeholder."""
    zf = zipfile.ZipFile(io.BytesIO(R.report_csv_bundle(_empty())))
    assert zf.namelist() == ["README.txt"]


def test_render_pdf_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """render_pdf raises PdfEngineUnavailableError when WeasyPrint is missing."""
    monkeypatch.setattr(R, "_load_weasyprint", lambda: (None, ImportError("no native libs")))
    with pytest.raises(R.PdfEngineUnavailableError):
        R.render_pdf(_report())


def test_render_pdf_available_returns_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """render_pdf returns PDF bytes when the engine is available."""

    class _FakeHTML:
        def __init__(self, string: str) -> None:
            self.string = string

        def write_pdf(self) -> bytes:
            assert "Case Alpha" in self.string
            return b"%PDF-1.7 fake"

    monkeypatch.setattr(R, "_load_weasyprint", lambda: (_FakeHTML, None))
    out = R.render_pdf(_report())
    assert out.startswith(b"%PDF")
