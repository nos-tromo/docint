"""Tests for report renderers (Markdown / HTML / PDF / JSON / CSV bundle)."""

import io
import json
import re
import zipfile
from typing import Any, cast

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
    assert "Entitäten" in md  # de translation of report_section_entities


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


def test_prose_items_flow_while_findings_stay_intact() -> None:
    """Page-break contract: prose items flow; only findings avoid in-page breaks.

    A summary or chat answer is often taller than a page. If it carries
    ``break-inside: avoid`` WeasyPrint pushes the whole block onto a fresh page,
    stranding the section heading on an almost-empty page (an orphaned heading)
    and leaving a large gap. So the prose artifacts (summary, chat answer) must
    flow, and the ``break-inside: avoid`` guard belongs only to the compact
    entity / hate-speech finding cards (``.item--card``).
    """
    htm = R.render_html(_report())  # chat (prose) + entity + hate (cards)
    # The break-avoid guard lives on the card modifier, not the base item rule.
    assert ".item--card" in htm
    assert "break-inside: avoid" in htm
    base_item_rule = re.search(r"\.item\s*\{([^}]*)\}", htm)
    assert base_item_rule is not None
    assert "break-inside" not in base_item_rule.group(1)  # the base item flows
    # Findings opt into staying intact; prose does not.
    card_report = _single_item_report(
        "entity_finding",
        {"entity_label": "E", "chunk_text": "x", "filename": "f", "row": 0, "entities": []},
    )
    assert 'class="item item--card"' in R.render_html(card_report)
    prose_report = _single_item_report("summary", {"collection": "c", "text": "long prose body"})
    prose_html = R.render_html(prose_report)
    assert 'class="item"' in prose_html  # prose keeps the plain item class …
    assert 'class="item item--card"' not in prose_html  # … never the break-avoid card modifier


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
    report: dict[str, Any] = {
        "id": 9,
        "title": "Ordered",
        "collection_name": "c",
        "created_at": "2026-06-20T10:00:00+00:00",
        "items": [
            {
                "id": 1,
                "artifact_type": "chat_answer",
                "note": None,
                "snapshot": {"user_text": "q", "model_response": "a", "sources": cast(list[dict[str, Any]], [])},
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
    # It rides the running header instead — prefixed with a discreet abbreviated label.
    refnum = re.search(r'<div class="running-refnum">(.*?)</div>', htm, re.S)
    assert refnum is not None
    assert refnum.group(1).strip() == f"{ui_string('report_label_reference_abbr')}: AZ-2026-42"
    assert ui_string("report_label_reference") not in htm  # the long "File reference" label never leaks in

    # No case file set → no running-header marker at all (the header stays empty).
    assert 'class="running-refnum"' not in R.render_html(_empty())


@pytest.mark.parametrize("locale", ["en", "de"])
def test_running_header_case_file_is_labeled_per_locale(monkeypatch: pytest.MonkeyPatch, locale: str) -> None:
    """The case-file header carries a discreet, localized abbreviation label.

    A bare number in the page corner reads like an artifact; a short prefix
    (``File:`` / ``Az.:``) makes it legible as a case reference. The label is a
    per-language string, so it differs between locales by design.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", locale)
    htm = R.render_html(_report())
    refnum = re.search(r'<div class="running-refnum">(.*?)</div>', htm, re.S)
    assert refnum is not None
    assert refnum.group(1).strip() == f"{ui_string('report_label_reference_abbr')}: AZ-2026-42"


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


def _single_item_report(artifact_type: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    """A minimal one-item report for exercising a single renderer in isolation."""
    return {
        "id": 1,
        "title": "T",
        "collection_name": "c",
        "created_at": "2026-06-20T10:00:00+00:00",
        "items": [{"id": 1, "artifact_type": artifact_type, "note": None, "snapshot": snapshot}],
    }


def test_html_renders_markdown_in_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summary Markdown (bold, bullets) is rendered to HTML, never shown raw."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report("summary", {"collection": "c", "text": "Lead in.\n\n* **alpha** point\n* beta"})
    htm = R.render_html(report)
    assert "<strong>alpha</strong>" in htm  # bold rendered
    assert "<li>" in htm  # bullets rendered
    assert "* **alpha**" not in htm  # the raw markdown markers are gone


def test_html_renders_markdown_in_chat_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat-answer Markdown is rendered to HTML."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report(
        "chat_answer", {"user_text": "q", "model_response": "It is **strongly** so.", "sources": []}
    )
    htm = R.render_html(report)
    assert "<strong>strongly</strong>" in htm
    assert "**strongly**" not in htm


def test_html_summary_renders_as_prose_not_evidence_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """A summary flows as prose, not inside the grey `.chunk` evidence box."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report("summary", {"collection": "c", "text": "plain summary body"})
    htm = R.render_html(report)
    assert "plain summary body" in htm
    assert 'class="chunk"' not in htm  # the only body here is the summary; it must not be boxed


def test_html_escapes_raw_markup_inside_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw HTML embedded in summary/chat Markdown is escaped, never injected."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report("summary", {"collection": "c", "text": "see <script>alert(1)</script>"})
    htm = R.render_html(report)
    assert "<script>alert" not in htm
    assert "&lt;script&gt;" in htm


def test_html_dedupes_entity_chips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated entities collapse to one chip each (case-insensitive)."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report(
        "entity_finding",
        {
            "chunk_id": "c1",
            "entity_label": "Männer [group]",
            "chunk_text": "…",
            "filename": "x.csv",
            "row": 0,
            "entities": [
                {"text": "Männer", "type": "group"},
                {"text": "männer", "type": "group"},  # case-variant duplicate
                {"text": "Männer", "type": "group"},  # exact duplicate
                {"text": "Volk", "type": "group"},
            ],
        },
    )
    htm = R.render_html(report)
    assert htm.count('class="badge"') == 2  # Männer + Volk only


def test_relevance_score_dropped_from_report_but_kept_in_csv() -> None:
    """The [score] is removed from the human-facing PDF/HTML/Markdown, but kept in the CSV data."""
    report = _report()  # its chat citation carries score 0.91 -> "[0.910]"
    assert "[0.910]" not in R.render_html(report)
    assert "[0.910]" not in R.render_markdown(report)
    zf = zipfile.ZipFile(io.BytesIO(R.report_csv_bundle(report)))
    assert "[0.910]" in zf.read("chat-answers.csv").decode("utf-8")


def _toc_report(show_toc: bool = True) -> dict[str, Any]:
    """A report carrying all four section types, for table-of-contents tests."""
    return {
        "id": 1,
        "title": "T",
        "collection_name": "c",
        "created_at": "2026-06-20T10:00:00+00:00",
        "show_toc": show_toc,
        "items": [
            {"id": 1, "artifact_type": "summary", "note": None, "snapshot": {"collection": "c", "text": "s"}},
            {
                "id": 2,
                "artifact_type": "chat_answer",
                "note": None,
                "snapshot": {"user_text": "q", "model_response": "a", "sources": []},
            },
            {
                "id": 3,
                "artifact_type": "entity_finding",
                "note": None,
                "snapshot": {"entity_label": "E", "chunk_text": "x", "filename": "f", "row": 0, "entities": []},
            },
            {
                "id": 4,
                "artifact_type": "hate_speech_finding",
                "note": None,
                "snapshot": {"category": "x", "confidence": "high", "reason": "r", "chunk_text": "x", "filename": "f"},
            },
        ],
    }


def test_html_toc_lists_present_sections_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With show_toc on, a contents block links every present section by anchor."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    htm = R.render_html(_toc_report(show_toc=True))
    assert 'class="toc"' in htm
    assert ui_string("report_section_toc") in htm
    for anchor in ("#sec-summaries", "#sec-chat", "#sec-entities", "#sec-hate"):
        assert f'href="{anchor}"' in htm
    assert 'id="sec-chat"' in htm  # the section heading carries the matching id
    assert "target-counter" in htm  # WeasyPrint page-number mechanism present in CSS


def test_html_toc_absent_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With show_toc off, no contents block is rendered."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    htm = R.render_html(_toc_report(show_toc=False))
    assert 'class="toc"' not in htm


def test_html_toc_lists_only_present_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contents block lists only sections that actually have content."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report("chat_answer", {"user_text": "q", "model_response": "a", "sources": []})
    report["show_toc"] = True
    htm = R.render_html(report)
    assert 'href="#sec-chat"' in htm
    assert 'href="#sec-entities"' not in htm
    assert 'href="#sec-summaries"' not in htm


def test_markdown_toc_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Markdown export carries a contents list (no page numbers) when enabled."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    md = R.render_markdown(_toc_report(show_toc=True))
    toc = ui_string("report_section_toc")
    assert toc in md
    assert md.index(toc) < md.index(ui_string("report_section_summaries"))  # leads the document


def test_markdown_toc_absent_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """No contents list in Markdown when the toggle is off."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    assert ui_string("report_section_toc") not in R.render_markdown(_toc_report(show_toc=False))


_OVERVIEW: dict[str, Any] = {
    "collection": "c1",
    "captured_at": "2026-07-06T10:00:00+00:00",
    "document_count": 2,
    "node_count": 9,
    "file_types": [{"label": "PDF", "count": 1}, {"label": "CSV", "count": 1}],
    "entity_types": ["ORG", "PER"],
    "documents": [
        {
            "filename": "a.pdf",
            "type_label": "PDF",
            "page_count": 4,
            "row_count": None,
            "node_count": 6,
            "file_hash": "0123456789abcdefff",
        },
        {
            "filename": "b.csv",
            "type_label": "CSV",
            "page_count": 0,
            "row_count": 30,
            "node_count": 3,
            "file_hash": "deadbeefcafebabe00",
        },
    ],
}


def _overview_report(**over: Any) -> dict[str, Any]:
    """A minimal report dict with the document-overview toggled on by default.

    Named distinctly from the module's ``_report()`` (the "Case Alpha" fixture
    used throughout this file) — reusing that name would shadow it, since
    Python resolves a bare-name call against whatever the module global is
    *at call time*, silently rebinding every existing ``_report()`` call to
    this smaller dict.
    """
    base: dict[str, Any] = {
        "title": "R",
        "items": [],
        "show_toc": True,
        "show_collection_overview": True,
        "collection_overview": _OVERVIEW,
    }
    base.update(over)
    return base


def test_overview_renders_last_in_markdown_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trailing overview section renders with its manifest table when enabled."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    md = R.render_markdown(_overview_report())
    assert "Document overview" in md
    assert "a.pdf" in md and "b.csv" in md
    assert "0123456789ab" in md and "0123456789abcdefff" not in md  # hash truncated to 12


def test_overview_omitted_when_toggled_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Toggling show_collection_overview off omits the section entirely."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    md = R.render_markdown(_overview_report(show_collection_overview=False))
    assert "Document overview" not in md


def test_overview_omitted_when_empty_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """An overview snapshot with no documents is omitted like an empty item section."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    empty: dict[str, Any] = {**_OVERVIEW, "documents": []}
    md = R.render_markdown(_overview_report(collection_overview=empty))
    assert "Document overview" not in md
    # items empty AND no overview -> the "empty report" copy actually renders.
    assert "This report has no items yet" in md


def test_overview_only_report_is_not_empty_and_appears_in_html_toc(monkeypatch: pytest.MonkeyPatch) -> None:
    """A report with only an overview (no items) is not treated as empty, and gets a TOC entry."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    htm = R.render_html(_overview_report(show_toc=True))
    assert 'id="sec-collection-overview"' in htm
    assert "This report has no items yet" not in htm  # overview counts as content
    assert "#sec-collection-overview" in htm  # TOC entry present
    assert "0123456789ab" in htm and "0123456789abcdefff" not in htm  # hash truncated to 12


def test_overview_renders_after_items_in_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trailing overview section appears after the item sections in output order."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    item = {
        "id": 1,
        "artifact_type": "summary",
        "note": None,
        "snapshot": {"collection": "c1", "text": "UNIQUE_ITEM_BODY_MARKER"},
    }
    md = R.render_markdown(_overview_report(items=[item]))
    assert "Document overview" in md
    assert "UNIQUE_ITEM_BODY_MARKER" in md  # the item body rendered …
    # Match the "## " section heading, not the "- " TOC entry (which precedes the
    # item body): the guarantee under test is that the overview *section* trails.
    assert md.index("## Document overview") > md.index("UNIQUE_ITEM_BODY_MARKER")


def test_csv_bundle_includes_overview_with_full_hash() -> None:
    """The CSV bundle carries collection-overview.csv with the untruncated hash."""
    zf = zipfile.ZipFile(io.BytesIO(R.report_csv_bundle(_overview_report())))
    assert "collection-overview.csv" in zf.namelist()
    body = zf.read("collection-overview.csv").decode()
    assert "a.pdf" in body and "0123456789abcdefff" in body  # full hash in CSV, unlike the display truncation


def test_csv_bundle_omits_overview_when_off() -> None:
    """No collection-overview.csv when the overview toggle is off."""
    zf = zipfile.ZipFile(io.BytesIO(R.report_csv_bundle(_overview_report(show_collection_overview=False))))
    assert "collection-overview.csv" not in zf.namelist()
