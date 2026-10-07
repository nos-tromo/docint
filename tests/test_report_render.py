"""Tests for report renderers (Markdown / HTML / PDF / JSON / CSV bundle)."""

import base64
import io
import json
import re
import zipfile
from typing import Any, cast

import pytest
from pdf_layout import (
    content_width_share,
    contents_entries,
    count_min_content_splits,
    hyphenated_text,
    pages_showing_a_finding_without_its_band,
    rows_alone_at_page_foot,
    rows_split_across_pages,
    text_beyond_its_cell,
    text_beyond_the_page,
    weasyprint_html,
)

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
                    "filename": "clip.mp4.nextext.jsonl",
                    "row": 2,
                    # A media-derived transcript segment: internal pipeline stamps
                    # plus the parent posting's reference fields (additive carry).
                    "reference_metadata": {
                        "network": "nextext",
                        "type": "transcript_segment",
                        "posting_uuid": "pu-1",
                        "posting_id": "P1",
                        "media_id": "P1",
                        "posting_network": "Facebook",
                        "posting_author": "Jane Poster",
                        "posting_author_id": "42",
                        "posting_vanity": "jane.poster",
                        "posting_timestamp": "2026-03-04 09:00:00+00",
                        "posting_url": "https://fb.example/p1",
                        "posting_text": "Original post body",
                        "timestamp": "00:00:08",
                        "text_id": "clip.mp4:2",
                        "language": "en",
                        "source_file": "clip.mp4",
                    },
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


def test_only_short_finding_rows_resist_page_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Page-break contract: items and finding tables flow; only their short rows move whole.

    A finding table (full chunk text + entity badges) is routinely taller than
    a page. A ``break-inside: avoid`` on the item, the table or a giant row
    makes WeasyPrint push it onto a fresh page, stranding the section heading
    on an almost-empty page and leaving page-sized gaps. So the guard sits on
    the short rows alone — provenance, reason, the picture and its words —
    which is what stops a label ending one page while its value starts the
    next. The chunk row and the entity badges stay breakable.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    htm = R.render_html(_report())  # chat (prose) + entity + hate findings
    assert ".item--card" not in htm  # the unbreakable-card modifier is gone
    base_item_rule = re.search(r"\.item\s*\{([^}]*)\}", htm)
    assert base_item_rule is not None
    assert "break-inside" not in base_item_rule.group(1)  # items flow
    guarded = [
        selector.strip()
        for selector, body in re.findall(r"(table\.finding[^{]*)\{([^}]*)\}", htm)
        if "break-inside" in body
    ]
    assert guarded == ["table.finding tr.f-keep, table.finding tr.f-media"]
    assert '<tr><td colspan="2" class="f-text">bad text</td></tr>' in htm  # the chunk row splits
    assert '<tr><td class="f-key">Entities</td>' in htm  # so do the entity badges
    assert '<tr class="f-keep"><td class="f-key">Source</td>' in htm
    # Headings keep their content: no orphaned section title at a page bottom.
    heading_rule = re.search(r"h2\.section\s*\{([^}]*)\}", htm)
    assert heading_rule is not None and "break-after: avoid" in heading_rule.group(1)
    # The manifest keeps its per-row guard (rows are single-line).
    manifest_rule = re.search(r"table\.manifest tr\s*\{([^}]*)\}", htm)
    assert manifest_rule is not None and "break-inside: avoid" in manifest_rule.group(1)


def test_a_finding_s_band_heads_its_table() -> None:
    """The band is the table's header group, so a finding that continues overleaf is named again."""
    htm = R.render_html(_report())
    assert htm.count('<thead><tr class="f-head">') == 2


def test_rules_separate_prose_items_and_space_separates_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hairline between two boxed findings printed alone at the top of a page; only prose keeps it."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _report()
    report["items"].append({"id": 4, "artifact_type": "chat_answer", "note": None, "snapshot": {"user_text": "Q2"}})
    htm = R.render_html(report)

    spacing = re.search(r"\.item \+ \.item\s*\{([^}]*)\}", htm)
    assert spacing is not None and "border" not in spacing.group(1)
    rule = re.search(r"\.item-prose \+ \.item-prose\s*\{([^}]*)\}", htm)
    assert rule is not None and "border-top" in rule.group(1)
    assert htm.count('<div class="item item-prose">') == 2  # both chat answers
    assert htm.count('<div class="item"><table class="finding">') == 2


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


def test_findings_carry_grouped_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Findings surface a grouped provenance block: source → posting → account.

    The report view is informative, not exhaustive: pipeline-internal fields
    (``network: nextext``, ``type``, UUIDs, ``text_id``) and duplicates
    (media ID equal to posting ID, the derived transcript filename shadowed by
    ``source_file``) never render; the full snapshot stays in the JSON export.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    for blob in (R.render_markdown(_report()), R.render_html(_report())):
        # Entity finding (a direct social row): bare fields feed posting/account.
        assert "Telegram" in blob and "alice" in blob
        # Hate finding (transcript segment): the parent posting's fields render …
        assert "Facebook" in blob
        assert "Jane Poster (@jane.poster · ID 42)" in blob  # account merged into one row
        assert "https://fb.example/p1" in blob
        assert "Original post body" in blob
        assert "ID P1" in blob  # the posting ID (the exact reference)
        # … while internal/duplicate fields are dropped from the report view.
        assert "nextext" not in blob
        assert "transcript_segment" not in blob
        assert "pu-1" not in blob and "u-1" not in blob  # docint-minted UUIDs
        assert "clip.mp4:2" not in blob  # text_id duplicates source file + position
        assert ui_string("report_label_media_id") not in blob  # media ID == posting ID
        assert "clip.mp4.nextext.jsonl" not in blob  # derived artifact; source_file wins
        assert "clip.mp4 · 00:00:08" in blob  # original media + in-media position


def test_media_id_renders_only_when_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """A media ID differing from the posting ID still renders (it adds information)."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _report()
    report["items"][2]["snapshot"]["reference_metadata"]["media_id"] = "M9"
    htm = R.render_html(report)
    assert f"{ui_string('report_label_media_id')} M9" in htm


def test_posting_text_renders_adjacent_to_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent posting's text sits directly under the chunk, before analysis/provenance rows."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    htm = R.render_html(_report())
    i_chunk = htm.index("bad text")
    i_posting_text = htm.index("Original post body")
    i_reason = htm.index("contains slur")
    i_source_row = htm.index("clip.mp4 · 00:00:08")
    assert i_chunk < i_posting_text < i_reason < i_source_row


def test_posting_text_suppressed_when_chunk_is_the_posting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finding on the posting itself doesn't repeat its text as a Posting text row.

    The chunk of a postings.csv finding *is* the posting's text; the dedicated
    row only appears for media-derived artifacts whose chunk differs from it.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _report()
    snap = report["items"][2]["snapshot"]
    snap["reference_metadata"]["posting_text"] = snap["chunk_text"]
    htm = R.render_html(report)
    md = R.render_markdown(report)
    assert ui_string("report_label_posting_text") not in htm
    assert f"| {ui_string('report_label_posting_text')} |" not in md
    # The chunk itself still renders.
    assert "bad text" in htm


def test_entity_findings_same_chunk_collapse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entity findings added per-entity for the same chunk merge into one block.

    The header names every entity label, the mention badges are merged, and the
    notes are joined; a finding on a different chunk stays its own block.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    snap = {
        "chunk_id": "c1",
        "chunk_text": "Acme met Bob",
        "filename": "a.pdf",
        "page": 2,
    }
    report: dict[str, Any] = {
        "id": 1,
        "title": "T",
        "collection_name": "c",
        "created_at": "2026-06-20T10:00:00+00:00",
        "items": [
            {
                "id": 1,
                "artifact_type": "entity_finding",
                "note": "first",
                "snapshot": {**snap, "entity_label": "Acme [ORG]", "entities": [{"text": "Acme", "type": "ORG"}]},
            },
            {
                "id": 2,
                "artifact_type": "entity_finding",
                "note": "second",
                "snapshot": {**snap, "entity_label": "Bob [PERSON]", "entities": [{"text": "Bob", "type": "PERSON"}]},
            },
            {
                "id": 3,
                "artifact_type": "entity_finding",
                "note": None,
                "snapshot": {
                    "chunk_id": "c2",
                    "entity_label": "Eve [PERSON]",
                    "chunk_text": "Eve elsewhere",
                    "filename": "a.pdf",
                    "page": 3,
                    "entities": [{"text": "Eve", "type": "PERSON"}],
                },
            },
        ],
    }
    htm = R.render_html(report)
    assert htm.count('<table class="finding">') == 2  # c1 collapsed + c2
    assert htm.count("Acme met Bob") == 1  # the shared chunk renders once
    assert "Acme [ORG] · Bob [PERSON]" in htm  # header names all entities
    assert "first · second" in htm  # notes merged
    md = R.render_markdown(report)
    assert md.count("Acme met Bob") == 1
    assert "Acme [ORG] · Bob [PERSON]" in md


def test_findings_render_as_single_table_each(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each finding is one table: full-width tag header band, full-width chunk row, the rest below."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _report()

    htm = R.render_html(report)
    # One table per finding (entity + hate); the tag rides a full-width shaded
    # header band, the chunk text a full-width row right under it.
    assert htm.count('<table class="finding">') == 2
    assert htm.count('class="f-head"') == 2
    assert '<tr class="f-head"><td colspan="2"><span class="f-num">#1</span>Acme [ORG]</td></tr>' in htm
    # A category outside the fixed enum shows as stored; the confidence is labelled.
    assert (
        '<tr class="f-head"><td colspan="2"><span class="f-num">#1</span><span class="badge">slur</span>'
        '<span class="badge conf-high">Confidence: high</span></td></tr>'
    ) in htm
    assert re.search(r'<td colspan="2" class="f-text">bad text</td>', htm)
    # The rest sits below as grouped label/value rows inside the same table.
    assert '<td class="f-key">Source</td><td class="f-val">a.pdf · Page 2</td>' in htm
    assert (
        '<td class="f-key">Posting</td>'
        '<td class="f-val">Facebook · 2026-03-04 09:00:00 (UTC) · ID P1\nhttps://fb.example/p1</td>'
    ) in htm

    md = R.render_markdown(report)
    # GFM table: tag + chunk text form the (prominent) header row.
    assert "| #1 · slur · Confidence: high | bad text |" in md
    assert "| #1 · Acme [ORG] | Acme met Bob <script>alert(1)</script> |" in md
    assert "| --- | --- |" in md
    # Grouped provenance rows; the multi-line posting value keeps the grid intact.
    assert "| Posting | Facebook · 2026-03-04 09:00:00 (UTC) · ID P1<br>https://fb.example/p1 |" in md
    assert "| Posting text | Original post body |" in md
    # The old exhaustive per-field metadata block is gone.
    assert "Posting Network" not in md
    assert "Posting UUID" not in md


def test_md_finding_table_cells_escape_pipes_and_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verbatim evidence text cannot break the Markdown table grid."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report(
        "hate_speech_finding",
        {
            "category": "x",
            "confidence": "high",
            "reason": "line one\nline two",
            "chunk_text": "a|b\nc",
            "filename": "f",
        },
    )
    md = R.render_markdown(report)
    assert "| #1 · x · Confidence: high | a\\|b<br>c |" in md
    assert "line one<br>line two" in md


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


def test_pdf_contents_name_the_page_each_section_starts_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every contents entry prints the page its section starts on — never 0."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    evidence = " ".join(f"evidence{i}" for i in range(250))[:1500]
    findings = [
        {"artifact_type": kind, "snapshot": {**snapshot, "chunk_text": evidence, "filename": "a.csv", "row": i}}
        for kind, snapshot in (
            ("entity_finding", {"entity_label": "Acme [ORG]"}),
            ("hate_speech_finding", {"category": "religion", "confidence": "high", "reason": "r"}),
        )
        for i in range(4)
    ]

    entries = contents_entries(
        html_cls(string=R.render_html({"title": "T", "show_toc": True, "items": findings})).render()
    )

    assert set(entries) == {"sec-entities", "sec-hate"}
    for target, (printed, starts) in entries.items():
        assert printed == str(starts), target
    assert entries["sec-hate"][1] > entries["sec-entities"][1], "sections share a page"


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


def test_pdf_overview_starts_its_own_page_after_the_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The overview chapter opens a fresh page rather than running on under the last item."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    item = {"id": 1, "artifact_type": "summary", "note": None, "snapshot": {"collection": "c1", "text": "Short."}}
    report = _overview_report(items=[item], show_toc=True)

    entries = contents_entries(html_cls(string=R.render_html(report)).render())

    assert entries["sec-summaries"][1] == 1
    assert entries["sec-collection-overview"] == ("2", 2)


def test_pdf_overview_alone_stays_under_the_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing before it, the overview does not leave the first page blank."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")

    entries = contents_entries(html_cls(string=R.render_html(_overview_report(show_toc=True))).render())

    assert entries["sec-collection-overview"] == ("1", 1)


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


def test_overview_csv_preserves_zero_counts() -> None:
    """A counted zero (0 pages/rows/nodes) renders as ``0``, never a blank cell.

    The snapshot distinguishes ``row_count: 0`` (an empty table) from
    ``row_count: None`` (no table); the CSV is the evidentiary artifact where
    that distinction must survive, so a real zero must not collapse to blank.
    """
    ov: dict[str, Any] = {
        **_OVERVIEW,
        "documents": [
            {
                "filename": "z.csv",
                "type_label": "CSV",
                "page_count": 0,
                "row_count": 0,
                "node_count": 0,
                "file_hash": "h0",
            }
        ],
    }
    zf = zipfile.ZipFile(io.BytesIO(R.report_csv_bundle(_overview_report(collection_overview=ov))))
    body = zf.read("collection-overview.csv").decode()
    # Columns: filename,type,pages,rows,nodes,hash -> the data row's counts are all "0".
    data_row = body.strip().splitlines()[-1].split(",")
    assert data_row[2:5] == ["0", "0", "0"]  # pages,rows,nodes are 0, not blank
    assert ",0,0,0," in body


# ---------------------------------------------------------------------------
# Thumbnails — visual evidence carried in the snapshot
# ---------------------------------------------------------------------------

_DATA_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="


def _report_with_thumbnails() -> dict[str, Any]:
    """A report whose chat source and hate finding carry frozen thumbnails."""
    report = _report()
    report["items"][0]["snapshot"]["sources"][0]["thumbnail"] = {
        "data_uri": _DATA_URI,
        "width": 320,
        "height": 180,
        "kind": "image",
    }
    report["items"][2]["snapshot"]["thumbnail"] = {
        "data_uri": _DATA_URI,
        "width": 320,
        "height": 180,
        "kind": "video_keyframe",
    }
    return report


def test_html_renders_finding_thumbnail_with_kind_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finding's thumbnail renders as an inline img row labeled by its kind."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    html = R.render_html(_report_with_thumbnails())
    assert f'<figure class="evidence"><img src="{_DATA_URI}"' in html
    assert ui_string("report_label_video_keyframe") in html


def test_html_renders_chat_source_thumbnails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chat answer's image sources render beneath the source list; a finding leads with its own."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    html = R.render_html(_report_with_thumbnails())
    assert html.count(f'<figure class="evidence"><img src="{_DATA_URI}"') == 1
    assert html.count(f'<figure class="evidence lead"><img src="{_DATA_URI}"') == 1
    assert ui_string("report_label_image_evidence") in html


def test_markdown_renders_thumbnails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markdown embeds the data URI as an inline image for chat and findings."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    md = R.render_markdown(_report_with_thumbnails())
    assert md.count(f"]({_DATA_URI})") == 2
    assert f"![{ui_string('report_label_video_keyframe')}]" in md


def test_non_image_data_uri_is_never_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshots are caller-supplied JSON: only data:image/* reaches an img src."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _report_with_thumbnails()
    report["items"][0]["snapshot"]["sources"][0]["thumbnail"]["data_uri"] = "javascript:alert(1)"
    report["items"][2]["snapshot"]["thumbnail"] = {"data_uri": "https://evil.example/x.jpg"}
    html = R.render_html(report)
    md = R.render_markdown(report)
    assert "javascript:alert(1)" not in html
    assert "evil.example" not in html
    assert '<figure class="evidence"' not in html
    assert "![" not in md


def test_csv_bundle_ignores_thumbnails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CSV bundle keeps its fixed columns; the base64 never leaks into a cell."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    payload = R.report_csv_bundle(_report_with_thumbnails())
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        for name in bundle.namelist():
            assert _DATA_URI not in bundle.read(name).decode("utf-8")


def _report_with_numbered_image_sources() -> dict[str, Any]:
    """A chat answer citing two images, numbered as the generator saw them."""
    report = _report()
    report["items"][0]["snapshot"]["sources"] = [
        {
            "filename": "chart.png",
            "citation_index": 1,
            "thumbnail": {"data_uri": _DATA_URI, "kind": "image"},
        },
        {
            "filename": "photo.jpg",
            "citation_index": 2,
            "thumbnail": {"data_uri": _DATA_URI, "kind": "image"},
        },
    ]
    return report


def test_html_chat_figures_carry_the_citation_number(monkeypatch: pytest.MonkeyPatch) -> None:
    """Side-by-side figures are captioned with the number the answer cites."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    html = R.render_html(_report_with_numbered_image_sources())
    assert '<div class="evidence-strip">' in html
    assert "<figcaption>[1] chart.png</figcaption>" in html
    assert "<figcaption>[2] photo.jpg</figcaption>" in html
    assert "<li>[1] chart.png</li>" in html


def test_html_finding_figure_is_captioned_with_its_kind_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A finding's figure says what it is — a picture or a video frame — and names its file in the provenance rows."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    html = R.render_html(_report_with_thumbnails())
    keyframe = ui_string("report_label_video_keyframe")
    assert f'alt="{keyframe}"><figcaption>{keyframe}</figcaption></figure>' in html
    assert "<figcaption>clip.mp4" not in html


def test_markdown_chat_figures_carry_the_citation_number(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markdown captions each figure on its own line beneath the image."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    md = R.render_markdown(_report_with_numbered_image_sources())
    assert f"![[1] chart.png]({_DATA_URI})" in md
    assert "*[1] chart.png*" in md
    assert "- [2] photo.jpg" in md


def test_sources_without_citation_index_are_not_renumbered(monkeypatch: pytest.MonkeyPatch) -> None:
    """An older snapshot keeps bare filenames — a made-up number would contradict the answer."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    html = R.render_html(_report_with_thumbnails())
    assert "<li>a.pdf (Page 2)</li>" in html
    assert "<figcaption>a.pdf</figcaption>" in html


def test_a_keyframe_finding_names_the_clip_it_came_from(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame is evidence about a video, and the row has to say which video.

    The linker stamps the clip on every artifact cut from it, so a keyframe
    reaches the report the way a transcript segment already did.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _report()
    report["items"][2]["snapshot"] = {
        "chunk_id": "k1",
        "category": "slur",
        "confidence": "high",
        "reason": "shown on screen",
        "chunk_text": "a frame caption",
        "filename": "clip.mp4",
        "reference_metadata": {
            "type": "keyframe",
            "source_file": "clip.mp4",
            "posting_network": "Facebook",
            "posting_author": "Jane Poster",
            "posting_timestamp": "2026-03-04 09:00:00+00",
        },
    }

    for blob in (R.render_markdown(report), R.render_html(report)):
        assert "clip.mp4" in blob


def test_a_document_figure_finding_names_its_document_and_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """A figure has no name of its own; the document and page are its address."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _report()
    report["items"][2]["snapshot"] = {
        "chunk_id": "f1",
        "category": "slur",
        "confidence": "low",
        "reason": "text in figure",
        "chunk_text": "a chart caption",
        "filename": "quarterly report.pdf",
        "page": 3,
        "reference_metadata": cast(dict[str, Any], {}),
    }

    for blob in (R.render_markdown(report), R.render_html(report)):
        assert f"quarterly report.pdf · {ui_string('report_label_page')} 3" in blob


_PRINTED = "PRINTED SLOGAN\nSECOND LINE"
_DESCRIPTION = "A poster held up in a town square."


def _image_finding(artifact_type: str, **extra: Any) -> dict[str, Any]:
    """A finding judged from an image, carrying its printed words, description and tags apart.

    Args:
        artifact_type: ``entity_finding`` or ``hate_speech_finding``.
        **extra: Snapshot keys to add or override.

    Returns:
        A one-item report.
    """
    snapshot: dict[str, Any] = {
        "chunk_id": "img-1",
        "chunk_text": f"{_PRINTED}\n\n{_DESCRIPTION}\n\nTags: poster, crowd",
        "filename": "poster.png",
        "image_id": "img-1",
        "ocr_text": _PRINTED,
        "image_description": _DESCRIPTION,
        "image_tags": ["poster", "crowd"],
        **extra,
    }
    if artifact_type == "entity_finding":
        snapshot |= {"entity_label": "Acme [ORG]", "entities": [{"text": "Acme", "type": "ORG"}]}
    else:
        snapshot |= {"category": "religion", "confidence": "high", "reason": "Endorses excluding a group."}
    return _single_item_report(artifact_type, snapshot)


@pytest.mark.parametrize("artifact_type", ["entity_finding", "hate_speech_finding"])
def test_image_finding_renders_its_parts_as_labelled_rows(monkeypatch: pytest.MonkeyPatch, artifact_type: str) -> None:
    """Printed words, description and tags each get a labelled row instead of one combined block."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _image_finding(artifact_type)

    htm = R.render_html(report)
    printed = f'<td class="f-key">Text in the image</td><td class="f-text">{_PRINTED}</td>'
    described = f'<td class="f-key">Image description</td><td class="f-val">{_DESCRIPTION}</td>'
    tagged = '<td class="f-key">Tags</td><td class="f-val">poster, crowd</td>'
    assert htm.index(printed) < htm.index(described) < htm.index(tagged)
    assert '<td colspan="2" class="f-text">' not in htm

    md = R.render_markdown(report)
    tag = "#1 · Acme [ORG]" if artifact_type == "entity_finding" else "#1 · Religion · Confidence: high"
    assert f"| {tag} |  |" in md
    printed_md = "| Text in the image | PRINTED SLOGAN<br>SECOND LINE |"
    described_md = f"| Image description | {_DESCRIPTION} |"
    assert md.index(printed_md) < md.index(described_md) < md.index("| Tags | poster, crowd |")
    assert "Tags: poster, crowd" not in md


def test_image_finding_translation_sits_under_the_printed_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """The translation of an image's printed words follows them, ahead of the description."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _image_finding(
        "hate_speech_finding", translation={"text": "TRANSLATED SLOGAN", "target_lang": "en", "model": "m"}
    )

    htm = R.render_html(report)
    translated = '<td class="f-key">Machine translation (→ English)</td><td class="f-val">TRANSLATED SLOGAN</td>'
    assert htm.index("Text in the image") < htm.index(translated) < htm.index("Image description")
    assert htm.count("TRANSLATED SLOGAN") == 1

    md = R.render_markdown(report)
    translated_md = "| Machine translation (→ English) | TRANSLATED SLOGAN |"
    assert md.index("| Text in the image |") < md.index(translated_md) < md.index("| Image description |")


def test_image_finding_without_printed_words_keeps_its_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing printed in the picture, the parts it has render and a translation stays where it always was."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _image_finding(
        "entity_finding",
        ocr_text="",
        image_tags=[],
        translation={"text": "TRANSLATED", "target_lang": "en", "model": "m"},
    )

    htm = R.render_html(report)
    assert "Text in the image" not in htm
    assert "Tags</td>" not in htm
    assert htm.index("Image description") < htm.index("Machine translation (→ English)")

    md = R.render_markdown(report)
    assert md.index("| Image description |") < md.index("| Machine translation (→ English) | TRANSLATED |")


def test_image_finding_part_labels_follow_the_response_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """The part labels are the report's locale, like every other label."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    htm = R.render_html(_image_finding("hate_speech_finding"))
    assert '<td class="f-key">Text im Bild</td>' in htm
    assert '<td class="f-key">Bildbeschreibung</td>' in htm
    assert '<td class="f-key">Schlagworte</td>' in htm


# --------------------------------------------------------------------------- #
# PDF layout (real WeasyPrint; skipped where its native libraries are absent)
# --------------------------------------------------------------------------- #
def _long_finding(text: str, **extra: Any) -> dict[str, Any]:
    """A one-finding report whose evidence cells hold ``text``."""
    return _single_item_report(
        "entity_finding",
        {"chunk_id": "c1", "entity_label": "Acme [ORG]", "chunk_text": text, "filename": "a.csv", "row": 1, **extra},
    )


def test_pdf_does_not_measure_a_finding_character_by_character(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sizing a finding's table never splits its evidence one character at a time.

    Auto table layout measures every cell's narrowest width, and text that may
    break anywhere is measured per character, each split re-laying out the rest
    of the text: quadratic per cell, which turned a report of a few dozen
    findings into a gateway timeout.
    """
    weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    text = " ".join(f"evidence{i}" for i in range(200))[:1500]
    report = _long_finding(text, translation={"text": text, "target_lang": "en", "model": "m"})
    splits = count_min_content_splits(monkeypatch)

    assert R.render_pdf(report).startswith(b"%PDF")
    # Per character, the chunk and its translation alone would take 3,000 splits.
    assert splits[0] < 100


def test_pdf_wraps_an_unbroken_token_inside_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URL or hash with no break opportunity wraps instead of pushing the table past the margin."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _long_finding(
        "x" * 400, reference_metadata={"network": "examplenet", "url": "https://example.invalid/" + "y" * 300}
    )

    assert text_beyond_the_page(html_cls(string=R.render_html(report)).render()) == []


def test_pdf_keeps_a_long_label_inside_the_key_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """A German label wider than the key column breaks there instead of running into its value."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    document = html_cls(string=R.render_html(_image_finding("entity_finding"))).render()

    assert text_beyond_its_cell(document, "f-key") == []


def test_pdf_gives_a_finding_s_evidence_most_of_the_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """The label column stays slim and the evidence beside it gets the page, as the colgroup lays out."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    document = html_cls(string=R.render_html(_long_finding("short evidence"))).render()

    assert max(content_width_share(document, "f-key")) < 0.2
    assert min(content_width_share(document, "f-val")) > 0.7


def test_pdf_never_hyphenates_a_label_that_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A label whose words fit the key column wraps at its spaces, never mid-word."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _long_finding("evidence", translation={"text": "translated evidence", "target_lang": "de", "model": "m"})
    document = html_cls(string=R.render_html(report)).render()

    assert hyphenated_text(document, "f-key") == []


# --------------------------------------------------------------------------- #
# A finding judged from a picture leads with it
# --------------------------------------------------------------------------- #
def _jpeg_data_uri(width: int, height: int) -> str:
    """A real JPEG data URI, so WeasyPrint lays a figure out at its true aspect ratio."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 200)).save(buffer, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


_THUMBNAIL = {"data_uri": _DATA_URI, "kind": "image"}


def test_image_finding_leads_with_its_picture_and_sets_its_words_beside_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The picture opens the finding; its printed words, description and tags share its row."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _image_finding(
        "hate_speech_finding", thumbnail=_THUMBNAIL, reference_metadata={"network": "examplenet", "text": "Post body"}
    )

    htm = R.render_html(report)
    media = htm.split('<tr class="f-media">')[1].split("</table></td></tr>")[0]
    assert media.index('<figure class="evidence lead">') < media.index("PRINTED SLOGAN")
    assert f'<div class="m-label">Text in the image</div><div class="m-text">{_PRINTED}</div>' in media
    assert f'<div class="m-label">Image description</div><div class="m-val">{_DESCRIPTION}</div>' in media
    assert '<div class="m-label">Tags</div><div class="m-val">poster, crowd</div>' in media
    assert "Text in the image</td>" not in htm  # no second, row-per-part copy
    # The picture row comes first, and the reason before the posting's own text.
    assert htm.index('<tbody><tr class="f-media">') < htm.index(">Reason<") < htm.index(">Posting text<")

    md = R.render_markdown(report)
    assert md.index(f"]({_DATA_URI})") < md.index("| Text in the image |") < md.index("| Reason |")
    assert md.index("| Reason |") < md.index("| Posting text |")


def test_image_finding_with_long_printed_text_gives_its_picture_a_row_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Words that could outgrow a page follow the picture as a row that may split; the picture row stays whole."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    printed = "\n".join(f"line {i}" for i in range(80))
    htm = R.render_html(_image_finding("hate_speech_finding", ocr_text=printed, thumbnail=_THUMBNAIL))

    assert 'class="media-grid"' not in htm
    assert '<tr class="f-media"><td colspan="2"><figure class="evidence lead">' in htm
    assert f'<tr><td class="f-key">Text in the image</td><td class="f-text">{printed}</td></tr>' in htm


def test_a_picture_frozen_without_its_parts_sets_the_finding_s_text_beside_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """An older snapshot names no parts: its text, and that text's translation, sit beside the picture."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report(
        "hate_speech_finding",
        {
            "category": "religion",
            "confidence": "low",
            "chunk_text": "words on the picture",
            "translation": {"text": "translated words", "target_lang": "en", "model": "m"},
            "thumbnail": _THUMBNAIL,
        },
    )

    htm = R.render_html(report)
    assert (
        '<td class="m-parts"><div class="m-text">words on the picture</div>'
        '<div class="m-label">Machine translation (→ English)</div><div class="m-val">translated words</div></td>'
    ) in htm
    assert '<td colspan="2" class="f-text">' not in htm


# --------------------------------------------------------------------------- #
# Order, numbers and labels
# --------------------------------------------------------------------------- #
def _dated_hate(chunk: str, timestamp: str | None) -> dict[str, Any]:
    """A hate-speech item whose posting carries ``timestamp`` (none when ``None``)."""
    reference: dict[str, str] = {"network": "examplenet"}
    if timestamp is not None:
        reference["timestamp"] = timestamp
    return {
        "id": chunk,
        "artifact_type": "hate_speech_finding",
        "note": None,
        "snapshot": {
            "chunk_id": chunk,
            "category": "other",
            "confidence": "low",
            "chunk_text": chunk,
            "reference_metadata": reference,
        },
    }


def test_hate_speech_findings_read_newest_first_and_are_numbered_in_that_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bulk add lands findings in paging order; the export reads them newest first, undated last."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = {
        "title": "T",
        "items": [
            _dated_hate("undated-first", None),
            _dated_hate("evening-utc", "2026-05-02T18:00:00Z"),
            _dated_hate("not-a-date", "yesterday"),
            _dated_hate("morning-plus-two", "2026-05-02T09:00:00+02:00"),  # 07:00 UTC
            _dated_hate("naive-noon", "2026-05-02T12:00:00"),  # read as UTC
            _dated_hate("day-before", "2026-05-01"),
        ],
    }
    expected = ["evening-utc", "naive-noon", "morning-plus-two", "day-before", "undated-first", "not-a-date"]

    for blob in (R.render_html(report), R.render_markdown(report)):
        positions = [blob.index(name) for name in expected]
        assert positions == sorted(positions)
    md = R.render_markdown(report)
    assert "| #1 · Other · Confidence: low | evening-utc |" in md
    assert "| #6 · Other · Confidence: low | not-a-date |" in md
    # The data exports keep the stored order.
    with zipfile.ZipFile(io.BytesIO(R.report_csv_bundle(report))) as bundle:
        rows = bundle.read("hate-speech.csv").decode("utf-8")
    assert rows.index("undated-first") < rows.index("evening-utc") < rows.index("day-before")


@pytest.mark.parametrize(
    ("locale", "raw", "shown"),
    [
        ("en", "2026-09-23T20:31:52+02:00", "2026-09-23 20:31:52 (UTC+02:00)"),
        ("de", "2026-09-23T20:31:52+02:00", "23.09.2026 20:31:52 (UTC+02:00)"),
        ("de", "2026-09-23T20:31:52Z", "23.09.2026 20:31:52 (UTC)"),
        ("de", "2026-09-23T20:31:52-05:30", "23.09.2026 20:31:52 (UTC-05:30)"),
        ("de", "2026-09-23 20:31:52", "23.09.2026 20:31:52"),
        ("de", "2026-09-23", "23.09.2026"),
        ("de", "2026-09-23T20:31", "23.09.2026 20:31"),  # no seconds invented
        ("de", "2026-09-23T20:31:52-05:30:15", "23.09.2026 20:31:52 (UTC-05:30:15)"),
        ("de", "1695470000", "1695470000"),
        ("de", "20260923", "20260923"),
        ("de", "yesterday", "yesterday"),
    ],
)
def test_posting_times_read_in_the_report_s_locale_with_their_own_offset(
    monkeypatch: pytest.MonkeyPatch, locale: str, raw: str, shown: str
) -> None:
    """A posting time is evidence: reformatted for the reader, never moved to another zone."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", locale)
    report = {"title": "T", "items": [_dated_hate("x", raw)]}
    assert f"| {ui_string('report_label_posting')} | examplenet · {shown} |" in R.render_markdown(report)


def test_creation_date_reads_in_the_report_s_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """The subheader's date follows the response language, like every other label."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    report = _single_item_report("summary", {"collection": "c", "text": "t"})
    assert "Erstellt: 20.06.2026" in R.render_html(report)
    assert "Erstellt: 20.06.2026" in R.render_markdown(report)


def test_hate_speech_band_reads_in_the_report_s_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """Category and confidence are protocol values; the reader sees their labels, the confidence named as such."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    report = _single_item_report(
        "hate_speech_finding", {"category": "sexual_orientation", "confidence": "medium", "chunk_text": "x"}
    )

    htm = R.render_html(report)
    assert (
        '<span class="badge">Sexuelle Orientierung</span><span class="badge conf-medium">Konfidenz: mittel</span>'
    ) in htm
    assert "| #1 · Sexuelle Orientierung · Konfidenz: mittel | x |" in R.render_markdown(report)


def test_an_unknown_confidence_shows_as_stored_and_never_reaches_a_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshots are caller-supplied JSON: an unexpected value is escaped text, not markup."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report(
        "hate_speech_finding", {"category": "religion", "confidence": 'x" onclick="y', "chunk_text": "x"}
    )
    assert '<span class="badge">Confidence: x&quot; onclick=&quot;y</span>' in R.render_html(report)


def test_contents_count_the_numbered_findings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contents count what the body numbers: entity findings on one chunk count once."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _report()
    report["show_toc"] = True
    entity = report["items"][1]["snapshot"]
    report["items"].append(
        {
            "id": 5,
            "artifact_type": "entity_finding",
            "note": None,
            "snapshot": {**entity, "entity_label": "Bob [PERSON]"},
        }
    )

    htm = R.render_html(report)
    assert ">Entity findings (1)</a>" in htm
    assert ">Hate-speech findings (1)</a>" in htm
    assert ">Chat answers</a>" in htm
    md = R.render_markdown(report)
    assert "- Entity findings (1)" in md
    assert "- Chat answers\n" in md


@pytest.mark.parametrize(("locale", "label"), [("en", "Page"), ("de", "Seite")])
def test_page_numbers_are_labelled_in_the_report_s_language(
    monkeypatch: pytest.MonkeyPatch, locale: str, label: str
) -> None:
    """The footer's page counter carries the locale's word, not a hard-coded English one."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", locale)
    assert f'content: "{label} " counter(page) " / " counter(pages);' in R.render_html(_report())


def test_summary_title_is_dropped_when_the_subheader_already_names_the_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A report belongs to one collection; repeating its name over the summary says nothing new."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    report = _single_item_report("summary", {"collection": "c", "text": "Summary body."})
    assert '<div class="item-title">' not in R.render_html(report)
    assert "### c" not in R.render_markdown(report)

    report["items"][0]["snapshot"]["collection"] = "other"
    assert '<div class="item-title">other</div>' in R.render_html(report)
    assert "### other" in R.render_markdown(report)


# --------------------------------------------------------------------------- #
# Page breaks (real WeasyPrint)
# --------------------------------------------------------------------------- #
def _uneven_findings_report(count: int) -> dict[str, Any]:
    """Findings of uneven height, a third of them pictures, so page breaks land all over them."""
    portrait, landscape = _jpeg_data_uri(432, 768), _jpeg_data_uri(768, 432)
    items = []
    for i in range(count):
        snapshot: dict[str, Any] = {
            "chunk_id": f"c{i}",
            "category": "other",
            "confidence": "low",
            "chunk_text": " ".join(f"evidence{j}" for j in range(5 + (i * 7) % 40)),
            "reason": " ".join(f"reason{j}" for j in range(8 + (i * 11) % 60)),
            "filename": "a.csv",
            "row": i,
            "reference_metadata": {
                "network": "examplenet",
                "timestamp": f"2026-05-{1 + i % 28:02d}T10:00:00Z",
                "url": f"https://example.invalid/{i}",
                "author": f"Person {i}",
            },
        }
        if i % 3 == 0:
            snapshot |= {
                "thumbnail": {"data_uri": portrait if i % 2 else landscape, "kind": "image"},
                "ocr_text": "\n".join(f"WORD {j}" for j in range(2 + i % 9)),
                "image_description": " ".join(f"described{j}" for j in range(10 + i % 30)),
            }
        items.append({"id": i, "artifact_type": "hate_speech_finding", "note": None, "snapshot": snapshot})
    return {"title": "T", "items": items}


def test_pdf_never_parts_a_label_from_its_value_or_a_picture_from_its_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short rows move whole and a finding's band never ends a page on its own."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    document = html_cls(string=R.render_html(_uneven_findings_report(40))).render()

    assert len(document.pages) > 5
    assert rows_split_across_pages(document, "f-keep") == []
    assert rows_split_across_pages(document, "f-media") == []
    assert rows_alone_at_page_foot(document, "f-head") == []
    assert pages_showing_a_finding_without_its_band(document) == []


@pytest.mark.parametrize("lines", [6, 120], ids=["beside the picture", "below the picture"])
def test_pdf_lays_out_a_picture_s_words_without_measuring_them_per_character(
    monkeypatch: pytest.MonkeyPatch, lines: int
) -> None:
    """The text beside (or below) a picture is sized without one split per character, as #624 requires."""
    weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    words = [f"evidence{i}" for i in range(200)]
    report = _image_finding(
        "hate_speech_finding",
        thumbnail={"data_uri": _jpeg_data_uri(432, 768), "kind": "image"},
        ocr_text="\n".join(" ".join(words[i::lines]) for i in range(lines))[:1500],
        image_description=" ".join(words)[:1200],
    )
    splits = count_min_content_splits(monkeypatch)

    assert R.render_pdf(report).startswith(b"%PDF")
    assert splits[0] < 100


def test_pdf_sets_the_document_overview_densely(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest lists every document of the collection; 120 of them take three pages, not four."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    documents = [
        {
            "filename": f"document_{i:03d}.pdf",
            "type_label": "PDF",
            "page_count": i % 9 + 1,
            "row_count": None,
            "node_count": 3,
            "file_hash": f"{i:064x}",
        }
        for i in range(120)
    ]
    report = _overview_report(collection_overview={**_OVERVIEW, "documents": documents, "document_count": 120})

    assert len(html_cls(string=R.render_html(report)).render().pages) <= 3


# --------------------------------------------------------------------------- #
# Height estimates behind the kept rows
# --------------------------------------------------------------------------- #
_CJK = "".join(chr(0x4E00 + (i * 37) % 2000) for i in range(1400))  # synthetic ideographs


def test_wide_characters_count_twice_toward_the_height_estimate() -> None:
    """A CJK glyph or an emoji takes about two narrow cells; counted as one, the estimate ran 1.4x short."""
    assert R._display_width("漢字ab") == 6
    assert R._display_width("😀x") == 3
    assert R._text_height_pt("漢" * 46, 46, 10.0) == 20.0


def test_wide_printed_text_too_tall_for_beside_the_picture_goes_below_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """1,400 ideographs fit the narrow-cell estimate but not the page; they are set below the picture."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    htm = R.render_html(_image_finding("hate_speech_finding", ocr_text=_CJK, thumbnail=_THUMBNAIL))
    assert 'class="media-grid"' not in htm
    assert '<tr class="f-media"><td colspan="2"><figure class="evidence lead">' in htm


def test_a_value_too_long_to_move_whole_splits_while_a_short_one_moves_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """A posting of many lines may not be kept whole; the short reason beside it still is."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    posting = "\n".join(f"posting line {i}" for i in range(60))
    report = _single_item_report(
        "hate_speech_finding",
        {
            "category": "other",
            "confidence": "low",
            "chunk_text": "short chunk",
            "reason": "A short reason.",
            "reference_metadata": {"network": "examplenet", "text": posting},
        },
    )

    htm = R.render_html(report)
    assert '<tr><td class="f-key">Posting text</td>' in htm
    assert '<tr class="f-keep"><td class="f-key">Reason</td>' in htm


def test_pdf_names_every_finding_on_every_page_it_reaches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wide picture text and a many-line posting no longer push a kept row past a page and lose the band."""
    html_cls = weasyprint_html()
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    filler = {"category": "other", "confidence": "low", "chunk_text": " ".join(["filler"] * 600), "reason": "r"}
    picture = {
        "category": "other",
        "confidence": "low",
        "chunk_text": "x",
        "ocr_text": _CJK,
        "image_tags": ["tag"] * 6,
        "reason": "A reason.",
        "thumbnail": {"data_uri": _jpeg_data_uri(432, 768), "kind": "image"},
    }
    posting = {
        "category": "other",
        "confidence": "low",
        "chunk_text": "short chunk",
        "reason": "r",
        "reference_metadata": {"network": "examplenet", "text": "\n".join(f"line {i}" for i in range(60))},
    }
    items = [
        {"id": i, "artifact_type": "hate_speech_finding", "note": None, "snapshot": snapshot}
        for i, snapshot in enumerate((filler, picture, filler, posting))
    ]
    document = html_cls(string=R.render_html({"title": "T", "items": items})).render()

    assert pages_showing_a_finding_without_its_band(document) == []
    assert rows_split_across_pages(document, "f-keep") == []
    assert rows_split_across_pages(document, "f-media") == []


# --------------------------------------------------------------------------- #
# Nothing outside the document is ever fetched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "data_uri",
    [
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",  # an SVG can reference further files
        "data:image/png;base64,AAAA)![x](https://example.invalid/y.png",  # breaks out of a Markdown image
        "data:image/png;base64,AA AA",
        "data:image/png,rawbytes",
        "https://example.invalid/x.png",
    ],
)
def test_a_thumbnail_must_be_a_base64_raster_image(data_uri: str) -> None:
    """Snapshots are caller-supplied JSON: only a base64 JPEG, PNG, WebP or GIF is ever embedded."""
    assert R._thumbnail_view({"thumbnail": {"data_uri": data_uri}}) is None


@pytest.mark.parametrize("kind", ["jpeg", "png", "webp", "gif"])
def test_raster_thumbnails_are_embedded(kind: str) -> None:
    """The thumbnail pipeline's own output passes."""
    assert R._thumbnail_view({"thumbnail": {"data_uri": f"data:image/{kind};base64,AAAA+/9="}}) is not None


def test_pdf_fetches_nothing_but_data_uris(tmp_path: Any) -> None:
    """A file or an SVG that names one is refused; the inline image beside them is drawn."""
    from PIL import Image

    local = tmp_path / "red.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(local)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><image href="{local.as_uri()}"/></svg>'
    svg_uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    document = f'<img src="{local.as_uri()}"><img src="{svg_uri}"><img src="{_jpeg_data_uri(8, 8)}">'

    pdf = weasyprint_html()(string=document).write_pdf()
    assert pdf.count(b"/Subtype /Image") == 1
