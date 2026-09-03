"""Tests for the extract renderers (Markdown, transcript text, HTML).

Every fixture is synthetic: invented handles, filenames and hashes.
"""

from __future__ import annotations

import pytest

from docint.core.extract.render import (
    appendix_numbers,
    extract_html,
    format_clock,
    format_short,
    index_markdown,
    text_direction,
    transcript_txt,
    unit_markdown,
)
from docint.core.extract.units import (
    Chunk,
    DocumentUnit,
    Figure,
    ImageUnit,
    MediaUnit,
    PostingUnit,
    Segment,
)


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render in English unless a test says otherwise."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


def _segments() -> list[Segment]:
    return [
        Segment(0, 0.0, 4.0, "00:00:00", "00:00:04", "SPEAKER_00", "en", "first line"),
        Segment(1, 4.0, 9.0, "00:00:04", "00:00:09", "SPEAKER_01", "en", "second line"),
    ]


def _body(document: str) -> str:
    """Return a rendered page's body, so an assertion cannot hit the CSS."""
    return document.split("</head>", 1)[1]


def _keyframe() -> Figure:
    return Figure(
        image_id="frame-1",
        kind="keyframe",
        time_sec=12.0,
        index=1,
        description="a room with a whiteboard",
        tags=("room", "whiteboard"),
        ocr_text="AGENDA",
        thumbnail_b64="AAAA",
    )


# --------------------------------------------------------------------------- #
# Time formatting
# --------------------------------------------------------------------------- #
def test_clock_formats() -> None:
    """Full stamps for transcripts, short ones for frames."""
    assert format_clock(0) == "00:00:00"
    assert format_clock(3661.4) == "01:01:01"
    assert format_short(72) == "01:12"
    assert format_clock(None) == ""
    assert format_short(None) == ""


# --------------------------------------------------------------------------- #
# Transcript text
# --------------------------------------------------------------------------- #
def test_transcript_txt_matches_the_nextext_block_layout() -> None:
    """A reader who has seen a Nextext transcript recognises this one."""
    rule = "=" * 40
    assert transcript_txt(_segments()) == (
        f"{rule}\n[00:00:00 - 00:00:04]  SPEAKER_00\n{rule}\nfirst line\n\n"
        f"{rule}\n[00:00:04 - 00:00:09]  SPEAKER_01\n{rule}\nsecond line\n"
    )


def test_transcript_txt_omits_an_absent_speaker() -> None:
    """An undiarized transcript carries no trailing speaker on its header."""
    segment = Segment(0, 0.0, 1.0, "00:00:00", "00:00:01", "", "en", "words")
    assert transcript_txt([segment]).splitlines()[1] == "[00:00:00 - 00:00:01]"


def test_transcript_txt_is_empty_without_segments() -> None:
    """A silent clip yields no file rather than an empty-looking one."""
    assert transcript_txt([]) == ""


def test_transcript_txt_falls_back_to_seconds_without_stamps() -> None:
    """A transcript stored before the stamps existed still reads as one."""
    segment = Segment(0, 5.0, 9.0, "", "", "", "en", "words")
    assert transcript_txt([segment]).splitlines()[1] == "[00:00:05 - 00:00:09]"


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def test_document_markdown_carries_its_text_and_figures() -> None:
    """A document renders its chunks in order, then its figures."""
    unit = DocumentUnit(
        key="h1",
        file_name="report.pdf",
        mimetype="application/pdf",
        chunks=[Chunk("a", 1, "opening paragraph"), Chunk("b", 2, "closing paragraph")],
        figures=[Figure(image_id="img-1", kind="figure", page_number=2, description="a chart", thumbnail_b64="AAAA")],
    )
    md = unit_markdown(unit)
    assert "# report.pdf" in md
    assert md.index("opening paragraph") < md.index("closing paragraph")
    assert "Figures" in md
    assert "a chart" in md
    assert "data:image/jpeg;base64,AAAA" in md


def test_document_markdown_warns_when_the_order_is_approximate() -> None:
    """A reader must not mistake storage order for the document's own."""
    unit = DocumentUnit(key="h1", file_name="notes.txt", chunks=[Chunk("a", None, "text")], approximate_order=True)
    assert "storage order" in unit_markdown(unit)


def test_media_markdown_renders_the_transcript_and_timed_frames() -> None:
    """The clip's words and what was on screen, each with a time."""
    unit = MediaUnit(key="m1", file_name="clip.mp4", segments=_segments(), keyframes=[_keyframe()])
    md = unit_markdown(unit)
    assert "[00:00:00 - 00:00:04] SPEAKER_00: first line" in md
    assert "00:12" in md
    assert "a room with a whiteboard" in md
    assert "AGENDA" in md


def test_the_taggers_keywords_are_never_rendered() -> None:
    """Tags are retrieval machinery; beside a caption they read as noise."""
    unit = MediaUnit(key="m1", file_name="clip.mp4", keyframes=[_keyframe()])
    md = unit_markdown(unit)
    document = extract_html([unit], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert "room, whiteboard" not in md
    assert "whiteboard" not in _body(document).replace("a room with a whiteboard", "")


def test_media_markdown_says_so_when_there_is_no_transcript() -> None:
    """A silent clip's frames are still evidence; the absence is stated."""
    unit = MediaUnit(key="m1", file_name="clip.mp4", keyframes=[_keyframe()])
    assert "No transcript" in unit_markdown(unit)


def test_keyframe_without_a_time_renders_its_index_instead() -> None:
    """An older Nextext gives no times, and the frame still numbers itself."""
    frame = Figure(image_id="f", kind="keyframe", index=2, description="a frame", thumbnail_b64="AAAA")
    md = unit_markdown(MediaUnit(key="m1", file_name="clip.mp4", keyframes=[frame]))
    assert "00:00" not in md
    assert "#3" in md


def test_posting_markdown_leads_with_its_provenance() -> None:
    """Network, author, time and link come before the words."""
    unit = PostingUnit(
        key="uuid-1",
        reference={
            "network": "examplenet",
            "author": "Example Account",
            "timestamp": "2026-01-02T03:04:05",
            "url": "https://example.invalid/p/1",
        },
        text="the posted words",
        media=[MediaUnit(key="uuid-1_0", file_name="clip.mp4", segments=_segments(), keyframes=[_keyframe()])],
    )
    md = unit_markdown(unit)
    assert "examplenet" in md
    assert "Example Account" in md
    assert "https://example.invalid/p/1" in md
    assert md.index("Example Account") < md.index("the posted words")
    assert "first line" in md


def test_posting_provenance_names_the_account_the_way_the_report_does() -> None:
    """The appendix and the report it belongs to must not disagree here."""
    unit = PostingUnit(
        key="uuid-1",
        file_name="postings.csv",
        row=4,
        reference={
            "network": "examplenet",
            "author": "Example Account",
            "author_id": "9900112233",
            "vanity": "exampleaccount",
            "timestamp": "2026-01-02T03:04:05",
        },
        text="the posted words",
    )
    md = unit_markdown(unit)
    document = extract_html([unit], collection="c", created_at="2026-01-02T03:04:05+00:00")
    for blob in (md, document):
        assert "Example Account (@exampleaccount · ID 9900112233)" in blob
        assert "postings.csv" in blob
    # "Generated" was the wrong label for a posting's own timestamp.
    assert "Generated" not in md


def test_image_markdown_puts_the_read_text_before_the_caption() -> None:
    """What a picture says outranks what it shows."""
    unit = ImageUnit(
        key="img-1",
        file_name="photo.jpg",
        figure=Figure(image_id="img-1", kind="image", description="a sign", ocr_text="OPEN", thumbnail_b64="AAAA"),
    )
    md = unit_markdown(unit)
    assert md.index("OPEN") < md.index("a sign")


def test_markdown_links_figures_by_path_when_one_is_given() -> None:
    """Inside a bundle a figure is a file, not 26KB of base64 per copy."""
    unit = MediaUnit(key="m1", file_name="clip.mp4", keyframes=[_keyframe()])
    md = unit_markdown(unit, figure_paths={"frame-1": "keyframes/frame_001.jpg"})
    assert "(keyframes/frame_001.jpg)" in md
    assert "base64" not in md


def test_markdown_headings_follow_the_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """German operators get German section headings."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    unit = MediaUnit(key="m1", file_name="clip.mp4", segments=_segments())
    assert "Transkript" in unit_markdown(unit)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
def test_index_lists_every_unit_with_its_folder() -> None:
    """The README is how a reader finds the file they want."""
    units = [
        DocumentUnit(key="h1", file_name="report.pdf", chunks=[Chunk("a", 1, "x")]),
        MediaUnit(key="m1", file_name="clip.mp4", segments=_segments()),
    ]
    md = index_markdown(
        units, collection="testcol", created_at="2026-01-02T03:04:05+00:00", paths={"h1": "documents/report"}
    )
    assert "testcol" in md
    assert "report.pdf" in md
    assert "documents/report" in md
    assert "clip.mp4" in md


def test_index_reports_a_skipped_pdf() -> None:
    """A missing extract.pdf must be explained, not merely absent."""
    md = index_markdown([], collection="c", created_at="2026-01-02T03:04:05+00:00", paths={}, pdf_skipped=True)
    assert "combined PDF was skipped" in md


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def test_html_is_self_contained_and_embeds_its_figures() -> None:
    """An exported page references nothing outside itself."""
    unit = MediaUnit(key="m1", file_name="clip.mp4", segments=_segments(), keyframes=[_keyframe()])
    document = extract_html([unit], collection="testcol", created_at="2026-01-02T03:04:05+00:00")
    assert document.startswith("<!DOCTYPE html>")
    assert "data:image/jpeg;base64,AAAA" in document
    assert "first line" in document
    assert "@page" in document


def test_html_escapes_evidence_text() -> None:
    """Ingested text is evidence, never markup."""
    unit = DocumentUnit(key="h1", file_name="x.txt", chunks=[Chunk("a", 1, "<script>alert(1)</script>")])
    document = extract_html([unit], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;" in document


def test_html_names_the_collection_and_date() -> None:
    """The page says what it is an extract of."""
    document = extract_html([], collection="testcol", created_at="2026-01-02T03:04:05+00:00")
    assert "testcol" in document
    assert "2026-01-02" in document


# --------------------------------------------------------------------------- #
# Appendix chrome
# --------------------------------------------------------------------------- #
def test_html_carries_the_reports_own_footer_disclaimer() -> None:
    """The appendix and the report it belongs to caveat themselves alike."""
    document = extract_html([], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert "AI-generated report" in document
    assert "machine-generated" not in document


def test_html_prints_the_case_file_and_operator_when_given() -> None:
    """An appendix says which file it belongs to, on every page."""
    document = extract_html(
        [],
        collection="c",
        created_at="2026-01-02T03:04:05+00:00",
        reference_number="AZ-12/26",
        operator="A. Analyst",
    )
    assert "AZ-12/26" in document
    assert "running-refnum" in document
    assert "A. Analyst" in document


def test_html_omits_the_case_file_when_there_is_none() -> None:
    """An unfiled extract prints no empty header, as the report does not."""
    document = extract_html([], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert "running-refnum" not in _body(document)


def test_units_are_numbered_so_a_finding_can_cite_one() -> None:
    """A curated report cites "appendix A.2"; the number must be there."""
    units = [
        DocumentUnit(key="h1", file_name="report.pdf", chunks=[Chunk("a", 1, "x")]),
        MediaUnit(key="m1", file_name="clip.mp4", segments=_segments()),
    ]
    numbers = appendix_numbers(units)
    assert numbers == {"h1": "A.1", "m1": "A.2"}
    document = extract_html(units, collection="c", created_at="2026-01-02T03:04:05+00:00", numbers=numbers)
    assert "A.2  clip.mp4" in document
    assert "A.1  report.pdf" in unit_markdown(units[0], numbers=numbers)


def test_html_opens_with_a_contents_block_naming_each_section_once() -> None:
    """The appendix opens the way the report does."""
    units = [
        DocumentUnit(key="h1", file_name="a.pdf", chunks=[Chunk("a", 1, "x")]),
        DocumentUnit(key="h2", file_name="b.pdf", chunks=[Chunk("b", 1, "y")]),
        MediaUnit(key="m1", file_name="clip.mp4", segments=_segments()),
    ]
    document = extract_html(units, collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert document.count('href="#sec-extract-documents"') == 1
    assert 'id="sec-extract-documents"' in document
    assert 'href="#sec-extract-media"' in document
    assert 'href="#sec-extract-postings"' not in document


def test_index_names_the_case_file_and_numbers_its_entries() -> None:
    """The README is the bundle's own index, so it agrees with the PDF."""
    units = [DocumentUnit(key="h1", file_name="report.pdf", chunks=[Chunk("a", 1, "x")])]
    md = index_markdown(
        units,
        collection="c",
        created_at="2026-01-02T03:04:05+00:00",
        paths={"h1": "documents/report"},
        reference_number="AZ-12/26",
        operator="A. Analyst",
        numbers=appendix_numbers(units),
    )
    assert "AZ-12/26" in md
    assert "A. Analyst" in md
    assert "A.1  report.pdf" in md


# --------------------------------------------------------------------------- #
# Transcript table and text direction
# --------------------------------------------------------------------------- #
def test_html_transcript_is_a_table_with_a_column_per_field() -> None:
    """A reader scans one speaker or one moment down a column."""
    unit = MediaUnit(key="m1", file_name="clip.mp4", segments=_segments())
    document = extract_html([unit], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert '<table class="transcript">' in document
    assert "<th>Time</th>" in document
    assert "<th>Speaker</th>" in document
    assert '<td class="t-time">00:00:00 - 00:00:04</td>' in document
    assert "SPEAKER_00" in document


def test_html_transcript_omits_the_speaker_column_when_undiarized() -> None:
    """An empty column would claim the transcript names speakers."""
    unit = MediaUnit(
        key="m1",
        file_name="clip.mp4",
        segments=[Segment(0, 0.0, 1.0, "00:00:00", "00:00:01", "", "en", "words")],
    )
    document = extract_html([unit], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert '<table class="transcript">' in document
    assert "<th>Speaker</th>" not in document
    assert "t-speaker" not in _body(document)


def test_right_to_left_evidence_declares_its_direction() -> None:
    """WeasyPrint ignores dir="auto", so the direction is decided here."""
    arabic = "وصل وزير الخارجية"
    assert text_direction(arabic) == "rtl"
    assert text_direction("plain words") == "ltr"
    assert text_direction("2026-01-02") == "ltr"
    unit = MediaUnit(
        key="m1",
        file_name="clip.mp4",
        segments=[Segment(0, 0.0, 1.0, "00:00:00", "00:00:01", "", "ar", arabic)],
    )
    document = extract_html([unit], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert f'<td class="t-text" dir="rtl">{arabic}</td>' in document


def test_posting_text_declares_its_direction_too() -> None:
    """The same displacement hit a posting's own words."""
    unit = PostingUnit(key="u1", reference={"network": "examplenet"}, text="وصل وزير الخارجية")
    document = extract_html([unit], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert '<pre class="evidence" dir="rtl">' in document


# --------------------------------------------------------------------------- #
# Figure layout
# --------------------------------------------------------------------------- #
def test_html_figures_put_the_picture_beside_its_description() -> None:
    """Stacked, the text left the page's right half empty per figure."""
    unit = MediaUnit(key="m1", file_name="clip.mp4", keyframes=[_keyframe()])
    document = extract_html([unit], collection="c", created_at="2026-01-02T03:04:05+00:00")
    assert '<table class="figures">' in document
    assert '<td class="fig">' in document
    assert '<p class="fig-label">Video keyframe 00:12</p>' in document
    assert "a room with a whiteboard" in document
    # The picture and its words must not be split across a page break.
    assert "table.figures tr { break-inside: avoid; }" in document


def test_a_figure_with_no_stored_thumbnail_says_so() -> None:
    """An absent picture is stated, never an empty cell."""
    figure = Figure(image_id="img-9", kind="image", file_name="photo.jpg", description="a sign")
    document = extract_html(
        [ImageUnit(key="img-9", file_name="photo.jpg", figure=figure)],
        collection="c",
        created_at="2026-01-02T03:04:05+00:00",
    )
    assert "The image itself was not stored." in document
    assert "a sign" in document
