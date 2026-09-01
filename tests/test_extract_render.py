"""Tests for the extract renderers (Markdown, transcript text, HTML).

Every fixture is synthetic: invented handles, filenames and hashes.
"""

from __future__ import annotations

import pytest

from docint.core.extract.render import (
    extract_html,
    format_clock,
    format_short,
    index_markdown,
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
    assert "[00:12]" in md
    assert "a room with a whiteboard" in md
    assert "AGENDA" in md
    assert "room, whiteboard" in md


def test_media_markdown_says_so_when_there_is_no_transcript() -> None:
    """A silent clip's frames are still evidence; the absence is stated."""
    unit = MediaUnit(key="m1", file_name="clip.mp4", keyframes=[_keyframe()])
    assert "No transcript" in unit_markdown(unit)


def test_keyframe_without_a_time_renders_its_index_instead() -> None:
    """An older Nextext gives no times, and the frame still numbers itself."""
    frame = Figure(image_id="f", kind="keyframe", index=2, description="a frame", thumbnail_b64="AAAA")
    md = unit_markdown(MediaUnit(key="m1", file_name="clip.mp4", keyframes=[frame]))
    assert "[00:" not in md
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
