"""Tests for assembling an extract into a ZIP bundle.

Fixtures are synthetic throughout: invented handles, filenames and hashes.
"""

from __future__ import annotations

import base64
import io
import zipfile
from datetime import UTC, datetime

import pytest

from docint.core.extract.bundle import build_bundle, build_single
from docint.core.extract.units import Chunk, DocumentUnit, Figure, MediaUnit, PostingUnit, Segment
from docint.utils.env_cfg import ExtractConfig

_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
_JPEG = base64.b64encode(b"\xff\xd8\xffjpeg").decode("ascii")


def _cfg(**overrides: int) -> ExtractConfig:
    base = {
        "retention_days": 7,
        "max_per_collection": 5,
        "pdf_max_units": 200,
        "pdf_max_figures": 400,
        "sync_max_units": 50,
    }
    base.update(overrides)
    return ExtractConfig(**base)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render in English so member names and notes are predictable."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


def _document() -> DocumentUnit:
    return DocumentUnit(
        key="a1b2c3d4e5f6",
        file_name="report.pdf",
        mimetype="application/pdf",
        chunks=[Chunk("p1", 1, "the body text")],
        figures=[Figure(image_id="img-1", kind="figure", page_number=1, description="a chart", thumbnail_b64=_JPEG)],
    )


def _media() -> MediaUnit:
    return MediaUnit(
        key="f1e2d3c4b5a6",
        file_name="clip.mp4",
        segments=[Segment(0, 0.0, 4.0, "00:00:00", "00:00:04", "SPEAKER_00", "en", "spoken words")],
        keyframes=[Figure(image_id="frame-1", kind="keyframe", time_sec=72.0, index=0, thumbnail_b64=_JPEG)],
    )


def _posting() -> PostingUnit:
    return PostingUnit(
        key="11112222-3333-4444-5555-666677778888",
        reference={
            "network": "examplenet",
            "author": "Example Account",
            "timestamp": "2026-01-02T03:04:05",
        },
        text="the posted words",
        images=[
            Figure(
                image_id="img-2", kind="image", file_name="Bild März.png", description="a picture", thumbnail_b64=_JPEG
            )
        ],
        media=[
            MediaUnit(
                key="media-1",
                file_name="post-clip.mp4",
                segments=[Segment(0, 0.0, 2.0, "00:00:00", "00:00:02", "", "en", "clip words")],
                keyframes=[Figure(image_id="frame-2", kind="keyframe", time_sec=1.0, thumbnail_b64=_JPEG)],
            )
        ],
    )


def _names(payload: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        return sorted(zf.namelist())


def _read(payload: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        return zf.read(name)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def test_bundle_lays_out_one_folder_per_unit() -> None:
    """Each unit gets its own folder, named for the source and its hash."""
    result = build_bundle(
        [_document(), _media(), _posting()],
        collection="testcol",
        cfg=_cfg(),
        pdf=None,
        now=_NOW,
    )
    names = _names(result.zip_bytes)
    root = "testcol-extract-20260102-0304"
    assert f"{root}/README.md" in names
    assert f"{root}/extract.md" in names
    assert f"{root}/documents/report.pdf-a1b2c3d4/extract.md" in names
    assert f"{root}/documents/report.pdf-a1b2c3d4/figures/report_page1_img-1.jpg" in names
    assert f"{root}/media/clip.mp4-f1e2d3c4/extract.md" in names
    assert f"{root}/media/clip.mp4-f1e2d3c4/clip.transcript.txt" in names
    assert f"{root}/media/clip.mp4-f1e2d3c4/keyframes/clip_frame_000_01-12.jpg" in names
    assert f"{root}/postings/examplenet/example-account/20260102-11112222/extract.md" in names
    assert f"{root}/postings/examplenet/example-account/20260102-11112222/post-clip.transcript.txt" in names


def test_figures_are_the_stored_thumbnail_bytes() -> None:
    """The bundle never re-fetches an original; it ships what was stored."""
    result = build_bundle([_document()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    name = next(n for n in _names(result.zip_bytes) if n.endswith("_img-1.jpg"))
    assert _read(result.zip_bytes, name) == b"\xff\xd8\xffjpeg"


def test_unit_markdown_links_its_figures_relatively() -> None:
    """Inside a folder a figure is a neighbouring file, not inline base64."""
    result = build_bundle([_document()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    name = next(n for n in _names(result.zip_bytes) if n.endswith("documents/report.pdf-a1b2c3d4/extract.md"))
    body = _read(result.zip_bytes, name).decode("utf-8")
    assert "(figures/report_page1_img-1.jpg)" in body
    assert "base64" not in body


def test_transcript_txt_uses_the_banner_layout() -> None:
    """The clip's transcript is the file a Nextext user already knows."""
    result = build_bundle([_media()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    name = next(n for n in _names(result.zip_bytes) if n.endswith(".transcript.txt"))
    assert _read(result.zip_bytes, name).decode("utf-8").startswith("=" * 40)


def test_a_silent_clip_writes_no_transcript_file() -> None:
    """An empty transcript.txt would read as a failed transcription."""
    unit = MediaUnit(key="m1", file_name="quiet.mp4", keyframes=[Figure("f", "keyframe", thumbnail_b64=_JPEG)])
    result = build_bundle([unit], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    assert not any(name.endswith(".transcript.txt") for name in _names(result.zip_bytes))


def test_readme_indexes_every_unit() -> None:
    """The README is how a reader finds a source's folder."""
    result = build_bundle([_document(), _media()], collection="testcol", cfg=_cfg(), pdf=None, now=_NOW)
    readme = _read(result.zip_bytes, "testcol-extract-20260102-0304/README.md").decode("utf-8")
    assert "documents/report.pdf-a1b2c3d4" in readme
    assert "media/clip.mp4-f1e2d3c4" in readme


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
def test_no_member_escapes_the_bundle_root() -> None:
    """A handle is untrusted input; it must never steer a write."""
    posting = PostingUnit(
        key="99998888-7777-6666-5555-444433332222",
        reference={"network": "../../etc", "author": "../../../root", "timestamp": "2026-01-02T03:04:05"},
        text="words",
    )
    result = build_bundle([posting], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    root = "c-extract-20260102-0304/"
    for name in _names(result.zip_bytes):
        assert name.startswith(root)
        assert ".." not in name


def test_a_nameless_unit_still_gets_a_folder() -> None:
    """A source with no filename must not collapse onto another's folder."""
    units = [DocumentUnit(key="hash-one", file_name=""), DocumentUnit(key="hash-two", file_name="")]
    result = build_bundle(units, collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    folders = {name.rsplit("/", 1)[0] for name in _names(result.zip_bytes) if "/documents/" in name}
    assert len(folders) == 2


def test_two_builds_at_the_same_instant_are_byte_identical() -> None:
    """A bundle is reproducible, so a diff between two runs means a data change."""
    first = build_bundle([_document(), _media()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    second = build_bundle([_document(), _media()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    assert first.zip_bytes == second.zip_bytes


# --------------------------------------------------------------------------- #
# PDF caps
# --------------------------------------------------------------------------- #
def test_pdf_is_written_when_an_engine_is_given() -> None:
    """With WeasyPrint available the bundle carries a paginated copy."""
    result = build_bundle([_document()], collection="c", cfg=_cfg(), pdf=lambda html: b"%PDF-1.7", now=_NOW)
    assert _read(result.zip_bytes, "c-extract-20260102-0304/extract.pdf") == b"%PDF-1.7"
    assert result.pdf_skipped is False


def test_pdf_is_skipped_above_the_unit_cap_and_the_readme_says_so() -> None:
    """A huge extract must not take the container's memory with it."""
    result = build_bundle(
        [_document(), _media()],
        collection="c",
        cfg=_cfg(pdf_max_units=1),
        pdf=lambda html: b"%PDF-1.7",
        now=_NOW,
    )
    assert result.pdf_skipped is True
    assert not any(name.endswith(".pdf") for name in _names(result.zip_bytes))
    readme = _read(result.zip_bytes, "c-extract-20260102-0304/README.md").decode("utf-8")
    assert "combined PDF was skipped" in readme


def test_pdf_is_skipped_above_the_figure_cap() -> None:
    """Figures, not units, are what actually fills the renderer's memory."""
    result = build_bundle(
        [_document()], collection="c", cfg=_cfg(pdf_max_figures=0), pdf=lambda html: b"%PDF", now=_NOW
    )
    assert result.pdf_skipped is True


def test_a_failing_pdf_engine_does_not_lose_the_bundle() -> None:
    """The written files are the point; the PDF is a convenience."""

    def _boom(html: str) -> bytes:
        raise RuntimeError("no pango")

    result = build_bundle([_document()], collection="c", cfg=_cfg(), pdf=_boom, now=_NOW)
    assert result.pdf_skipped is True
    assert "c-extract-20260102-0304/extract.md" in _names(result.zip_bytes)


def test_counts_report_what_went_in() -> None:
    """The sidecar's counts are what the SPA shows beside a stored extract."""
    result = build_bundle([_document(), _media(), _posting()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    assert result.counts == {"documents": 1, "media": 1, "postings": 1, "images": 0, "figures": 4}


# --------------------------------------------------------------------------- #
# Single-source downloads
# --------------------------------------------------------------------------- #
def test_single_markdown_inlines_its_figures() -> None:
    """A lone .md file has no folder to hold images, so they ride inside it."""
    body, media_type = build_single([_document()], "md", collection="c", now=_NOW)
    assert media_type.startswith("text/markdown")
    assert "base64" in body.decode("utf-8")


def test_single_zip_holds_only_the_requested_units() -> None:
    """A per-source download is that source, not the collection."""
    body, media_type = build_single([_media()], "zip", collection="c", now=_NOW)
    assert media_type == "application/zip"
    assert all("/media/" in name or name.endswith(("README.md", "extract.md")) for name in _names(body))


def test_single_pdf_uses_the_injected_engine() -> None:
    """The PDF path is the same engine the report exports use."""
    body, media_type = build_single([_document()], "pdf", collection="c", now=_NOW, pdf=lambda html: b"%PDF-1.7")
    assert media_type == "application/pdf"
    assert body == b"%PDF-1.7"


def test_the_bundle_numbers_its_units_for_citation() -> None:
    """A curated finding cites "appendix A.2"; the README must agree."""
    result = build_bundle([_document(), _media()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    readme = _read(result.zip_bytes, next(n for n in _names(result.zip_bytes) if n.endswith("README.md")))
    combined = _read(
        result.zip_bytes, next(n for n in _names(result.zip_bytes) if n.endswith("c-extract-20260102-0304/extract.md"))
    )
    assert "A.1  report.pdf" in readme.decode("utf-8")
    assert "A.2  clip.mp4" in readme.decode("utf-8")
    assert "# A.2  clip.mp4" in combined.decode("utf-8")


def test_the_case_file_reaches_the_readme_and_the_pdf() -> None:
    """An appendix is filed under the report it belongs to."""
    rendered: list[str] = []

    def _pdf(html: str) -> bytes:
        rendered.append(html)
        return b"%PDF-1.7"

    result = build_bundle(
        [_document()],
        collection="c",
        cfg=_cfg(),
        pdf=_pdf,
        now=_NOW,
        reference_number="AZ-12/26",
        operator="A. Analyst",
    )
    readme = _read(result.zip_bytes, next(n for n in _names(result.zip_bytes) if n.endswith("README.md")))
    assert "AZ-12/26" in readme.decode("utf-8")
    assert "A. Analyst" in readme.decode("utf-8")
    assert "AZ-12/26" in rendered[0]


def test_a_single_source_download_numbers_from_the_top() -> None:
    """One source rendered alone is its own document, so it starts at A.1."""
    body, _media_type = build_single([_media()], "md", collection="c", now=_NOW)
    assert body.decode("utf-8").startswith("# A.1  clip.mp4")


# --------------------------------------------------------------------------- #
# File names
# --------------------------------------------------------------------------- #
def test_a_picture_keeps_the_name_the_export_shipped_it_under() -> None:
    """An analyst looks for the file name, not a content hash.

    Non-ASCII survives: a slug would turn ``Bild März.png`` into something
    nobody exported and nobody can search for.
    """
    result = build_bundle([_posting()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    assert any(name.endswith("/media/Bild März.jpg") for name in _names(result.zip_bytes))


def test_a_thumbnail_is_named_for_its_own_format_not_the_originals() -> None:
    """The stored bytes are JPEG; a ``.png`` name would misdescribe them."""
    result = build_bundle([_posting()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    assert not any(name.endswith(".png") for name in _names(result.zip_bytes))


def test_two_pictures_sharing_one_name_are_both_written() -> None:
    """A bundle that wrote one over the other would silently lose evidence."""
    posting = _posting()
    posting.images.append(Figure(image_id="img-3", kind="image", file_name="Bild März.png", thumbnail_b64=_JPEG))
    result = build_bundle([posting], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    pictures = [name for name in _names(result.zip_bytes) if "/media/" in name]
    assert len(pictures) == len(set(pictures)) == 2


def test_a_keyframe_names_the_clip_it_was_sampled_from() -> None:
    """Two clips on one posting both start at frame 000; only the clip separates them."""
    result = build_bundle([_posting()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    assert any(name.endswith("/keyframes/post-clip_frame_000_00-01.jpg") for name in _names(result.zip_bytes))


def test_a_document_figure_names_its_document_and_page() -> None:
    """A figure was cut out of a page and never had a name of its own."""
    result = build_bundle([_document()], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    assert any(name.endswith("/figures/report_page1_img-1.jpg") for name in _names(result.zip_bytes))


def test_a_file_name_can_never_steer_a_write() -> None:
    """A file name is untrusted input, like a handle."""
    unit = DocumentUnit(
        key="hash-1",
        file_name="../../etc/passwd",
        figures=[Figure(image_id="img-4", kind="figure", file_name="../evil.png", thumbnail_b64=_JPEG)],
    )
    result = build_bundle([unit], collection="c", cfg=_cfg(), pdf=None, now=_NOW)
    root = "c-extract-20260102-0304/"
    for name in _names(result.zip_bytes):
        assert name.startswith(root)
        assert ".." not in name
    assert any("documents/passwd-hash-1/" in name for name in _names(result.zip_bytes))
