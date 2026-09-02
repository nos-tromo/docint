"""Tests for partitioning a collection's points into extractable units.

Every payload here is synthetic: invented hashes, an invented handle and
invented filenames.
"""

from __future__ import annotations

import json
from typing import Any

from docint.core.extract.units import (
    DocumentUnit,
    ImageUnit,
    MediaUnit,
    PostingUnit,
    handle_from_url,
    partition,
    resolve_target,
)


def _node(text: str, **extra: Any) -> str:
    """Serialize a llama-index-style node blob carrying ``text``."""
    return json.dumps({"text": text, **extra})


def _chunk(point: str, file_hash: str, text: str, **extra: Any) -> tuple[str, dict[str, Any]]:
    """Build a synthetic document-chunk payload."""
    node_extra = {k: v for k, v in extra.items() if k == "start_char_idx"}
    payload: dict[str, Any] = {
        "file_hash": file_hash,
        "file_name": "report.pdf",
        "mimetype": "application/pdf",
        "_node_content": _node(text, **node_extra),
    }
    payload.update({k: v for k, v in extra.items() if k != "start_char_idx"})
    return point, payload


def _segment(point: str, index: int, text: str, **extra: Any) -> tuple[str, dict[str, Any]]:
    """Build a synthetic transcript-segment payload."""
    payload: dict[str, Any] = {
        "docint_doc_kind": "transcript_segment",
        "sentence_index": index,
        "start_seconds": float(index),
        "end_seconds": float(index) + 1.0,
        "start_ts": "00:00:00",
        "end_ts": "00:00:01",
        "speaker": "SPEAKER_00",
        "source_file": "clip.mp4",
        "_node_content": _node(text),
    }
    payload.update(extra)
    return point, payload


def _image_point(point: str, image_id: str, **extra: Any) -> tuple[str, dict[str, Any]]:
    """Build a synthetic ``_images`` companion payload."""
    payload: dict[str, Any] = {
        "image_id": image_id,
        "llm_description": "a description",
        "llm_tags": ["one", "two"],
        "ocr_text": "",
        "thumbnail_b64": "AAAA",
        "thumbnail_mime": "image/jpeg",
    }
    payload.update(extra)
    return point, payload


def _posting_row(point: str, uuid: str, text: str) -> tuple[str, dict[str, Any]]:
    """Build a synthetic postings-table row payload."""
    return point, {
        "source": "table",
        "file_hash": "table-hash",
        "file_name": "postings.csv",
        "reference_metadata": {
            "uuid": uuid,
            "type": "posting",
            "network": "examplenet",
            "author": "Example Account",
            "author_id": "acct-1",
            "timestamp": "2026-01-02T03:04:05",
            "url": "https://example.invalid/p/1",
            "text": text,
        },
        "_node_content": _node(text),
    }


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def test_document_chunks_read_in_page_then_offset_order() -> None:
    """Pages come first, and within a page the character offset decides."""
    points = [
        _chunk("c", "h1", "second of page one", page=1, start_char_idx=500),
        _chunk("a", "h1", "page two", page=2, start_char_idx=0),
        _chunk("b", "h1", "first of page one", page=1, start_char_idx=0),
    ]
    (unit,) = partition(points, [])
    assert isinstance(unit, DocumentUnit)
    assert [c.text for c in unit.chunks] == ["first of page one", "second of page one", "page two"]
    assert unit.approximate_order is False


def test_coarse_parent_chunks_are_dropped() -> None:
    """A hierarchical collection would otherwise emit its text twice."""
    points = [
        _chunk("a", "h1", "the fine chunk", page=1, docint_hier_type="fine"),
        _chunk("b", "h1", "the coarse parent", page=1, docint_hier_type="coarse"),
    ]
    (unit,) = partition(points, [])
    assert isinstance(unit, DocumentUnit)
    assert [c.text for c in unit.chunks] == ["the fine chunk"]


def test_order_is_flagged_approximate_without_pages_or_offsets() -> None:
    """A reader must be told when the order is only the point ids."""
    points = [
        ("b", {"file_hash": "h1", "file_name": "notes.txt", "_node_content": _node("second")}),
        ("a", {"file_hash": "h1", "file_name": "notes.txt", "_node_content": _node("first")}),
    ]
    (unit,) = partition(points, [])
    assert isinstance(unit, DocumentUnit)
    assert unit.approximate_order is True
    assert [c.text for c in unit.chunks] == ["first", "second"]


def test_document_figures_join_by_source_doc_id() -> None:
    """A PDF's extracted figures hang off the document's own hash."""
    points = [_chunk("a", "h1", "body", page=1)]
    images = [_image_point("i1", "img-1", source_type="document", source_doc_id="h1", page_number=3)]
    (unit,) = partition(points, images)
    assert isinstance(unit, DocumentUnit)
    assert [f.image_id for f in unit.figures] == ["img-1"]
    assert unit.figures[0].page_number == 3


# --------------------------------------------------------------------------- #
# Standalone media
# --------------------------------------------------------------------------- #
def test_standalone_media_groups_segments_and_keyframes_by_hash() -> None:
    """A clip's transcript and its frames are one unit, keyed by the media hash."""
    points = [
        _segment("s2", 1, "second line", file_hash="m1", media_file_hash="m1"),
        _segment("s1", 0, "first line", file_hash="m1", media_file_hash="m1"),
    ]
    images = [
        _image_point("k2", "f2", source_type="video_keyframe", source_doc_id="m1", keyframe_time_sec=9.0),
        _image_point("k1", "f1", source_type="video_keyframe", source_doc_id="m1", keyframe_time_sec=1.0),
    ]
    (unit,) = partition(points, images)
    assert isinstance(unit, MediaUnit)
    assert unit.key == "m1"
    assert [s.text for s in unit.segments] == ["first line", "second line"]
    assert [f.time_sec for f in unit.keyframes] == [1.0, 9.0]


def test_untimed_keyframes_fall_back_to_their_index() -> None:
    """Frames from an older Nextext still order, just without times."""
    images = [
        _image_point("k2", "f2", source_type="video_keyframe", source_doc_id="m1", keyframe_index=1),
        _image_point("k1", "f1", source_type="video_keyframe", source_doc_id="m1", keyframe_index=0),
    ]
    (unit,) = partition([], images)
    assert isinstance(unit, MediaUnit)
    assert [f.image_id for f in unit.keyframes] == ["f1", "f2"]


# --------------------------------------------------------------------------- #
# Postings
# --------------------------------------------------------------------------- #
def test_posting_unit_gathers_its_media_by_posting_uuid() -> None:
    """A post, its picture, its clip's transcript and that clip's frames are one unit."""
    points = [
        _posting_row("p1", "uuid-1", "a post"),
        _segment("s1", 0, "spoken words", posting_uuid="uuid-1", media_id="uuid-1_0", file_hash="transient"),
    ]
    images = [
        _image_point(
            "i1",
            "img-1",
            source_type="social_media",
            source_doc_id="uuid-1",
            posting_uuid="uuid-1",
            media_id="uuid-1_1",
            reference_metadata={"type": "image"},
        ),
        _image_point(
            "k1",
            "frame-1",
            source_type="social_media",
            source_doc_id="uuid-1",
            posting_uuid="uuid-1",
            media_id="uuid-1_0",
            keyframe_time_sec=2.0,
            reference_metadata={"type": "keyframe"},
        ),
    ]
    (unit,) = partition(points, images)
    assert isinstance(unit, PostingUnit)
    assert unit.key == "uuid-1"
    assert unit.reference["author"] == "Example Account"
    assert [f.image_id for f in unit.images] == ["img-1"]
    assert [m.key for m in unit.media] == ["uuid-1_0"]
    assert [s.text for s in unit.media[0].segments] == ["spoken words"]
    assert [f.image_id for f in unit.media[0].keyframes] == ["frame-1"]


def test_social_transcript_never_groups_by_file_hash() -> None:
    """Its file_hash is a transient JSONL's, so two clips must not merge."""
    points = [
        _segment("s1", 0, "clip one", posting_uuid="uuid-1", media_id="uuid-1_0", file_hash="same"),
        _segment("s2", 0, "clip two", posting_uuid="uuid-1", media_id="uuid-1_1", file_hash="same"),
    ]
    (unit,) = partition(points, [])
    assert isinstance(unit, PostingUnit)
    assert [m.key for m in unit.media] == ["uuid-1_0", "uuid-1_1"]


# --------------------------------------------------------------------------- #
# Standalone images
# --------------------------------------------------------------------------- #
def test_standalone_image_joins_its_companion_point_by_image_id() -> None:
    """The file's caption node and its CLIP point are one unit, not two."""
    points = [
        (
            "n1",
            {
                "file_hash": "img-1",
                "file_name": "photo.jpg",
                "image_id": "img-1",
                "mimetype": "image/jpeg",
                "_node_content": _node("a description"),
            },
        )
    ]
    images = [_image_point("i1", "img-1", source_type="standalone", ocr_text="printed words")]
    (unit,) = partition(points, images)
    assert isinstance(unit, ImageUnit)
    assert unit.key == "img-1"
    assert unit.figure is not None
    assert unit.figure.ocr_text == "printed words"


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
def test_resolve_target_finds_a_document_by_hash() -> None:
    """A file hash from the Inspector's table addresses its own unit."""
    units = partition([_chunk("a", "h1", "body", page=1)], [])
    assert [u.key for u in resolve_target(units, "h1")] == ["h1"]


def test_resolve_target_expands_a_postings_table_into_its_postings() -> None:
    """A table file is not one document; it is every post recorded in it."""
    units = partition([_posting_row("p1", "uuid-1", "a"), _posting_row("p2", "uuid-2", "b")], [])
    assert sorted(u.key for u in resolve_target(units, "table-hash")) == ["uuid-1", "uuid-2"]


def test_resolve_target_accepts_a_posting_uuid() -> None:
    """One post is addressable on its own."""
    units = partition([_posting_row("p1", "uuid-1", "a")], [])
    assert [u.key for u in resolve_target(units, "uuid-1")] == ["uuid-1"]


def test_resolve_target_returns_nothing_for_an_unknown_id() -> None:
    """An id the collection does not hold is a 404, not an empty bundle."""
    units = partition([_chunk("a", "h1", "body", page=1)], [])
    assert resolve_target(units, "nope") == []


def test_units_sort_deterministically_regardless_of_input_order() -> None:
    """Two builds of the same collection must lay out identically."""
    points = [_chunk("a", "h2", "b", page=1), _chunk("b", "h1", "a", page=1)]
    assert [u.key for u in partition(points, [])] == [u.key for u in partition(list(reversed(points)), [])]


# --------------------------------------------------------------------------- #
# Posting identity
# --------------------------------------------------------------------------- #
def test_a_postings_title_distinguishes_two_posts_by_one_author() -> None:
    """Eight posts by one account are otherwise eight identical headings."""
    unit = PostingUnit(key="u1", reference={"author": "authorname", "timestamp": "2025-05-19T14:51:03+00:00"})
    assert unit.title == "authorname · 2025-05-19 14:51"


def test_a_postings_title_falls_back_when_a_field_is_missing() -> None:
    """Neither half is guaranteed; the title never becomes a bare separator."""
    assert PostingUnit(key="u1", reference={"author": "authorname"}).title == "authorname"
    assert PostingUnit(key="u1", reference={"timestamp": "2025-05-19T14:51:03"}).title == "2025-05-19 14:51"
    assert PostingUnit(key="u1", reference={}).title == "u1"


def test_a_handle_is_read_back_out_of_a_posting_url() -> None:
    """A chat-style export names the account only in its permalink."""
    assert handle_from_url("https://x.com/vanityname/status/4400000000000000004") == "vanityname"
    assert handle_from_url("https://www.twitter.com/vanityname/status/1") == "vanityname"


def test_a_url_that_names_no_account_yields_no_handle() -> None:
    """A wrong guess here would put a route in the report's account row."""
    assert handle_from_url("https://x.com/i/status/1") == ""
    assert handle_from_url("https://example.invalid/vanityname/status/1") == ""
    assert handle_from_url("https://x.com") == ""
    assert handle_from_url("") == ""


def test_a_chat_export_gains_its_handle_from_the_url() -> None:
    """The messages schema carries no handle column, so the link supplies it."""
    payload = {
        "reference_metadata": {
            "network": "x",
            "type": "posting",
            "uuid": "u1",
            "author": "authorname",
            "url": "https://x.com/vanityname/status/4400000000000000004",
        },
        "table": {"row_index": 4},
        "text": "the posted words",
    }
    (unit,) = partition([("p1", payload)], [])
    assert isinstance(unit, PostingUnit)
    assert unit.reference["vanity"] == "vanityname"
    assert unit.row == 4


def test_an_exports_own_handle_is_never_overwritten_by_the_url() -> None:
    """A declared handle is data; the URL-derived one is only a fallback."""
    payload = {
        "reference_metadata": {
            "network": "x",
            "type": "posting",
            "uuid": "u1",
            "vanity": "declared",
            "url": "https://x.com/fromurl/status/1",
        },
        "text": "words",
    }
    (unit,) = partition([("p1", payload)], [])
    assert isinstance(unit, PostingUnit)
    assert unit.reference["vanity"] == "declared"
