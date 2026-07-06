"""Tests for social-media linker join core logic."""

from pathlib import Path

import pandas as pd

from docint.core.ingest.social_linker import (
    _derive_posting_id,
    build_posting_index,
    resolve_media_rows,
    strip_counter,
)


def test_strip_counter_drops_trailing_numeric_segment() -> None:
    """Test that strip_counter removes trailing _<digits> segment."""
    assert strip_counter("2603434334845655437_44657421320_0") == "2603434334845655437_44657421320"
    assert strip_counter("2603434334845655437_44657421320_12") == "2603434334845655437_44657421320"


def test_resolve_matches_by_known_posting_id_in_flat_dir(tmp_path: Path) -> None:
    """Test matching by known posting ID with the file directly in the manifest dir."""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    # ORPHAN_0 is skipped as an orphan before file resolution.
    # P_1_1 exercises the missing-file skip branch (known posting, file not on disk).
    media = pd.DataFrame(
        {
            "Media ID": ["P_1_0", "ORPHAN_0", "P_1_1"],
            "Exported media filename": ["a.jpg", "ignored.jpg", "nope.jpg"],
        }
    )

    links = resolve_media_rows(media, build_posting_index(postings), tmp_path)
    assert len(links) == 1
    assert links[0].posting_uuid == "uuid-1"
    assert links[0].media_id == "P_1_0"
    assert links[0].path == img


def test_resolve_does_not_find_file_in_different_directory(tmp_path: Path) -> None:
    """A basename that exists only in a different directory is not resolved.

    Proves the flat model cannot reach outside ``media_dir`` even for a bare
    basename with no path component: ``x.jpg`` lives in a sibling directory,
    not in the media directory passed to ``resolve_media_rows``, so the row
    must be skipped rather than ingested.
    """
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "x.jpg").write_bytes(b"\xff\xd8\xff")

    batch = tmp_path / "batch"
    batch.mkdir()

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    media = pd.DataFrame({"Media ID": ["P_1_0"], "Exported media filename": ["x.jpg"]})

    links = resolve_media_rows(media, build_posting_index(postings), batch)
    assert links == []


def test_resolve_media_rows_aggregates_skips_for_large_manifest(tmp_path: Path) -> None:
    """A large manifest with only a few present files logs ONE summary, not per row.

    Guards robustness for the real-world drop-in shape: a full media.csv (tens of
    thousands of rows) placed in a batch that physically contains only a handful of
    the referenced files. Per-row skip logging would flood; resolution must stay quiet.
    """
    from loguru import logger

    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    n_orphan, n_missing = 500, 300
    media = pd.DataFrame(
        {
            "Media ID": (
                ["P_1_0"]  # linkable: known posting + file present
                + [f"ORPHAN_{i}_0" for i in range(n_orphan)]  # unknown posting
                + [f"P_1_{i}" for i in range(1, n_missing + 1)]  # known posting, file absent
            ),
            "Exported media filename": (["a.jpg"] + ["x.jpg"] * n_orphan + ["missing.jpg"] * n_missing),
        }
    )

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG", format="{level}|{message}")
    try:
        links = resolve_media_rows(media, build_posting_index(postings), tmp_path)
    finally:
        logger.remove(sink_id)

    assert len(links) == 1  # only the present, posting-matched file links
    # Robustness: exactly one aggregated summary line, not ~800 per-row lines.
    assert len(lines) == 1
    assert lines[0].startswith("INFO|")
    assert "media linked" in lines[0]


def test_derive_posting_id_prefers_network_id() -> None:
    """The Network ID column is the join key when it names a known posting."""
    posting_uuids = {"POST_1": "uuid-1"}
    # strip_counter(Media ID) would yield the wrong id here; Network ID is correct.
    assert _derive_posting_id("POST_1", "ALBUM_9_POST_1_ACCT", posting_uuids) == "POST_1"


def test_derive_posting_id_falls_back_to_media_id_then_strip_counter() -> None:
    """Falls back to the raw Media ID, then strip_counter(Media ID); None if neither."""
    posting_uuids = {"POST_1": "uuid-1"}
    assert _derive_posting_id("", "POST_1", posting_uuids) == "POST_1"  # raw Media ID == Posting ID
    assert _derive_posting_id("", "POST_1_0", posting_uuids) == "POST_1"  # <Posting ID>_<counter>
    assert _derive_posting_id("NOPE", "ALSO_NOPE", posting_uuids) is None


def test_resolve_media_rows_links_via_network_id(tmp_path: Path) -> None:
    """A row whose Network ID (not strip_counter) matches a posting links correctly."""
    (tmp_path / "clip.jpg").write_bytes(b"\xff\xd8\xff")
    postings = pd.DataFrame({"Posting ID": ["3745_779"], "UUID": ["uuid-1"]})
    # Media ID is <album>_<posting>_<account>: strip_counter -> "18525_3745" (wrong);
    # the true parent Posting ID is carried in Network ID.
    media = pd.DataFrame(
        {
            "Media ID": ["18525_3745_779"],
            "Network ID": ["3745_779"],
            "Exported media filename": ["clip.jpg"],
        }
    )
    links = resolve_media_rows(media, build_posting_index(postings), tmp_path)
    assert len(links) == 1
    assert links[0].posting_uuid == "uuid-1"
    assert links[0].posting_id == "3745_779"
    assert links[0].media_id == "18525_3745_779"
