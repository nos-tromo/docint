"""Tests for social-media linker join core logic."""

from pathlib import Path

import pandas as pd

from docint.core.ingest.social_linker import (
    build_file_index,
    build_posting_index,
    resolve_media_rows,
    strip_counter,
)


def test_strip_counter_drops_trailing_numeric_segment() -> None:
    """Test that strip_counter removes trailing _<digits> segment."""
    assert strip_counter("2603434334845655437_44657421320_0") == "2603434334845655437_44657421320"
    assert strip_counter("2603434334845655437_44657421320_12") == "2603434334845655437_44657421320"


def test_resolve_matches_by_known_posting_id_and_finds_file_recursively(tmp_path: Path) -> None:
    """Test matching by known posting ID and recursive file resolution."""
    (tmp_path / "media" / "sub").mkdir(parents=True)
    img = tmp_path / "media" / "sub" / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    media = pd.DataFrame({"Media ID": ["P_1_0", "ORPHAN_0"], "Exported media filename": ["a.jpg", "missing.jpg"]})

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        build_file_index(tmp_path),
        tables_dir=tmp_path,
    )
    assert len(links) == 1
    assert links[0].posting_uuid == "uuid-1"
    assert links[0].media_id == "P_1_0"
    assert links[0].path == img
