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
    # ORPHAN_0 is skipped as an orphan before file resolution.
    # P_1_1 exercises the missing-file skip branch (known posting, file not on disk).
    media = pd.DataFrame(
        {
            "Media ID": ["P_1_0", "ORPHAN_0", "P_1_1"],
            "Exported media filename": ["a.jpg", "ignored.jpg", "nope.jpg"],
        }
    )

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


def test_resolve_media_rows_aggregates_skips_for_large_manifest(tmp_path: Path) -> None:
    """A large manifest with only a few present files logs ONE summary, not per row.

    Guards robustness for the real-world drop-in shape: a full media.csv (tens of
    thousands of rows) placed in a batch that physically contains only a handful of
    the referenced files. Per-row skip logging would flood; resolution must stay quiet.
    """
    from loguru import logger

    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "a.jpg").write_bytes(b"\xff\xd8\xff")

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
        links = resolve_media_rows(
            media,
            build_posting_index(postings),
            build_file_index(tmp_path),
            tables_dir=tmp_path,
            root=tmp_path,
        )
    finally:
        logger.remove(sink_id)

    assert len(links) == 1  # only the present, posting-matched file links
    # Robustness: exactly one aggregated summary line, not ~800 per-row lines.
    assert len(lines) == 1
    assert lines[0].startswith("INFO|")
    assert "media linked" in lines[0]
