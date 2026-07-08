"""Tests for Nextext transcript source_file falling back to extra_info."""

from pathlib import Path

from docint.core.readers.json import CustomJSONReader


def _write(tmp_path: Path) -> Path:
    """Write a two-segment Nextext JSONL fixture with no per-segment source_file.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        Path to the written ``clip.mp4.nextext.jsonl`` fixture.
    """
    jsonl = tmp_path / "clip.mp4.nextext.jsonl"
    jsonl.write_text(
        '{"text":"hello","start_seconds":0,"end_seconds":1}\n{"text":"world","start_seconds":1,"end_seconds":2}\n',
        encoding="utf-8",
    )
    return jsonl


def test_source_file_falls_back_to_extra_info(tmp_path: Path) -> None:
    """Segments with no source_file of their own inherit it from extra_info.

    Mirrors a standalone media ingest: the caller passes the original clip
    name via ``extra_info["source_file"]`` and every segment's top-level
    ``source_file`` and ``reference_metadata["source_file"]`` should carry
    it, alongside other passed-through extra_info keys like ``file_hash``.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    docs = list(
        CustomJSONReader(is_jsonl=True).iter_documents(
            _write(tmp_path),
            extra_info={"source_file": "clip.mp4", "file_hash": "hash-1"},
        )
    )
    assert len(docs) == 2
    for doc in docs:
        assert doc.metadata["source_file"] == "clip.mp4"
        assert doc.metadata["reference_metadata"]["source_file"] == "clip.mp4"
        assert doc.metadata["file_hash"] == "hash-1"


def test_no_source_file_anywhere_omits_it(tmp_path: Path) -> None:
    """Regression: with no source_file anywhere, reference_metadata omits it.

    The social ingestion path passes no ``source_file`` in extra_info (only
    posting ids), so this must keep behaving exactly as before the fallback
    was introduced.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    docs = list(CustomJSONReader(is_jsonl=True).iter_documents(_write(tmp_path)))
    assert "source_file" not in docs[0].metadata["reference_metadata"]


def test_segment_source_file_wins_over_extra_info(tmp_path: Path) -> None:
    """A segment's own source_file takes precedence over the extra_info fallback.

    Guards the ``segment.get(...) or base.get(...)`` operand order: an accidental
    swap would let the caller's fallback clobber the transcript's real value.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    jsonl = tmp_path / "clip.mp4.nextext.jsonl"
    jsonl.write_text(
        '{"text":"hi","start_seconds":0,"end_seconds":1,"source_file":"original.mp4"}\n',
        encoding="utf-8",
    )
    docs = list(
        CustomJSONReader(is_jsonl=True).iter_documents(
            jsonl,
            extra_info={"source_file": "fallback.mp4"},
        )
    )
    assert docs[0].metadata["source_file"] == "original.mp4"
    assert docs[0].metadata["reference_metadata"]["source_file"] == "original.mp4"
