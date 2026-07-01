"""Tests for the Nextext transcript caching functionality in IngestManifest."""

from pathlib import Path

from docint.core.storage.ingest_manifest import IngestManifest


def test_cache_and_get_nextext_transcript(tmp_path: Path) -> None:
    """Caching and retrieval of Nextext transcripts should be scoped by (collection, file_hash)."""
    manifest = IngestManifest(tmp_path / "m.db")
    assert manifest.get_nextext_transcript("c", "h1") is None
    manifest.cache_nextext_transcript("c", "h1", '{"text":"x"}\n')
    assert manifest.get_nextext_transcript("c", "h1") == '{"text":"x"}\n'
    assert manifest.get_nextext_transcript("other", "h1") is None  # scoped by (collection, file_hash)
    manifest.close()


def test_cache_survives_reopen(tmp_path: Path) -> None:
    """Cached Nextext transcripts should persist across manifest reopens (migration guard survives)."""
    db = tmp_path / "m.db"
    m1 = IngestManifest(db)
    m1.cache_nextext_transcript("c", "h1", "data\n")
    m1.close()
    m2 = IngestManifest(db)  # migration guard runs again; column persists
    assert m2.get_nextext_transcript("c", "h1") == "data\n"
    m2.close()


def test_cache_only_row_not_marked_completed(tmp_path: Path) -> None:
    """Cache-only rows (no prior ingestion) should not appear in completed_files()."""
    manifest = IngestManifest(tmp_path / "m.db")
    # Insert a cache row for a hash with no prior row
    manifest.cache_nextext_transcript("c", "h", "data\n")
    # Verify it is NOT in completed files
    assert "h" not in manifest.completed_files("c")
    # But the transcript is still retrievable
    assert manifest.get_nextext_transcript("c", "h") == "data\n"
    manifest.close()
