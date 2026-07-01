"""Tests for the is_media_manifest table-detection helper."""

from docint.core.readers.tables import is_media_manifest


def test_is_media_manifest_detects_join_columns() -> None:
    """Recognises tables whose headers include both required join-column names."""
    assert is_media_manifest(["Media ID", "Exported media filename", "Extra"])
    assert is_media_manifest(["media id", "exported media filename"])


def test_is_media_manifest_rejects_postings() -> None:
    """Returns False for tables that lack the required media-manifest headers."""
    assert not is_media_manifest(["UUID", "Posting ID", "Text Content"])
    assert not is_media_manifest(["Media ID"])
