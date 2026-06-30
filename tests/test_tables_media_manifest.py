from docint.core.readers.tables import is_media_manifest


def test_is_media_manifest_detects_join_columns() -> None:
    assert is_media_manifest(["Media ID", "Exported media filename", "Extra"])
    assert is_media_manifest(["media id", "exported media filename"])


def test_is_media_manifest_rejects_postings() -> None:
    assert not is_media_manifest(["UUID", "Posting ID", "Text Content"])
    assert not is_media_manifest(["Media ID"])
