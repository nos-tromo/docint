"""Unit tests for the _attach_posting_group helper in docint.core.rag."""

from docint.core.rag import _attach_posting_group


def test_attach_posting_group_from_reference_metadata() -> None:
    """Groups sources by posting UUID from top-level key and reference_metadata."""
    sources = [
        {"text": "post", "reference_metadata": {"uuid": "u1"}},
        {"text": "caption", "posting_uuid": "u1"},
        {"text": "unrelated"},
    ]
    grouped = _attach_posting_group(sources)
    assert grouped[0]["posting_group"] == "u1"
    assert grouped[1]["posting_group"] == "u1"
    assert "posting_group" not in grouped[2]


def test_attach_posting_group_image_source_no_reference_metadata() -> None:
    """Image sources with a top-level posting_uuid (no reference_metadata) are grouped.

    After Fix 2, _retrieve_image_sources populates posting_uuid on image source
    dicts so that _attach_posting_group can group them with their sibling posts.
    """
    sources = [
        {"source": "image", "posting_uuid": "u1", "text": "a red banner"},
        {"source": "image", "text": "unlinked image"},
    ]
    grouped = _attach_posting_group(sources)
    assert grouped[0]["posting_group"] == "u1", "Image source with posting_uuid must receive posting_group"
    assert "posting_group" not in grouped[1], "Image source without posting_uuid must not receive posting_group"
