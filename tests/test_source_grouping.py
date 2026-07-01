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
