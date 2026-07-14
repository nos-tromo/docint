"""Tests for the shared reference-metadata field rendering."""

from __future__ import annotations

from docint.utils.reference_metadata import (
    REFERENCE_METADATA_FIELDS,
    format_reference_metadata_block,
    reference_metadata_items,
)


def test_format_reference_metadata_block_renders_detected_language() -> None:
    """The detected source language renders as its own labeled row, after Language."""
    src = {
        "reference_metadata": {
            "language": "en",
            "detected_language": "de",
        }
    }
    block = format_reference_metadata_block(src)
    assert "- Language: en" in block
    assert "- Detected Language: de" in block
    # Detected Language is ordered immediately after Language.
    assert block.index("Detected Language") > block.index("- Language:")


def test_posting_reference_fields_registered_and_labeled() -> None:
    """The posting reference fields are registered, ordered, and labeled."""
    expected = {
        "url": "URL",
        "posting_network": "Posting Network",
        "posting_author": "Posting Author",
        "posting_author_id": "Posting Author ID",
        "posting_vanity": "Posting Vanity",
        "posting_timestamp": "Posting Timestamp",
        "posting_url": "Posting URL",
        "posting_text": "Posting Text",
    }
    for key, label in expected.items():
        assert REFERENCE_METADATA_FIELDS.get(key) == label
    # Posting context is grouped right after the link ids.
    keys = list(REFERENCE_METADATA_FIELDS)
    assert keys.index("url") == keys.index("media_id") + 1


def test_reference_metadata_items_render_posting_fields() -> None:
    """posting_* values render as labeled rows alongside the artifact's own fields."""
    src = {
        "reference_metadata": {
            "network": "nextext",
            "type": "transcript_segment",
            "posting_network": "Facebook",
            "posting_author": "Jane Poster",
            "posting_url": "https://fb.example/p1",
            "posting_text": "Original post body",
        }
    }
    items = dict(reference_metadata_items(src))
    assert items["Network"] == "nextext"
    assert items["Posting Network"] == "Facebook"
    assert items["Posting Author"] == "Jane Poster"
    assert items["Posting URL"] == "https://fb.example/p1"
    assert items["Posting Text"] == "Original post body"
