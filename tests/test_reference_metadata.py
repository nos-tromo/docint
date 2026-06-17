"""Tests for the shared reference-metadata field rendering."""

from __future__ import annotations

from docint.utils.reference_metadata import format_reference_metadata_block


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
