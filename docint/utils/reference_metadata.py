"""Shared reference-metadata field definitions."""

from __future__ import annotations

REFERENCE_METADATA_FIELDS: dict[str, str] = {
    "network": "Network",
    "type": "Type",
    "uuid": "UUID",
    "timestamp": "Timestamp",
    "author": "Author",
    "author_id": "Author ID",
    "vanity": "Vanity",
    "text": "Text",
    "text_id": "Text ID",
    "parent_text": "Parent Text",
    "anchor_text": "Anchor Text",
    # Transcript-specific fields surfaced for Nextext JSONL segments.
    "start_ts": "Start",
    "end_ts": "End",
    "speaker": "Speaker",
    "language": "Language",
    "source_file": "Source File",
}
