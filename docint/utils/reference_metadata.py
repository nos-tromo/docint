"""Shared reference-metadata field definitions."""

from __future__ import annotations

from typing import Any

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
    "speaker": "Speaker",
    "language": "Language",
    "source_file": "Source File",
}


def format_reference_metadata_block(
    src: dict[str, Any],
    *,
    include_text: bool = True,
) -> str:
    """Render a source's reference metadata as a multiline text block.

    Iterates :data:`REFERENCE_METADATA_FIELDS` in order, skipping ``None`` and
    empty-after-strip values. Used by the CLI export and the response validator
    to surface stable citation fields (Network, UUID, Timestamp, Author, ...)
    to consumers that would otherwise see only the raw chunk text.

    Args:
        src (dict[str, Any]): Source dictionary containing an optional
            ``reference_metadata`` sub-dict.
        include_text (bool): Whether to include the ``text`` field. Defaults to
            ``True``; pass ``False`` when the body text is rendered separately.

    Returns:
        str: Newline-joined ``- {Label}: {value}`` lines, or ``""`` when no
        reference metadata is present.
    """
    raw = src.get("reference_metadata")
    if not isinstance(raw, dict):
        return ""

    lines: list[str] = []
    for key, label in REFERENCE_METADATA_FIELDS.items():
        if not include_text and key == "text":
            continue
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lines.append(f"- {label}: {text}")
    return "\n".join(lines)
