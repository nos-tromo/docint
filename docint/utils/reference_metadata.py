"""Shared reference-metadata field definitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

REFERENCE_METADATA_FIELDS: dict[str, str] = {
    "network": "Network",
    "type": "Type",
    "uuid": "UUID",
    "posting_uuid": "Posting UUID",
    "posting_id": "Posting ID",
    "media_id": "Media ID",
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
    "detected_language": "Detected Language",
    "source_file": "Source File",
}

# Fields whose contents duplicate (or dwarf) the body text rendered separately;
# callers that already show the chunk text skip these in the metadata block.
BODY_TEXT_FIELDS: frozenset[str] = frozenset({"text", "parent_text", "anchor_text"})


def reference_metadata_items(
    src: dict[str, Any],
    *,
    skip_keys: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """Return a source's reference metadata as ordered ``(label, value)`` pairs.

    Iterates :data:`REFERENCE_METADATA_FIELDS` in order, omitting any key in
    ``skip_keys`` as well as ``None`` / empty-after-strip values. Shared by the
    plain-text block renderer below and the report renderers, so field order and
    selection stay in lockstep across every export.

    Args:
        src (dict[str, Any]): Source dictionary containing an optional
            ``reference_metadata`` sub-dict.
        skip_keys (Iterable[str]): Field keys to omit (e.g.
            :data:`BODY_TEXT_FIELDS` when the body text is rendered separately).

    Returns:
        list[tuple[str, str]]: Ordered ``(label, value)`` pairs, or ``[]`` when
        no reference metadata is present.
    """
    raw = src.get("reference_metadata")
    if not isinstance(raw, dict):
        return []
    skip = set(skip_keys)
    items: list[tuple[str, str]] = []
    for key, label in REFERENCE_METADATA_FIELDS.items():
        if key in skip:
            continue
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        items.append((label, text))
    return items


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
    skip_keys = () if include_text else ("text",)
    return "\n".join(f"- {label}: {value}" for label, value in reference_metadata_items(src, skip_keys=skip_keys))
