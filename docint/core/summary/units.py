"""Map-unit discovery for the tree summarizer.

Partitions a collection's raw Qdrant payloads into the units the map stage
summarizes: one unit per document for document-ish content, one unit per
coarse author/hour bucket for row-level social content. Pure and
dependency-free (no Qdrant, no RAG) so it is trivially unit-testable; the
orchestration in :mod:`docint.core.rag` streams ``(point_id, payload)``
pairs in and fetches member texts back out by point id.

``is_social_payload`` is a payload-level counterpart to
``RAG._is_social_payload``, which remains on RAG for collection profiling
(``_infer_collection_profile``). ``diversity_bucket`` has no RAG
counterpart — the source-level helper it was ported from existed only for
the (now-removed) sampling summarizer's social-source selection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class MapUnit:
    """One unit of content the map stage summarizes independently.

    Attributes:
        unit_key: Stable identity — ``doc:{file_hash}``, ``doc:name:{filename}``
            for hash-less documents, or ``social:{bucket}``.
        kind: ``"document"`` or ``"social_bucket"``.
        label: Human-readable name shown in the map prompt and diagnostics
            (filename, or ``author @ hour-bucket``).
        fingerprint: SHA-256 over ordered member identities and text hashes;
            changes iff the unit's content changes. Validates map-cache entries.
        member_ids: Qdrant point ids belonging to this unit, in deterministic
            reading order.
    """

    unit_key: str
    kind: str
    label: str
    fingerprint: str
    member_ids: tuple[str, ...]


def _reference_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload's reference metadata dict, tolerating JSON strings.

    Args:
        payload: Raw Qdrant payload.

    Returns:
        dict[str, Any]: Parsed reference metadata; ``{}`` when absent/invalid.
    """
    ref = payload.get("reference_metadata")
    if isinstance(ref, str):
        try:
            ref = json.loads(ref)
        except (TypeError, ValueError):
            return {}
    return ref if isinstance(ref, dict) else {}


def is_social_payload(payload: dict[str, Any]) -> bool:
    """Return whether a raw payload looks like a row-level social post.

    Args:
        payload: Raw Qdrant payload.

    Returns:
        bool: ``True`` for table-sourced payloads carrying social reference
        metadata (type/network/author/author_id/text_id).
    """
    if not isinstance(payload, dict):
        return False
    if str(payload.get("source") or payload.get("source_type") or "") != "table":
        return False
    ref = _reference_metadata(payload)
    return any(str(ref.get(key) or "").strip() for key in ("type", "network", "author", "author_id", "text_id"))


def diversity_bucket(payload: dict[str, Any]) -> str:
    """Return the coarse ``author::hour`` bucket for a social payload.

    Args:
        payload: Raw Qdrant payload.

    Returns:
        str: ``{author-lowercase}::{YYYY-MM-DDTHH}`` with ``unknown`` fallbacks.
    """
    ref = _reference_metadata(payload)
    author = str(ref.get("author_id") or ref.get("author") or "unknown").strip()
    timestamp_raw = str(ref.get("timestamp") or "").strip()
    time_bucket = "unknown"
    if timestamp_raw:
        try:
            parsed = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
            time_bucket = parsed.astimezone(UTC).strftime("%Y-%m-%dT%H")
        except ValueError:
            time_bucket = timestamp_raw[:13]
    return f"{author.lower()}::{time_bucket}"


#: Suffix of the transient NDJSON ``media_transcribe`` writes beside a clip so
#: ``CustomJSONReader``, which reads from a path, can parse Nextext's answer.
#: The file is unlinked immediately, but a segment ingested before the clip's
#: own name was stamped carries this as its file name — which is why the read
#: side strips it rather than showing a path that never survived the ingest.
TRANSIENT_TRANSCRIPT_SUFFIX = ".nextext.jsonl"


def source_file_name(payload: dict[str, Any]) -> str:
    """Return the name of the file a payload's content came from.

    One rule for every artifact, because every caller wants the same thing: the
    name the file had in the export. ``source_file`` is the media file a
    transcript segment or keyframe was cut from and therefore wins; a document
    chunk or a picture names itself under ``file_name``/``filename``; an
    ``_images`` point may name only a path.

    Args:
        payload (dict[str, Any]): Raw Qdrant payload.

    Returns:
        str: The original file name, or ``""`` when the payload names none.
    """
    for key in ("source_file", "file_name", "filename"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value.removesuffix(TRANSIENT_TRANSCRIPT_SUFFIX)
    path = str(payload.get("source_path") or "").strip()
    return path.rsplit("/", 1)[-1].removesuffix(TRANSIENT_TRANSCRIPT_SUFFIX) if path else ""


def source_file_hash(payload: dict[str, Any]) -> str:
    """Return the hash of the stored file a payload's content came from.

    One rule for every artifact, because the preview and the session ZIP both
    resolve a source by this hash. A chunk carries ``file_hash``; an ``_images``
    point carries none, so a keyframe names the clip it was cut from
    (``media_file_hash``) and a still image names itself (``image_id``, its
    content sha256). ``source_doc_id`` names a *file* only for a document
    figure -- on a social artifact it is the posting's uuid, which no store
    can resolve, so it is skipped whenever it merely repeats ``posting_uuid``.

    Args:
        payload (dict[str, Any]): Raw Qdrant payload.

    Returns:
        str: The file hash, or ``""`` when the payload names none.
    """
    for key in ("file_hash", "media_file_hash"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    source_doc_id = str(payload.get("source_doc_id") or "").strip()
    if source_doc_id and source_doc_id != str(payload.get("posting_uuid") or "").strip():
        return source_doc_id
    return str(payload.get("image_id") or "").strip()


def payload_text(payload: dict[str, Any]) -> str:
    """Best-effort node text from a raw Qdrant payload.

    Raw payloads do NOT carry a top-level ``text`` key: llama-index's
    ``node_to_metadata_dict`` folds the node (text included) into a JSON
    string under ``_node_content``. This mirrors
    ``RAG._extract_payload_text`` (kept separate to preserve this module's
    purity) — without the fallback every fingerprint hashes the empty
    string and every fetched chunk is blank.

    Args:
        payload: Raw Qdrant payload.

    Returns:
        str: Extracted text, or ``""`` when unavailable.
    """
    for key in ("text", "chunk_text", "chunk", "content"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    node_content = payload.get("_node_content")
    node_data: dict[str, Any] | None = None
    if isinstance(node_content, dict):
        node_data = node_content
    elif isinstance(node_content, str) and node_content.strip():
        try:
            parsed = json.loads(node_content)
            if isinstance(parsed, dict):
                node_data = parsed
        except (TypeError, ValueError):
            node_data = None

    if isinstance(node_data, dict):
        for scope in (node_data, node_data.get("metadata")):
            if not isinstance(scope, dict):
                continue
            for key in ("text", "chunk_text", "chunk", "content"):
                candidate = scope.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
    return ""


def _member_sort_key(payload: dict[str, Any], point_id: str) -> tuple[int, int, str]:
    """Build a deterministic reading-order key for a unit member.

    Args:
        payload: Raw Qdrant payload.
        point_id: The member's Qdrant point id (final tie-break).

    Returns:
        tuple[int, int, str]: ``(page, row, point_id)`` with 0 fallbacks.
    """

    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    table = payload.get("table")
    row = payload.get("row")
    if row is None and isinstance(table, dict):
        row = table.get("row_index")
    return (_as_int(payload.get("page")), _as_int(row), point_id)


def _text_hash(payload: dict[str, Any]) -> str:
    """Hash a member's text content for the unit fingerprint.

    Args:
        payload: Raw Qdrant payload.

    Returns:
        str: Short SHA-256 hex digest of the member's text.
    """
    return hashlib.sha256(payload_text(payload).encode("utf-8")).hexdigest()[:16]


def partition_units(points: Iterable[tuple[str, dict[str, Any]]]) -> list[MapUnit]:
    """Partition raw collection points into map units.

    Document-ish payloads group by ``file_hash`` (falling back to filename);
    social payloads group by :func:`diversity_bucket`. Output order and
    per-unit fingerprints are deterministic regardless of input order.

    Args:
        points: ``(point_id, payload)`` pairs for every point in the
            collection.

    Returns:
        list[MapUnit]: Units sorted by ``unit_key``.
    """
    # unit_key -> (kind, label, [(sort_key, point_id, text_hash)])
    groups: dict[str, tuple[str, str, list[tuple[tuple[int, int, str], str, str]]]] = {}
    for point_id, payload in points:
        if not isinstance(payload, dict):
            continue
        if is_social_payload(payload):
            bucket = diversity_bucket(payload)
            unit_key = f"social:{bucket}"
            kind = "social_bucket"
            # Label keeps the author's ORIGINAL casing (the bucket key is
            # lowercased for grouping only) — it is shown to the model and
            # to the operator.
            ref = _reference_metadata(payload)
            display_author = str(ref.get("author_id") or ref.get("author") or "unknown").strip() or "unknown"
            _, _, hour = bucket.partition("::")
            label = f"{display_author} @ {hour}"
        else:
            file_hash = str(payload.get("file_hash") or "").strip()
            filename = str(payload.get("filename") or payload.get("file_name") or "").strip()
            if file_hash:
                unit_key = f"doc:{file_hash}"
            elif filename:
                unit_key = f"doc:name:{filename}"
            else:
                continue
            kind = "document"
            label = filename or unit_key
        entry = groups.setdefault(unit_key, (kind, label, []))
        entry[2].append((_member_sort_key(payload, str(point_id)), str(point_id), _text_hash(payload)))

    units: list[MapUnit] = []
    for unit_key in sorted(groups):
        kind, label, members = groups[unit_key]
        members.sort(key=lambda item: item[0])
        member_ids = tuple(point_id for _, point_id, _ in members)
        digest = hashlib.sha256()
        for _, point_id, text_hash in members:
            digest.update(point_id.encode("utf-8"))
            digest.update(text_hash.encode("utf-8"))
        units.append(
            MapUnit(
                unit_key=unit_key,
                kind=kind,
                label=label,
                fingerprint=digest.hexdigest(),
                member_ids=member_ids,
            )
        )
    return units
