"""Pure aggregation of a collection's document list into a frozen overview snapshot.

The snapshot is stored on a report (``Report.collection_overview_snapshot``) and
rendered as the trailing "Document overview" section. Kept dependency-free (no
Qdrant, no RAG) so it is trivially unit-testable; the API layer fetches the
document list via :meth:`docint.core.rag.RAG.list_documents` under a scoped
collection and passes it here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Python mirror of frontend/src/lib/documentFormat.ts::mimeLabel — keep in sync.
_MIME_LABELS: dict[str, str] = {
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
    "image/tiff": "TIFF",
    "application/pdf": "PDF",
    "text/csv": "CSV",
    "application/json": "JSON",
    "application/x-ndjson": "NDJSON",
    "application/ld+json": "JSON-LD",
    "text/plain": "TXT",
    "text/markdown": "MD",
    "text/rtf": "RTF",
    "application/rtf": "RTF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "application/msword": "DOC",
    "video/mp4": "MP4",
    "audio/mpeg": "MP3",
}


def mime_label(mimetype: str | None) -> str:
    """Return a compact upper-cased label for a MIME type ("—" when missing).

    Mirrors the frontend ``mimeLabel`` so the report manifest and the live
    Inspector show identical type labels. Unknown types degrade to the subtype
    with ``x-`` / ``vnd.`` prefixes stripped, capped at 12 characters.
    """
    if not mimetype:
        return "—"
    key = mimetype.strip().lower()
    if not key:
        return "—"
    if key in _MIME_LABELS:
        return _MIME_LABELS[key]
    subtype = key.split(";")[0].split("/")[-1]
    cleaned = subtype.removeprefix("x-").removeprefix("vnd.")
    if not cleaned:
        return "—"
    return cleaned[:12].upper()


def build_collection_overview(
    documents: list[dict[str, Any]],
    collection: str,
    captured_at: datetime,
) -> dict[str, Any]:
    """Aggregate a raw ``rag.list_documents()`` list into a frozen overview snapshot.

    Args:
        documents: Raw document dicts from :meth:`RAG.list_documents`.
        collection: The report's logical collection name.
        captured_at: Point-in-time stamp (UTC) for this capture.

    Returns:
        The snapshot dict. Keys are English (protocol); ``documents`` is sorted
        by filename and ``file_types`` is ordered most-common-first then
        alphabetically. ``max_rows`` is normalized to ``row_count`` (``None``
        when absent).
    """
    type_counts: dict[str, int] = {}
    entity_types: set[str] = set()
    node_total = 0
    rows: list[dict[str, Any]] = []

    for doc in documents:
        mimetype = doc.get("mimetype")
        label = mime_label(mimetype)
        type_counts[label] = type_counts.get(label, 0) + 1
        node_count = int(doc.get("node_count") or 0)
        node_total += node_count
        for et in doc.get("entity_types") or []:
            if et:
                entity_types.add(str(et))
        raw_rows = doc.get("max_rows")
        row_count = int(raw_rows) if isinstance(raw_rows, (int, float)) else None
        rows.append(
            {
                "filename": str(doc.get("filename") or ""),
                "mimetype": mimetype,
                "type_label": label,
                "page_count": int(doc.get("page_count") or 0),
                "row_count": row_count,
                "node_count": node_count,
                "file_hash": str(doc.get("file_hash") or ""),
            }
        )

    file_types = [
        {"label": label, "count": count} for label, count in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    rows.sort(key=lambda r: r["filename"])

    return {
        "collection": collection,
        "captured_at": captured_at.isoformat(),
        "document_count": len(documents),
        "node_count": node_total,
        "file_types": file_types,
        "entity_types": sorted(entity_types),
        "documents": rows,
    }
