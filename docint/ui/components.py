"""Shared data-processing helpers and reusable rendering components.

This module contains all NER / source data logic previously in ``app.py``,
plus extracted rendering helpers that are reused across pages.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import requests
import streamlit as st

from docint.ui.state import BACKEND_HOST, BACKEND_PUBLIC_HOST

DEFAULT_ARCHIVE_NAME = "source"


# ---------------------------------------------------------------------------
# Data-processing helpers (pure functions – no Streamlit calls)
# ---------------------------------------------------------------------------


def format_score(score: Any) -> str:
    """Format score values for display.

    Args:
        score: The score value to format.

    Returns:
        Formatted score string.
    """
    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return "—"


def normalize_entities(entities: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Return sanitised entity payloads.

    Args:
        entities: Iterable of entity dicts or ``None``.

    Returns:
        List of normalised entity dicts.
    """
    normalized: list[dict[str, Any]] = []
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        text_val = str(ent.get("text") or "").strip()
        if not text_val:
            continue
        normalized.append(
            {
                "text": text_val,
                "type": ent.get("type") or ent.get("label"),
                "score": ent.get("score"),
            }
        )
    return normalized


def normalize_relations(relations: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Return sanitised relation payloads.

    Args:
        relations: Iterable of relation dicts or ``None``.

    Returns:
        List of normalised relation dicts.
    """
    normalized: list[dict[str, Any]] = []
    for rel in relations or []:
        if not isinstance(rel, dict):
            continue
        head = str(rel.get("head") or rel.get("subject") or "").strip()
        tail = str(rel.get("tail") or rel.get("object") or "").strip()
        if not head or not tail:
            continue
        normalized.append(
            {
                "head": head,
                "tail": tail,
                "label": rel.get("label") or rel.get("type"),
                "score": rel.get("score"),
            }
        )
    return normalized


def source_label(src: dict) -> str:
    """Build a compact label for a source row.

    Args:
        src: Source dictionary with possible keys ``filename``, ``file_path``,
            ``page``, ``row``.

    Returns:
        A human-readable string label.
    """
    filename_val = src.get("filename") or src.get("file_path") or "Unknown"
    filename = str(filename_val).strip() or "Unknown"
    parts: list[str] = []
    if src.get("page") is not None:
        parts.append(f"p{src['page']}")
    if src.get("row") is not None:
        parts.append(f"row {src['row']}")
    return f"{filename} ({', '.join(parts)})" if parts else filename


def aggregate_ner(
    sources: Iterable[dict] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate entities and relations across source payloads.

    Deduplicates by normalised key, tracks best scores, counts, and
    originating source files.

    Args:
        sources: Iterable of source dictionaries containing ``entities``
            and ``relations``.

    Returns:
        A tuple ``(entities_sorted, relations_sorted)`` where each list
        item carries ``text``, ``type``, ``best_score``, ``count``,
        ``files``, and ``occurrences``.
    """
    entity_index: dict[tuple[str, str], dict[str, Any]] = {}
    relation_index: dict[tuple[str, str, str], dict[str, Any]] = {}

    for src in sources or []:
        if not isinstance(src, dict):
            continue
        label = source_label(src)
        entities = normalize_entities(src.get("entities"))
        relations = normalize_relations(src.get("relations"))

        for ent in entities:
            text_val = str(ent.get("text") or "")
            type_val = str(ent.get("type") or "")
            ent_key: tuple[str, str] = (text_val.lower(), type_val.lower())
            if ent_key not in entity_index:
                entity_index[ent_key] = {
                    "text": text_val,
                    "type": ent.get("type"),
                    "best_score": ent.get("score"),
                    "count": 0,
                    "files": set(),
                    "occurrences": [],
                }
            entry = entity_index[ent_key]
            entry["count"] += 1
            entry["files"].add(label)
            if ent.get("score") is not None:
                prev = entry.get("best_score")
                entry["best_score"] = (
                    max(prev, ent["score"]) if prev is not None else ent["score"]
                )
            entry["occurrences"].append({"source": label, "score": ent.get("score")})

        for rel in relations:
            head_val = str(rel.get("head") or "")
            label_val = str(rel.get("label") or "")
            tail_val = str(rel.get("tail") or "")
            rel_key: tuple[str, str, str] = (
                head_val.lower(),
                label_val.lower(),
                tail_val.lower(),
            )
            if rel_key not in relation_index:
                relation_index[rel_key] = {
                    "head": head_val,
                    "tail": tail_val,
                    "label": rel.get("label"),
                    "best_score": rel.get("score"),
                    "count": 0,
                    "files": set(),
                    "occurrences": [],
                }
            entry = relation_index[rel_key]
            entry["count"] += 1
            entry["files"].add(label)
            if rel.get("score") is not None:
                prev = entry.get("best_score")
                entry["best_score"] = (
                    max(prev, rel["score"]) if prev is not None else rel["score"]
                )
            entry["occurrences"].append({"source": label, "score": rel.get("score")})

    entities_sorted: list[dict[str, Any]] = sorted(
        [{**v, "files": sorted(v["files"])} for v in entity_index.values()],
        key=lambda item: (
            -int(item.get("count", 0) or 0),
            str(item.get("text") or "").lower(),
        ),
    )
    relations_sorted: list[dict[str, Any]] = sorted(
        [{**v, "files": sorted(v["files"])} for v in relation_index.values()],
        key=lambda item: (
            -int(item.get("count", 0) or 0),
            str(item.get("head") or "").lower(),
            str(item.get("label") or ""),
        ),
    )
    return entities_sorted, relations_sorted


def build_entity_histogram_data(
    entities: Iterable[dict[str, Any]] | None,
    *,
    top_k: int = 15,
    include_type: bool = True,
    max_label_len: int | None = None,
) -> dict[str, int]:
    """Return a label->mention mapping for top entities.

    Args:
        entities: Aggregated entity rows.
        top_k: Maximum number of bars to emit.

    Returns:
        Mapping suitable for ``st.bar_chart``.
    """
    rows = list(entities or [])
    rows.sort(
        key=lambda item: (
            -int(item.get("count", item.get("mentions", 0)) or 0),
            str(item.get("text") or "").lower(),
        )
    )
    chart: dict[str, int] = {}
    for ent in rows[: max(1, int(top_k))]:
        text = str(ent.get("text") or "Unknown")
        if max_label_len is not None and max_label_len > 0:
            if len(text) > max_label_len:
                text = f"{text[: max_label_len - 1].rstrip()}…"
        kind = str(ent.get("type") or "Unlabeled")
        label = f"{text} ({kind})" if include_type else text
        base_label = label
        suffix = 2
        while label in chart:
            label = f"{base_label} #{suffix}"
            suffix += 1
        chart[label] = int(ent.get("count", ent.get("mentions", 0)) or 0)
    return chart


def filter_entities(
    entities: Iterable[dict[str, Any]] | None,
    *,
    query: str = "",
    entity_type: str | None = None,
    min_mentions: int = 1,
    sort_by: str = "mentions",
) -> list[dict[str, Any]]:
    """Filter and sort aggregated entity rows.

    Args:
        entities: Aggregated entity rows.
        query: Case-insensitive substring applied to entity text.
        entity_type: Optional type filter (case-insensitive).
        min_mentions: Minimum mention count.
        sort_by: ``mentions`` or ``score``.

    Returns:
        Filtered and sorted entity rows.
    """
    query_l = str(query or "").strip().lower()
    type_l = str(entity_type or "").strip().lower()

    rows: list[dict[str, Any]] = []
    for ent in entities or []:
        text = str(ent.get("text") or "")
        kind = str(ent.get("type") or "Unlabeled")
        mentions = int(ent.get("count", ent.get("mentions", 0)) or 0)
        if mentions < max(1, int(min_mentions)):
            continue
        if type_l and kind.lower() != type_l:
            continue
        if query_l and query_l not in text.lower():
            continue
        rows.append(ent)

    if sort_by == "score":
        rows.sort(
            key=lambda item: (
                -(float(item.get("best_score") or 0.0)),
                -int(item.get("count", item.get("mentions", 0)) or 0),
                str(item.get("text") or "").lower(),
            )
        )
    else:
        rows.sort(
            key=lambda item: (
                -int(item.get("count", item.get("mentions", 0)) or 0),
                -(float(item.get("best_score") or 0.0)),
                str(item.get("text") or "").lower(),
            )
        )
    return rows


def entity_density_by_document(
    sources: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Compute entity mention density per document.

    Args:
        sources: Source rows with optional ``filename`` and ``entities`` keys.

    Returns:
        Rows sorted by entity mentions desc.
    """
    docs: dict[str, dict[str, Any]] = {}
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        filename = str(src.get("filename") or src.get("file_path") or "Unknown")
        row = docs.setdefault(
            filename,
            {
                "filename": filename,
                "ie_sources": 0,
                "entity_mentions": 0,
                "unique_entities": set(),
            },
        )
        entities = normalize_entities(src.get("entities"))
        relations = normalize_relations(src.get("relations"))
        if entities or relations:
            row["ie_sources"] += 1
        for ent in entities:
            row["entity_mentions"] += 1
            text = str(ent.get("text") or "").lower()
            kind = str(ent.get("type") or "unlabeled").lower()
            row["unique_entities"].add((text, kind))

    rows: list[dict[str, Any]] = []
    for value in docs.values():
        ie_sources = int(value["ie_sources"])
        entity_mentions = int(value["entity_mentions"])
        rows.append(
            {
                "filename": value["filename"],
                "ie_sources": ie_sources,
                "entity_mentions": entity_mentions,
                "unique_entities": len(value["unique_entities"]),
                "entity_density": (
                    float(entity_mentions) / float(ie_sources) if ie_sources else 0.0
                ),
            }
        )
    rows.sort(key=lambda item: (-int(item["entity_mentions"]), item["filename"]))
    return rows


def response_validation_summary(
    *,
    validation_checked: bool | None,
    validation_mismatch: bool | None,
    validation_reason: str | None,
) -> tuple[str, str, str | None] | None:
    """Build a user-facing response-validation summary.

    Args:
        validation_checked: Whether validation executed.
        validation_mismatch: Whether validation found a mismatch.
        validation_reason: Optional validator reason.

    Returns:
        Optional tuple ``(level, title, detail)`` where ``level`` is one of
        ``success``, ``warning``, or ``info``.
    """
    if (
        validation_checked is None
        and validation_mismatch is None
        and not validation_reason
    ):
        return None

    if validation_checked is True and validation_mismatch is True:
        detail = (
            validation_reason
            or "Potential mismatch between answer and retrieved sources."
        )
        return (
            "warning",
            "⚠️ Response validation flagged a potential mismatch.",
            detail,
        )

    if validation_checked is True:
        return ("success", "✅ Response validation passed.", None)

    detail = (
        validation_reason or "Validation was skipped or unavailable for this response."
    )
    return ("info", "ℹ️ Response validation was not completed.", detail)


def summary_diagnostics_summary(
    summary_diagnostics: dict[str, Any] | None,
) -> tuple[str, str | None] | None:
    """Build a user-facing summary-coverage diagnostics message.

    Args:
        summary_diagnostics: Optional summary diagnostics payload from backend.

    Returns:
        Optional tuple ``(title, detail)``.
    """
    if not isinstance(summary_diagnostics, dict):
        return None

    try:
        total_documents = int(summary_diagnostics.get("total_documents", 0) or 0)
        covered_documents = int(summary_diagnostics.get("covered_documents", 0) or 0)
        coverage_ratio = float(summary_diagnostics.get("coverage_ratio", 0.0) or 0.0)
        coverage_target = float(summary_diagnostics.get("coverage_target", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None

    doc_label = "document" if total_documents == 1 else "documents"
    title = (
        "Summary coverage: "
        f"{covered_documents}/{total_documents} {doc_label} "
        f"({coverage_ratio:.0%}, target {coverage_target:.0%})"
    )
    uncovered = summary_diagnostics.get("uncovered_documents")
    if not isinstance(uncovered, list) or not uncovered:
        return (title, None)
    detail = "Uncovered documents: " + ", ".join(str(item) for item in uncovered)
    return (title, detail)


def collect_session_referenced_sources(
    messages: Iterable[dict[str, Any]] | None,
    analysis_sources: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Collect referenced source rows from the current session context.

    Args:
        messages: Current chat-session messages.
        analysis_sources: Current analysis/summary source rows.

    Returns:
        Combined source rows tagged with ``context`` set to ``chat`` or
        ``analysis``.
    """
    collected: list[dict[str, Any]] = []

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        for src in message.get("sources") or []:
            if not isinstance(src, dict):
                continue
            collected.append({**src, "context": "chat"})

    for src in analysis_sources or []:
        if not isinstance(src, dict):
            continue
        collected.append({**src, "context": "analysis"})

    return collected


def unique_referenced_sources(
    sources: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Deduplicate source rows while preserving first-seen ordering.

    Args:
        sources: Source rows from chat and/or analysis contexts. Each row may
            include ``file_hash``, ``file_path``, ``filename``, and ``context``
            keys.

    Returns:
        Unique source rows keyed by file hash when available, else by file path
        or filename.
    """
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str]] = []

    for src in sources or []:
        if not isinstance(src, dict):
            continue
        key = _source_identity_key(src)
        if key is None:
            continue
        if key not in unique:
            unique[key] = dict(src)
            ordered_keys.append(key)
            continue

        existing = unique[key]
        contexts = {
            str(existing.get("context") or "").strip(),
            str(src.get("context") or "").strip(),
        }
        contexts.discard("")
        if contexts:
            existing["context"] = ", ".join(sorted(contexts))

    return [unique[key] for key in ordered_keys]


def build_source_files_zip(
    collection: str,
    sources: Iterable[dict[str, Any]] | None,
    *,
    fetch_source: Callable[[str, str], bytes] | None = None,
) -> tuple[bytes | None, list[str]]:
    """Build a ZIP archive for unique referenced source files.

    Args:
        collection: Active collection name used for backend download requests.
        sources: Source rows from the current session context. Each row may
            include ``file_hash``, ``file_path``, ``filename``, and ``context``
            keys.
        fetch_source: Optional file-fetching callback for tests.

    Returns:
        Tuple ``(zip_bytes, warnings)``. ``zip_bytes`` is ``None`` when no files
        could be archived successfully.
    """
    collection_name = str(collection or "").strip()
    if not collection_name:
        return (None, ["No collection is selected for source export."])

    fetcher = fetch_source or _fetch_source_file
    warnings: list[str] = []
    unique_sources = unique_referenced_sources(sources)
    if not unique_sources:
        return (None, ["No referenced source files are available."])

    buffer = BytesIO()
    written = 0
    used_names: set[str] = set()

    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for src in unique_sources:
            filename = _source_archive_name(src)
            file_hash = str(src.get("file_hash") or "").strip()
            if not file_hash:
                warnings.append(f"{filename}: original file is unavailable.")
                continue

            try:
                content = fetcher(collection_name, file_hash)
            except (requests.RequestException, ValueError) as exc:
                warnings.append(f"{filename}: {exc}")
                continue

            if not isinstance(content, (bytes, bytearray)) or not content:
                warnings.append(f"{filename}: downloaded file was empty.")
                continue

            archive_name = _unique_archive_name(filename, used_names)
            archive.writestr(archive_name, bytes(content))
            written += 1

    if written == 0:
        return (None, warnings or ["Could not create the source ZIP archive."])

    return (buffer.getvalue(), warnings)


# ---------------------------------------------------------------------------
# Rendering helpers (use Streamlit)
# ---------------------------------------------------------------------------


def render_entities_relations(src: dict[str, Any]) -> None:
    """Render entities and relations for a single source.

    Args:
        src: Source dictionary containing ``entities`` and ``relations``.
    """
    entities = normalize_entities(src.get("entities"))
    relations = normalize_relations(src.get("relations"))

    if not entities and not relations:
        return

    col_entities, col_relations = st.columns(2)
    if entities:
        with col_entities:
            st.caption("Entities")
            for ent in entities:
                score = format_score(ent.get("score"))
                lbl = ent.get("type") or "Unlabeled"
                st.markdown(f"- **{ent['text']}** ({lbl}) — score {score}")

    if relations:
        with col_relations:
            st.caption("Relations")
            for rel in relations:
                score = format_score(rel.get("score"))
                lbl = rel.get("label") or "rel"
                st.markdown(
                    f"- **{rel['head']}** — _{lbl}_ → **{rel['tail']}** (score {score})"
                )


def render_ner_overview(sources: list[dict[str, Any]]) -> None:
    """Show aggregated NER results for a set of sources.

    Args:
        sources: List of source dicts with ``entities`` and ``relations``.
    """
    entities, relations = aggregate_ner(sources)

    metrics = st.columns(2)
    metrics[0].metric("Unique entities", len(entities))
    metrics[1].metric("Unique relations", len(relations))

    if entities:
        st.markdown("#### Entities")
        st.dataframe(
            {
                "Entity": [e["text"] for e in entities],
                "Type": [e.get("type") or "Unlabeled" for e in entities],
                "Mentions": [e["count"] for e in entities],
                "Best score": [format_score(e.get("best_score")) for e in entities],
                "Sources": [", ".join(e["files"]) for e in entities],
            },
            width="stretch",
            hide_index=True,
        )
        for ent in entities:
            with st.expander(
                f"{ent['text']} ({ent.get('type') or 'Unlabeled'}) "
                f"— {ent['count']} mention(s)"
            ):
                for occ in ent["occurrences"]:
                    st.markdown(
                        f"- {occ['source']} (score {format_score(occ.get('score'))})"
                    )
    else:
        st.caption("No entities detected.")

    if relations:
        st.markdown("#### Relations")
        st.dataframe(
            {
                "Head": [r["head"] for r in relations],
                "Label": [r.get("label") or "rel" for r in relations],
                "Tail": [r["tail"] for r in relations],
                "Mentions": [r["count"] for r in relations],
                "Best score": [format_score(r.get("best_score")) for r in relations],
                "Sources": [", ".join(r["files"]) for r in relations],
            },
            width="stretch",
            hide_index=True,
        )
        for rel in relations:
            with st.expander(
                f"{rel['head']} — {rel.get('label') or 'rel'} → {rel['tail']} "
                f"({rel['count']} mention(s))"
            ):
                for occ in rel["occurrences"]:
                    st.markdown(
                        f"- {occ['source']} (score {format_score(occ.get('score'))})"
                    )
    else:
        st.caption("No relations detected.")


def render_source_item(src: dict[str, Any], collection: str) -> None:
    """Render a single source item (filename, score, preview, download link).

    Used by the Chat and Analysis pages to display retrieval sources.

    Args:
        src: Source dictionary from the backend.
        collection: Currently selected collection name.
    """
    loc = ""
    if src.get("page") is not None:
        loc += f" (Page {src['page']})"
    if src.get("row") is not None:
        loc += f" (Row {src['row']})"

    score = ""
    if src.get("score"):
        score = f" — Score: {src['score']:.2f}"

    st.markdown(f"**{src.get('filename')}{loc}**{score}")
    st.caption(src.get("preview_text", ""))

    if src.get("file_hash"):
        link = (
            f"{BACKEND_PUBLIC_HOST}/sources/preview"
            f"?collection={collection}&file_hash={src['file_hash']}"
        )
        st.markdown(
            f'<a href="{link}" target="_blank">Download / View Original</a>',
            unsafe_allow_html=True,
        )

    render_entities_relations(src)
    st.divider()


def render_response_validation(
    *,
    validation_checked: bool | None,
    validation_mismatch: bool | None,
    validation_reason: str | None,
) -> None:
    """Render response-validation status and reason when available.

    Args:
        validation_checked: Whether validation executed.
        validation_mismatch: Whether validation found a mismatch.
        validation_reason: Optional validator reason.
    """
    summary = response_validation_summary(
        validation_checked=validation_checked,
        validation_mismatch=validation_mismatch,
        validation_reason=validation_reason,
    )
    if summary is None:
        return

    level, title, detail = summary
    if level == "warning":
        st.warning(title)
    elif level == "success":
        st.success(title)
    else:
        st.info(title)

    if detail:
        st.caption(detail)


def render_summary_diagnostics(summary_diagnostics: dict[str, Any] | None) -> None:
    """Render summary coverage diagnostics when available.

    Args:
        summary_diagnostics: Optional summary diagnostics payload.
    """
    summary = summary_diagnostics_summary(summary_diagnostics)
    if summary is None:
        return

    title, detail = summary
    st.caption(title)
    if detail:
        st.caption(detail)


def _fetch_source_file(collection: str, file_hash: str) -> bytes:
    """Download a single original source file from the backend."""
    response = requests.get(
        f"{BACKEND_HOST}/sources/preview",
        params={"collection": collection, "file_hash": file_hash},
        timeout=60,
    )
    if response.status_code != 200:
        raise ValueError(f"download failed ({response.status_code})")
    return response.content


def _source_archive_name(src: dict[str, Any]) -> str:
    """Return a safe archive name for a referenced source row."""
    raw_name = str(
        src.get("filename") or src.get("file_path") or DEFAULT_ARCHIVE_NAME
    ).strip()
    safe_name = Path(raw_name).name.strip()
    return safe_name or DEFAULT_ARCHIVE_NAME


def _unique_archive_name(filename: str, used_names: set[str]) -> str:
    """Return a unique ZIP entry name while preserving the original filename."""
    candidate = Path(str(filename or DEFAULT_ARCHIVE_NAME)).name or DEFAULT_ARCHIVE_NAME
    stem = Path(candidate).stem or DEFAULT_ARCHIVE_NAME
    suffix = Path(candidate).suffix
    unique_name = candidate
    counter = 2
    while unique_name in used_names:
        unique_name = f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(unique_name)
    return unique_name


def _source_identity_key(src: dict[str, Any]) -> tuple[str, str] | None:
    """Return a tagged key used to identify unique referenced source files."""
    file_hash = str(src.get("file_hash") or "").strip()
    if file_hash:
        return ("file_hash", file_hash)

    file_path = str(src.get("file_path") or "").strip()
    if file_path:
        return ("file_path", file_path)

    filename = str(src.get("filename") or "").strip()
    if filename:
        return ("filename", filename)

    return None
