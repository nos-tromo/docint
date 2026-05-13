"""Pure data-shaping helpers for NER aggregation.

These functions were extracted from the legacy UI layer to preserve their
test coverage. They contain no UI dependencies and are safe to call from
any context.
"""

from __future__ import annotations

from typing import Any, Iterable

from docint.core.ner import aggregate_ner_sources, entity_cluster_key


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalize_entities(entities: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Return sanitised entity payloads.

    Args:
        entities (Iterable[Any] | None): Iterable of entity dicts or ``None``.

    Returns:
        list[dict[str, Any]]: List of normalised entity dicts.
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
                "type": ent.get("type") or ent.get("label") or "Unlabeled",
                "score": ent.get("score"),
            }
        )
    return normalized


def normalize_relations(relations: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Return sanitised relation payloads.

    Args:
        relations (Iterable[Any] | None): Iterable of relation dicts or ``None``.

    Returns:
        list[dict[str, Any]]: List of normalised relation dicts.
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


def source_label(src: dict[str, Any]) -> str:
    """Build a compact label for a source row.

    Args:
        src (dict[str, Any]): Source dictionary with possible keys ``filename``,
            ``file_path``, ``page``, ``row``.

    Returns:
        str: A human-readable string label.
    """
    filename_val = src.get("filename") or src.get("file_path") or "Unknown"
    filename = str(filename_val).strip() or "Unknown"
    parts: list[str] = []
    if src.get("page") is not None:
        parts.append(f"p{src['page']}")
    if src.get("row") is not None:
        parts.append(f"row {src['row']}")
    return f"{filename} ({', '.join(parts)})" if parts else filename


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_ner(
    sources: Iterable[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate entities and relations across source payloads.

    Deduplicates by normalised key, tracks best scores, counts, and
    originating source files.

    Args:
        sources (Iterable[dict[str, Any]] | None): Iterable of source
            dictionaries containing ``entities`` and ``relations``.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: A tuple
        ``(entities_sorted, relations_sorted)`` where each list item
        carries ``text``, ``type``, ``best_score``, ``count``, ``files``,
        and ``occurrences``.
    """
    source_rows = [src for src in sources or [] if isinstance(src, dict)]
    aggregate = aggregate_ner_sources(source_rows)
    entity_index: dict[str, dict[str, Any]] = {}
    relation_index: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in list(aggregate.get("entities") or []):
        key = str(row.get("key") or "")
        entity_index[key] = {
            "text": row.get("text"),
            "type": row.get("type"),
            "best_score": row.get("best_score"),
            "count": int(row.get("mentions", 0) or 0),
            "files": set(),
            "occurrences": [],
            "variant_count": int(row.get("variant_count", 0) or 0),
            "variants": list(row.get("variants") or []),
        }

    text_to_keys = dict(aggregate.get("text_to_keys") or {})
    compact_to_keys = dict(aggregate.get("compact_to_keys") or {})

    def _resolve_entity(text: str) -> tuple[str | None, str]:
        """Map raw ``text`` to a canonical entity key and display form."""
        lowered = str(text or "").strip().lower()
        compact = "".join(ch for ch in lowered if ch.isalnum())
        for candidates in (
            text_to_keys.get(lowered, []),
            compact_to_keys.get(compact, []),
        ):
            if len(candidates) == 1:
                key = candidates[0]
                entry = entity_index.get(key)
                if entry is not None:
                    return key, str(entry.get("text") or text)
        return None, text

    for src in source_rows:
        lbl = source_label(src)
        entities = normalize_entities(src.get("entities"))
        relations = normalize_relations(src.get("relations"))

        for ent in entities:
            text_val = str(ent.get("text") or "")
            type_val = str(ent.get("type") or "")
            ent_key = entity_cluster_key(text_val, type_val)
            entry = entity_index.get(ent_key)
            if entry is None:
                continue
            entry["files"].add(lbl)
            entry["occurrences"].append(
                {"source": lbl, "score": ent.get("score"), "text": text_val}
            )

        for rel in relations:
            head_raw = str(rel.get("head") or "")
            label_val = str(rel.get("label") or "")
            tail_raw = str(rel.get("tail") or "")
            head_key, head_val = _resolve_entity(head_raw)
            tail_key, tail_val = _resolve_entity(tail_raw)
            rel_key: tuple[str, str, str] = (
                head_key or head_val.lower(),
                label_val.lower(),
                tail_key or tail_val.lower(),
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
            rel_entry = relation_index[rel_key]
            rel_entry["count"] += 1
            rel_entry["files"].add(lbl)
            if rel.get("score") is not None:
                prev = rel_entry.get("best_score")
                rel_entry["best_score"] = (
                    max(prev, rel["score"]) if prev is not None else rel["score"]
                )
            rel_entry["occurrences"].append({"source": lbl, "score": rel.get("score")})

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
