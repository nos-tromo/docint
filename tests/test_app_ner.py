"""Tests for the NER aggregation logic (ported from the Streamlit app module)."""

from typing import Any

from docint.core.ner import aggregate_ner_sources
from docint.utils.ner_aggregate import aggregate_ner


def test_aggregate_ner_deduplicates_and_tracks_sources() -> None:
    """aggregate_ner dedupes entities/relations, sums counts, and tracks source files+locations."""
    sources: list[dict[str, Any]] = [
        {
            "filename": "docA.pdf",
            "page": 1,
            "entities": [
                {"text": "Acme Corp", "type": "ORG", "score": 0.9},
            ],
            "relations": [
                {
                    "head": "Acme Corp",
                    "tail": "Widget",
                    "label": "sells",
                    "score": 0.7,
                }
            ],
        },
        {
            "filename": "docA.pdf",
            "row": 2,
            "entities": [
                {"text": "acme corp", "type": "ORG", "score": 0.8},
                {"text": "Rivertown", "type": "LOC"},
            ],
            "relations": [
                {
                    "head": "ACME CORP",
                    "tail": "Widget",
                    "label": "sells",
                    "score": 0.6,
                }
            ],
        },
        {
            "filename": "docB.pdf",
            "entities": [{"text": "Rivertown", "type": "LOC", "score": 0.5}],
        },
    ]

    entities, relations = aggregate_ner(sources)

    assert len(entities) == 2
    acme = entities[0]
    rivertown = entities[1]

    assert acme["text"] == "Acme Corp"
    assert acme["type"] == "ORG"
    assert acme["count"] == 2
    assert acme["best_score"] == 0.9
    assert acme["files"] == ["docA.pdf (p1)", "docA.pdf (row 2)"]

    assert rivertown["text"] == "Rivertown"
    assert rivertown["type"] == "LOC"
    assert rivertown["count"] == 2
    assert rivertown["best_score"] == 0.5
    assert rivertown["files"] == ["docA.pdf (row 2)", "docB.pdf"]

    assert len(relations) == 1
    rel = relations[0]
    assert rel["head"] == "Acme Corp"
    assert rel["tail"] == "Widget"
    assert rel["label"] == "sells"
    assert rel["count"] == 2
    assert rel["best_score"] == 0.7
    assert rel["files"] == ["docA.pdf (p1)", "docA.pdf (row 2)"]


def test_aggregate_ner_condenses_orthographic_entity_variants() -> None:
    """UI aggregation should condense orthographic variants into one entity row."""
    sources: list[dict[str, Any]] = [
        {
            "filename": "docA.pdf",
            "page": 1,
            "entities": [{"text": "Parteitag", "type": "EVENT", "score": 0.7}],
        },
        {
            "filename": "docB.pdf",
            "row": 2,
            "entities": [{"text": "Partei Tag", "type": "EVENT", "score": 0.9}],
        },
    ]

    entities, _ = aggregate_ner(sources)

    assert len(entities) == 1
    entity = entities[0]
    assert entity["text"] == "Partei Tag"
    assert entity["count"] == 2
    assert entity["variant_count"] == 2
    assert {row["text"] for row in entity["variants"]} == {"Parteitag", "Partei Tag"}
    assert entity["files"] == ["docA.pdf (p1)", "docB.pdf (row 2)"]


def _resolved_index() -> dict[str, Any]:
    """Build a resolved index mapping USA/United States to one entity.

    Returns:
        dict[str, Any]: Index shaped like ``EntityStore.load_alias_index``
        output plus the ``case_normalize`` flag aggregation reads.
    """
    return {
        "alias_to_id": {
            ("usa", "loc"): "ent-1",
            ("united states", "loc"): "ent-1",
        },
        "canonical": {"ent-1": "USA"},
        "case_normalize": True,
    }


def test_resolved_mode_merges_semantic_variants() -> None:
    """Resolved mode collapses USA + United States via the entity index."""
    sources: list[dict[str, Any]] = [
        {"filename": "a.pdf", "entities": [{"text": "USA", "type": "loc", "score": 0.9}]},
        {"filename": "b.pdf", "entities": [{"text": "United States", "type": "loc", "score": 0.8}]},
    ]

    aggregate = aggregate_ner_sources(sources, entity_merge_mode="resolved", resolved_index=_resolved_index())

    assert len(aggregate["entities"]) == 1
    entity = aggregate["entities"][0]
    assert entity["text"] == "USA"
    assert entity["mentions"] == 2
    assert entity["variant_count"] == 2
    assert {v["text"] for v in entity["variants"]} == {"USA", "United States"}


def test_orthographic_mode_keeps_semantic_variants_separate() -> None:
    """Without resolution, USA and United States remain distinct entities."""
    sources: list[dict[str, Any]] = [
        {"filename": "a.pdf", "entities": [{"text": "USA", "type": "loc", "score": 0.9}]},
        {"filename": "b.pdf", "entities": [{"text": "United States", "type": "loc", "score": 0.8}]},
    ]

    aggregate = aggregate_ner_sources(sources, entity_merge_mode="orthographic")

    assert len(aggregate["entities"]) == 2


def test_resolved_mode_falls_back_to_orthographic_for_unmapped() -> None:
    """An entity absent from the index is keyed orthographically, not dropped."""
    sources: list[dict[str, Any]] = [
        {"filename": "a.pdf", "entities": [{"text": "USA", "type": "loc", "score": 0.9}]},
        {"filename": "b.pdf", "entities": [{"text": "Canada", "type": "loc", "score": 0.8}]},
    ]

    aggregate = aggregate_ner_sources(sources, entity_merge_mode="resolved", resolved_index=_resolved_index())

    texts = {e["text"] for e in aggregate["entities"]}
    assert texts == {"USA", "Canada"}
