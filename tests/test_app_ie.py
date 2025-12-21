from typing import Any

from docint import app


def test_aggregate_ie_deduplicates_and_tracks_sources() -> None:
    """
    Tests that the _aggregate_ie method correctly deduplicates entities and relations,
    aggregates their counts and best scores, and tracks the source files and locations.
    """
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

    entities, relations = app._aggregate_ie(sources)

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
