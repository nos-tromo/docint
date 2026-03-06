from docint.ui.analysis import (
    _entity_chunks_to_txt,
    _entity_related_chunks,
    _graphviz_dot,
    _hate_speech_chunks_to_txt,
    _update_summary_metadata,
)


def test_update_summary_metadata_tracks_diagnostics_and_validation() -> None:
    """Analysis metadata updater should persist validation and diagnostics values."""
    result = {
        "validation_checked": None,
        "validation_mismatch": None,
        "validation_reason": None,
        "summary_diagnostics": None,
    }
    event = {
        "validation_checked": True,
        "validation_mismatch": False,
        "validation_reason": None,
        "summary_diagnostics": {
            "total_documents": 5,
            "covered_documents": 4,
            "coverage_ratio": 0.8,
            "coverage_target": 0.7,
            "uncovered_documents": ["doc5.pdf"],
        },
    }

    _update_summary_metadata(result, event)

    assert result["validation_checked"] is True
    assert result["validation_mismatch"] is False
    assert result["validation_reason"] is None
    assert result["summary_diagnostics"] == event["summary_diagnostics"]


def test_entity_related_chunks_matches_existing_ner_sources() -> None:
    """Entity chunk mapping should reuse NER sources and return matching chunks only."""
    sources = [
        {
            "filename": "docA.pdf",
            "page": 1,
            "chunk_id": "n1",
            "chunk_text": "Acme opened a new office in Rivertown.",
            "entities": [{"text": "Acme"}, {"text": "Rivertown"}],
        },
        {
            "filename": "docB.pdf",
            "page": 2,
            "chunk_id": "n2",
            "chunk_text": "Completely unrelated text.",
            "entities": [{"text": "Elsewhere"}],
        },
    ]

    rows = _entity_related_chunks("Acme", sources)

    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "n1"
    assert "Acme opened" in rows[0]["chunk_text"]


def test_graphviz_dot_highlights_selected_node() -> None:
    """Entity graph DOT rendering should contain nodes, edges, and selected styling."""
    dot = _graphviz_dot(
        {
            "nodes": [
                {"id": "acme::org", "text": "Acme", "mentions": 2},
                {"id": "river::loc", "text": "Rivertown", "mentions": 1},
            ],
            "edges": [
                {
                    "source": "acme::org",
                    "target": "river::loc",
                    "label": "located_in",
                    "weight": 2,
                }
            ],
        },
        selected_entity="Acme",
    )

    assert '"acme::org"' in dot
    assert "located_in (2)" in dot
    assert 'fillcolor="#90CAF9"' in dot


def test_chunk_download_text_helpers_include_metadata() -> None:
    """Download text helpers should include chunk metadata and body text."""
    entity_text = _entity_chunks_to_txt(
        "Acme",
        [
            {
                "filename": "docA.pdf",
                "page": 1,
                "row": None,
                "chunk_id": "c1",
                "chunk_text": "Acme content",
            }
        ],
    )
    hate_text = _hate_speech_chunks_to_txt(
        [
            {
                "source_ref": "docB.pdf",
                "page": 3,
                "chunk_id": "c9",
                "category": "ethnicity",
                "confidence": "high",
                "reason": "Derogatory language",
                "chunk_text": "flagged chunk body",
            }
        ]
    )

    assert "chunk_id=c1" in entity_text
    assert "Acme content" in entity_text
    assert "category=ethnicity" in hate_text
    assert "flagged chunk body" in hate_text
