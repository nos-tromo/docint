"""Tests for analysis UI helper functions."""

from docint.ui.analysis import (
    _dot_escape,
    _entity_chunks_to_txt,
    _entity_related_chunks,
    _graph_connected_subgraph,
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


def test_entity_related_chunks_supports_name_key_and_missing_chunk_text() -> None:
    """Chunk matching should support name/key aliases and keep metadata-only rows."""
    sources = [
        {
            "filename": "docA.pdf",
            "page": 3,
            "chunk_id": "n3",
            "text": "Deutschland appears in this source row.",
            "entities": [{"name": "Deutschland", "type": "LOC"}],
        },
        {
            "filename": "docB.pdf",
            "page": 4,
            "chunk_id": "n4",
            "entities": [{"key": "deutschland::loc", "type": "LOC"}],
        },
    ]

    rows = _entity_related_chunks("Deutschland", sources)

    assert len(rows) == 2
    assert rows[0]["chunk_id"] == "n3"
    assert "Deutschland appears" in rows[0]["chunk_text"]
    assert rows[1]["chunk_id"] == "n4"
    assert rows[1]["chunk_text"] == ""


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


def test_graph_connected_subgraph_drops_isolated_nodes() -> None:
    """Graph render helper should keep only nodes that are connected by edges."""
    graph = {
        "nodes": [
            {"id": "a::org", "text": "A", "mentions": 2},
            {"id": "b::org", "text": "B", "mentions": 1},
            {"id": "c::org", "text": "C", "mentions": 1},
        ],
        "edges": [{"source": "a::org", "target": "b::org", "weight": 1}],
        "meta": {"node_count": 3, "edge_count": 1},
    }

    connected = _graph_connected_subgraph(graph)

    assert [node["id"] for node in connected["nodes"]] == ["a::org", "b::org"]
    assert connected["edges"] == graph["edges"]


def test_graph_connected_subgraph_returns_empty_when_no_edges() -> None:
    """Graph render helper should avoid plotting isolated-only node rows."""
    connected = _graph_connected_subgraph(
        {
            "nodes": [{"id": "a::org", "text": "A", "mentions": 1}],
            "edges": [],
            "meta": {"node_count": 1, "edge_count": 0},
        }
    )

    assert connected["nodes"] == []
    assert connected["edges"] == []


def test_dot_escape_handles_quotes_backslashes_and_newlines() -> None:
    """DOT escaping helper should sanitize strings used in labels/ids."""
    escaped = _dot_escape('A "quote" \\ path\nnext')
    assert escaped == 'A \\"quote\\" \\\\ path\\nnext'


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
