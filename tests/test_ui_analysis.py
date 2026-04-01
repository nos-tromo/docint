"""Tests for analysis UI helper functions."""

from docint.ui.analysis import (
    _entity_chunks_to_txt,
    _highlight_entity_text,
    _entity_related_chunks,
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
    assert rows[0]["reference_metadata"] is None


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


def test_entity_related_chunks_supports_condensed_entity_rows() -> None:
    """Chunk matching should use aggregated variant metadata when available."""
    sources = [
        {
            "filename": "docA.pdf",
            "page": 1,
            "chunk_id": "n1",
            "chunk_text": "Parteitag mention.",
            "entities": [{"text": "Parteitag", "type": "EVENT"}],
        },
        {
            "filename": "docB.pdf",
            "page": 2,
            "chunk_id": "n2",
            "chunk_text": "Partei Tag mention.",
            "entities": [{"text": "Partei Tag", "type": "EVENT"}],
        },
    ]

    rows = _entity_related_chunks(
        {
            "key": "parteitag::event",
            "text": "Partei Tag",
            "variants": [
                {"text": "Parteitag"},
                {"text": "Partei Tag"},
            ],
        },
        sources,
    )

    assert [row["chunk_id"] for row in rows] == ["n1", "n2"]


def test_highlight_entity_text_marks_all_variants() -> None:
    """Highlighted entity text should mark all matched orthographic variants."""
    highlighted = _highlight_entity_text(
        "Parteitag and Partei Tag are both present.",
        {
            "text": "Partei Tag",
            "variants": [{"text": "Parteitag"}, {"text": "Partei Tag"}],
        },
    )

    assert "<mark" in highlighted
    assert "Parteitag" in highlighted
    assert "Partei Tag" in highlighted


def test_highlight_entity_text_escapes_non_matching_html() -> None:
    """Highlight helper should escape raw HTML outside of highlighted terms."""
    highlighted = _highlight_entity_text("<b>Acme</b> & Co.", "Acme")

    assert "&lt;b&gt;" in highlighted
    assert "&amp; Co." in highlighted
    assert "<mark" in highlighted


def test_chunk_download_text_helpers_include_metadata() -> None:
    """Download text helpers should include metadata without body duplication."""
    entity_text = _entity_chunks_to_txt(
        "Acme",
        [
            {
                "filename": "docA.pdf",
                "page": 1,
                "row": None,
                "chunk_id": "c1",
                "chunk_text": "Acme content",
                "reference_metadata": {
                    "network": "Telegram",
                    "type": "comment",
                    "timestamp": "2026-01-02T10:00:00Z",
                    "author": "Alice",
                    "author_id": "a1",
                    "vanity": "alice-v",
                    "text": "Acme content",
                    "text_id": "c1",
                },
            }
        ],
    )
    hate_text = _hate_speech_chunks_to_txt(
        [
            {
                "source_ref": "docB.pdf",
                "page": None,
                "row": 5,
                "chunk_id": "c9",
                "category": "ethnicity",
                "confidence": "high",
                "reason": "Derogatory language",
                "chunk_text": "flagged chunk body",
                "reference_metadata": {
                    "network": "Facebook",
                    "type": "posting",
                    "timestamp": "2026-01-02T10:00:00Z",
                    "author": "Bob",
                    "author_id": "b2",
                    "vanity": "bob-v",
                    "text": "flagged chunk body",
                    "text_id": "p1",
                },
            }
        ]
    )

    assert "chunk_id=c1" in entity_text
    assert "Acme content" in entity_text
    assert "- Network: Telegram" in entity_text
    assert "- Text: Acme content" not in entity_text
    assert entity_text.count("Acme content") == 1
    assert "- category: ethnicity" in hate_text
    assert "flagged chunk body" in hate_text
    assert "- row: 5" in hate_text
    assert "- Text ID: p1" in hate_text
