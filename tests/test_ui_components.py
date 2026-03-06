from __future__ import annotations

from docint.ui.components import (
    aggregate_ner,
    build_entity_histogram_data,
    entity_density_by_document,
    filter_entities,
    response_validation_summary,
    summary_diagnostics_summary,
)


def test_build_entity_histogram_data_orders_by_mentions() -> None:
    """Histogram helper should order and cap entities by mention count."""
    entities = [
        {"text": "B", "type": "ORG", "count": 2},
        {"text": "A", "type": "ORG", "count": 5},
        {"text": "C", "type": "LOC", "count": 1},
    ]
    chart = build_entity_histogram_data(entities, top_k=2)
    assert list(chart.keys())[0].startswith("A")
    assert len(chart) == 2
    assert chart["A (ORG)"] == 5


def test_filter_entities_supports_query_type_and_score_sort() -> None:
    """Entity filtering should support query/type filters and score sorting."""
    entities = [
        {"text": "Acme", "type": "ORG", "count": 4, "best_score": 0.7},
        {"text": "Acme Labs", "type": "ORG", "count": 2, "best_score": 0.9},
        {"text": "Rivertown", "type": "LOC", "count": 3, "best_score": 0.5},
    ]
    rows = filter_entities(
        entities,
        query="acme",
        entity_type="org",
        min_mentions=2,
        sort_by="score",
    )
    assert len(rows) == 2
    assert rows[0]["text"] == "Acme Labs"


def test_entity_density_by_document_counts_mentions_and_sources() -> None:
    """Document density helper should compute mention density by filename."""
    sources = [
        {
            "filename": "a.pdf",
            "entities": [
                {"text": "Acme", "type": "ORG"},
                {"text": "Acme", "type": "ORG"},
            ],
        },
        {"filename": "a.pdf", "entities": [{"text": "Rivertown", "type": "LOC"}]},
        {"filename": "b.pdf", "relations": [{"head": "Acme", "tail": "Widget"}]},
    ]
    rows = entity_density_by_document(sources)
    assert rows[0]["filename"] == "a.pdf"
    assert rows[0]["entity_mentions"] == 3
    assert rows[0]["ie_sources"] == 2
    assert rows[0]["entity_density"] == 1.5


def test_filter_entities_handles_aggregate_ner_payload_shape() -> None:
    """Filter helper should work with aggregate_ner outputs."""
    sources = [
        {"filename": "a.pdf", "entities": [{"text": "Acme", "type": "ORG"}]},
        {"filename": "b.pdf", "entities": [{"text": "acme", "type": "ORG"}]},
        {"filename": "c.pdf", "entities": [{"text": "Berlin", "type": "LOC"}]},
    ]
    entities, _ = aggregate_ner(sources)
    filtered = filter_entities(entities, query="ac", min_mentions=2, sort_by="mentions")
    assert len(filtered) == 1
    assert filtered[0]["text"] == "Acme"


def test_response_validation_summary_warning_payload() -> None:
    """Validation summary should return warning details for mismatches."""
    summary = response_validation_summary(
        validation_checked=True,
        validation_mismatch=True,
        validation_reason="source mismatch",
    )
    assert summary == (
        "warning",
        "⚠️ Response validation flagged a potential mismatch.",
        "source mismatch",
    )


def test_response_validation_summary_success_payload() -> None:
    """Validation summary should return success when check passes."""
    summary = response_validation_summary(
        validation_checked=True,
        validation_mismatch=False,
        validation_reason=None,
    )
    assert summary == ("success", "✅ Response validation passed.", None)


def test_response_validation_summary_skipped_payload() -> None:
    """Validation summary should explain skipped/unavailable validation."""
    summary = response_validation_summary(
        validation_checked=False,
        validation_mismatch=None,
        validation_reason=None,
    )
    assert summary == (
        "info",
        "ℹ️ Response validation was not completed.",
        "Validation was skipped or unavailable for this response.",
    )


def test_summary_diagnostics_summary_formats_payload() -> None:
    """Summary diagnostics helper should build readable title/detail."""
    summary = summary_diagnostics_summary(
        {
            "total_documents": 10,
            "covered_documents": 8,
            "coverage_ratio": 0.8,
            "coverage_target": 0.7,
            "uncovered_documents": ["c.pdf", "d.pdf"],
        }
    )
    assert summary == (
        "Summary coverage: 8/10 documents (80%, target 70%)",
        "Uncovered documents: c.pdf, d.pdf",
    )


def test_summary_diagnostics_summary_handles_missing_payload() -> None:
    """Summary diagnostics helper should ignore missing/invalid payloads."""
    assert summary_diagnostics_summary(None) is None
    assert summary_diagnostics_summary({"coverage_ratio": object()}) is None


def test_summary_diagnostics_summary_uses_singular_for_one_document() -> None:
    """Summary diagnostics helper should use singular label for one document."""
    summary = summary_diagnostics_summary(
        {
            "total_documents": 1,
            "covered_documents": 1,
            "coverage_ratio": 1.0,
            "coverage_target": 0.7,
            "uncovered_documents": [],
        }
    )
    assert summary == ("Summary coverage: 1/1 document (100%, target 70%)", None)
