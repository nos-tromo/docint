"""Tests for reusable UI component helpers."""

from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

from docint.ui.components import (
    aggregate_ner,
    build_source_files_zip,
    build_entity_histogram_data,
    collect_session_referenced_sources,
    entity_density_by_document,
    filter_entities,
    response_validation_summary,
    summary_diagnostics_summary,
    unique_referenced_sources,
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


def test_collect_session_referenced_sources_combines_chat_and_analysis() -> None:
    """Session source collector should merge chat and analysis references."""
    combined = collect_session_referenced_sources(
        [
            {
                "role": "assistant",
                "sources": [
                    {"filename": "chat.pdf", "file_hash": "hash-chat"},
                    {"filename": "shared.pdf", "file_hash": "hash-shared"},
                ],
            },
            {"role": "user", "content": "ignore me"},
        ],
        [
            {"filename": "analysis.pdf", "file_hash": "hash-analysis"},
            {"filename": "shared.pdf", "file_hash": "hash-shared"},
        ],
    )

    assert [row["context"] for row in combined] == [
        "chat",
        "chat",
        "analysis",
        "analysis",
    ]
    assert combined[0]["filename"] == "chat.pdf"
    assert combined[-1]["filename"] == "shared.pdf"


def test_unique_referenced_sources_deduplicates_by_hash_and_merges_contexts() -> None:
    """Unique source helper should collapse duplicate files across contexts."""
    unique = unique_referenced_sources(
        [
            {"filename": "shared.pdf", "file_hash": "same", "context": "chat"},
            {"filename": "shared.pdf", "file_hash": "same", "context": "analysis"},
            {"filename": "other.pdf", "file_hash": "other", "context": "analysis"},
        ]
    )

    assert len(unique) == 2
    assert unique[0]["filename"] == "shared.pdf"
    assert unique[0]["context"] == "analysis, chat"
    assert unique[1]["filename"] == "other.pdf"


def test_build_source_files_zip_deduplicates_entries_and_keeps_names() -> None:
    """ZIP builder should keep source names and avoid duplicate file entries."""

    def _fetch_source(_: str, file_hash: str) -> bytes:
        return {
            "hash-a": b"alpha",
            "hash-b": b"beta",
            "hash-c": b"gamma",
        }[file_hash]

    zip_bytes, warnings = build_source_files_zip(
        "alpha",
        [
            {"filename": "report.pdf", "file_hash": "hash-a", "context": "chat"},
            {"filename": "report.pdf", "file_hash": "hash-a", "context": "analysis"},
            {"filename": "report.pdf", "file_hash": "hash-b", "context": "analysis"},
            {
                "filename": "nested/path/data.csv",
                "file_hash": "hash-c",
                "context": "chat",
            },
        ],
        fetch_source=_fetch_source,
    )

    assert warnings == []
    assert zip_bytes is not None
    with ZipFile(BytesIO(zip_bytes), "r") as archive:
        assert sorted(archive.namelist()) == ["data.csv", "report.pdf", "report_2.pdf"]
        assert archive.read("report.pdf") == b"alpha"
        assert archive.read("report_2.pdf") == b"beta"
        assert archive.read("data.csv") == b"gamma"


def test_build_source_files_zip_handles_missing_hashes_gracefully() -> None:
    """ZIP builder should fail visibly when no downloadable sources are available."""
    zip_bytes, warnings = build_source_files_zip(
        "alpha",
        [{"filename": "report.pdf", "context": "chat"}],
        fetch_source=lambda *_: b"",
    )

    assert zip_bytes is None
    assert warnings == ["report.pdf: original file is unavailable."]
