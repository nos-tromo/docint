"""Tests for collection_overview aggregation."""

from datetime import UTC, datetime

from docint.core.collection_overview import build_collection_overview, mime_label


def test_mime_label_known_and_fallback_and_missing() -> None:
    """Test mime_label for known types, fallback, and missing/empty inputs."""
    assert mime_label("application/pdf") == "PDF"
    assert mime_label("application/x-ndjson") == "NDJSON"
    assert mime_label("application/vnd.acme.thing") == "ACME.THING"  # fallback strips the vnd. prefix
    assert mime_label("application/x-something") == "SOMETHING"  # unknown x- type: fallback strips x-
    assert mime_label("application/vnd.superlongvendortypename") == "SUPERLONGVEN"  # 12-char cap
    assert mime_label(None) == "—"
    assert mime_label("  ") == "—"


def test_build_overview_aggregates_counts_types_and_normalizes_rows() -> None:
    """Test build_collection_overview aggregation, sorting, and max_rows normalization."""
    docs = [
        {
            "filename": "b.pdf",
            "mimetype": "application/pdf",
            "file_hash": "hb",
            "node_count": 3,
            "page_count": 5,
            "entity_types": ["PER", "ORG"],
        },
        {
            "filename": "a.csv",
            "mimetype": "text/csv",
            "file_hash": "ha",
            "node_count": 2,
            "page_count": 0,
            "max_rows": 40,
            "entity_types": ["ORG"],
        },
        {
            "filename": "c.jpg",
            "mimetype": "image/jpeg",
            "file_hash": "hc",
            "node_count": 1,
            "page_count": 0,
            "entity_types": [],
        },
        {
            "filename": "d.pdf",
            "mimetype": "application/pdf",
            "file_hash": "hd",
            "node_count": 4,
            "page_count": 2,
            "entity_types": [],
        },
    ]
    ov = build_collection_overview(docs, "case1", datetime(2026, 7, 6, tzinfo=UTC))
    assert ov["collection"] == "case1"
    assert ov["document_count"] == 4
    assert ov["node_count"] == 10
    # sorted by filename
    assert [d["filename"] for d in ov["documents"]] == ["a.csv", "b.pdf", "c.jpg", "d.pdf"]
    # max_rows -> row_count; page-only doc has row_count None
    csv_row = next(d for d in ov["documents"] if d["filename"] == "a.csv")
    assert csv_row["row_count"] == 40 and csv_row["page_count"] == 0
    pdf_row = next(d for d in ov["documents"] if d["filename"] == "b.pdf")
    assert pdf_row["row_count"] is None and pdf_row["type_label"] == "PDF"
    # entity_types = sorted distinct union
    assert ov["entity_types"] == ["ORG", "PER"]
    # file_types most-common-first (PDF count=2), then alphabetical on ties
    assert [ft["label"] for ft in ov["file_types"]] == ["PDF", "CSV", "JPEG"]
    assert ov["file_types"][0] == {"label": "PDF", "count": 2}
    assert ov["captured_at"].startswith("2026-07-06")


def test_build_overview_empty() -> None:
    """Test build_collection_overview with an empty document list."""
    ov = build_collection_overview([], "empty", datetime(2026, 7, 6, tzinfo=UTC))
    assert ov["document_count"] == 0 and ov["documents"] == [] and ov["node_count"] == 0


def test_row_count_zero_preserved_distinct_from_absent() -> None:
    """Test that max_rows: 0 normalizes to row_count 0, not None (distinct from absent)."""
    ov = build_collection_overview(
        [{"filename": "z.csv", "mimetype": "text/csv", "file_hash": "h", "node_count": 1, "max_rows": 0}],
        "c",
        datetime(2026, 7, 6, tzinfo=UTC),
    )
    assert ov["documents"][0]["row_count"] == 0  # zero rows, distinct from absent (None)
