"""Unit partitioning for the tree summarizer. Pure — no Qdrant, no RAG."""

import json
from typing import Any

from docint.core.summary.units import diversity_bucket, is_social_payload, partition_units, payload_text


def _doc_payload(file_hash: str, filename: str, text: str, page: int = 1) -> dict[str, Any]:
    return {
        "file_hash": file_hash,
        "filename": filename,
        "page": page,
        "text": text,
        "source": "document",
    }


def _social_payload(author: str, ts: str, text: str, text_id: str) -> dict[str, Any]:
    return {
        "source": "table",
        "text": text,
        "reference_metadata": {
            "type": "post",
            "network": "examplenet",
            "author": author,
            "timestamp": ts,
            "text_id": text_id,
        },
    }


def test_documents_partition_one_unit_per_file_hash() -> None:
    """Documents partition by file hash with members ordered by page."""
    points = [
        ("p1", _doc_payload("hashA", "alpha.pdf", "one", page=2)),
        ("p2", _doc_payload("hashA", "alpha.pdf", "two", page=1)),
        ("p3", _doc_payload("hashB", "beta.pdf", "three")),
    ]
    units = partition_units(points)
    assert len(units) == 2
    by_key = {u.unit_key: u for u in units}
    alpha = by_key["doc:hashA"]
    assert alpha.kind == "document"
    assert alpha.label == "alpha.pdf"
    # Members ordered by page: p2 (page 1) before p1 (page 2).
    assert alpha.member_ids == ("p2", "p1")


def test_social_rows_partition_by_author_time_bucket() -> None:
    """Social rows partition by author and hour bucket."""
    points = [
        ("p1", _social_payload("user_one", "2026-01-01T10:15:00Z", "post a", "t1")),
        ("p2", _social_payload("user_one", "2026-01-01T10:45:00Z", "post b", "t2")),
        ("p3", _social_payload("user_two", "2026-01-01T10:00:00Z", "post c", "t3")),
    ]
    units = partition_units(points)
    assert len(units) == 2
    kinds = {u.kind for u in units}
    assert kinds == {"social_bucket"}
    sizes = sorted(len(u.member_ids) for u in units)
    assert sizes == [1, 2]


def test_mixed_collection_yields_both_kinds() -> None:
    """Mixed collections yield both document and social_bucket kinds."""
    points = [
        ("p1", _doc_payload("hashA", "alpha.pdf", "text")),
        ("p2", _social_payload("user_one", "2026-01-01T10:15:00Z", "post", "t1")),
    ]
    units = partition_units(points)
    assert {u.kind for u in units} == {"document", "social_bucket"}


def test_fingerprint_changes_with_content() -> None:
    """Fingerprints differ when content changes and match when content is identical."""
    a1 = partition_units([("p1", _doc_payload("hashA", "alpha.pdf", "one"))])[0]
    a2 = partition_units([("p1", _doc_payload("hashA", "alpha.pdf", "CHANGED"))])[0]
    same = partition_units([("p1", _doc_payload("hashA", "alpha.pdf", "one"))])[0]
    assert a1.fingerprint != a2.fingerprint
    assert a1.fingerprint == same.fingerprint


def test_partition_is_deterministic_regardless_of_input_order() -> None:
    """Partition output is deterministic regardless of input order."""
    points = [
        ("p1", _doc_payload("hashA", "alpha.pdf", "one", page=1)),
        ("p2", _doc_payload("hashA", "alpha.pdf", "two", page=2)),
        ("p3", _doc_payload("hashB", "beta.pdf", "x")),
    ]
    forward = partition_units(points)
    backward = partition_units(list(reversed(points)))
    assert [u.unit_key for u in forward] == [u.unit_key for u in backward]
    assert [u.fingerprint for u in forward] == [u.fingerprint for u in backward]


def test_doc_without_file_hash_falls_back_to_filename_key() -> None:
    """Documents without file_hash fall back to filename-based keys."""
    units = partition_units([("p1", {"filename": "loose.txt", "text": "hi", "source": "document"})])
    assert units[0].unit_key == "doc:name:loose.txt"


def test_reference_metadata_as_json_string_is_tolerated() -> None:
    """Reference metadata as JSON string is parsed and handled correctly."""
    payload = {
        "source": "table",
        "text": "post",
        "reference_metadata": '{"type": "post", "author": "user_one", "timestamp": "2026-01-01T10:00:00Z"}',
    }
    assert is_social_payload(payload)
    assert diversity_bucket(payload).startswith("user_one::")


def test_bucket_unknown_author_and_time() -> None:
    """Buckets with missing metadata default to 'unknown'."""
    assert diversity_bucket({"reference_metadata": {}}) == "unknown::unknown"


# Fix round 1 tests: text extraction, row ordering, and label casing


def test_payload_text_extracts_from_node_content_json_string() -> None:
    """Text under _node_content as a JSON string is extracted."""
    payload_with_content = {
        "_node_content": json.dumps({"text": "real content from node"}),
    }
    payload_empty = {
        "_node_content": json.dumps({}),
    }
    assert payload_text(payload_with_content) == "real content from node"
    assert payload_text(payload_empty) == ""


def test_payload_text_extracts_from_node_content_dict() -> None:
    """_node_content as a dict (not a string) is also handled."""
    payload_with_dict = {
        "_node_content": {"text": "direct dict content"},
    }
    assert payload_text(payload_with_dict) == "direct dict content"


def test_payload_text_with_metadata_scope() -> None:
    """Text nested under _node_content.metadata is extracted."""
    payload = {
        "_node_content": json.dumps(
            {
                "metadata": {"text": "nested in metadata"},
            }
        ),
    }
    assert payload_text(payload) == "nested in metadata"


def test_payload_text_prefers_top_level_text() -> None:
    """Top-level text key is preferred over _node_content."""
    payload = {
        "text": "top level",
        "_node_content": json.dumps({"text": "from node"}),
    }
    assert payload_text(payload) == "top level"


def test_payload_text_returns_empty_string_when_no_text() -> None:
    """payload_text returns empty string for payload with no recoverable text."""
    assert payload_text({}) == ""
    assert payload_text({"_node_content": json.dumps({})}) == ""
    assert payload_text({"_node_content": ""}) == ""


def test_fingerprint_differs_with_node_content_text() -> None:
    """Two units with different _node_content text have different fingerprints."""
    payload_a = {
        "file_hash": "hashA",
        "filename": "doc.pdf",
        "_node_content": json.dumps({"text": "content A"}),
        "source": "document",
    }
    payload_b = {
        "file_hash": "hashA",
        "filename": "doc.pdf",
        "_node_content": json.dumps({"text": "content B"}),
        "source": "document",
    }
    unit_a = partition_units([("p1", payload_a)])[0]
    unit_b = partition_units([("p1", payload_b)])[0]
    assert unit_a.fingerprint != unit_b.fingerprint


def test_node_content_vs_empty_content_fingerprint() -> None:
    """A unit built from _node_content differs from one with empty content."""
    payload_with_content = {
        "file_hash": "hashX",
        "filename": "doc.pdf",
        "_node_content": json.dumps({"text": "has text"}),
        "source": "document",
    }
    payload_empty = {
        "file_hash": "hashX",
        "filename": "doc.pdf",
        "source": "document",
    }
    unit_with = partition_units([("p1", payload_with_content)])[0]
    unit_empty = partition_units([("p1", payload_empty)])[0]
    assert unit_with.fingerprint != unit_empty.fingerprint


def test_table_row_ordering_by_nested_row_index() -> None:
    """Table-row payloads ordered by table.row_index produce members in row order."""
    points = [
        (
            "p1",
            {
                "file_hash": "hashA",
                "filename": "table.csv",
                "text": "row 2",
                "table": {"row_index": 2},
                "source": "document",
            },
        ),
        (
            "p2",
            {
                "file_hash": "hashA",
                "filename": "table.csv",
                "text": "row 0",
                "table": {"row_index": 0},
                "source": "document",
            },
        ),
        (
            "p3",
            {
                "file_hash": "hashA",
                "filename": "table.csv",
                "text": "row 1",
                "table": {"row_index": 1},
                "source": "document",
            },
        ),
    ]
    units = partition_units(points)
    assert len(units) == 1
    # Members should be ordered by row_index: p2 (0), p3 (1), p1 (2)
    assert units[0].member_ids == ("p2", "p3", "p1")


def test_social_label_preserves_author_casing() -> None:
    """Social unit labels preserve original author casing while unit_key is lowercased."""
    points = [
        (
            "p1",
            _social_payload("JaneDoe", "2026-01-01T10:15:00Z", "post from jane", "t1"),
        ),
    ]
    units = partition_units(points)
    assert len(units) == 1
    unit = units[0]
    # unit_key should have lowercased author
    assert unit.unit_key.startswith("social:janedoe::")
    # label should preserve original casing
    assert unit.label.startswith("JaneDoe @")
