"""Unit partitioning for the tree summarizer. Pure — no Qdrant, no RAG."""

from typing import Any

from docint.core.summary.units import MapUnit, diversity_bucket, is_social_payload, partition_units


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


def test_documents_partition_one_unit_per_file_hash():
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


def test_social_rows_partition_by_author_time_bucket():
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


def test_mixed_collection_yields_both_kinds():
    points = [
        ("p1", _doc_payload("hashA", "alpha.pdf", "text")),
        ("p2", _social_payload("user_one", "2026-01-01T10:15:00Z", "post", "t1")),
    ]
    units = partition_units(points)
    assert {u.kind for u in units} == {"document", "social_bucket"}


def test_fingerprint_changes_with_content():
    a1 = partition_units([("p1", _doc_payload("hashA", "alpha.pdf", "one"))])[0]
    a2 = partition_units([("p1", _doc_payload("hashA", "alpha.pdf", "CHANGED"))])[0]
    same = partition_units([("p1", _doc_payload("hashA", "alpha.pdf", "one"))])[0]
    assert a1.fingerprint != a2.fingerprint
    assert a1.fingerprint == same.fingerprint


def test_partition_is_deterministic_regardless_of_input_order():
    points = [
        ("p1", _doc_payload("hashA", "alpha.pdf", "one", page=1)),
        ("p2", _doc_payload("hashA", "alpha.pdf", "two", page=2)),
        ("p3", _doc_payload("hashB", "beta.pdf", "x")),
    ]
    forward = partition_units(points)
    backward = partition_units(list(reversed(points)))
    assert [u.unit_key for u in forward] == [u.unit_key for u in backward]
    assert [u.fingerprint for u in forward] == [u.fingerprint for u in backward]


def test_doc_without_file_hash_falls_back_to_filename_key():
    units = partition_units([("p1", {"filename": "loose.txt", "text": "hi", "source": "document"})])
    assert units[0].unit_key == "doc:name:loose.txt"


def test_reference_metadata_as_json_string_is_tolerated():
    payload = {
        "source": "table",
        "text": "post",
        "reference_metadata": '{"type": "post", "author": "user_one", "timestamp": "2026-01-01T10:00:00Z"}',
    }
    assert is_social_payload(payload)
    assert diversity_bucket(payload).startswith("user_one::")


def test_bucket_unknown_author_and_time():
    assert diversity_bucket({"reference_metadata": {}}) == "unknown::unknown"
