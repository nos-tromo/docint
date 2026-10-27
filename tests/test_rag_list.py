"""Tests for RAG collection listing functionality."""

from typing import Any
from unittest.mock import MagicMock

from docint.core.rag import RAG


def test_list_documents() -> None:
    """Test listing documents from RAG with mocked Qdrant client."""
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()

    # Mock payload
    point1 = MagicMock()
    point1.payload = {
        "origin": {
            "filename": "file1.pdf",
            "file_hash": "hash1",
            "mimetype": "application/pdf",
        },
        "page": 1,
        "mimetype": "application/pdf",
    }

    point2 = MagicMock()
    point2.payload = {
        "origin": {"filename": "file1.pdf", "file_hash": "hash1"},
        "page": 2,
    }

    point3 = MagicMock()
    point3.payload = {
        "origin": {"filename": "file2.txt", "file_hash": "hash2"},
        "page": None,
    }

    point4 = MagicMock()
    point4.payload = {
        "origin": {
            "filename": "data.csv",
            "file_hash": "hash3",
            "filetype": "text/csv",
        },
        "table": {"n_rows": 150},
    }

    point5 = MagicMock()
    point5.payload = {
        "origin": {
            "filename": "transcript.jsonl",
            "file_hash": "hash4",
            "mimetype": "application/x-ndjson",
        },
        "end_seconds": 125.5,
    }

    # Complex nested json structure (Docling)
    point6 = MagicMock()
    point6.payload = {
        "file_name": "complex.json",
        "doc_items": [{"prov": [{"page_no": 5}]}],
        "entities": [{"type": "ORG", "text": "Acme"}, {"label": "LOC", "text": "City"}],
    }

    # scroll returns (points, next_offset)
    rag._qdrant_client.scroll = MagicMock(
        side_effect=[
            ([point5, point1, point2], 100),
            ([point3, point4, point6], 200),
            ([], None),
        ]
    )

    docs = rag.list_documents()

    # Sort order is by filename ['complex.json', 'data.csv', 'file1.pdf',
    # 'file2.txt', 'transcript.jsonl']
    assert len(docs) == 5

    doc_complex = docs[0]
    assert doc_complex["filename"] == "complex.json"
    assert doc_complex["page_count"] == 1
    assert "LOC" in doc_complex["entity_types"]
    assert "ORG" in doc_complex["entity_types"]

    doc_csv = docs[1]
    assert doc_csv["filename"] == "data.csv"
    assert doc_csv["mimetype"] == "text/csv"
    assert doc_csv["max_rows"] == 150

    doc_pdf = docs[2]
    assert doc_pdf["filename"] == "file1.pdf"
    assert doc_pdf["page_count"] == 2
    assert doc_pdf["mimetype"] == "application/pdf"

    doc_txt = docs[3]
    assert doc_txt["filename"] == "file2.txt"
    assert doc_txt["page_count"] == 0

    doc_transcript = docs[4]
    assert doc_transcript["filename"] == "transcript.jsonl"
    assert doc_transcript["mimetype"] == "application/x-ndjson"
    assert doc_transcript["max_duration"] == 125.5


def test_list_documents_terminates_when_cursor_exhausted() -> None:
    """list_documents must stop once the scroll cursor is exhausted.

    Regression for an infinite loop: real Qdrant returns the final partial page
    *with* a ``None`` next-page offset (it does not emit a trailing empty page).
    The loop only broke on empty ``points``, so after the last page ``offset``
    reset to ``None`` and re-scrolled page 1 forever. This reproduces real
    Qdrant pagination (no trailing empty page; ``offset=None`` re-yields page 1)
    and asserts the scan terminates — chat and summary reach this method via
    ``start_session`` -> ``build_query_engine`` -> ``_infer_collection_profile``,
    so the loop hung both end to end.
    """
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()

    point1 = MagicMock()
    point1.payload = {"origin": {"filename": "a.pdf", "file_hash": "h1"}, "page": 1}
    point2 = MagicMock()
    point2.payload = {"origin": {"filename": "b.pdf", "file_hash": "h2"}, "page": 1}

    # page 1 carries a cursor; the final page returns offset=None (no empty
    # trailing page). offset=None always re-yields page 1, as real Qdrant does.
    page1: tuple[list[MagicMock], Any] = ([point1], "cursor-1")
    page2: tuple[list[MagicMock], Any] = ([point2], None)
    calls = {"n": 0}

    def fake_scroll(**kwargs: Any) -> tuple[list[MagicMock], Any]:
        calls["n"] += 1
        if calls["n"] > 10:
            raise AssertionError("list_documents did not terminate — infinite scroll loop")
        return page1 if kwargs.get("offset") is None else page2

    rag._qdrant_client.scroll = fake_scroll

    docs = rag.list_documents()

    assert calls["n"] == 2  # page 1 (cursor) -> page 2 (offset=None) -> stop
    assert sorted(d["filename"] for d in docs) == ["a.pdf", "b.pdf"]


def test_list_documents_tallies_topic_values_per_document() -> None:
    """Each document counts its points per topic value, most frequent first, across runs."""
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = MagicMock()
    tagged = [
        ("a.pdf", "r1:2"),
        ("a.pdf", "r1:1"),
        ("a.pdf", "r1:2"),
        ("a.pdf", "r0:7"),
        ("a.pdf", "r1:1"),
        ("b.pdf", None),
    ]
    points = []
    for name, topic in tagged:
        point = MagicMock()
        point.payload = {"file_name": name, **({"docint_topic": topic} if topic else {})}
        points.append(point)
    rag._qdrant_client.scroll = MagicMock(return_value=(points, None))

    docs = rag.list_documents()

    assert docs[0]["topics"] == [
        {"topic": "r1:1", "count": 2},
        {"topic": "r1:2", "count": 2},
        {"topic": "r0:7", "count": 1},
    ]
    assert "topics" not in docs[1]
