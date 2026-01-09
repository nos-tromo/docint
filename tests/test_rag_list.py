from unittest.mock import MagicMock

from docint.core.rag import RAG


def test_list_documents() -> None:
    """
    Test listing documents from RAG with mocked Qdrant client.
    """
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
            "filename": "audio.mp3",
            "file_hash": "hash4",
            "mimetype": "audio/mpeg",
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

    # Sort order is by filename ['audio.mp3', 'complex.json', 'data.csv', 'file1.pdf', 'file2.txt']
    assert len(docs) == 5

    doc_audio = docs[0]
    assert doc_audio["filename"] == "audio.mp3"
    assert doc_audio["mimetype"] == "audio/mpeg"
    assert doc_audio["max_duration"] == 125.5

    doc_complex = docs[1]
    assert doc_complex["filename"] == "complex.json"
    assert doc_complex["page_count"] == 1
    assert "LOC" in doc_complex["entity_types"]
    assert "ORG" in doc_complex["entity_types"]

    doc_csv = docs[2]
    assert doc_csv["filename"] == "data.csv"
    assert doc_csv["mimetype"] == "text/csv"
    assert doc_csv["max_rows"] == 150

    doc_pdf = docs[3]
    assert doc_pdf["filename"] == "file1.pdf"
    assert doc_pdf["page_count"] == 2
    assert doc_pdf["mimetype"] == "application/pdf"

    doc_txt = docs[4]
    assert doc_txt["filename"] == "file2.txt"
    assert doc_txt["page_count"] == 0
