"""Phase 2 streaming-reader tests: iter_documents API parity with load_data.

Pins the contract that:

* TableReader, CustomJSONReader, and ImageReader all expose a generator
  ``iter_documents`` entry point alongside the eager-list ``load_data``.
* ``load_data`` is a thin shim over the iterator and produces the same
  output sequence.
* The iterator does NOT exhaust its source up front — for files with
  many rows/segments, partial consumption produces only the prefix.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import GeneratorType

import pandas as pd
import pytest

from docint.core.readers.json import CustomJSONReader
from docint.core.readers.tables import TableReader


def _write_nextext_jsonl(path: Path, n_segments: int) -> None:
    """Write *n_segments* lines of Nextext-shaped JSONL to *path*."""
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n_segments):
            fh.write(
                json.dumps(
                    {
                        "text": f"segment {i}",
                        "sentence_index": i,
                        "start_ts": "00:00:00",
                        "end_ts": "00:00:01",
                        "start_seconds": float(i),
                        "end_seconds": float(i) + 1.0,
                        "speaker": "S0",
                        "whisper_task": "transcribe",
                        "whisper_language": "en",
                        "source_file": "audio.mp3",
                        "source_file_hash": "sha256:abc",
                    }
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# CustomJSONReader (Nextext + generic JSON)
# ---------------------------------------------------------------------------


def test_custom_json_iter_documents_is_a_generator(tmp_path: Path) -> None:
    """``iter_documents`` should return a Python generator object."""
    jsonl = tmp_path / "t.jsonl"
    _write_nextext_jsonl(jsonl, n_segments=3)

    reader = CustomJSONReader(is_jsonl=True)
    iterator = reader.iter_documents(jsonl)

    assert isinstance(iterator, GeneratorType)
    assert inspect.isgenerator(iterator)


def test_custom_json_iter_documents_matches_load_data_for_nextext(
    tmp_path: Path,
) -> None:
    """``load_data`` should produce the same ordered output as ``iter_documents``."""
    jsonl = tmp_path / "t.jsonl"
    _write_nextext_jsonl(jsonl, n_segments=4)

    reader = CustomJSONReader(is_jsonl=True)
    eager = reader.load_data(jsonl)
    streamed = list(reader.iter_documents(jsonl))

    assert len(eager) == len(streamed) == 4
    for a, b in zip(eager, streamed):
        assert a.text == b.text
        assert a.metadata.get("sentence_index") == b.metadata.get(
            "sentence_index"
        )
        assert a.metadata.get("file_hash") == b.metadata.get("file_hash")


def test_custom_json_iter_documents_streams_lazily(tmp_path: Path) -> None:
    """Partial consumption should not exhaust the underlying file."""
    jsonl = tmp_path / "t.jsonl"
    _write_nextext_jsonl(jsonl, n_segments=20)

    reader = CustomJSONReader(is_jsonl=True)
    iterator = reader.iter_documents(jsonl)

    first_two = [next(iterator), next(iterator)]
    assert first_two[0].metadata["sentence_index"] == 0
    assert first_two[1].metadata["sentence_index"] == 1
    # The generator is still alive and has more segments to yield.
    third = next(iterator)
    assert third.metadata["sentence_index"] == 2


def test_custom_json_iter_documents_validates_unsupported_suffix(
    tmp_path: Path,
) -> None:
    """An unsupported suffix should raise ``ValueError`` on first iteration."""
    bad = tmp_path / "table.csv"
    bad.write_text("col\nval\n", encoding="utf-8")

    reader = CustomJSONReader()
    with pytest.raises(ValueError, match=r"\.json"):
        list(reader.iter_documents(bad))


# ---------------------------------------------------------------------------
# TableReader (CSV / Parquet)
# ---------------------------------------------------------------------------


def _write_csv(path: Path, n_rows: int) -> None:
    """Write a CSV with *n_rows* of dummy data."""
    rows = [{"id": i, "text": f"row {i}", "n": i * 2} for i in range(n_rows)]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_table_iter_documents_is_a_generator(tmp_path: Path) -> None:
    """``iter_documents`` should return a Python generator object."""
    csv = tmp_path / "rows.csv"
    _write_csv(csv, n_rows=5)

    reader = TableReader(text_cols=["text"])
    iterator = reader.iter_documents(csv)

    assert isinstance(iterator, GeneratorType)


def test_table_iter_documents_matches_load_data(tmp_path: Path) -> None:
    """The eager shim should match the streaming generator output."""
    csv = tmp_path / "rows.csv"
    _write_csv(csv, n_rows=8)

    reader = TableReader(text_cols=["text"])
    eager = reader.load_data(csv)
    streamed = list(reader.iter_documents(csv))

    assert len(eager) == len(streamed) == 8
    for a, b in zip(eager, streamed):
        assert a.text == b.text
        assert a.metadata.get("table", {}).get("row_index") == b.metadata.get(
            "table", {}
        ).get("row_index")


def test_table_iter_documents_streams_lazily(tmp_path: Path) -> None:
    """Pulling only the first row should not require exhausting the table."""
    csv = tmp_path / "rows.csv"
    _write_csv(csv, n_rows=50)

    reader = TableReader(text_cols=["text"])
    iterator = reader.iter_documents(csv)

    first = next(iterator)
    second = next(iterator)
    assert first.text == "row 0"
    assert second.text == "row 1"
    # generator still produces more rows
    third = next(iterator)
    assert third.text == "row 2"


def test_table_iter_documents_respects_limit(tmp_path: Path) -> None:
    """The ``limit`` field should cap iteration mid-file."""
    csv = tmp_path / "rows.csv"
    _write_csv(csv, n_rows=100)

    reader = TableReader(text_cols=["text"], limit=5)
    docs = list(reader.iter_documents(csv))
    assert len(docs) == 5


def test_table_iter_documents_unsupported_suffix(tmp_path: Path) -> None:
    """Unsupported file types should raise ``ValueError`` on first iteration."""
    bad = tmp_path / "thing.bin"
    bad.write_bytes(b"\x00\x01\x02")

    reader = TableReader(text_cols=["text"])
    with pytest.raises(ValueError, match="Unsupported table type"):
        list(reader.iter_documents(bad))


# ---------------------------------------------------------------------------
# ImageReader (single-file uniformity)
# ---------------------------------------------------------------------------


def test_image_iter_documents_yields_one_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ImageReader.iter_documents should yield exactly one document for a file."""
    from docint.core.readers.images import ImageReader

    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # not real, but bytes are read

    class FakeRecord:
        llm_description = "a cat"
        llm_tags = ["cat", "animal"]
        image_id = "img-1"
        point_id = "pt-1"
        error = None
        status = "ok"

    class FakeService:
        def ingest_image(self, asset, context=None):
            return FakeRecord()

    reader = ImageReader(image_ingestion_service=FakeService())  # type: ignore[arg-type]
    docs = list(reader.iter_documents(img))
    assert len(docs) == 1
    assert "cat" in docs[0].text


def test_image_load_data_matches_iter_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list shim should produce the same single document the iterator yields."""
    from docint.core.readers.images import ImageReader

    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    class FakeRecord:
        llm_description = "a dog"
        llm_tags: list[str] = []
        image_id = ""
        point_id = ""
        error = None
        status = ""

    class FakeService:
        def ingest_image(self, asset, context=None):
            return FakeRecord()

    reader = ImageReader(image_ingestion_service=FakeService())  # type: ignore[arg-type]
    eager = reader.load_data(img)
    streamed = list(reader.iter_documents(img))
    assert len(eager) == len(streamed) == 1
    assert eager[0].text == streamed[0].text
