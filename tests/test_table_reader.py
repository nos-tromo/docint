"""Tests for table ingestion behavior in ``TableReader``."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from docint.core.readers.tables import TableReader


def test_table_reader_autodetects_semicolon_csv(tmp_path: Path) -> None:
    """CSV files using semicolon delimiters are parsed correctly.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """
    csv_path = tmp_path / "de_style.csv"
    csv_path.write_text("title;category\nHallo Welt;politik\n", encoding="utf-8")

    reader = TableReader(text_cols=["title"])
    docs = reader.load_data(csv_path)

    assert len(docs) == 1
    assert docs[0].text == "Hallo Welt"
    assert docs[0].metadata["category"] == "politik"
    assert docs[0].metadata["table"]["columns"] == ["title", "category"]
    assert docs[0].metadata["ft"]["csv"]["sep"] == ";"


def test_table_reader_autodetects_comma_csv(tmp_path: Path) -> None:
    """CSV files using comma delimiters are parsed correctly.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """
    csv_path = tmp_path / "intl_style.csv"
    csv_path.write_text("title,category\nHello World,news\n", encoding="utf-8")

    reader = TableReader(text_cols=["title"])
    docs = reader.load_data(csv_path)

    assert len(docs) == 1
    assert docs[0].text == "Hello World"
    assert docs[0].metadata["category"] == "news"
    assert docs[0].metadata["table"]["columns"] == ["title", "category"]
    assert docs[0].metadata["ft"]["csv"]["sep"] == ","


def test_table_reader_uses_explicit_csv_separator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``csv_sep`` overrides auto-detection.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    csv_path = tmp_path / "explicit_sep.csv"
    csv_path.write_text("title;category\nBonjour;fr\n", encoding="utf-8")

    reader = TableReader(text_cols=["title"], csv_sep=";")

    def _fail_detection(_file_path: Path) -> str:
        raise AssertionError("Delimiter auto-detection should not be used")

    monkeypatch.setattr(reader, "_detect_csv_separator", _fail_detection)
    docs = reader.load_data(csv_path)

    assert len(docs) == 1
    assert docs[0].text == "Bonjour"
    assert docs[0].metadata["category"] == "fr"
    assert docs[0].metadata["ft"]["csv"]["sep"] == ";"


def test_table_reader_reads_a_former_social_export_as_a_plain_table(
    tmp_path: Path,
) -> None:
    """A social export's exact former header set is just a table now.

    Social exports are ingested from ``me-dossier/1`` JSON by the social
    linker; no header set gives a table rows with social reference metadata.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """
    csv_path = tmp_path / "messages.csv"
    csv_path.write_text(
        (
            "UUID,Chat ID,Sender,Timestamp,Text,Tags,URL,Chat Group,Answers Count,Reply To,Network\n"
            "u1,chat-1,Bob,2026-02-03T11:00:00Z,Message body,tag,https://example.com,"
            "group-1,5,root,Signal\n"
        ),
        encoding="utf-8",
    )

    docs = TableReader(text_cols=["Text"], id_col="Chat ID").load_data(csv_path)

    assert len(docs) == 1
    assert docs[0].text == "Message body"
    assert docs[0].doc_id == "chat-1"
    assert "style" not in docs[0].metadata["table"]
    assert "reference_metadata" not in docs[0].metadata


def test_table_reader_sanitizes_excel_time_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Excel-sourced ``datetime.time`` cells sanitize to ISO strings.

    Reproduces the ingestion crash reported in the ``testdata-1`` run:
    ``pd.read_excel`` surfaces Excel time cells as ``datetime.time``
    instances which were previously planted verbatim into node metadata
    and later exploded inside ``SQLiteKVStore.put_all``'s ``json.dumps``.

    Monkeypatches ``pd.read_excel`` so the test does not depend on an
    ``openpyxl`` fixture or an on-disk ``.xlsx``. Asserts both that the
    resulting metadata survives ``json.dumps`` and that the time column
    appears as its ``HH:MM:SS`` ISO representation.

    Args:
        tmp_path: Pytest-provided temporary directory for the synthetic
            Excel placeholder file.
        monkeypatch: Pytest monkeypatch fixture used to replace
            ``pd.read_excel`` with an in-memory DataFrame.
    """
    df = pd.DataFrame(
        {
            "title": ["morning standup", "afternoon sync"],
            "starts_at": [datetime.time(8, 30), datetime.time(14, 15)],
        }
    )
    monkeypatch.setattr(
        "docint.core.readers.tables.pd.read_excel",
        lambda *args, **kwargs: df.copy(),
    )

    xlsx_placeholder = tmp_path / "schedule.xlsx"
    xlsx_placeholder.write_bytes(b"")  # content is irrelevant — read is stubbed

    docs = TableReader(text_cols=["title"]).load_data(xlsx_placeholder)

    assert len(docs) == 2
    # Metadata must be JSON-serializable — this is the regression guard.
    serialized = json.dumps(docs[0].metadata)
    assert "08:30:00" in serialized
    assert docs[0].metadata["starts_at"] == "08:30:00"
    assert docs[1].metadata["starts_at"] == "14:15:00"
