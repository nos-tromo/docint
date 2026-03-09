"""Tests for table ingestion behavior in ``TableReader``."""

from __future__ import annotations

from pathlib import Path

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


def test_table_reader_uses_explicit_csv_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
