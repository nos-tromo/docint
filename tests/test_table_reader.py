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


def test_table_reader_detects_social_media_comments_schema(tmp_path: Path) -> None:
    """Exact comments schema should attach normalized reference metadata.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """
    csv_path = tmp_path / "comments.csv"
    csv_path.write_text(
        (
            "UUID,Comment ID,Network Object ID,URL,Crawled at,Network,Text Content,"
            "Timestamp,Tags,Author ID,Author,Vanity Name,Replies Count,Reactions Count,"
            "Parent Comment Text,Parent Comment ID,Posting Text,Posting ID\n"
            "u1,c1,n1,https://example.com,2026-01-01,Telegram,Comment body,"
            "2026-01-02T10:00:00Z,tag,a1,Alice,alice-v,2,3,parent,pc1,post,p1\n"
        ),
        encoding="utf-8",
    )

    docs = TableReader().load_data(csv_path)

    assert len(docs) == 1
    assert docs[0].text == "Comment body"
    assert docs[0].doc_id == "c1"
    assert docs[0].metadata["table"]["style"] == "comments"
    assert docs[0].metadata["reference_metadata"] == {
        "network": "Telegram",
        "type": "comment",
        "uuid": "u1",
        "timestamp": "2026-01-02T10:00:00Z",
        "author": "Alice",
        "author_id": "a1",
        "vanity": "alice-v",
        "text": "Comment body",
        "text_id": "c1",
        "parent_text": "parent",
        "anchor_text": "post",
    }


def test_table_reader_detects_social_media_messages_schema(tmp_path: Path) -> None:
    """Exact messages schema should normalize missing fields to ``None``.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """
    csv_path = tmp_path / "messages.csv"
    csv_path.write_text(
        (
            "UUID,Chat ID,Sender,Timestamp,Text,Tags,URL,Chat Group,Answers Count,"
            "Reply To,Network\n"
            "u1,chat-1,Bob,2026-02-03T11:00:00Z,Message body,tag,https://example.com,"
            "group-1,5,root,Signal\n"
        ),
        encoding="utf-8",
    )

    docs = TableReader().load_data(csv_path)

    assert len(docs) == 1
    assert docs[0].text == "Message body"
    assert docs[0].doc_id == "chat-1"
    assert docs[0].metadata["table"]["style"] == "messages"
    assert docs[0].metadata["reference_metadata"] == {
        "network": "Signal",
        "type": "message",
        "uuid": "u1",
        "timestamp": "2026-02-03T11:00:00Z",
        "author": "Bob",
        "author_id": None,
        "vanity": None,
        "text": "Message body",
        "text_id": "chat-1",
        "parent_text": "root",
        "anchor_text": None,
    }


def test_table_reader_detects_social_media_postings_schema_out_of_order(
    tmp_path: Path,
) -> None:
    """Exact header-set matching should ignore column ordering.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """
    csv_path = tmp_path / "postings.csv"
    csv_path.write_text(
        (
            "Text Content,Posting ID,UUID,URL,Date last updated,Timestamp,Timezone,"
            "Crawled at,Postings Connections,Network Posting ID,Location,Author ID,"
            "Author,Vanity Name,Co-Author,Quoted User,Expected Reactions,"
            "Collected Reactions,Expected Comments,Collected Comments,Network,"
            "Posted in Group,Task,Filename,Tags\n"
            "Posting body,p-1,u1,https://example.com,2026-02-01,2026-02-02T12:00:00Z,"
            "UTC,2026-02-03,3,np1,Berlin,a9,Carol,carol-v,Dan,Eve,5,4,2,1,Facebook,"
            "group-x,task-1,file.csv,tag\n"
        ),
        encoding="utf-8",
    )

    docs = TableReader().load_data(csv_path)

    assert len(docs) == 1
    assert docs[0].text == "Posting body"
    assert docs[0].doc_id == "p-1"
    assert docs[0].metadata["table"]["style"] == "postings"
    assert docs[0].metadata["reference_metadata"]["type"] == "posting"
    assert docs[0].metadata["reference_metadata"]["network"] == "Facebook"
    assert docs[0].metadata["reference_metadata"]["uuid"] == "u1"
    assert docs[0].metadata["reference_metadata"]["parent_text"] is None
    assert docs[0].metadata["reference_metadata"]["anchor_text"] is None


def test_table_reader_preserves_generic_behavior_for_non_matching_schema(
    tmp_path: Path,
) -> None:
    """Tables with near-matching headers should not trigger the specialized profile.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
    """
    csv_path = tmp_path / "near_match.csv"
    csv_path.write_text(
        (
            "UUID,Chat ID,Sender,Timestamp,Text,Tags,URL,Chat Group,Answers Count,"
            "Reply To,Network,Extra Column\n"
            "u1,chat-1,Bob,2026-02-03T11:00:00Z,Message body,tag,https://example.com,"
            "group-1,5,root,Signal,extra\n"
        ),
        encoding="utf-8",
    )

    docs = TableReader(text_cols=["Text"], id_col="Chat ID").load_data(csv_path)

    assert len(docs) == 1
    assert docs[0].text == "Message body"
    assert docs[0].doc_id == "chat-1"
    assert "style" not in docs[0].metadata["table"]
    assert "reference_metadata" not in docs[0].metadata


def test_table_reader_sanitizes_excel_time_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
