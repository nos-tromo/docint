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
        "timestamp": "2026-01-02T10:00:00Z",
        "author": "Alice",
        "author_id": "a1",
        "vanity": "alice-v",
        "text": "Comment body",
        "text_id": "c1",
    }
    assert docs[0].metadata["graph"]["record_kind"] == "comment"
    assert docs[0].metadata["graph"]["record_id"] == "c1"
    assert docs[0].metadata["graph"]["thread_id"] == "p1"
    assert docs[0].metadata["graph"]["parent_record_id"] == "pc1"
    assert docs[0].metadata["graph"]["domain"] == "example.com"
    assert docs[0].metadata["graph_search_text"]


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
        "timestamp": "2026-02-03T11:00:00Z",
        "author": "Bob",
        "author_id": None,
        "vanity": None,
        "text": "Message body",
        "text_id": "chat-1",
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
    assert docs[0].metadata["graph"]["record_kind"] == "posting"
    assert docs[0].metadata["graph"]["platform"] == "Facebook"
    assert docs[0].metadata["graph"]["tags"] == ["tag"]


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
