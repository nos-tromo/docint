"""Unit tests for the search field whitelist and its payload indexes."""

from __future__ import annotations

import types
from typing import Any

import pytest
from qdrant_client import models

from docint.core.search.fields import (
    DEFAULT_SEARCH_FIELD,
    IMAGE_LANE_FIELDS,
    SEARCH_FIELDS,
    UnknownSearchFieldError,
    ensure_field_indexes,
    field_index_kind,
    search_payload_key,
)
from docint.core.search.index import SEARCH_TEXT_FIELD, search_index_params


class _FakeClient:
    """Records index calls and serves a configurable payload schema."""

    def __init__(self, schema: dict[str, str] | None = None, *, fail: bool = False) -> None:
        self.fail = fail
        self.schema = dict(schema or {})
        self.created: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    def get_collection(self, collection_name: str) -> Any:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        payload_schema = {key: types.SimpleNamespace(data_type=kind, params=None) for key, kind in self.schema.items()}
        return types.SimpleNamespace(payload_schema=payload_schema)

    def create_payload_index(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.created.append(kwargs)
        self.schema[kwargs["field_name"]] = "text"

    def delete_payload_index(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.deleted.append(kwargs)
        self.schema.pop(kwargs["field_name"], None)


def test_text_is_the_first_and_default_field() -> None:
    """The picker's default is the first whitelist entry, and it is the chunk text."""
    assert DEFAULT_SEARCH_FIELD == "text"
    assert next(iter(SEARCH_FIELDS)) == "text"
    assert SEARCH_FIELDS["text"] == SEARCH_TEXT_FIELD


def test_search_payload_key_maps_short_names_to_payload_paths() -> None:
    """Each short name resolves to its dotted payload path."""
    assert search_payload_key("author") == "reference_metadata.author"
    assert search_payload_key("author_id") == "reference_metadata.author_id"
    assert search_payload_key("file_name") == "file_name"


def test_search_payload_key_rejects_unknown_names() -> None:
    """A raw payload path must not pass through — the whitelist is closed."""
    with pytest.raises(UnknownSearchFieldError):
        search_payload_key("reference_metadata.author")


def test_image_lane_fields_only_name_keys_an_image_point_carries() -> None:
    """The companion's points carry no speaker or author; text and posting fields they do."""
    assert "text" in IMAGE_LANE_FIELDS
    assert "posting_author" in IMAGE_LANE_FIELDS
    assert "speaker" not in IMAGE_LANE_FIELDS
    assert "author" not in IMAGE_LANE_FIELDS
    assert IMAGE_LANE_FIELDS <= set(SEARCH_FIELDS)


def test_field_index_kind_reports_the_schema_type_lowercased() -> None:
    """The kind is the schema's data type as a lowercase string, or None when unindexed."""
    client = _FakeClient({"reference_metadata.author": models.PayloadSchemaType.KEYWORD})
    assert field_index_kind(client, "col", "reference_metadata.author") == "keyword"
    assert field_index_kind(client, "col", "reference_metadata.network") is None


def test_field_index_kind_is_none_when_qdrant_is_unreachable() -> None:
    """An outage reads as unindexed, never as indexed."""
    assert field_index_kind(_FakeClient(fail=True), "col", "file_name") is None


def test_ensure_field_indexes_creates_one_text_index_per_metadata_field() -> None:
    """Every non-text field gets the same prefix/lowercase index search_text has."""
    client = _FakeClient()
    assert ensure_field_indexes(client, "col") is True
    created = {c["field_name"] for c in client.created}
    assert created == {key for name, key in SEARCH_FIELDS.items() if name != "text"}
    assert all(c["field_schema"] == search_index_params() for c in client.created)
    assert all(c["wait"] is True for c in client.created)
    assert client.deleted == []


def test_ensure_field_indexes_replaces_a_keyword_index() -> None:
    """Qdrant holds one index per field: a facet-era KEYWORD index is dropped first."""
    client = _FakeClient({"reference_metadata.author": models.PayloadSchemaType.KEYWORD})
    assert ensure_field_indexes(client, "col") is True
    assert [d["field_name"] for d in client.deleted] == ["reference_metadata.author"]
    assert client.deleted[0]["wait"] is True
    assert "reference_metadata.author" in {c["field_name"] for c in client.created}


def test_ensure_field_indexes_leaves_an_existing_text_index_alone() -> None:
    """Re-running the backport must not rebuild indexes that already exist."""
    client = _FakeClient({key: "text" for name, key in SEARCH_FIELDS.items() if name != "text"})
    assert ensure_field_indexes(client, "col") is True
    assert client.created == []
    assert client.deleted == []


def test_ensure_field_indexes_never_touches_search_text() -> None:
    """The search_text index is owned by index.py; this function must not recreate it."""
    client = _FakeClient()
    ensure_field_indexes(client, "col")
    assert SEARCH_TEXT_FIELD not in {c["field_name"] for c in client.created}


def test_ensure_field_indexes_is_fail_soft() -> None:
    """A Qdrant outage degrades to False rather than raising."""
    assert ensure_field_indexes(_FakeClient(fail=True), "col") is False
