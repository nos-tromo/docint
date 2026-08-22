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
    FieldSpec,
    UnknownSearchFieldError,
    ensure_field_indexes,
    field_index_kind,
    field_indexes_ready,
    search_field_spec,
)
from docint.core.search.fulltext import uuid_match_forms, value_match_forms
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
        schema = kwargs["field_schema"]
        self.schema[kwargs["field_name"]] = "keyword" if schema == models.PayloadSchemaType.KEYWORD else "text"

    def delete_payload_index(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.deleted.append(kwargs)
        self.schema.pop(kwargs["field_name"], None)


def _all_expected() -> list[tuple[str, str]]:
    """Return every (payload key, expected index kind) pair outside the text field."""
    pairs: list[tuple[str, str]] = []
    for name, spec in SEARCH_FIELDS.items():
        if name == DEFAULT_SEARCH_FIELD:
            continue
        pairs.extend(spec.indexed_keys())
    return pairs


def test_text_is_the_first_and_default_field() -> None:
    """The picker's default is the first whitelist entry, and it is the chunk text."""
    assert DEFAULT_SEARCH_FIELD == "text"
    assert next(iter(SEARCH_FIELDS)) == "text"
    assert SEARCH_FIELDS["text"].text_keys == (SEARCH_TEXT_FIELD,)


def test_the_picker_offers_exactly_four_options() -> None:
    """Type, speaker, language and file were dropped; ids folded into author; uuid added.

    File went because filtering by filename is a metadata-filter concern the
    chat filters already cover; uuid came because it is the sole identifier of
    a single posting artifact.
    """
    assert list(SEARCH_FIELDS) == ["text", "author", "network", "uuid"]


def test_uuid_matches_by_value_only() -> None:
    """A uuid is an identifier, never prose.

    Exact match on the posting's own uuid and on the posting_uuid every
    derived artifact carries — no prefixing.
    """
    spec = search_field_spec("uuid")
    assert spec.text_keys == ()
    assert spec.value_keys == ("reference_metadata.uuid", "posting_uuid")
    assert spec.value_forms is uuid_match_forms


def test_author_keeps_the_default_identifier_forms() -> None:
    """Only uuid needs dash handling; author ids keep the int/str duality."""
    assert search_field_spec("author").value_forms is value_match_forms


def test_author_covers_name_vanity_and_the_posting_equivalents() -> None:
    """One 'Author' option answers "everything this person posted", however they are named.

    A social artifact (an image, a transcript segment) carries its parent
    posting's author under ``posting_*`` rather than its own, so leaving those
    out would find the post but not the media hanging off it.
    """
    spec = search_field_spec("author")
    assert spec.text_keys == (
        "reference_metadata.author",
        "reference_metadata.vanity",
        "reference_metadata.posting_author",
        "reference_metadata.posting_vanity",
    )


def test_author_matches_ids_by_value_not_by_text() -> None:
    """Author ids are numeric in Qdrant, and MatchText cannot match a number.

    This is the whole reason ids get their own matcher: a text index over an
    integer field indexes zero points, so the id lane has to be an exact
    value match instead.
    """
    spec = search_field_spec("author")
    assert spec.value_keys == (
        "reference_metadata.author_id",
        "reference_metadata.posting_author_id",
    )


def test_search_field_spec_rejects_unknown_names() -> None:
    """A raw payload path must not pass through — the whitelist is closed."""
    with pytest.raises(UnknownSearchFieldError):
        search_field_spec("reference_metadata.author")


def test_retired_field_names_are_no_longer_accepted() -> None:
    """The dropped options must fail closed rather than silently searching nothing."""
    for retired in ("author_id", "posting_author", "type", "speaker", "language", "file_name"):
        with pytest.raises(UnknownSearchFieldError):
            search_field_spec(retired)


def test_indexed_keys_pairs_each_key_with_the_index_its_matcher_needs() -> None:
    """A text key needs a TEXT index; a value key needs a KEYWORD one."""
    assert search_field_spec("author").indexed_keys() == (
        ("reference_metadata.author", "text"),
        ("reference_metadata.vanity", "text"),
        ("reference_metadata.posting_author", "text"),
        ("reference_metadata.posting_vanity", "text"),
        ("reference_metadata.author_id", "keyword"),
        ("reference_metadata.posting_author_id", "keyword"),
    )


def test_image_lane_fields_only_name_fields_an_image_point_carries() -> None:
    """A companion point has a caption, its parent posting's author and its posting_uuid."""
    assert IMAGE_LANE_FIELDS == frozenset({"text", "author", "uuid"})
    assert IMAGE_LANE_FIELDS <= set(SEARCH_FIELDS)


def test_field_index_kind_reports_the_schema_type_lowercased() -> None:
    """The kind is the schema's data type as a lowercase string, or None when unindexed."""
    client = _FakeClient({"reference_metadata.author": models.PayloadSchemaType.TEXT})
    assert field_index_kind(client, "col", "reference_metadata.author") == "text"
    assert field_index_kind(client, "col", "reference_metadata.network") is None


def test_field_index_kind_is_none_when_qdrant_is_unreachable() -> None:
    """An outage reads as unindexed, never as indexed."""
    assert field_index_kind(_FakeClient(fail=True), "col", "reference_metadata.network") is None


def test_ensure_field_indexes_creates_the_matcher_specific_index_per_key() -> None:
    """Text keys get the prefix/lowercase TEXT params; id keys get KEYWORD."""
    client = _FakeClient()
    assert ensure_field_indexes(client, "col") is True
    created = {c["field_name"]: c["field_schema"] for c in client.created}
    assert set(created) == {key for key, _ in _all_expected()}
    assert created["reference_metadata.author"] == search_index_params()
    assert created["reference_metadata.vanity"] == search_index_params()
    assert created["reference_metadata.author_id"] == models.PayloadSchemaType.KEYWORD
    assert created["reference_metadata.posting_author_id"] == models.PayloadSchemaType.KEYWORD
    assert all(c["wait"] is True for c in client.created)
    assert client.deleted == []


def test_ensure_field_indexes_replaces_a_text_index_on_an_id_key() -> None:
    """The TEXT index this feature originally put on author_id indexes zero points.

    It is worse than useless — it made an id search silently return nothing —
    so it must be replaced with the KEYWORD index a value match can use.
    """
    client = _FakeClient({"reference_metadata.author_id": models.PayloadSchemaType.TEXT})
    assert ensure_field_indexes(client, "col") is True
    assert [d["field_name"] for d in client.deleted] == ["reference_metadata.author_id"]
    created = {c["field_name"]: c["field_schema"] for c in client.created}
    assert created["reference_metadata.author_id"] == models.PayloadSchemaType.KEYWORD


def test_ensure_field_indexes_leaves_the_pre_existing_posting_uuid_index_alone() -> None:
    """create_index already KEYWORD-indexes posting_uuid; the uuid field must reuse it.

    Only the posting's own ``reference_metadata.uuid`` is new. Deleting and
    recreating posting_uuid would be a pointless rebuild on every collection.
    """
    client = _FakeClient({"posting_uuid": models.PayloadSchemaType.KEYWORD})
    assert ensure_field_indexes(client, "col") is True
    assert "posting_uuid" not in {d["field_name"] for d in client.deleted}
    assert "posting_uuid" not in {c["field_name"] for c in client.created}
    created = {c["field_name"]: c["field_schema"] for c in client.created}
    assert created["reference_metadata.uuid"] == models.PayloadSchemaType.KEYWORD


def test_ensure_field_indexes_replaces_a_keyword_index_on_a_text_key() -> None:
    """Qdrant holds one index per field, so a facet-era KEYWORD index is dropped first."""
    client = _FakeClient({"reference_metadata.author": models.PayloadSchemaType.KEYWORD})
    assert ensure_field_indexes(client, "col") is True
    assert "reference_metadata.author" in {d["field_name"] for d in client.deleted}
    created = {c["field_name"]: c["field_schema"] for c in client.created}
    assert created["reference_metadata.author"] == search_index_params()


def test_ensure_field_indexes_leaves_correct_indexes_alone() -> None:
    """Re-running the backport must not rebuild indexes that are already right."""
    client = _FakeClient({key: kind for key, kind in _all_expected()})
    assert ensure_field_indexes(client, "col") is True
    assert client.created == []
    assert client.deleted == []


def test_ensure_field_indexes_reads_the_schema_once() -> None:
    """One get_collection per call, not one per key."""
    client = _FakeClient()
    calls = 0
    original = client.get_collection

    def counting(collection_name: str) -> Any:
        nonlocal calls
        calls += 1
        return original(collection_name=collection_name)

    client.get_collection = counting  # type: ignore[method-assign]
    ensure_field_indexes(client, "col")
    assert calls == 1


def test_ensure_field_indexes_never_touches_search_text() -> None:
    """The search_text index is owned by index.py; this function must not recreate it."""
    client = _FakeClient()
    ensure_field_indexes(client, "col")
    assert SEARCH_TEXT_FIELD not in {c["field_name"] for c in client.created}


def test_ensure_field_indexes_is_fail_soft() -> None:
    """A Qdrant outage degrades to False rather than raising."""
    assert ensure_field_indexes(_FakeClient(fail=True), "col") is False


def test_field_indexes_ready_requires_every_key_to_carry_its_own_kind() -> None:
    """A field is searchable only when all of its keys are indexed the right way."""
    client = _FakeClient({key: kind for key, kind in _all_expected()})
    assert field_indexes_ready(client, "col", "author") is True

    partial = _FakeClient({key: kind for key, kind in _all_expected() if key != "reference_metadata.vanity"})
    assert field_indexes_ready(partial, "col", "author") is False


def test_field_indexes_ready_rejects_an_id_key_indexed_as_text() -> None:
    """The exact failure that shipped: a TEXT index on a numeric key matches nothing."""
    schema = {key: kind for key, kind in _all_expected()}
    schema["reference_metadata.author_id"] = "text"
    assert field_indexes_ready(_FakeClient(schema), "col", "author") is False


def test_field_indexes_ready_is_trivially_true_for_the_text_field() -> None:
    """`search_text` has its own index, checked separately by search_index_status."""
    assert field_indexes_ready(_FakeClient(), "col", "text") is True


def test_field_spec_is_hashable_and_frozen() -> None:
    """The whitelist is module-level constant data and must not be mutable."""
    spec = FieldSpec(text_keys=("a",), value_keys=("b",))
    with pytest.raises(AttributeError):
        spec.text_keys = ("c",)  # type: ignore[misc]
