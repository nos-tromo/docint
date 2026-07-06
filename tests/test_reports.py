"""Tests for owner-scoped curated reports (ReportManager)."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.core.state.base import Base
from docint.core.state.report_manager import ReportManager


@pytest.fixture
def report_manager() -> Generator[ReportManager, None, None]:
    """ReportManager bound to an in-memory SQLite DB.

    Mirrors the fixture in ``test_session_manager_ownership.py``. The
    ``_SessionMaker`` is injected directly so the manager never touches the
    RAG's real session store.

    Returns:
        Generator[ReportManager, None, None]: The ReportManager instance.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)

    rm = ReportManager(rag=MagicMock())
    rm._SessionMaker = SessionMaker
    yield rm
    engine.dispose()


def _ok(value: dict[str, Any] | None) -> dict[str, Any]:
    """Assert an owner-scoped lookup returned a row and narrow it for the type checker."""
    assert value is not None
    return value


def _entity_item(chunk_id: str, *, text: str = "hello") -> dict[str, Any]:
    """Build an add_item kwargs dict for an entity finding."""
    return {
        "artifact_type": "entity_finding",
        "dedupe_key": f"entity:{chunk_id}",
        "snapshot": {"chunk_id": chunk_id, "chunk_text": text, "entity_label": "Acme [ORG]"},
    }


def test_create_and_get_round_trip(report_manager: ReportManager) -> None:
    """A created report is retrievable with an empty item list."""
    created = report_manager.create_report(title="Case A", owner="alice", collection_name="docs")
    assert created["title"] == "Case A"
    assert created["collection_name"] == "docs"
    assert created["items"] == []
    assert created["item_count"] == 0

    fetched = _ok(report_manager.get_report(created["id"], "alice"))
    assert fetched["id"] == created["id"]


def test_create_defaults_blank_title(report_manager: ReportManager) -> None:
    """An empty title falls back to a sensible default."""
    created = report_manager.create_report(title="", owner="alice")
    assert created["title"] == "Untitled report"


def test_create_and_update_case_metadata(report_manager: ReportManager) -> None:
    """Operator and reference number persist on create and patch independently."""
    created = report_manager.create_report(title="A", owner="alice", operator="Jane", reference_number="AZ-1")
    assert created["operator"] == "Jane"
    assert created["reference_number"] == "AZ-1"

    updated = _ok(report_manager.update_report(created["id"], "alice", operator="John"))
    assert updated["operator"] == "John"
    # A field not included in the patch is preserved.
    assert updated["reference_number"] == "AZ-1"


def test_show_toc_defaults_on_and_can_be_disabled(report_manager: ReportManager) -> None:
    """The table-of-contents flag defaults on, persists when toggled, and survives a fetch."""
    created = report_manager.create_report(title="A", owner="alice")
    assert created["show_toc"] is True

    updated = _ok(report_manager.update_report(created["id"], "alice", show_toc=False))
    assert updated["show_toc"] is False
    assert _ok(report_manager.get_report(created["id"], "alice"))["show_toc"] is False


def test_list_reports_is_owner_scoped(report_manager: ReportManager) -> None:
    """list_reports only returns reports owned by the caller."""
    report_manager.create_report(title="A", owner="alice")
    report_manager.create_report(title="B", owner="bob")

    alice = report_manager.list_reports("alice")
    bob = report_manager.list_reports("bob")

    assert {r["title"] for r in alice} == {"A"}
    assert {r["title"] for r in bob} == {"B"}


def test_list_reports_collection_filter(report_manager: ReportManager) -> None:
    """list_reports filters by collection when requested."""
    report_manager.create_report(title="A", owner="alice", collection_name="docs")
    report_manager.create_report(title="B", owner="alice", collection_name="other")

    only_docs = report_manager.list_reports("alice", collection="docs")
    assert {r["title"] for r in only_docs} == {"A"}


def test_get_update_delete_cross_owner_is_not_found(report_manager: ReportManager) -> None:
    """A report owned by A is invisible/immutable to B."""
    created = report_manager.create_report(title="A", owner="alice")
    rid = created["id"]

    assert report_manager.get_report(rid, "bob") is None
    assert report_manager.update_report(rid, "bob", title="hacked") is None
    assert report_manager.delete_report(rid, "bob") is False
    # Still intact for the owner.
    assert _ok(report_manager.get_report(rid, "alice"))["title"] == "A"


def test_update_and_delete_report(report_manager: ReportManager) -> None:
    """The owner can rename and delete a report (cascading its items)."""
    created = report_manager.create_report(title="A", owner="alice")
    rid = created["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))

    renamed = _ok(report_manager.update_report(rid, "alice", title="Renamed"))
    assert renamed["title"] == "Renamed"

    assert report_manager.delete_report(rid, "alice") is True
    assert report_manager.get_report(rid, "alice") is None


def test_add_item_appends_and_increments_position(report_manager: ReportManager) -> None:
    """Items are appended with monotonically increasing positions."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    first = _ok(report_manager.add_item(rid, "alice", **_entity_item("c1")))
    second = _ok(report_manager.add_item(rid, "alice", **_entity_item("c2")))
    assert first["position"] == 0
    assert second["position"] == 1

    report = _ok(report_manager.get_report(rid, "alice"))
    assert [i["dedupe_key"] for i in report["items"]] == ["entity:c1", "entity:c2"]


def test_add_item_is_idempotent_by_dedupe_key(report_manager: ReportManager) -> None:
    """Re-adding the same dedupe_key is a no-op returning the existing item."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    first = _ok(report_manager.add_item(rid, "alice", **_entity_item("c1")))
    again = _ok(report_manager.add_item(rid, "alice", **_entity_item("c1", text="changed")))

    assert again["id"] == first["id"]
    report = _ok(report_manager.get_report(rid, "alice"))
    assert len(report["items"]) == 1
    # The original snapshot is preserved (the duplicate add did not overwrite).
    assert report["items"][0]["snapshot"]["chunk_text"] == "hello"


def test_type_prefixed_dedupe_allows_same_chunk_across_types(report_manager: ReportManager) -> None:
    """The same chunk_id can be both an entity finding and a hate-speech finding."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))
    report_manager.add_item(
        rid,
        "alice",
        artifact_type="hate_speech_finding",
        dedupe_key="hate:c1",
        snapshot={"chunk_id": "c1", "category": "slur", "confidence": "high"},
    )

    report = _ok(report_manager.get_report(rid, "alice"))
    assert len(report["items"]) == 2
    assert {i["dedupe_key"] for i in report["items"]} == {"entity:c1", "hate:c1"}


def test_snapshot_is_frozen_at_add_time(report_manager: ReportManager) -> None:
    """Mutating the source dict after add does not change the stored snapshot."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    snap = {"chunk_id": "c1", "chunk_text": "original"}
    report_manager.add_item(rid, "alice", artifact_type="entity_finding", dedupe_key="entity:c1", snapshot=snap)
    snap["chunk_text"] = "mutated-after-add"

    report = _ok(report_manager.get_report(rid, "alice"))
    assert report["items"][0]["snapshot"]["chunk_text"] == "original"


def test_add_item_cross_owner_is_not_found(report_manager: ReportManager) -> None:
    """B cannot add items to A's report."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    assert report_manager.add_item(rid, "bob", **_entity_item("c1")) is None


def test_remove_item(report_manager: ReportManager) -> None:
    """Owner can remove an item; cross-owner and wrong-report removals fail."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    item = _ok(report_manager.add_item(rid, "alice", **_entity_item("c1")))

    assert report_manager.remove_item(rid, "bob", item["id"]) is False
    assert report_manager.remove_item(rid, "alice", item["id"]) is True
    assert _ok(report_manager.get_report(rid, "alice"))["items"] == []


def test_remove_item_wrong_report(report_manager: ReportManager) -> None:
    """An item id from another report cannot be removed via this report."""
    rid_a = report_manager.create_report(title="A", owner="alice")["id"]
    rid_b = report_manager.create_report(title="B", owner="alice")["id"]
    item = _ok(report_manager.add_item(rid_a, "alice", **_entity_item("c1")))
    assert report_manager.remove_item(rid_b, "alice", item["id"]) is False


def test_annotate_item(report_manager: ReportManager) -> None:
    """A note can be set and cleared on an item."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    item = _ok(report_manager.add_item(rid, "alice", **_entity_item("c1")))

    annotated = _ok(report_manager.annotate_item(rid, "alice", item["id"], note="key evidence"))
    assert annotated["note"] == "key evidence"
    cleared = _ok(report_manager.annotate_item(rid, "alice", item["id"], note=None))
    assert cleared["note"] is None


def test_reorder_items(report_manager: ReportManager) -> None:
    """Reordering reassigns positions to match the requested order."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    a = _ok(report_manager.add_item(rid, "alice", **_entity_item("c1")))
    b = _ok(report_manager.add_item(rid, "alice", **_entity_item("c2")))
    c = _ok(report_manager.add_item(rid, "alice", **_entity_item("c3")))

    report_manager.reorder_items(rid, "alice", [c["id"], a["id"], b["id"]])

    report = _ok(report_manager.get_report(rid, "alice"))
    assert [i["id"] for i in report["items"]] == [c["id"], a["id"], b["id"]]
    assert [i["position"] for i in report["items"]] == [0, 1, 2]


def test_reorder_items_partial_keeps_remainder(report_manager: ReportManager) -> None:
    """Items omitted from the reorder list keep their relative order after."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    a = _ok(report_manager.add_item(rid, "alice", **_entity_item("c1")))
    b = _ok(report_manager.add_item(rid, "alice", **_entity_item("c2")))
    c = _ok(report_manager.add_item(rid, "alice", **_entity_item("c3")))

    # Only mention c first; a and b should follow in their original order.
    report_manager.reorder_items(rid, "alice", [c["id"]])

    report = _ok(report_manager.get_report(rid, "alice"))
    assert [i["id"] for i in report["items"]] == [c["id"], a["id"], b["id"]]


def test_list_dedupe_keys_is_owner_scoped(report_manager: ReportManager) -> None:
    """list_dedupe_keys returns keys for the owner and [] cross-owner."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))
    report_manager.add_item(rid, "alice", **_entity_item("c2"))

    assert set(report_manager.list_dedupe_keys(rid, "alice")) == {"entity:c1", "entity:c2"}
    assert report_manager.list_dedupe_keys(rid, "bob") == []


def test_report_model_has_collection_overview_columns() -> None:
    """The Report ORM exposes the two document-overview columns."""
    from docint.core.state.report import Report

    assert "show_collection_overview" in Report.__table__.columns
    assert "collection_overview_snapshot" in Report.__table__.columns


def test_migration_backfills_overview_columns_on_legacy_reports_table() -> None:
    """`_ensure_report_columns` adds the new columns to a pre-existing table."""
    from sqlalchemy import create_engine, inspect, text

    from docint.core.state.base import _ensure_report_columns

    engine = create_engine("sqlite://")  # in-memory
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE reports (id INTEGER PRIMARY KEY, title TEXT)"))
    _ensure_report_columns(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("reports")}
    assert {"show_collection_overview", "collection_overview_snapshot"} <= cols


def test_new_report_defaults_collection_overview_on(report_manager: ReportManager) -> None:
    """A freshly created report opts into the document overview by default."""
    created = report_manager.create_report(title="Case 1", owner="alice")
    assert created["show_collection_overview"] is True
    assert created["collection_overview"] is None


def test_toggle_and_snapshot_roundtrip(report_manager: ReportManager) -> None:
    """The overview toggle and snapshot setter round-trip and are owner-gated."""
    r = report_manager.create_report(title="C", owner="alice")
    off = _ok(report_manager.update_report(r["id"], "alice", show_collection_overview=False))
    assert off["show_collection_overview"] is False

    snap: dict[str, Any] = {"collection": "c", "documents": [{"filename": "a.pdf"}], "document_count": 1}
    stored = _ok(report_manager.set_collection_overview_snapshot(r["id"], "alice", snap))
    assert stored["collection_overview"]["document_count"] == 1

    # cross-owner is a no-op miss (returns None)
    assert report_manager.set_collection_overview_snapshot(r["id"], "mallory", snap) is None
    # clearing sets it back to None
    cleared = _ok(report_manager.set_collection_overview_snapshot(r["id"], "alice", None))
    assert cleared["collection_overview"] is None
