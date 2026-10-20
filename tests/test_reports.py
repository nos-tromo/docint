"""Tests for owner-scoped curated reports (ReportManager)."""

import json
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, delete, event, func, insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from docint.core.state.base import Base, _make_session_maker
from docint.core.state.report import Report
from docint.core.state.report_item import ReportItem
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


def _entity_item(
    chunk_id: str,
    *,
    text: str = "hello",
    translation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an add_item kwargs dict for an entity finding.

    ``translation`` is nested inside the snapshot as the SPA sends it, and
    omitted when None, like an untranslated finding.
    """
    snapshot: dict[str, Any] = {"chunk_id": chunk_id, "chunk_text": text, "entity_label": "Acme [ORG]"}
    if translation is not None:
        snapshot["translation"] = translation
    return {
        "artifact_type": "entity_finding",
        "dedupe_key": f"entity:{chunk_id}",
        "snapshot": snapshot,
    }


def _translation(text: str = "hallo") -> dict[str, Any]:
    """Build a frozen translation payload in the shape the SPA sends."""
    return {"text": text, "target_lang": "de", "model": "test-model"}


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


def test_set_snapshot_empty_dict_is_stored_not_cleared(report_manager: ReportManager) -> None:
    """An empty dict persists as ``{}``; only ``None`` clears the snapshot."""
    r = report_manager.create_report(title="C", owner="alice")
    stored = _ok(report_manager.set_collection_overview_snapshot(r["id"], "alice", {}))
    assert stored["collection_overview"] == {}  # {} persists as an empty dict...
    cleared = _ok(report_manager.set_collection_overview_snapshot(r["id"], "alice", None))
    assert cleared["collection_overview"] is None  # ...only None clears


# ---------------------------------------------------------------------------
# Batch add — "Add all" from an Analysis section
# ---------------------------------------------------------------------------


def test_add_items_appends_all_with_sequential_positions(report_manager: ReportManager) -> None:
    """A batch appends every item, positions continuing from the existing count."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c0"))

    result = report_manager.add_items(rid, "alice", items=[_entity_item("c1"), _entity_item("c2"), _entity_item("c3")])

    assert result is not None
    assert (result["added"], result["skipped"]) == (3, 0)
    report = _ok(report_manager.get_report(rid, "alice"))
    assert [i["dedupe_key"] for i in report["items"]] == ["entity:c0", "entity:c1", "entity:c2", "entity:c3"]
    assert [i["position"] for i in report["items"]] == [0, 1, 2, 3]


def test_add_items_skips_keys_already_present(report_manager: ReportManager) -> None:
    """Items whose dedupe key is already in the report count as skipped, not added."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))

    result = report_manager.add_items(rid, "alice", items=[_entity_item("c1", text="changed"), _entity_item("c2")])

    assert result is not None
    assert (result["added"], result["skipped"]) == (1, 1)
    report = _ok(report_manager.get_report(rid, "alice"))
    assert len(report["items"]) == 2
    # The pre-existing snapshot is untouched by the batch.
    assert report["items"][0]["snapshot"]["chunk_text"] == "hello"


def test_add_items_deduplicates_within_one_batch(report_manager: ReportManager) -> None:
    """A key repeated inside the same batch is stored once."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]

    result = report_manager.add_items(rid, "alice", items=[_entity_item("c1"), _entity_item("c1")])

    assert result is not None
    assert (result["added"], result["skipped"]) == (1, 1)
    assert len(_ok(report_manager.get_report(rid, "alice"))["items"]) == 1


def test_add_items_reports_item_count(report_manager: ReportManager) -> None:
    """The result carries the report's total item count after the batch."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    result = report_manager.add_items(rid, "alice", items=[_entity_item("c1"), _entity_item("c2")])
    assert result is not None
    assert result["item_count"] == 2


def test_add_items_cross_owner_is_not_found(report_manager: ReportManager) -> None:
    """A batch against another owner's report returns None (the 404 seam)."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    assert report_manager.add_items(rid, "bob", items=[_entity_item("c1")]) is None


def test_add_items_empty_batch_is_a_no_op(report_manager: ReportManager) -> None:
    """An empty item list neither fails nor touches the report."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    result = report_manager.add_items(rid, "alice", items=[])
    assert result is not None
    assert (result["added"], result["skipped"], result["item_count"]) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Batch add — translation backfill into snapshots the report already holds
# ---------------------------------------------------------------------------


def test_add_items_backfills_a_translation_into_an_existing_snapshot(report_manager: ReportManager) -> None:
    """A duplicate carrying a translation amends the stored snapshot instead of being inert."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))

    result = report_manager.add_items(rid, "alice", items=[_entity_item("c1", translation=_translation())])

    assert result is not None
    assert (result["added"], result["updated"], result["skipped"]) == (0, 1, 0)
    stored = _ok(report_manager.get_report(rid, "alice"))["items"][0]["snapshot"]
    assert stored["translation"] == _translation()
    # Additive: the rest of the frozen snapshot is exactly as it was added.
    assert stored["chunk_text"] == "hello"
    assert stored["entity_label"] == "Acme [ORG]"


def test_add_items_never_overwrites_a_stored_translation(report_manager: ReportManager) -> None:
    """A snapshot that already carries a translation keeps it; the incoming one is dropped."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1", translation=_translation("erste")))

    result = report_manager.add_items(rid, "alice", items=[_entity_item("c1", translation=_translation("zweite"))])

    assert result is not None
    assert (result["added"], result["updated"], result["skipped"]) == (0, 0, 1)
    stored = _ok(report_manager.get_report(rid, "alice"))["items"][0]["snapshot"]
    assert stored["translation"]["text"] == "erste"


def test_add_items_duplicate_without_translation_is_still_skipped(report_manager: ReportManager) -> None:
    """The plain duplicate path is unchanged — nothing is written back."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))

    result = report_manager.add_items(rid, "alice", items=[_entity_item("c1", text="changed")])

    assert result is not None
    assert (result["added"], result["updated"], result["skipped"]) == (0, 0, 1)
    stored = _ok(report_manager.get_report(rid, "alice"))["items"][0]["snapshot"]
    assert stored["chunk_text"] == "hello"
    assert "translation" not in stored


def test_add_items_counts_add_update_and_skip_to_the_batch_length(report_manager: ReportManager) -> None:
    """Every entry lands in exactly one of the three counts."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))
    report_manager.add_item(rid, "alice", **_entity_item("c2"))

    items = [
        _entity_item("c1", translation=_translation()),  # backfilled
        _entity_item("c2"),  # plain duplicate
        _entity_item("c3", translation=_translation()),  # new
    ]
    result = report_manager.add_items(rid, "alice", items=items)

    assert result is not None
    assert (result["added"], result["updated"], result["skipped"]) == (1, 1, 1)
    assert result["added"] + result["updated"] + result["skipped"] == len(items)


def test_add_items_backfill_repeated_in_one_batch_merges_once(report_manager: ReportManager) -> None:
    """The same key twice in one batch backfills once; the second is a plain skip."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))

    result = report_manager.add_items(
        rid,
        "alice",
        items=[
            _entity_item("c1", translation=_translation("erste")),
            _entity_item("c1", translation=_translation("zweite")),
        ],
    )

    assert result is not None
    assert (result["added"], result["updated"], result["skipped"]) == (0, 1, 1)
    stored = _ok(report_manager.get_report(rid, "alice"))["items"][0]["snapshot"]
    assert stored["translation"]["text"] == "erste"


def test_add_items_backfill_touches_updated_at(report_manager: ReportManager) -> None:
    """An update-only batch bumps the report's timestamp — it did change the report."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))
    before = _ok(report_manager.get_report(rid, "alice"))["updated_at"]

    report_manager.add_items(rid, "alice", items=[_entity_item("c1", translation=_translation())])

    after = _ok(report_manager.get_report(rid, "alice"))["updated_at"]
    assert after >= before


def test_add_items_backfill_ignores_a_non_dict_translation(report_manager: ReportManager) -> None:
    """Only a translation object amends a snapshot; a stray scalar is not a translation."""
    rid = report_manager.create_report(title="A", owner="alice")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))

    item = _entity_item("c1")
    item["snapshot"]["translation"] = "hallo"
    result = report_manager.add_items(rid, "alice", items=[item])

    assert result is not None
    assert (result["added"], result["updated"], result["skipped"]) == (0, 0, 1)
    assert "translation" not in _ok(report_manager.get_report(rid, "alice"))["items"][0]["snapshot"]


def test_deleting_a_collection_deletes_its_reports_and_their_items(report_manager: ReportManager) -> None:
    """The collection's reports go with it: nothing derived from a deleted collection is kept."""
    doomed = report_manager.create_report(title="A", owner="alice", collection_name="docs")["id"]
    report_manager.add_item(doomed, "alice", **_entity_item("c1"))
    report_manager.add_item(doomed, "alice", **_entity_item("c2"))
    other_collection = report_manager.create_report(title="B", owner="alice", collection_name="other")["id"]

    assert report_manager.delete_reports_for_collection("alice", "docs") == [doomed]

    assert report_manager.get_report(doomed, "alice") is None
    assert report_manager.get_report(other_collection, "alice") is not None
    with report_manager._session_scope() as s:
        assert s.query(ReportItem).count() == 0


def test_deleting_a_collection_spares_another_owners_same_named_reports(report_manager: ReportManager) -> None:
    """Report collection names are logical, so the owner is what keeps Bob's ``docs`` apart from Alice's."""
    alice = report_manager.create_report(title="A", owner="alice", collection_name="docs")["id"]
    bob = report_manager.create_report(title="B", owner="bob", collection_name="docs")["id"]

    report_manager.delete_reports_for_collection("alice", "docs")

    assert report_manager.get_report(alice, "alice") is None
    assert report_manager.get_report(bob, "bob") is not None


def test_deleting_a_collection_spares_reports_scoped_to_none(report_manager: ReportManager) -> None:
    """A report started outside any collection belongs to no collection's cascade."""
    unscoped = report_manager.create_report(title="A", owner="alice", collection_name=None)["id"]

    assert report_manager.delete_reports_for_collection("alice", "docs") == []

    assert report_manager.get_report(unscoped, "alice") is not None


def test_report_meta_names_an_owned_report_without_its_items(report_manager: ReportManager) -> None:
    """The light lookup keeps get_report's owner gate and formats its timestamps the same way."""
    rid = report_manager.create_report(title="Case Alpha", owner="alice", collection_name="docs")["id"]
    report_manager.add_item(rid, "alice", **_entity_item("c1"))
    full = _ok(report_manager.get_report(rid, "alice"))

    assert report_manager.get_report_meta(rid, "alice") == {
        "id": rid,
        "title": "Case Alpha",
        "collection_name": "docs",
        "created_at": full["created_at"],
        "updated_at": full["updated_at"],
    }
    assert report_manager.get_report_meta(rid, "bob") is None
    assert report_manager.get_report_meta(rid + 1, "alice") is None


def test_report_meta_moves_with_every_change(report_manager: ReportManager) -> None:
    """A rendered PDF is current only while updated_at stands still, so every edit must move it."""
    rid = report_manager.create_report(title="A", owner="alice", collection_name="docs")["id"]
    stamps = [_ok(report_manager.get_report_meta(rid, "alice"))["updated_at"]]
    for change in (
        lambda: report_manager.add_item(rid, "alice", **_entity_item("c1")),
        lambda: report_manager.update_report(rid, "alice", title="B"),
    ):
        time.sleep(0.002)
        change()
        stamps.append(_ok(report_manager.get_report_meta(rid, "alice"))["updated_at"])

    assert len(set(stamps)) == 3


@contextmanager
def _on_first_commit(report_manager: ReportManager, change: Callable[[Session], None]) -> Iterator[list[bool]]:
    """Apply ``change`` through a second session right after the next commit, once.

    That moment is where another request can slip in before a write reads its
    record back; the hook makes the interleaving deterministic.
    """
    maker = report_manager._SessionMaker
    assert maker is not None
    fired: list[bool] = []

    def hook(_session: Session) -> None:
        if fired:
            return
        fired.append(True)
        with maker() as other:
            change(other)
            other.commit()

    event.listen(maker, "after_commit", hook)
    try:
        yield fired
    finally:
        event.remove(maker, "after_commit", hook)


def _bobs_item(report_id: int, item_id: int | None = None) -> ReportItem:
    """A finding in one of Bob's reports."""
    snapshot = json.dumps({"chunk_id": "bob-chunk", "chunk_text": "Bob's evidence"})
    return ReportItem(
        id=item_id,
        report_id=report_id,
        artifact_type="entity_finding",
        dedupe_key="entity:bob-chunk",
        snapshot=snapshot,
        position=0,
    )


def _give_the_newest_report_to_bob(other: Session) -> None:
    """Delete the newest report and let one of Bob's take its id, as SQLite does."""
    newest = other.query(func.max(Report.id)).scalar()
    other.query(ReportItem).filter(ReportItem.report_id == newest).delete()
    other.query(Report).filter(Report.id == newest).delete()
    other.add(Report(id=newest, title="Bob's case", owner="bob", collection_name="bobs"))
    other.add(_bobs_item(newest))


def _give_the_newest_item_to(bob_report_id: int) -> Callable[[Session], None]:
    """Delete the newest item and let a finding in Bob's report take its id."""

    def change(other: Session) -> None:
        newest = other.query(func.max(ReportItem.id)).scalar()
        other.query(ReportItem).filter(ReportItem.id == newest).delete()
        other.add(_bobs_item(bob_report_id, item_id=newest))

    return change


@pytest.mark.parametrize(
    ("write", "unassigned", "items"),
    [
        pytest.param(
            lambda rm, rid: rm.create_report(title="Alice's case", owner="alice", collection_name="docs"),
            False,
            [],
            id="create_report",
        ),
        pytest.param(
            lambda rm, rid: rm.update_report(rid, "alice", title="Alice's case"),
            False,
            ["entity:alice-chunk"],
            id="update_report",
        ),
        pytest.param(
            lambda rm, rid: rm.set_collection_overview_snapshot(rid, "alice", {"document_count": 1}),
            False,
            ["entity:alice-chunk"],
            id="set_collection_overview_snapshot",
        ),
        pytest.param(
            lambda rm, rid: rm.reorder_items(rid, "alice", []),
            False,
            ["entity:alice-chunk"],
            id="reorder_items",
        ),
        pytest.param(
            lambda rm, rid: rm.assign_collection(rid, "alice", "docs"),
            True,
            ["entity:alice-chunk"],
            id="assign_collection",
        ),
    ],
)
def test_a_report_write_never_returns_another_owners_report(
    report_manager: ReportManager,
    write: Callable[[ReportManager, int], dict[str, Any] | None],
    unassigned: bool,
    items: list[str],
) -> None:
    """A write reads its report back before it commits, so an id reused right after cannot come back.

    SQLite hands a deleted highest id to the next insert. Read back after the
    commit, the report could be deleted and another owner's could take its id
    in between, and that owner's report and evidence would be returned.
    """
    collection = None if unassigned else "docs"
    rid = report_manager.create_report(title="Alice's case", owner="alice", collection_name=collection)["id"]
    report_manager.add_item(rid, "alice", **_entity_item("alice-chunk"))

    with _on_first_commit(report_manager, _give_the_newest_report_to_bob) as fired:
        written = _ok(write(report_manager, rid))

    assert fired
    assert (written["title"], written["collection_name"]) == ("Alice's case", "docs")
    assert [i["dedupe_key"] for i in written["items"]] == items


@pytest.mark.parametrize(
    ("write", "dedupe_key"),
    [
        pytest.param(
            lambda rm, rid, iid: rm.add_item(rid, "alice", **_entity_item("alice-new")),
            "entity:alice-new",
            id="add_item",
        ),
        pytest.param(
            lambda rm, rid, iid: rm.annotate_item(rid, "alice", iid, note="checked"),
            "entity:alice-chunk",
            id="annotate_item",
        ),
    ],
)
def test_an_item_write_never_returns_another_owners_item(
    report_manager: ReportManager,
    write: Callable[[ReportManager, int, int], dict[str, Any] | None],
    dedupe_key: str,
) -> None:
    """An item is read back before its write commits, so a reused item id cannot return another owner's finding."""
    bob_rid = report_manager.create_report(title="Bob's case", owner="bob", collection_name="bobs")["id"]
    rid = report_manager.create_report(title="Alice's case", owner="alice", collection_name="docs")["id"]
    iid = _ok(report_manager.add_item(rid, "alice", **_entity_item("alice-chunk")))["id"]

    with _on_first_commit(report_manager, _give_the_newest_item_to(bob_rid)) as fired:
        written = _ok(write(report_manager, rid, iid))

    assert fired
    assert written["dedupe_key"] == dedupe_key
    assert written["snapshot"]["chunk_id"] != "bob-chunk"


@pytest.fixture
def stored_report_manager(tmp_path: Path) -> Iterator[tuple[ReportManager, str]]:
    """A ReportManager over a file-backed store, built as production builds it.

    Its sessions get their own connections, so another request can be played
    against a write on a second connection, as two requests would be.
    """
    url = f"sqlite:///{tmp_path / 'sessions.sqlite3'}"
    rm = ReportManager(rag=MagicMock())
    rm._SessionMaker = _make_session_maker(url)
    yield rm, url
    rm._SessionMaker.kw["bind"].dispose()


@contextmanager
def _another_request_before_the_write(
    report_manager: ReportManager, url: str, rid: int, iid: int
) -> Iterator[list[str]]:
    """Just before a write's first change, after its ownership check, let another request give the ids to Bob.

    The other request deletes the report and stores one of Bob's under the same
    report and item ids, as SQLite's id reuse allows. It runs on its own
    connection with a short busy timeout, so it either gets in (``interleaved``)
    or finds the write lock already taken (``blocked``).
    """
    maker = report_manager._SessionMaker
    assert maker is not None
    other = create_engine(url, connect_args={"timeout": 0.05})
    outcome: list[str] = []

    def hook(_conn: Any, _cursor: Any, statement: str, *_rest: Any) -> None:
        if outcome or not statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            return
        try:
            with other.begin() as conn:
                conn.execute(delete(ReportItem).where(ReportItem.report_id == rid))
                conn.execute(delete(Report).where(Report.id == rid))
                conn.execute(insert(Report).values(id=rid, title="Bob's case", owner="bob", collection_name="docs"))
                conn.execute(
                    insert(ReportItem).values(
                        id=iid,
                        report_id=rid,
                        artifact_type="entity_finding",
                        dedupe_key="entity:bob-chunk",
                        snapshot=json.dumps({"chunk_id": "bob-chunk"}),
                        position=0,
                    )
                )
            outcome.append("interleaved")
        except OperationalError:
            outcome.append("blocked")

    engine = maker.kw["bind"]
    event.listen(engine, "before_cursor_execute", hook)
    try:
        yield outcome
    finally:
        event.remove(engine, "before_cursor_execute", hook)
        other.dispose()


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda rm, rid, iid: rm.update_report(rid, "alice", title="Renamed"), id="update_report"),
        pytest.param(
            lambda rm, rid, iid: rm.set_collection_overview_snapshot(rid, "alice", {"document_count": 1}),
            id="set_collection_overview_snapshot",
        ),
        pytest.param(lambda rm, rid, iid: rm.reorder_items(rid, "alice", [iid]), id="reorder_items"),
        pytest.param(lambda rm, rid, iid: rm.delete_report(rid, "alice"), id="delete_report"),
        pytest.param(
            lambda rm, rid, iid: rm.delete_reports_for_collection("alice", "docs"), id="delete_reports_for_collection"
        ),
        pytest.param(lambda rm, rid, iid: rm.add_item(rid, "alice", **_entity_item("alice-new")), id="add_item"),
        pytest.param(
            lambda rm, rid, iid: rm.add_items(rid, "alice", items=[_entity_item("alice-new")]), id="add_items"
        ),
        pytest.param(lambda rm, rid, iid: rm.annotate_item(rid, "alice", iid, note="checked"), id="annotate_item"),
        pytest.param(lambda rm, rid, iid: rm.remove_item(rid, "alice", iid), id="remove_item"),
    ],
)
def test_a_write_holds_the_lock_from_its_ownership_check(
    stored_report_manager: tuple[ReportManager, str], write: Callable[[ReportManager, int, int], Any]
) -> None:
    """No other request can change a report between a write's ownership check and the write itself.

    If the check took no lock, the report could be deleted in that gap and one
    of Bob's take its id, and the write would land on Bob's report: Codex
    reproduced Alice renaming it and receiving its overview.
    """
    rm, url = stored_report_manager
    rid = rm.create_report(title="Alice's case", owner="alice", collection_name="docs")["id"]
    iid = _ok(rm.add_item(rid, "alice", **_entity_item("alice-chunk")))["id"]

    with _another_request_before_the_write(rm, url, rid, iid) as outcome:
        write(rm, rid, iid)

    assert outcome == ["blocked"]
    assert rm.get_report_meta(rid, "bob") is None


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(
            lambda rm, rid, iid: rm.update_report(rid, "alice", title="Renamed", collection="other"),
            id="update_report",
        ),
        pytest.param(
            lambda rm, rid, iid: rm.set_collection_overview_snapshot(
                rid, "alice", {"document_count": 1}, collection="other"
            ),
            id="set_collection_overview_snapshot",
        ),
        pytest.param(
            lambda rm, rid, iid: rm.reorder_items(rid, "alice", [iid], collection="other"), id="reorder_items"
        ),
        pytest.param(lambda rm, rid, iid: rm.delete_report(rid, "alice", collection="other"), id="delete_report"),
        pytest.param(
            lambda rm, rid, iid: rm.add_item(rid, "alice", collection="other", **_entity_item("alice-new")),
            id="add_item",
        ),
        pytest.param(
            lambda rm, rid, iid: rm.add_items(rid, "alice", items=[_entity_item("alice-new")], collection="other"),
            id="add_items",
        ),
        pytest.param(
            lambda rm, rid, iid: rm.annotate_item(rid, "alice", iid, note="checked", collection="other"),
            id="annotate_item",
        ),
        pytest.param(lambda rm, rid, iid: rm.remove_item(rid, "alice", iid, collection="other"), id="remove_item"),
    ],
)
def test_a_write_refuses_a_report_from_another_collection(
    report_manager: ReportManager, write: Callable[[ReportManager, int, int], Any]
) -> None:
    """Inside the lock a report counts as not found unless it is in the request's collection too.

    The route checks the collection first, in its own transaction. Between that
    check and the write, the report's id could pass to another of the caller's
    reports, in another collection.
    """
    rid = report_manager.create_report(title="Alice's case", owner="alice", collection_name="docs")["id"]
    iid = _ok(report_manager.add_item(rid, "alice", **_entity_item("alice-chunk")))["id"]
    before = report_manager.get_report(rid, "alice")

    assert not write(report_manager, rid, iid)
    assert report_manager.get_report(rid, "alice") == before
