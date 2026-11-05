"""Unit tests for CollectionOwnerManager: per-user collection ownership + namespacing.

The manager is the source of truth for the ``(owner, logical) -> physical``
mapping that makes each user's Qdrant collections their own. Mirrors the
owner-scoped posture of :class:`ReportManager` (cross-owner access is "not
found", never an error).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from docint.core.state.base import Base
from docint.core.state.collection_owner_manager import (
    CollectionOwnerManager,
    InvalidCollectionNameError,
    physical_collection_name,
)
from docint.core.state.collection_ownership import CollectionOwnership

T0 = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


class _Stub:
    """Minimal RAG stand-in exposing only the session-store URL."""

    session_store = "sqlite://"


@pytest.fixture
def mgr() -> CollectionOwnerManager:
    """A manager backed by a shared in-memory SQLite DB (StaticPool)."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    m = CollectionOwnerManager(rag=cast(Any, _Stub()))
    m._SessionMaker = sessionmaker(bind=engine)
    return m


def test_register_is_idempotent_and_returns_stable_physical(mgr: CollectionOwnerManager) -> None:
    """Re-registering the same (owner, logical) returns the same physical, no duplicate row."""
    p1 = mgr.register("alice", "mydocs")
    p2 = mgr.register("alice", "mydocs")
    assert p1 == p2
    assert mgr.list_for("alice") == ["mydocs"]


def test_same_logical_distinct_owners_distinct_physical(mgr: CollectionOwnerManager) -> None:
    """Alice's 'mydocs' and Bob's 'mydocs' map to different physical collections."""
    pa = mgr.register("alice", "mydocs")
    pb = mgr.register("bob", "mydocs")
    assert pa != pb
    assert mgr.resolve("alice", "mydocs") == pa
    assert mgr.resolve("bob", "mydocs") == pb


def test_resolve_unowned_returns_none(mgr: CollectionOwnerManager) -> None:
    """Resolve is the access gate: a non-owner (or unknown name) gets None."""
    mgr.register("alice", "mydocs")
    assert mgr.resolve("bob", "mydocs") is None
    assert mgr.resolve("alice", "missing") is None


def test_list_for_is_scoped_and_sorted(mgr: CollectionOwnerManager) -> None:
    """list_for returns only the owner's logical names, sorted."""
    mgr.register("alice", "b")
    mgr.register("alice", "a")
    mgr.register("bob", "c")
    assert mgr.list_for("alice") == ["a", "b"]
    assert mgr.list_for("bob") == ["c"]
    assert mgr.list_for("carol") == []


def test_delete_removes_mapping_and_returns_physical(mgr: CollectionOwnerManager) -> None:
    """Delete returns the physical name (so the caller can drop the Qdrant collection)."""
    p = mgr.register("alice", "mydocs")
    assert mgr.delete("alice", "mydocs") == p
    assert mgr.resolve("alice", "mydocs") is None
    assert mgr.list_for("alice") == []
    assert mgr.delete("alice", "mydocs") is None


def test_delete_is_owner_scoped(mgr: CollectionOwnerManager) -> None:
    """A non-owner cannot delete someone else's mapping."""
    mgr.register("alice", "mydocs")
    assert mgr.delete("bob", "mydocs") is None
    assert mgr.resolve("alice", "mydocs") is not None


def test_backfill_legacy_assigns_bare_names_to_default_owner(mgr: CollectionOwnerManager) -> None:
    """Pre-existing (ownerless) collections become owned by the default identity, keeping their bare name."""
    mgr.backfill_legacy(["legacyA", "legacyB"], default_owner="operator")
    assert mgr.list_for("operator") == ["legacyA", "legacyB"]
    # No Qdrant rename: the legacy physical name equals the logical name.
    assert mgr.resolve("operator", "legacyA") == "legacyA"


def test_backfill_legacy_is_idempotent_and_preserves_existing(mgr: CollectionOwnerManager) -> None:
    """Backfill never clobbers an already-registered mapping and is safe to re-run."""
    mgr.register("alice", "mydocs")
    physical = mgr.resolve("alice", "mydocs")
    assert physical is not None
    mgr.backfill_legacy([physical, "legacyA"], default_owner="operator")
    mgr.backfill_legacy([physical, "legacyA"], default_owner="operator")
    assert mgr.resolve("alice", "mydocs") == physical
    assert mgr.list_for("operator") == ["legacyA"]


def test_list_all_returns_every_owner_sorted(mgr: CollectionOwnerManager) -> None:
    """list_all exposes the whole table as (owner, logical), owner-then-name sorted."""
    mgr.register("bob", "zeta")
    mgr.register("alice", "beta")
    mgr.register("alice", "alpha")
    mgr.backfill_legacy(["legacy-docs"], "operator")

    assert mgr.list_all() == [
        ("alice", "alpha"),
        ("alice", "beta"),
        ("bob", "zeta"),
        ("operator", "legacy-docs"),
    ]


@pytest.mark.parametrize(
    ("logical", "offending"),
    [
        ("2026-05-03 amanahMaz [media] - Test #549", ["#"]),
        ("a/b", ["/"]),
        ("x?y", ["?"]),
        ("50%", ["%"]),
        ("back\\slash", ["\\"]),
        ("tab\there", ["\t"]),
        ("a#b/c", ["#", "/"]),
    ],
)
def test_a_name_that_breaks_the_qdrant_url_is_refused(
    mgr: CollectionOwnerManager, logical: str, offending: list[str]
) -> None:
    """qdrant-client formats the name into the URL path unencoded, so these would address another collection."""
    with pytest.raises(InvalidCollectionNameError) as excinfo:
        physical_collection_name("alice", logical)
    assert excinfo.value.offending == offending
    assert "cannot be used" in str(excinfo.value)
    with pytest.raises(InvalidCollectionNameError):
        mgr.register("alice", logical)
    assert mgr.list_for("alice") == []


@pytest.mark.parametrize("logical", ["", "   "])
def test_a_blank_name_is_refused(logical: str) -> None:
    """A blank logical name would mint a physical name that is only the owner prefix."""
    with pytest.raises(InvalidCollectionNameError, match="must not be empty"):
        physical_collection_name("alice", logical)


@pytest.mark.parametrize("logical", [".", ".."])
def test_a_dot_segment_name_is_refused(mgr: CollectionOwnerManager, logical: str) -> None:
    """Browsers and nginx collapse it out of ``/collections/{name}/…``, so no route could ever reach it."""
    with pytest.raises(InvalidCollectionNameError, match=r"'\.' or '\.\.'"):
        mgr.register("alice", logical)
    assert mgr.list_for("alice") == []


def test_names_that_merely_contain_dots_stay_allowed(mgr: CollectionOwnerManager) -> None:
    """Only a whole ``.`` or ``..`` segment is collapsed; dots inside a name are ordinary."""
    assert mgr.register("alice", "a..b").endswith("__a..b")
    assert mgr.register("alice", ".hidden").endswith("__.hidden")


def test_spaces_brackets_and_unicode_stay_allowed(mgr: CollectionOwnerManager) -> None:
    """Existing collections carry these, and Qdrant addresses them fine."""
    physical = mgr.register("alice", "2026-05-03 amanahMaz [media] - Test 549 ü")
    assert physical.endswith("__2026-05-03 amanahMaz [media] - Test 549 ü")


# --- Retention clock (docs/retention.md) ---


def _stamp(mgr: CollectionOwnerManager, owner: str, logical: str) -> datetime | None:
    """The recorded last activity of one collection."""
    return next(row.last_activity_at for row in mgr.list_activity(owner) if row.logical == logical)


@contextmanager
def _locked_scope(self: CollectionOwnerManager) -> Iterator[Session]:
    """A session scope whose store refuses every write, like a locked SQLite file."""
    raise OperationalError("UPDATE collection_owners", {}, Exception("database is locked"))
    yield  # pragma: no cover


def test_register_starts_the_clock(mgr: CollectionOwnerManager) -> None:
    """A new collection is active the moment it is created."""
    before = datetime.now(UTC) - timedelta(seconds=1)
    physical = mgr.register("alice", "mydocs")

    [row] = mgr.list_activity("alice")

    assert (row.owner, row.logical, row.physical) == ("alice", "mydocs", physical)
    assert row.last_activity_at is not None
    assert row.last_activity_at.tzinfo is UTC
    assert row.last_activity_at >= before


def test_list_activity_is_owner_scoped(mgr: CollectionOwnerManager) -> None:
    """A caller's listing never carries another owner's collections."""
    mgr.register("alice", "a")
    mgr.register("bob", "b")

    assert [row.logical for row in mgr.list_activity("alice")] == ["a"]
    assert [(row.owner, row.logical) for row in mgr.list_all_activity()] == [("alice", "a"), ("bob", "b")]


def test_an_unstamped_collection_lists_without_a_stamp(mgr: CollectionOwnerManager) -> None:
    """``NULL`` survives the read as ``None`` — it is what "never expires" is made of."""
    physical = mgr.register("alice", "mydocs")
    with mgr._session_scope() as s:
        s.query(CollectionOwnership).filter(CollectionOwnership.physical_name == physical).update(
            {CollectionOwnership.last_activity_at: None}
        )
        s.commit()

    assert _stamp(mgr, "alice", "mydocs") is None


def test_get_activity_reads_one_collections_clock(mgr: CollectionOwnerManager) -> None:
    """The sweep reads a collection's owner and logical name by its physical name."""
    physical = mgr.register("alice", "mydocs")

    row = mgr.get_activity(physical)

    assert row is not None
    assert (row.owner, row.logical) == ("alice", "mydocs")
    assert mgr.get_activity("u000000000000__missing") is None


def test_created_at_reads_the_row_as_utc_and_changes_when_recreated(mgr: CollectionOwnerManager) -> None:
    """A recreated collection reads a new creation time; a missing one reads none."""
    physical = mgr.register("alice", "mydocs")
    first = mgr.created_at(physical)

    assert first is not None
    assert first.tzinfo is UTC
    assert mgr.created_at("u000000000000__missing") is None

    mgr.delete("alice", "mydocs")
    assert mgr.register("alice", "mydocs") == physical
    second = mgr.created_at(physical)
    assert second is not None
    assert second > first
