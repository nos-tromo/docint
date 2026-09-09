"""Unit tests for CollectionOwnerManager: per-user collection ownership + namespacing.

The manager is the source of truth for the ``(owner, logical) -> physical``
mapping that makes each user's Qdrant collections their own. Mirrors the
owner-scoped posture of :class:`ReportManager` (cross-owner access is "not
found", never an error).
"""

from typing import Any, cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from docint.core.state.base import Base
from docint.core.state.collection_owner_manager import (
    CollectionOwnerManager,
    InvalidCollectionNameError,
    physical_collection_name,
)


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


def test_spaces_brackets_and_unicode_stay_allowed(mgr: CollectionOwnerManager) -> None:
    """Existing collections carry these, and Qdrant addresses them fine."""
    physical = mgr.register("alice", "2026-05-03 amanahMaz [media] - Test 549 ü")
    assert physical.endswith("__2026-05-03 amanahMaz [media] - Test 549 ü")
