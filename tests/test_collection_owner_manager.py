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
from docint.core.state.collection_owner_manager import CollectionOwnerManager


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
