"""Tests for the Qdrant-backed EntityStore.

These run against a real in-memory ``QdrantClient(":memory:")`` rather than a
mock, so they exercise the actual vector search, payload filter, scroll and
``set_payload`` behaviour the production store relies on.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from docint.core.entities.store import EntityStore


@pytest.fixture
def store() -> EntityStore:
    """Build an EntityStore over a fresh in-memory Qdrant collection.

    Returns:
        EntityStore: A store whose collection has been ensured.
    """
    client = QdrantClient(location=":memory:")
    store = EntityStore(client, collection="docs_entities", dim=2, embed_model="test-embed")
    store.ensure_collection()
    return store


def test_ensure_collection_is_idempotent(store: EntityStore) -> None:
    """Calling ensure_collection twice does not raise."""
    store.ensure_collection()
    assert store.client.collection_exists("docs_entities")


def test_mint_then_cluster_finds_near_vector(store: EntityStore) -> None:
    """A minted entity is found when clustering a near-identical vector."""
    entity_id = store.mint_entity("Africa", [1.0, 0.0], "loc")
    hits = store.cluster_candidates([0.99, 0.02], threshold=0.86, k=5, entity_type="loc")
    assert [h["id"] for h in hits] == [entity_id]
    assert hits[0]["canonical_name"] == "Africa"
    assert hits[0]["score"] >= 0.86


def test_cluster_excludes_below_threshold(store: EntityStore) -> None:
    """An orthogonal vector yields no candidates."""
    store.mint_entity("Africa", [1.0, 0.0], "loc")
    assert store.cluster_candidates([0.0, 1.0], threshold=0.86, k=5, entity_type="loc") == []


def test_cluster_is_type_blocked(store: EntityStore) -> None:
    """A candidate of another type is never returned."""
    store.mint_entity("Africa", [1.0, 0.0], "loc")
    assert store.cluster_candidates([1.0, 0.0], threshold=0.86, k=5, entity_type="person") == []


def test_lookup_alias_exact_surface_and_type(store: EntityStore) -> None:
    """lookup_alias matches the exact stored surface within its type block."""
    entity_id = store.mint_entity("Africa", [1.0, 0.0], "loc")
    assert store.lookup_alias("Africa", "loc") == entity_id
    assert store.lookup_alias("africa", "loc") is None
    assert store.lookup_alias("Africa", "person") is None


def test_attach_alias_makes_surface_resolvable(store: EntityStore) -> None:
    """An attached surface becomes findable via lookup_alias."""
    entity_id = store.mint_entity("Africa", [1.0, 0.0], "loc")
    store.attach_alias(entity_id, "African continent")
    assert store.lookup_alias("African continent", "loc") == entity_id


def test_load_alias_index_normalizes_all_surfaces(store: EntityStore) -> None:
    """The read-path index maps normalized aliases+canonical to the entity."""
    entity_id = store.mint_entity("Africa", [1.0, 0.0], "loc")
    store.attach_alias(entity_id, "African Continent")

    alias_to_id, canonical = store.load_alias_index(case_normalize=True)

    assert alias_to_id[("africa", "loc")] == entity_id
    assert alias_to_id[("african continent", "loc")] == entity_id
    assert canonical[entity_id] == "Africa"
