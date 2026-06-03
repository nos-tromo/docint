"""Tests for the pure entity-resolution pipeline.

These exercise :mod:`docint.core.entities.resolution` against an
in-memory fake store plus fake embed / chat callables, mirroring the
behaviours pinned by chorus's ``tests/ingestion/test_resolution.py``:
type-blocking, mint-vs-attach decision tree, conservative LLM
tie-break, idempotent re-runs and the in-run case-variant cache.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from docint.core.entities.resolution import (
    ResolutionSummary,
    SurfaceMention,
    llm_tiebreaker,
    normalize_surface,
    resolve_collection,
    resolve_surface,
)
from docint.utils.env_cfg import ResolutionConfig


def _cfg(
    *,
    threshold: float = 0.86,
    llm: bool = True,
    case_normalize: bool = True,
    k: int = 5,
) -> ResolutionConfig:
    """Build a ``ResolutionConfig`` for tests.

    Args:
        threshold: Cosine merge threshold.
        llm: Whether the LLM tie-break is enabled.
        case_normalize: Whether the in-run cache casefolds surfaces.
        k: Candidate fan-out.

    Returns:
        ResolutionConfig: The configured dataclass.
    """
    return ResolutionConfig(
        embed_cluster_threshold=threshold,
        llm_tiebreak_enabled=llm,
        case_normalize=case_normalize,
        vector_k=k,
    )


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        float: Cosine similarity, or 0.0 when either vector is zero.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class FakeStore:
    """In-memory reference implementation of the EntityStore protocol."""

    def __init__(self) -> None:
        """Initialize empty entity and alias indexes."""
        self.entities: dict[str, dict[str, Any]] = {}
        self.alias_to_id: dict[tuple[str, str], str] = {}
        self._counter = 0

    def lookup_alias(self, surface: str, entity_type: str) -> str | None:
        """Return the entity id for an exact ``(surface, type)`` alias."""
        return self.alias_to_id.get((surface, entity_type.lower()))

    def cluster_candidates(
        self, embedding: list[float], threshold: float, k: int, entity_type: str
    ) -> list[dict[str, Any]]:
        """Return type-matched entities scoring at or above ``threshold``."""
        cands: list[dict[str, Any]] = []
        for eid, ent in self.entities.items():
            if ent["type"].lower() != entity_type.lower():
                continue
            score = _cos(embedding, ent["vector"])
            if score >= threshold:
                cands.append({"id": eid, "canonical_name": ent["canonical_name"], "type": ent["type"], "score": score})
        cands.sort(key=lambda c: -c["score"])
        return cands[:k]

    def mint_entity(self, surface: str, embedding: list[float], entity_type: str) -> str:
        """Create a new canonical entity seeded with ``surface``."""
        self._counter += 1
        eid = f"e{self._counter}"
        self.entities[eid] = {
            "canonical_name": surface,
            "type": entity_type,
            "vector": list(embedding),
            "aliases": {surface},
        }
        self.alias_to_id[(surface, entity_type.lower())] = eid
        return eid

    def attach_alias(self, entity_id: str, surface: str) -> None:
        """Attach ``surface`` to an existing entity."""
        ent = self.entities[entity_id]
        ent["aliases"].add(surface)
        self.alias_to_id[(surface, ent["type"].lower())] = entity_id


def _embed_fn(vectors: dict[str, list[float]]) -> Callable[[list[str]], list[list[float]]]:
    """Build a deterministic embed callable from a surface->vector map.

    Args:
        vectors: Mapping of surface text to its embedding vector.

    Returns:
        Callable[[list[str]], list[list[float]]]: Batch embed function.
    """

    def embed(texts: list[str]) -> list[list[float]]:
        return [vectors[t] for t in texts]

    return embed


def _chat_fn(reply: str) -> Callable[[str], str]:
    """Build a chat callable returning a fixed reply.

    Args:
        reply: The text the fake LLM returns for any prompt.

    Returns:
        Callable[[str], str]: Chat function.
    """
    return lambda _prompt: reply


# --- normalize_surface -------------------------------------------------------


def test_normalize_surface_strips_and_casefolds_when_enabled() -> None:
    """With case-normalize on, surfaces are stripped and casefolded."""
    assert normalize_surface("  Africa ", case_normalize=True) == "africa"
    assert normalize_surface("AFRICA", case_normalize=True) == "africa"


def test_normalize_surface_strips_only_when_case_disabled() -> None:
    """With case-normalize off, only surrounding whitespace is removed."""
    assert normalize_surface("  Africa ", case_normalize=False) == "Africa"


# --- llm_tiebreaker ----------------------------------------------------------


def _candidates() -> list[dict[str, Any]]:
    """Two candidate entities for tie-break tests.

    Returns:
        list[dict[str, Any]]: Candidate rows with distinct ids.
    """
    return [
        {"id": "e-1", "canonical_name": "Berlin", "type": "loc", "score": 0.9},
        {"id": "e-2", "canonical_name": "Berlin, NH", "type": "loc", "score": 0.88},
    ]


def test_llm_tiebreaker_returns_single_named_id() -> None:
    """A reply naming exactly one candidate id resolves to that id."""
    chosen = llm_tiebreaker("Berlin", _candidates(), chat_fn=_chat_fn("e-1"), prompt_header="pick one")
    assert chosen == "e-1"


def test_llm_tiebreaker_abstains_on_none() -> None:
    """A NONE reply yields no match (caller will mint)."""
    chosen = llm_tiebreaker("Berlin", _candidates(), chat_fn=_chat_fn("NONE"), prompt_header="pick one")
    assert chosen is None


def test_llm_tiebreaker_abstains_on_ambiguous_reply() -> None:
    """A reply naming several ids abstains rather than guessing."""
    chosen = llm_tiebreaker("Berlin", _candidates(), chat_fn=_chat_fn("e-1 e-2"), prompt_header="pick one")
    assert chosen is None


def test_llm_tiebreaker_uses_exact_id_not_substring() -> None:
    """``e-1`` must not be matched by a reply containing only ``e-12``."""
    cands = [{"id": "e-1", "canonical_name": "A", "type": "loc", "score": 0.9}]
    chosen = llm_tiebreaker("x", cands, chat_fn=_chat_fn("e-12"), prompt_header="pick one")
    assert chosen is None


# --- resolve_surface ---------------------------------------------------------


def test_resolve_surface_mints_when_no_candidates() -> None:
    """An unseen surface with no candidates mints a new entity."""
    store = FakeStore()
    entity_id, method = resolve_surface(
        store, "Africa", [1.0, 0.0], _cfg(), entity_type="loc", chat_fn=_chat_fn("NONE"), prompt_header="h"
    )
    assert method == "minted"
    assert store.entities[entity_id]["canonical_name"] == "Africa"


def test_resolve_surface_attaches_to_single_candidate() -> None:
    """A lone above-threshold candidate is attached without the LLM."""
    store = FakeStore()
    store.mint_entity("Africa", [1.0, 0.0], "loc")
    entity_id, method = resolve_surface(
        store, "africa", [0.99, 0.05], _cfg(), entity_type="loc", chat_fn=_chat_fn("NONE"), prompt_header="h"
    )
    assert method == "vector_single"
    assert store.entities[entity_id]["canonical_name"] == "Africa"


def test_resolve_surface_uses_llm_when_multiple_candidates() -> None:
    """With >1 candidate the LLM picks the winning entity id."""
    store = FakeStore()
    a = store.mint_entity("Berlin", [1.0, 0.0], "loc")
    store.mint_entity("Berlin City", [0.98, 0.1], "loc")
    entity_id, method = resolve_surface(
        store, "berlin", [0.99, 0.02], _cfg(), entity_type="loc", chat_fn=_chat_fn(a), prompt_header="h"
    )
    assert method == "vector_llm"
    assert entity_id == a


def test_resolve_surface_mints_when_llm_abstains() -> None:
    """If the LLM abstains amid ambiguity, mint rather than force-merge."""
    store = FakeStore()
    store.mint_entity("Berlin", [1.0, 0.0], "loc")
    store.mint_entity("Berlin City", [0.98, 0.1], "loc")
    before = len(store.entities)
    _id, method = resolve_surface(
        store, "berlin", [0.99, 0.02], _cfg(), entity_type="loc", chat_fn=_chat_fn("NONE"), prompt_header="h"
    )
    assert method == "minted"
    assert len(store.entities) == before + 1


def test_resolve_surface_attaches_top_score_when_tiebreak_disabled() -> None:
    """With the LLM off, ambiguity attaches to the top-scoring candidate."""
    store = FakeStore()
    top = store.mint_entity("Berlin", [1.0, 0.0], "loc")
    store.mint_entity("Berlin City", [0.9, 0.2], "loc")
    entity_id, method = resolve_surface(
        store,
        "berlin",
        [0.99, 0.02],
        _cfg(llm=False),
        entity_type="loc",
        chat_fn=_chat_fn("NONE"),
        prompt_header="h",
    )
    assert method == "vector_topk"
    assert entity_id == top


def test_resolve_surface_is_idempotent() -> None:
    """Re-resolving an already-attached surface is a skip."""
    store = FakeStore()
    minted, _ = resolve_surface(
        store, "Africa", [1.0, 0.0], _cfg(), entity_type="loc", chat_fn=_chat_fn("NONE"), prompt_header="h"
    )
    again, method = resolve_surface(
        store, "Africa", [1.0, 0.0], _cfg(), entity_type="loc", chat_fn=_chat_fn("NONE"), prompt_header="h"
    )
    assert method == "skipped"
    assert again == minted


# --- resolve_collection ------------------------------------------------------


def test_resolve_collection_clusters_case_variants_via_cache() -> None:
    """Case variants collapse to one entity through the in-run cache."""
    store = FakeStore()
    surfaces = [
        SurfaceMention("Berlin", "loc", 3),
        SurfaceMention("berlin", "loc", 1),
    ]
    embed = _embed_fn({"Berlin": [1.0, 0.0], "berlin": [1.0, 0.0]})
    summary = resolve_collection(
        store, surfaces, embed_fn=embed, chat_fn=_chat_fn("NONE"), prompt_header="h", cfg=_cfg()
    )
    assert len(store.entities) == 1
    assert summary.minted == 1
    assert summary.attached == 1


def test_resolve_collection_does_not_merge_across_types() -> None:
    """The same surface under two types yields two distinct entities."""
    store = FakeStore()
    surfaces = [
        SurfaceMention("Jordan", "loc", 3),
        SurfaceMention("Jordan", "person", 2),
    ]
    embed = _embed_fn({"Jordan": [1.0, 0.0]})
    resolve_collection(store, surfaces, embed_fn=embed, chat_fn=_chat_fn("NONE"), prompt_header="h", cfg=_cfg())
    assert len(store.entities) == 2
    assert {e["type"] for e in store.entities.values()} == {"loc", "person"}


def test_resolve_collection_canonical_is_most_mentioned() -> None:
    """Most-mentioned surface mints first and becomes the canonical name."""
    store = FakeStore()
    surfaces = [
        SurfaceMention("United States", "loc", 2),
        SurfaceMention("USA", "loc", 5),
    ]
    embed = _embed_fn({"USA": [1.0, 0.0], "United States": [0.98, 0.2]})
    resolve_collection(store, surfaces, embed_fn=embed, chat_fn=_chat_fn("NONE"), prompt_header="h", cfg=_cfg())
    assert len(store.entities) == 1
    (only,) = store.entities.values()
    assert only["canonical_name"] == "USA"
    assert only["aliases"] == {"USA", "United States"}


def test_resolve_collection_rerun_is_all_skipped() -> None:
    """A second pass over resolved surfaces mints and attaches nothing."""
    store = FakeStore()
    surfaces = [SurfaceMention("USA", "loc", 5), SurfaceMention("United States", "loc", 2)]
    embed = _embed_fn({"USA": [1.0, 0.0], "United States": [0.98, 0.2]})
    resolve_collection(store, surfaces, embed_fn=embed, chat_fn=_chat_fn("NONE"), prompt_header="h", cfg=_cfg())
    summary = resolve_collection(
        store, surfaces, embed_fn=embed, chat_fn=_chat_fn("NONE"), prompt_header="h", cfg=_cfg()
    )
    assert summary.minted == 0
    assert summary.attached == 0
    assert summary.skipped == 2


def test_resolve_collection_empty_input_does_not_embed() -> None:
    """No surfaces means no embed call and an empty summary."""

    def explode(_texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed_fn must not be called for empty input")

    summary = resolve_collection(
        FakeStore(), [], embed_fn=explode, chat_fn=_chat_fn("NONE"), prompt_header="h", cfg=_cfg()
    )
    assert isinstance(summary, ResolutionSummary)
    assert summary.processed == 0
