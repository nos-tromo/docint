"""Qdrant-backed store for durable canonical entities.

Each canonical entity is one point in a hidden ``{collection}_entities``
companion collection:

- id      = entity UUID
- vector  = canonical-name embedding (dense, cosine; same model/dim as the
  base collection so name vectors are comparable)
- payload = ``{canonical_name, type, aliases: [surface forms], embed_model}``

This is the docint analogue of chorus's ``:Entity`` node plus its
``:Alias -[:RESOLVED_TO]-> :Entity`` links — except the alias->entity mapping
lives in the entity's ``aliases`` payload list rather than on separate nodes,
so chunk payloads stay immutable.

The class satisfies
:class:`docint.core.entities.resolution.EntityStoreProtocol`.
"""

from __future__ import annotations

import uuid
import warnings
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from docint.core.entities.resolution import Candidate, normalize_surface
from docint.core.storage.utils import qdrant_collection_exists


def _aliases_of(payload: dict[str, Any]) -> list[str]:
    """Return the alias list from an entity payload defensively.

    Args:
        payload (dict[str, Any]): Entity point payload.

    Returns:
        list[str]: Alias surface forms (empty when absent/malformed).
    """
    raw = payload.get("aliases")
    if not isinstance(raw, list):
        return []
    return [str(a) for a in raw if a]


class EntityStore:
    """Persist and query canonical entities in a Qdrant companion collection."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection: str,
        dim: int,
        embed_model: str | None = None,
    ) -> None:
        """Initialize the store.

        Args:
            client (QdrantClient): Qdrant client (shared with the RAG layer).
            collection (str): Companion collection name (``{base}_entities``).
            dim (int): Embedding dimension for the name vectors.
            embed_model (str | None): Embedding model id stamped on minted
                entities for provenance (normalized to ``""`` when unset).
        """
        self.client = client
        self.collection = collection
        self.dim = int(dim)
        self.embed_model = embed_model or ""
        self._raw_alias_index: dict[tuple[str, str], str] | None = None

    def ensure_collection(self) -> None:
        """Create the companion collection if it does not yet exist."""
        if qdrant_collection_exists(self.client, self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qdrant_models.VectorParams(
                size=self.dim,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        try:
            # The KEYWORD index speeds up the type filter on a real Qdrant
            # server; it is a no-op (and warns) in local/in-memory mode, so the
            # informational warning is suppressed without hiding real errors.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Payload indexes have no effect", category=UserWarning)
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name="type",
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                )
        except Exception as exc:
            logger.debug("Payload index on 'type' skipped for '{}': {}", self.collection, exc)
        logger.info("Created entity collection '{}' (vector_size={}).", self.collection, self.dim)

    def _scroll_all(self) -> list[Any]:
        """Scroll every point in the companion collection.

        Returns:
            list[Any]: All stored entity points with payloads.
        """
        records: list[Any] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(points)
            if offset is None or not points:
                break
        return records

    def _ensure_raw_index(self) -> dict[tuple[str, str], str]:
        """Lazily load the exact ``(surface, type) -> id`` alias index.

        Returns:
            dict[tuple[str, str], str]: Exact-surface alias index.
        """
        if self._raw_alias_index is not None:
            return self._raw_alias_index
        index: dict[tuple[str, str], str] = {}
        for point in self._scroll_all():
            payload = dict(getattr(point, "payload", {}) or {})
            entity_id = str(getattr(point, "id", ""))
            entity_type = str(payload.get("type") or "")
            for alias in _aliases_of(payload):
                index[(alias, entity_type)] = entity_id
        self._raw_alias_index = index
        return index

    def lookup_alias(self, surface: str, entity_type: str) -> str | None:
        """Return the entity id for an exact ``(surface, type)`` alias.

        Args:
            surface (str): Exact raw surface form.
            entity_type (str): Type block.

        Returns:
            str | None: Entity id, or ``None`` when not previously resolved.
        """
        return self._ensure_raw_index().get((surface, entity_type))

    def cluster_candidates(self, embedding: list[float], threshold: float, k: int, entity_type: str) -> list[Candidate]:
        """Return up to ``k`` same-type entities scoring ``>= threshold``.

        Args:
            embedding (list[float]): Query name embedding.
            threshold (float): Minimum cosine score to keep a candidate.
            k (int): Maximum number of nearest candidates to consider.
            entity_type (str): Type block (hard filter — no cross-type leak).

        Returns:
            list[Candidate]: Candidates sorted by descending score.
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(embedding),
            limit=max(1, int(k)),
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="type",
                        match=qdrant_models.MatchValue(value=entity_type),
                    )
                ]
            ),
            with_payload=True,
        )
        candidates: list[Candidate] = []
        for point in response.points:
            score = float(point.score)
            if score < threshold:
                continue
            payload = dict(getattr(point, "payload", {}) or {})
            candidates.append(
                {
                    "id": str(point.id),
                    "canonical_name": str(payload.get("canonical_name") or ""),
                    "type": str(payload.get("type") or ""),
                    "score": score,
                }
            )
        return candidates

    def mint_entity(self, surface: str, embedding: list[float], entity_type: str) -> str:
        """Create a new canonical entity seeded with ``surface``.

        Args:
            surface (str): Canonical surface form (the first/most-mentioned).
            embedding (list[float]): Name embedding used as the point vector.
            entity_type (str): Entity type block.

        Returns:
            str: The new entity UUID.
        """
        entity_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection,
            wait=True,
            points=[
                qdrant_models.PointStruct(
                    id=entity_id,
                    vector=list(embedding),
                    payload={
                        "canonical_name": surface,
                        "type": entity_type,
                        "aliases": [surface],
                        "embed_model": self.embed_model,
                    },
                )
            ],
        )
        if self._raw_alias_index is not None:
            self._raw_alias_index[(surface, entity_type)] = entity_id
        return entity_id

    def attach_alias(self, entity_id: str, surface: str) -> None:
        """Attach ``surface`` to an existing entity's alias list.

        Args:
            entity_id (str): Target entity id.
            surface (str): Surface form to attach.
        """
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[entity_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return
        payload = dict(getattr(records[0], "payload", {}) or {})
        aliases = _aliases_of(payload)
        if surface not in aliases:
            aliases.append(surface)
            self.client.set_payload(
                collection_name=self.collection,
                payload={"aliases": aliases},
                points=[entity_id],
            )
        if self._raw_alias_index is not None:
            self._raw_alias_index[(surface, str(payload.get("type") or ""))] = entity_id

    def load_alias_index(self, *, case_normalize: bool) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
        """Build the read-path index for resolved-mode aggregation.

        Args:
            case_normalize (bool): Whether to casefold surfaces (must match the
                value used at resolve time so chunk occurrences map correctly).

        Returns:
            tuple[dict[tuple[str, str], str], dict[str, str]]:
                ``(alias_to_id, canonical)`` where ``alias_to_id`` maps
                ``(normalized_surface, type_lower) -> entity_id`` for every
                alias and canonical name, and ``canonical`` maps
                ``entity_id -> canonical_name``.
        """
        alias_to_id: dict[tuple[str, str], str] = {}
        canonical: dict[str, str] = {}
        for point in self._scroll_all():
            payload = dict(getattr(point, "payload", {}) or {})
            entity_id = str(getattr(point, "id", ""))
            entity_type = str(payload.get("type") or "")
            canonical_name = str(payload.get("canonical_name") or "")
            canonical[entity_id] = canonical_name
            surfaces = _aliases_of(payload)
            if canonical_name:
                surfaces.append(canonical_name)
            for surface in surfaces:
                key = (normalize_surface(surface, case_normalize=case_normalize), entity_type.lower())
                alias_to_id[key] = entity_id
        return alias_to_id, canonical
