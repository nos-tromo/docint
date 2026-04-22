"""Shared Qdrant storage utilities."""

from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient


def qdrant_collection_exists(
    client: QdrantClient | None,
    collection_name: str,
) -> bool:
    """Return whether a Qdrant collection exists.

    Args:
        client: The Qdrant client instance.  Returns ``False`` when *None*.
        collection_name: Name of the collection to check.

    Returns:
        ``True`` if the collection exists, ``False`` otherwise.
    """
    if client is None:
        return False
    try:
        return bool(client.collection_exists(collection_name))
    except Exception as exc:
        logger.warning(
            "Collection existence check failed for '{}': {}",
            collection_name,
            exc,
        )
        return False
