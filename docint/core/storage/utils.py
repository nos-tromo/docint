"""Shared Qdrant storage utilities."""

from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from docint.utils.env_cfg import QdrantQuantizationConfig, load_quantization_env


def build_quantization_config(
    cfg: QdrantQuantizationConfig | None = None,
) -> qdrant_models.TurboQuantization | None:
    """Build the Qdrant quantization payload from configuration.

    Single source of truth for the TurboQuant wire shape: every collection
    creation site and the startup reconcile pass use this builder.

    Args:
        cfg: Quantization settings; loaded from the environment when *None*.

    Returns:
        The ``TurboQuantization`` payload, or ``None`` when quantization
        is disabled (``QDRANT_QUANTIZATION=none``).
    """
    resolved = cfg if cfg is not None else load_quantization_env()
    if resolved.mode != "turbo":
        return None
    return qdrant_models.TurboQuantization(
        turbo=qdrant_models.TurboQuantQuantizationConfig(
            bits=qdrant_models.TurboQuantBitSize(resolved.bits),
            always_ram=resolved.always_ram,
        )
    )


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
