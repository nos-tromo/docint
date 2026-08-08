"""CLI entry point for backfilling the full-text search field.

Mirrors ``docint.cli.resolve``: a thin terminal wrapper that populates the
``search_text`` payload field across an already-ingested collection and creates
its index. Payload-only — no re-embedding, no inference, no model downloads —
so it is safe on an airgapped host, and re-running it is cheap because
populated points are skipped.

Takes the *logical* collection name shown in the app; the physical Qdrant name
is owner-namespaced and resolved here.
"""

import sys
from pathlib import Path

from loguru import logger

from docint.cli._collection import CollectionNotFoundError, resolve_collection_name
from docint.core.rag import RAG
from docint.core.search.index import backfill_search_text, ensure_search_index
from docint.utils.env_cfg import set_offline_env
from docint.utils.logger_cfg import init_logger


def get_collection() -> str:
    """Get user input for the collection name.

    Returns:
        str: The collection name as entered.
    """
    return input("Enter collection name: ").strip()


def build_search_index(collection: str) -> None:
    """Create the search index and backfill the field for one collection.

    Every failure here exits non-zero rather than reporting a count. The
    backfill's scroll is deliberately fail-soft, so a collection that does not
    exist yields ``0 scanned, 0 written`` — which, announced as a total, reads
    exactly like an already-migrated collection. An operator working through a
    list would tick it off.

    Args:
        collection (str): Logical or physical collection name.

    Raises:
        SystemExit: When the name cannot be resolved, the payload index cannot
            be created, or the collection yielded no points at all.
    """
    rag = RAG(qdrant_collection=collection)
    try:
        try:
            physical = resolve_collection_name(rag, collection)
        except CollectionNotFoundError as exc:
            logger.error("{}", exc)
            raise SystemExit(1) from exc

        if physical != collection:
            logger.info("Resolved '{}' to the physical collection '{}'.", collection, physical)

        if not ensure_search_index(rag.qdrant_client, physical):
            logger.error(
                "Could not create the search index on '{}'. Search would be case-sensitive on "
                "non-ASCII text without it, so this is a hard failure — is Qdrant reachable?",
                physical,
            )
            raise SystemExit(1)

        summary = backfill_search_text(
            rag.qdrant_client,
            physical,
            extract_text=RAG._extract_payload_text,
            progress=lambda msg: logger.info(msg),
        )
    finally:
        rag.unload_models()

    if summary.scanned == 0:
        logger.error(
            "Scanned no points in '{}'. An empty result here means the scroll failed, not that "
            "the collection is already indexed — check the warnings above.",
            physical,
        )
        raise SystemExit(1)

    logger.info(
        "Search index ready for '{}': {} scanned, {} written, {} already populated, {} without text.",
        physical,
        summary.scanned,
        summary.written,
        summary.skipped,
        summary.empty,
    )


def main() -> None:
    """Main function for the search-index CLI."""
    init_logger()
    set_offline_env()
    build_search_index(get_collection())


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
