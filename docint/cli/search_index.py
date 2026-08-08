"""CLI entry point for backfilling the full-text search field.

Mirrors ``docint.cli.resolve``: a thin terminal wrapper that populates the
``search_text`` payload field across an already-ingested collection and creates
its index. Payload-only — no re-embedding, no inference, no model downloads —
so it is safe on an airgapped host, and re-running it is cheap because
populated points are skipped.
"""

import sys
from pathlib import Path

from loguru import logger

from docint.core.rag import RAG
from docint.core.search.index import backfill_search_text, ensure_search_index
from docint.utils.env_cfg import set_offline_env
from docint.utils.logger_cfg import init_logger


def get_collection() -> str:
    """Get user input for the Qdrant collection name.

    Returns:
        str: Qdrant collection name.
    """
    return input("Enter Qdrant collection name: ").strip()


def build_search_index(qdrant_col: str) -> None:
    """Create the search index and backfill the field for one collection.

    Args:
        qdrant_col (str): Qdrant collection name.
    """
    rag = RAG(qdrant_collection=qdrant_col)
    try:
        if not ensure_search_index(rag.qdrant_client, qdrant_col):
            logger.warning(
                "Could not create the search index on '{}' — is Qdrant reachable?",
                qdrant_col,
            )
        summary = backfill_search_text(
            rag.qdrant_client,
            qdrant_col,
            extract_text=RAG._extract_payload_text,
            progress=lambda msg: logger.info(msg),
        )
    finally:
        rag.unload_models()
    logger.info(
        "Search index ready for '{}': {} scanned, {} written, {} skipped.",
        qdrant_col,
        summary.scanned,
        summary.written,
        summary.skipped,
    )


def main() -> None:
    """Main function for the search-index CLI."""
    init_logger()
    set_offline_env()
    build_search_index(get_collection())


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
