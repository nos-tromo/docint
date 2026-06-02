"""CLI entry point for resolving entities in a collection.

Mirrors ``docint.cli.ingest``: a thin terminal wrapper around
``RAG.resolve_entities`` that merges semantically-equivalent named entities
into durable canonical records (the hidden ``{collection}_entities`` store).
Re-runnable and idempotent — surfaces already resolved are skipped.
"""

import sys
from pathlib import Path

from loguru import logger

from docint.core.rag import RAG
from docint.utils.env_cfg import set_offline_env
from docint.utils.logger_cfg import init_logger


def get_collection() -> str:
    """Get user input for the Qdrant collection name.

    Returns:
        str: Qdrant collection name.
    """
    return input("Enter Qdrant collection name: ").strip()


def resolve_entities(qdrant_col: str) -> None:
    """Resolve entities for one collection into durable canonicals.

    Args:
        qdrant_col (str): Qdrant collection name.

    Notes:
        Query engine creation is skipped (the headless path) so large
        generation/reranker models are not loaded for a resolution job.
    """
    rag = RAG(qdrant_collection=qdrant_col)
    try:
        summary = rag.resolve_entities(progress_callback=lambda msg: logger.info(msg))
    finally:
        rag.unload_models()
    logger.info(
        "Resolution complete for '{}': {} processed, {} minted, {} attached, {} skipped.",
        qdrant_col,
        summary.processed,
        summary.minted,
        summary.attached,
        summary.skipped,
    )


def main() -> None:
    """Main function for the entity-resolution CLI."""
    init_logger()
    set_offline_env()
    qdrant_col = get_collection()
    resolve_entities(qdrant_col)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
