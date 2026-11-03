"""CLI entry point for resolving entities in a collection.

Mirrors ``docint.cli.ingest``: a thin terminal wrapper around
``RAG.resolve_entities`` that merges semantically-equivalent named entities
into durable canonical records (the hidden ``{collection}_entities`` store).
Re-runnable and idempotent — surfaces already resolved are skipped.

A run is one audited ``combine`` (``entities.resolve``), with the surfaces'
embeddings and the tie-break prompts as its ``ai_call`` transfers.
"""

import sys
from pathlib import Path

from loguru import logger

from docint.cli._audit import collection_criteria, session
from docint.cli._collection import CollectionNotFoundError, resolve_collection_name
from docint.core.audit import CliAudit, metered
from docint.core.rag import RAG
from docint.utils.audit_event import Target
from docint.utils.env_cfg import set_offline_env
from docint.utils.logger_cfg import init_logger


def get_collection() -> str:
    """Get user input for the Qdrant collection name.

    Returns:
        str: The collection name as entered.
    """
    return input("Enter collection name: ").strip()


def resolve_entities(qdrant_col: str, *, audit: CliAudit | None = None) -> None:
    """Resolve entities for one collection into durable canonicals.

    Args:
        qdrant_col (str): Logical or physical collection name.
        audit (CliAudit | None): The command's audit chain; one is opened for
            this run when omitted.

    Raises:
        SystemExit: When the name cannot be resolved to one collection.

    Notes:
        Query engine creation is skipped (the headless path) so large
        generation/reranker models are not loaded for a resolution job.
    """
    typed = qdrant_col
    with session(audit, "resolve", "combine", "entities.resolve", also=("ai_call",)) as cli:
        op = cli.operation("combine", "entities.resolve", container=typed)
        with metered(op) as meter:
            rag = RAG(qdrant_collection=qdrant_col)
            try:
                try:
                    # The app shows *logical* names; the physical Qdrant name is
                    # owner-namespaced, and passing the logical one straight through
                    # just 404s.
                    physical = resolve_collection_name(rag, qdrant_col)
                except CollectionNotFoundError as exc:
                    logger.error("{}", exc)
                    raise SystemExit(1) from exc
                if physical != qdrant_col:
                    logger.info("Resolved '{}' to the physical collection '{}'.", qdrant_col, physical)
                    qdrant_col = physical
                    rag = RAG(qdrant_collection=physical)
                op.criteria.update(collection_criteria(typed, physical))
                summary = rag.resolve_entities(progress_callback=lambda msg: logger.info(msg))
            finally:
                rag.unload_models()
        op.record(
            target=Target(kind="entity", count=summary.entities_touched),
            criteria={
                "processed": summary.processed,
                "minted": summary.minted,
                "attached": summary.attached,
                "skipped": summary.skipped,
            },
        )
        op.transfers(meter)
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
