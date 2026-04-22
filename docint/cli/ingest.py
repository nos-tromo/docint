import sys
from pathlib import Path
from typing import Callable

from loguru import logger

from docint.core.rag import RAG, EmptyIngestionError
from docint.utils.env_cfg import load_path_env, set_offline_env
from docint.utils.logger_cfg import init_logger


def get_collection() -> str:
    """Get user inputs for the Qdrant collection name.

    Returns:
        str: Qdrant collection name.
    """
    return input("Enter Qdrant collection name: ").strip()


def ingest_docs(
    qdrant_col: str,
    data_dir: Path,
    hybrid: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Ingest documents from the specified directory into the Qdrant collection.

    Args:
        qdrant_col (str): Qdrant collection name.
        data_dir (Path): Path to the data directory.
        hybrid (bool): Whether to enable hybrid search. Defaults to True.
        progress_callback (Callable[[str], None] | None): Optional callback for
            reporting ingestion progress.

    Raises:
        EmptyIngestionError: When the source directory yielded no usable
            content for a fresh collection. Re-raised so programmatic
            callers (e.g. the streaming ingest API) can branch on the
            soft-empty outcome without parsing log messages. The terminal
            CLI :func:`main` catches this and exits cleanly.

    Notes:
        The CLI skips query engine creation so that large generation and reranker models
        are not loaded unnecessarily during ingestion jobs.
    """
    rag = RAG(qdrant_collection=qdrant_col, enable_hybrid=hybrid)
    try:
        rag.ingest_docs(
            data_dir,
            build_query_engine=False,
            progress_callback=progress_callback or (lambda msg: logger.info(msg)),
        )
    finally:
        rag.unload_models()
    logger.info("Ingestion complete.")


def main() -> None:
    """Main function for the ingestion CLI.

    Catches :class:`EmptyIngestionError` so that a no-content ingestion
    surfaces as a warning and a clean exit rather than a traceback or
    non-zero exit code — the underlying ``RAG`` already removed the
    empty SQLite KV store and retained the uploaded source files.
    """
    init_logger()
    set_offline_env()
    data_path = load_path_env().data
    qdrant_col = get_collection()
    try:
        ingest_docs(qdrant_col, data_path)
    except EmptyIngestionError as exc:
        logger.warning(
            "Ingestion produced no content for '{}'. Empty KV store has been "
            "removed; uploaded source files are retained.",
            exc.collection_name,
        )


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
