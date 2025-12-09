import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from docint.core.rag import RAG
from docint.utils.env_cfg import load_path_env, set_offline_env
from docint.utils.logging_cfg import setup_logging


def get_collection() -> str:
    """
    Get user inputs for the Qdrant collection name.

    Returns:
        str: Qdrant collection name.
    """
    return input("Enter Qdrant collection name: ").strip()


def ingest_docs(
    qdrant_col: str,
    data_dir: Path,
    hybrid: bool = True,
    table_row_limit: int | None = None,
    table_row_filter: str | None = None,
) -> None:
    """
    Ingest documents from the specified directory into the Qdrant collection.

    Args:
        qdrant_col (str): Qdrant collection name.
        data_dir (Path): Path to the data directory.
        hybrid (bool): Whether to enable hybrid search (default: True).
        table_row_limit (int | None): Optional limit applied to tabular rows.
        table_row_filter (str | None): Optional pandas-compatible query string to filter rows.

    Notes:
        The CLI skips query engine creation so that large generation and reranker models
        are not loaded unnecessarily during ingestion jobs.
    """
    rag = RAG(
        qdrant_collection=qdrant_col,
        enable_hybrid=hybrid,
        table_row_limit=table_row_limit,
        table_row_filter=table_row_filter,
    )
    # Avoid loading chat/query models during offline ingestion jobs.
    rag.ingest_docs(data_dir, build_query_engine=False)
    logger.info("Ingestion complete.")


def main() -> None:
    load_dotenv()
    setup_logging()
    set_offline_env()
    data_path = load_path_env().data
    qdrant_col = get_collection()
    ingest_docs(qdrant_col, data_path)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
