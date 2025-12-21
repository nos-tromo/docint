import sys
from pathlib import Path
from typing import Callable

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
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """
    Ingest documents from the specified directory into the Qdrant collection.

    Args:
        qdrant_col (str): Qdrant collection name.
        data_dir (Path): Path to the data directory.
        hybrid (bool): Whether to enable hybrid search (default: True).
        progress_callback (Callable[[str], None] | None): Optional callback for
            reporting ingestion progress.

    Notes:
        The CLI skips query engine creation so that large generation and reranker models
        are not loaded unnecessarily during ingestion jobs.
    """
    rag = RAG(
        qdrant_collection=qdrant_col,
        enable_hybrid=hybrid,
    )
    rag.ingest_docs(
        data_dir,
        build_query_engine=False,
        progress_callback=progress_callback or (lambda msg: logger.info(msg)),
    )
    rag.unload_models()
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
