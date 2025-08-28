import logging
from pathlib import Path

from wizard.modules.rag import RAG
from wizard.utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def get_inputs() -> tuple[str, Path]:
    """
    Get user inputs for Qdrant collection name and data directory.

    Raises:
        ValueError: If the data directory does not exist.

    Returns:
        tuple[str, Path]: Qdrant collection name and data directory path.
    """
    qdrant_col = input("Enter Qdrant collection name: ").strip()
    data_dir = Path(input("Enter path to data directory [default: ./backend/data]: ").strip() or "data")
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    return qdrant_col, data_dir


def ingest_docs(qdrant_col: str, data_dir: Path) -> None:
    """
    Ingest documents from the specified directory into the Qdrant collection.

    Args:
        qdrant_col (str): Qdrant collection name.
        data_dir (Path): Path to the data directory.
    """
    rag = RAG(qdrant_collection=qdrant_col, enable_hybrid=True)
    rag.ingest_docs(data_dir)
    logger.info("Ingestion complete.")


def main() -> None:
    qdrant_col, data_dir = get_inputs()
    ingest_docs(qdrant_col, data_dir)


if __name__ == "__main__":
    main()
