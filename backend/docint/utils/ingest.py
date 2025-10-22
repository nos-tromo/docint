import logging
import os
from pathlib import Path

from docint.core.rag import RAG
from docint.utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH")


def get_inputs() -> tuple[str, Path]:
    """
    Get user inputs for Milvus collection name and data directory.

    Raises:
        ValueError: If the data directory does not exist.

    Returns:
        tuple[str, Path]: Milvus collection name and data directory path.
    """
    collection = input("Enter Milvus collection name: ").strip()
    data_dir = Path(DATA_PATH) if DATA_PATH else Path.home() / "docint" / "data"
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    return collection, data_dir


def ingest_docs(collection: str, data_dir: Path, hybrid: bool = True) -> None:
    """
    Ingest documents from the specified directory into the Milvus collection.

    Args:
        collection (str): Milvus collection name.
        data_dir (Path): Path to the data directory.
        hybrid (bool): Whether to enable hybrid search (default: True).
    """
    rag = RAG(milvus_collection=collection, enable_hybrid=hybrid)
    rag.ingest_docs(data_dir)
    logger.info("Ingestion complete.")


def main() -> None:
    collection, data_dir = get_inputs()
    ingest_docs(collection, data_dir)


if __name__ == "__main__":
    main()
