import json
import logging
from pathlib import Path
from time import time

from docint.core.rag import RAG
from docint.utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


RESULTS_DIR: Path = Path.home() / "docint" / "results"


def _get_col_name() -> str:
    """
    Prompts the user to enter a collection name.

    Returns:
        str: The entered collection name.
    """
    return input("Enter collection name: ")


def _store_output(
    filename: str, data: dict | list, out_dir: str | Path = RESULTS_DIR
) -> None:
    """
    Stores the output data to a JSON file.

    Args:
        filename (str): The name of the output file (without extension).
        data (dict | list): The data to store.
        out_dir (str | Path, optional): The directory to store the output file. Defaults to RESULTS_DIR.
    """
    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True)

    if isinstance(data, dict):
        with open(out_dir / f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    elif isinstance(data, list):
        # Detect if list elements are nodes and use .to_dict()
        serializable = []
        for item in data:
            if hasattr(item, "to_dict"):  # works for LlamaIndex BaseNode
                serializable.append(item.to_dict())
            else:
                serializable.append(str(item))

        with open(out_dir / f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info("Results stored in %s", out_dir / f"{filename}.json")


def rag_pipeline() -> RAG:
    """
    Initializes a Retrieval-Augmented Generation (RAG) session.

    Returns:
        RAG: The initialized RAG instance.
    """
    logger.info("Initializing RAG pipeline...")
    rag = RAG(qdrant_collection=_get_col_name())
    rag.create_index()
    rag.create_query_engine()
    return rag


def load_queries(q_path: Path = Path("queries.txt")) -> list[str]:
    """
    Loads query strings from a text file. If the file does not exist, it creates a default one.

    Args:
        q_path (Path, optional): The path to the query file. Defaults to Path("queries.txt").

    Returns:
        list[str]: The list of query strings.
    """
    q_path = Path(q_path).resolve()
    if q_path.exists():
        logger.info("Loading queries from %s", q_path)
        with open(q_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        logger.info("Creating default query file at %s", q_path)
        default_query = "Summarize the content with a maximum of 15 sentences."
        with open(q_path, "w", encoding="utf-8") as f:
            f.write(default_query + "\n")
        return [default_query]


def run_query(rag: RAG, query: str, index: int) -> None:
    """
    Runs a query against the RAG instance and stores the result.

    Args:
        rag (RAG): The RAG instance to query.
        query (str): The query string.
        index (int): The index of the query (for logging and output purposes).
    """
    logger.info("Running query %d: %s", index, query)
    result = rag.run_query(query)
    timestamp = str(int(time()))
    _store_output(f"{timestamp}_{index}_result", result)


def main() -> None:
    """
    Main entry point for the CLI. Initializes the RAG pipeline, loads queries, and processes each query.
    """
    rag = rag_pipeline()
    queries = load_queries()
    for index, query in enumerate(queries, start=1):
        run_query(rag=rag, query=query, index=index)
    logger.info("All queries processed.")


if __name__ == "__main__":
    main()
