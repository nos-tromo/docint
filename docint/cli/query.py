import json
import sys
from pathlib import Path
from time import time

from dotenv import load_dotenv
from loguru import logger

from docint.core.rag import RAG
from docint.utils.env_cfg import load_path_env, set_offline_env
from docint.utils.logging_cfg import setup_logging


def _get_col_name() -> str:
    """
    Prompts the user to enter a collection name.

    Returns:
        str: The entered collection name.
    """
    return input("Enter collection name: ")


def _store_output(filename: str, data: dict | list, output_path: str | Path) -> None:
    """
    Stores the output data to a JSON file.

    Args:
        filename (str): The name of the output file (without extension).
        data (dict | list): The data to store.
        output_path (str | Path): The directory to store the output file.
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path).expanduser()

    if not output_path.exists():
        logger.info("Creating output directory at {}", output_path)
        output_path.mkdir(exist_ok=True)

    if isinstance(data, dict):
        with open(output_path / f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    elif isinstance(data, list):
        # Detect if list elements are nodes and use .to_dict()
        serializable = []
        for item in data:
            if hasattr(item, "to_dict"):
                serializable.append(item.to_dict())
            else:
                serializable.append(str(item))

        with open(output_path / f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info("Results stored in {}", output_path / f"{filename}.json")


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


def load_queries(q_path: str | Path) -> list[str]:
    """
    Loads query strings from a text file. Defaults to creating a file with a default query if none exists.

    Args:
        q_path (str | Path): The path to the query text file.

    Returns:
        list[str]: The list of query strings.
    """
    if not isinstance(q_path, Path):
        q_path = Path(q_path).expanduser()

    if q_path.exists():
        logger.info("Loading queries from {}", q_path)
        with open(q_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        logger.info("Creating default query file at {}", q_path)
        default_query = "Summarize the content with a maximum of 15 sentences."
        with open(q_path, "w", encoding="utf-8") as f:
            f.write(default_query + "\n")
        return [default_query]


def run_query(rag: RAG, query: str, index: int, output_path: str | Path) -> None:
    """
    Runs a query against the RAG instance and stores the result.

    Args:
        rag (RAG): The RAG instance to query.
        query (str): The query string.
        index (int): The index of the query (for logging and output purposes).
        output_path (str | Path): The directory to store the output file.
    """
    logger.info("Running query {}: {}", index, query)
    result = rag.run_query(query)
    timestamp = str(int(time()))
    _store_output(
        filename=f"{timestamp}_{index}_result", data=result, output_path=output_path
    )


def main() -> None:
    """
    Main entry point for the CLI. Initializes the RAG pipeline, loads queries, and processes each query.
    """
    load_dotenv()
    setup_logging()
    set_offline_env()
    rag = rag_pipeline()
    path_config = load_path_env()
    queries = load_queries(q_path=path_config.queries)
    for index, query in enumerate(queries, start=1):
        run_query(rag=rag, query=query, index=index, output_path=path_config.results)
    rag.unload_models()
    logger.info("All queries processed.")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
