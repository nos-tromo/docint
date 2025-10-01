import json
import logging
from pathlib import Path
from time import time

from wizard.modules.rag import RAG
from wizard.utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _get_col_name() -> str:
    return input("Enter collection name: ")


def _store_output(filename: str, data: dict | list, out_dir: str | Path = "results") -> None:
    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
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


def rag_session() -> RAG:
    logger.info("Initializing RAG session...")
    rag = RAG(qdrant_collection=_get_col_name())
    rag.create_index()
    rag.create_query_engine()
    rag.start_session()
    return rag


def load_queries(q_path: Path = Path("queries.txt")) -> list[str]:
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
    logger.info("Running query %d: %s", index, query)
    result = rag.chat(query)
    timestamp = str(int(time()))
    _store_output(f"{timestamp}_{index}_result", result)


def main() -> None:
    rag = rag_session()
    queries = load_queries()
    for index, query in enumerate(queries, start=1):
        run_query(rag, query, index)
    logger.info("All queries processed.")


if __name__ == "__main__":
    main()
