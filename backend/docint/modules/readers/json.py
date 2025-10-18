import logging
from pathlib import Path

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader

from docint.utils.mimetype import get_mimetype

logger = logging.getLogger(__name__)


class CustomJSONReader(BaseReader):
    """
    Custom JSON reader to handle specific JSON structures.
    """

    def __init__(
        self,
        levels_back: int | None = 0,
        collapse_length: int | None = None,
        ensure_ascii: bool = False,
        is_jsonl: bool = False,
        clean_json: bool = True,
    ) -> None:
        """
        Initializes the CustomJSONReader with specific parameters.

        Args:
            levels_back (int | None, optional): _levels_back description_. Defaults to 0.
            collapse_length (int | None, optional): _collapse_length description_. Defaults to None.
            ensure_ascii (bool, optional): _description_. Defaults to False.
            is_jsonl (bool, optional): _is_jsonl description_. Defaults to False.
            clean_json (bool, optional): _clean_json description_. Defaults to True.
        """
        self.json_reader = JSONReader(
            levels_back=levels_back,
            collapse_length=collapse_length,
            ensure_ascii=ensure_ascii,
            is_jsonl=is_jsonl,
            clean_json=clean_json,
        )

    def load_data(self, file: str | Path, **kwargs) -> list[Document]:
        """
        Load data from a JSON file and return a list of Document objects.

        Args:
            file (str | Path): The path to the JSON file.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: A list of Document objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid JSON or JSONL file.
        """
        file_path = Path(file) if not isinstance(file, Path) else file

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError(
                f"Expected a .json or .jsonl file but got: {file_path.suffix}"
            )

        filename = file_path.name
        mimetype = get_mimetype(file_path)

        logger.info("[CustomJSONReader] Loading JSON file: %s", file_path)

        return self.json_reader.load_data(
            input_file=file_path,
            extra_info={
                "file_path": str(file_path),
                "file_name": filename,
                "filename": filename,
                "file_type": mimetype,
                "mimetype": mimetype,
                "source": "json",
                "origin": {
                    "filename": filename,
                    "mimetype": mimetype,
                },
            },
        )
