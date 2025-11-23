import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from loguru import logger

from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype


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
        schema_sample_size: int = 200,
        list_sample_size: int = 50,
    ) -> None:
        """
        Initializes the CustomJSONReader with specific parameters.

        Args:
            levels_back (int | None, optional): The number of levels to go back in the JSON structure. Defaults to 0.
            collapse_length (int | None, optional): The maximum length of collapsed text. Defaults to None.
            ensure_ascii (bool, optional): Whether to ensure ASCII encoding. Defaults to False.
            is_jsonl (bool, optional): Whether the input is in JSONL format. Defaults to False.
            clean_json (bool, optional): Whether to clean the JSON data. Defaults to True.
        """
        self.json_reader = JSONReader(
            levels_back=levels_back,
            collapse_length=collapse_length,
            ensure_ascii=ensure_ascii,
            is_jsonl=is_jsonl,
            clean_json=clean_json,
        )
        self.is_jsonl = is_jsonl
        self.schema_sample_size = max(schema_sample_size, 0)
        self.list_sample_size = max(list_sample_size, 0)

    def _sample_list_items(self, values: Sequence[Any]) -> Iterable[Any]:
        """
        Samples a subset of items from a list.

        Args:
            values (Sequence[Any]): The list of values to sample from.

        Returns:
            Iterable[Any]: A sampled subset of the input list.
        """
        if self.list_sample_size == 0:
            return []
        total = len(values)
        if total <= self.list_sample_size:
            return values
        if self.list_sample_size == 1:
            return [values[0]]
        step = total / float(self.list_sample_size)
        sampled: list[Any] = []
        position = 0.0
        for _ in range(self.list_sample_size):
            sampled.append(values[int(position)])
            position += step
            if position >= total:
                break
        return sampled

    def _collect_nested_keys(self, data: Any, prefix: str = "") -> set[str]:
        """
        Collects nested keys from a JSON-like structure.

        Args:
            data (Any): The JSON-like data to inspect.
            prefix (str, optional): The prefix for nested keys. Defaults to "".

        Returns:
            set[str]: A set of all nested keys found in the data.
        """
        keys: set[str] = set()
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                keys.add(path)
                keys.update(self._collect_nested_keys(value, path))
        elif isinstance(data, list):
            for item in self._sample_list_items(data):
                keys.update(self._collect_nested_keys(item, prefix))
        return keys

    def _infer_schema(self, file_path: Path, is_jsonl: bool) -> dict[str, list[str]]:
        """
        Infers the schema of the JSON data by collecting nested keys.

        Args:
            file_path (Path): The path to the JSON file.
            is_jsonl (bool): Whether the file is in JSONL format.

        Returns:
            dict[str, list[str]]: A dictionary containing inferred schema information.

        Raises:
            OSError: If there is an error reading the file.
            json.JSONDecodeError: If the file content is not valid JSON.
        """
        nested_keys: set[str] = set()
        try:
            if is_jsonl:
                rng = random.Random(file_path.stat().st_size)
                reservoir: list[Any] = []
                with file_path.open("r", encoding="utf-8") as handle:
                    for idx, line in enumerate(handle):
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if self.schema_sample_size <= 0:
                            break
                        if len(reservoir) < self.schema_sample_size:
                            reservoir.append(payload)
                        else:
                            j = rng.randint(0, idx)
                            if j < self.schema_sample_size:
                                reservoir[j] = payload
                for sample in reservoir:
                    nested_keys.update(self._collect_nested_keys(sample))
            else:
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    nested_keys.update(self._collect_nested_keys(payload))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unable to infer JSON schema for %s: %s", file_path, exc)
        return {"nested_keys": sorted(nested_keys)}

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
        provided_info = kwargs.get("extra_info", {})
        file_hash = (
            provided_info.get("file_hash") if isinstance(provided_info, dict) else None
        )
        if file_hash is None:
            file_hash = compute_file_hash(file_path)

        if not file_path.exists():
            logger.error("FileNotFoundError: File not found: {}", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in {".json", ".jsonl"}:
            logger.error(
                "ValueError: Expected a .json or .jsonl file but got: {}",
                file_path.suffix,
            )
            raise ValueError(
                f"Expected a .json or .jsonl file but got: {file_path.suffix}"
            )

        file_path_str = str(file_path)
        filename = file_path.name
        mimetype = get_mimetype(file_path)

        logger.info("[CustomJSONReader] Loading JSON file: {}", file_path_str)

        schema_info = self._infer_schema(file_path, self.is_jsonl)

        extra_info = {
            "file_path": file_path_str,
            "file_name": filename,
            "filename": filename,
            "file_type": mimetype,
            "mimetype": mimetype,
            "source": "json",
            "origin": {
                "filename": filename,
                "mimetype": mimetype,
            },
            "schema": schema_info,
        }
        if isinstance(provided_info, dict):
            extra_info.update(provided_info)
        ensure_file_hash(extra_info, file_hash=file_hash, path=file_path)

        return self.json_reader.load_data(
            input_file=file_path_str,
            extra_info=extra_info,
        )
