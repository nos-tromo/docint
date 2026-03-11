from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, cast

import pandas as pd
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from loguru import logger

from docint.utils.hashing import compute_file_hash, ensure_file_hash
from docint.utils.mimetype import get_mimetype
from docint.utils.reference_metadata import REFERENCE_METADATA_FIELDS

RowFilter = Callable[[dict], bool]
ORIGINAL_INDEX_COL = "_original_row_index"


@dataclass(frozen=True, slots=True)
class TableSchemaProfile:
    """Declarative profile for exact-match specialized table schemas."""

    style: str
    headers: tuple[str, ...]
    text_col: str
    id_col: str
    reference_mapping: dict[str, str | None]

    @property
    def normalized_headers(self) -> set[str]:
        """Return the normalized header set used for exact matching.

        Returns:
            set[str]: A set of normalized column names for schema matching.
        """
        return {_normalize_column_name(header) for header in self.headers}


def _normalize_column_name(value: Any) -> str:
    """Normalize a column name for exact schema-set matching.

    Args:
        value (Any): The column name to normalize.

    Returns:
        str: The normalized column name.
    """
    return str(value or "").strip().casefold()


@dataclass(slots=True)
class TableReader(BaseReader):
    """Reads tabular data from CSV, TSV, XLSX, or Parquet files.

    Args:
        BaseReader (_type_): _base class for all readers_
        text_cols (list[str] | str | None, optional): Column(s) to use for text content.
            If None, will guess based on column names or content.
            Defaults to None.
        metadata_cols (list[str] | set[str] | str | None, optional): Columns to include in metadata.
            If None, all columns except text_cols will be included.
            If a string, it will be treated as a single column name.
            Defaults to None.
        id_col (str | None, optional): Column to use as document ID.
            If None, no ID will be set.
            Defaults to None.
        combine_with (str, optional): String to join multiple text columns.
            Defaults to "\n".
        row_filter (RowFilter | None, optional): Function to filter rows.
            If None, all rows will be included.
            Defaults to None.
        row_query (str | None, optional): Pandas-style query applied before iterating rows.
            If provided, rows that do not satisfy the expression are excluded.
            Defaults to None.
        limit (int | None, optional): Maximum number of rows to read.
            If None, all rows will be read.
            Defaults to None.
        auto_text_guess (bool, optional): Whether to automatically guess text columns.
            If True, will use a heuristic to find likely text columns.
            If False, requires text_cols to be set.
            Defaults to True.
        encoding (str, optional): Encoding to use for reading files.
            Defaults to "utf-8".
        excel_sheet (str | int | None, optional): Sheet name or index for XLSX files.
            If None, will read the first sheet.
            Defaults to None.
        csv_sep (str | None, optional): Separator for CSV/TSV files.
            If None, TSV files use "\t" and CSV files use automatic delimiter
            detection with a fallback to ",".

    Raises:
        ValueError: If an unsupported file type is provided.

    Returns:
        _type_: A list of Document objects, each containing text and metadata.
    """

    # Accept either a single column name or a list; normalize in __post_init__
    text_cols: list[str] | str | None = None
    metadata_cols: list[str] | set[str] | str | None = None
    id_col: str | None = None
    combine_with: str = "\n"
    row_filter: RowFilter | None = None
    limit: int | None = None
    row_query: str | None = None
    auto_text_guess: bool = True
    encoding: str = "utf-8"
    excel_sheet: str | int | None = None  # for XLSX
    csv_sep: str | None = None  # allow overriding delimiter
    schema_profiles: ClassVar[tuple[TableSchemaProfile, ...]] = (
        TableSchemaProfile(
            style="comments",
            headers=(
                "UUID",
                "Comment ID",
                "Network Object ID",
                "URL",
                "Crawled at",
                "Network",
                "Text Content",
                "Timestamp",
                "Tags",
                "Author ID",
                "Author",
                "Vanity Name",
                "Replies Count",
                "Reactions Count",
                "Parent Comment Text",
                "Parent Comment ID",
                "Posting Text",
                "Posting ID",
            ),
            text_col="Text Content",
            id_col="Comment ID",
            reference_mapping={
                "network": "Network",
                "type": None,
                "timestamp": "Timestamp",
                "author": "Author",
                "author_id": "Author ID",
                "vanity": "Vanity Name",
                "text": "Text Content",
                "text_id": "Comment ID",
            },
        ),
        TableSchemaProfile(
            style="messages",
            headers=(
                "UUID",
                "Chat ID",
                "Sender",
                "Timestamp",
                "Text",
                "Tags",
                "URL",
                "Chat Group",
                "Answers Count",
                "Reply To",
                "Network",
            ),
            text_col="Text",
            id_col="Chat ID",
            reference_mapping={
                "network": "Network",
                "type": None,
                "timestamp": "Timestamp",
                "author": "Sender",
                "author_id": None,
                "vanity": None,
                "text": "Text",
                "text_id": "Chat ID",
            },
        ),
        TableSchemaProfile(
            style="postings",
            headers=(
                "UUID",
                "Posting ID",
                "URL",
                "Date last updated",
                "Timestamp",
                "Timezone",
                "Crawled at",
                "Postings Connections",
                "Network Posting ID",
                "Location",
                "Author ID",
                "Author",
                "Vanity Name",
                "Co-Author",
                "Quoted User",
                "Expected Reactions",
                "Collected Reactions",
                "Expected Comments",
                "Collected Comments",
                "Network",
                "Posted in Group",
                "Task",
                "Text Content",
                "Filename",
                "Tags",
            ),
            text_col="Text Content",
            id_col="Posting ID",
            reference_mapping={
                "network": "Network",
                "type": None,
                "timestamp": "Timestamp",
                "author": "Author",
                "author_id": "Author ID",
                "vanity": "Vanity Name",
                "text": "Text Content",
                "text_id": "Posting ID",
            },
        ),
    )

    def __post_init__(self) -> None:
        """Normalize configuration options."""
        # Normalize config
        if isinstance(self.text_cols, str):
            self.text_cols = [self.text_cols]
        # convert iterables in a future refactor if needed; today we accept str|list only
        if isinstance(self.metadata_cols, str):
            self.metadata_cols = {self.metadata_cols}
        if isinstance(self.metadata_cols, set):
            self.metadata_cols = list(self.metadata_cols)

        if self.combine_with is None:
            self.combine_with = "\n"  # guardrail

    # --- Helpers ---
    def _guess_text_cols(self, df: pd.DataFrame) -> list[str]:
        """Guess the text columns in a DataFrame based on common patterns.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            list[str]: A list of column names that are likely to contain text.
        """
        preferred = [
            "title",
            "headline",
            "text",
            "content",
            "body",
            "description",
            "abstract",
            "notes",
            "message",
            "summary",
        ]
        hits = [c for c in df.columns if c.lower() in preferred]
        if hits:
            order = {n: i for i, n in enumerate(preferred)}
            hits.sort(key=lambda c: order.get(c.lower(), 999))
            return hits
        # fallback: string-like with longer average length
        candidates: list[tuple[str, float]] = []
        for c in df.columns:
            if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
                s = df[c].dropna().astype(str).head(200)
                if not s.empty:
                    avg = s.str.len().mean()
                    if avg >= 12:
                        candidates.append((c, avg))
        candidates.sort(key=lambda x: -x[1])
        return [c for c, _ in candidates[:3]] or [df.columns[0]]

    def _combine_text(self, row: pd.Series, cols: list[str]) -> str:
        """Combine text from specified columns in a DataFrame row.

        Args:
            row (pd.Series): The input DataFrame row.
            cols (list[str]): The list of column names to combine.

        Returns:
            str: The combined text from the specified columns.
        """
        parts: list[str] = []
        for c in cols:
            v = row.get(c, "")
            if pd.isna(v):
                continue
            parts.append(str(v))
        return self.combine_with.join(parts).strip()

    def _detect_csv_separator(self, file_path: Path) -> str:
        """Detect the delimiter used by a CSV file.

        Args:
            file_path (Path): Path to the CSV file.

        Returns:
            str: The detected delimiter, or "," as a safe fallback.
        """
        default_separator = ","
        candidates = [",", ";", "\t", "|"]
        try:
            sample = file_path.read_text(encoding=self.encoding, errors="replace")[
                :8192
            ]
        except OSError:
            return default_separator

        if not sample.strip():
            return default_separator

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
            if dialect.delimiter in candidates:
                return dialect.delimiter
        except csv.Error:
            pass

        counts = {delimiter: sample.count(delimiter) for delimiter in candidates}
        detected = max(counts, key=lambda delimiter: counts[delimiter])
        return detected if counts[detected] > 0 else default_separator

    @classmethod
    def _detect_schema_profile(
        cls, columns: list[str] | pd.Index
    ) -> tuple[TableSchemaProfile | None, dict[str, str]]:
        """Return the matching specialized schema profile for a table, if any.

        Args:
            cls: The TableReader class.
            columns (list[str] | pd.Index): The list of column names to match against known
                schema profiles.

        Returns:
            tuple[TableSchemaProfile | None, dict[str, str]]: A tuple containing the matching
            schema profile (or None if no match is found) and a mapping of normalized column
            names to their original names.
        """
        original_columns = [str(column) for column in columns]
        normalized_map = {
            _normalize_column_name(column): column for column in original_columns
        }
        normalized_headers = set(normalized_map)
        for profile in cls.schema_profiles:
            if normalized_headers == profile.normalized_headers:
                return profile, normalized_map
        return None, normalized_map

    @staticmethod
    def _build_reference_metadata(
        *,
        profile: TableSchemaProfile,
        row_dict: dict[str, Any],
        normalized_map: dict[str, str],
    ) -> dict[str, Any]:
        """Build the stable reference-metadata block for a specialized row.

        Args:
            profile (TableSchemaProfile): The matched schema profile for the table.
            row_dict (dict[str, Any]): The dictionary representation of the current row.
            normalized_map (dict[str, str]): A mapping of normalized column names to their original names

        Returns:
            dict[str, Any]: A dictionary containing the extracted reference metadata fields based on the profile's
        """
        metadata: dict[str, Any] = {}
        for key in REFERENCE_METADATA_FIELDS.keys():
            if key == "type":
                metadata[key] = profile.style.rstrip("s")
                continue
            source_column = profile.reference_mapping.get(key)
            if source_column is None:
                metadata[key] = None
                continue
            original_column = normalized_map.get(_normalize_column_name(source_column))
            metadata[key] = row_dict.get(original_column) if original_column else None
        return metadata

    def load_data(self, file: str | Path, **kwargs) -> list[Document]:
        """Load data from a file into a list of Document objects.

        Args:
            file (str | Path): The path to the file to load.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Document]: A list of Document objects representing the loaded data.

        Raises:
            ValueError: If the file type is unsupported.
        """
        logger.info("[TableReader] Loading table from {}", file)
        file_path = Path(file) if not isinstance(file, Path) else file
        suffix = file_path.suffix.lower()
        extra_info_arg = kwargs.get("extra_info", {})
        extra_info: dict[str, Any] = (
            extra_info_arg if isinstance(extra_info_arg, dict) else {}
        )

        file_hash = extra_info.get("file_hash")
        if file_hash is None:
            file_hash = compute_file_hash(file_path)
        mimetype = get_mimetype(file_path)

        # --- Load to DataFrame ---
        if suffix in {".csv", ".tsv"}:
            if self.csv_sep:
                sep = self.csv_sep
            elif suffix == ".tsv":
                sep = "\t"
            else:
                sep = self._detect_csv_separator(file_path)
            df = pd.read_csv(file_path, sep=sep, encoding=self.encoding)
            ft_extras = {"csv": {"sep": sep}}
        elif suffix in {".xlsx", ".xls"}:
            # If excel_sheet is None, pandas would return a dict of DataFrames, which
            # breaks later when we call reset_index on it. Default to first sheet (0)
            # unless the user explicitly set a sheet name/index.
            effective_sheet = self.excel_sheet if self.excel_sheet is not None else 0
            df = pd.read_excel(file_path, sheet_name=effective_sheet)
            sheet = str(effective_sheet)  # Ensure sheet is a string
            ft_extras = {"excel": {"sheet": sheet}}
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
            ft_extras = {"parquet": {}}
        else:
            logger.error("ValueError: Unsupported table type: {}", suffix)
            raise ValueError(f"Unsupported table type: {suffix}")

        row_query_applied = False
        row_query_error: str | None = None
        if self.row_query:
            try:
                df = df.query(self.row_query)
                row_query_applied = True
            except Exception as exc:  # pragma: no cover - log and continue
                row_query_error = str(exc)
                logger.warning(
                    "Failed to apply row_query='{}' to {}: {}",
                    self.row_query,
                    file_path,
                    exc,
                )

        schema_profile, normalized_columns = self._detect_schema_profile(df.columns)
        df[ORIGINAL_INDEX_COL] = df.index
        df = df.reset_index(drop=True)
        effective_id_col: str | None
        if schema_profile is not None:
            text_cols = [
                normalized_columns[_normalize_column_name(schema_profile.text_col)]
            ]
            effective_id_col = normalized_columns[
                _normalize_column_name(schema_profile.id_col)
            ]
        else:
            if self.text_cols is None:
                text_cols = self._guess_text_cols(df)
            elif isinstance(self.text_cols, str):
                text_cols = [self.text_cols]
            else:
                text_cols = self.text_cols
            effective_id_col = self.id_col
        meta_cols = (
            [c for c in df.columns if c not in set(text_cols)]
            if self.metadata_cols is None
            else [c for c in df.columns if c in self.metadata_cols]
        )
        if ORIGINAL_INDEX_COL in meta_cols:
            meta_cols.remove(ORIGINAL_INDEX_COL)

        docs: list[Document] = []
        n_rows, n_cols = len(df), len(df.columns)
        columns = [c for c in df.columns if c != ORIGINAL_INDEX_COL]
        column_types = {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
            if col != ORIGINAL_INDEX_COL
        }
        count = 0

        for i, row in df.iterrows():
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            original_row_index = row_dict.pop(ORIGINAL_INDEX_COL, None)
            if self.row_filter and not self.row_filter(row_dict):
                continue

            content = self._combine_text(row, text_cols)

            if not content.strip():
                continue

            table_info: dict[str, Any] = {
                "columns": columns,
                "column_types": column_types,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "row_index": i,
                "original_row_index": original_row_index,
                "row_query": self.row_query,
                "row_query_applied": row_query_applied,
                "row_limit": self.limit,
            }
            if row_query_error:
                table_info["row_query_error"] = row_query_error
            if schema_profile is not None:
                table_info["style"] = schema_profile.style

            metadata: dict[str, Any] = {
                "origin": {
                    "filename": file_path.name,
                    "filetype": mimetype or "",
                },
                "source": "table",
                "table": table_info,
                "ft": ft_extras,
            }
            if extra_info:
                metadata.update(extra_info)
            if schema_profile is not None:
                metadata["reference_metadata"] = self._build_reference_metadata(
                    profile=schema_profile,
                    row_dict=cast(dict[str, Any], row_dict),
                    normalized_map=normalized_columns,
                )

            for k in meta_cols:
                metadata[k] = row_dict.get(k, "")

            ensure_file_hash(metadata, file_hash=file_hash)

            # Only set doc_id if present; passing None triggers Pydantic validation in some versions
            if effective_id_col and row_dict.get(effective_id_col) is not None:
                docs.append(
                    Document(
                        text=content,
                        metadata=metadata,
                        doc_id=str(row_dict[effective_id_col]),
                    )
                )
            else:
                docs.append(
                    Document(
                        text=content,
                        metadata=metadata,
                    )
                )
            count += 1
            if self.limit and count >= self.limit:
                break

        return docs
