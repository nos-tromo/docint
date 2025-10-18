from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

from docint.utils.mimetype import get_mimetype

PathLike = str | Path
CleanFn = Callable[[str], str]
RowFilter = Callable[[dict], bool]


def basic_clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


@dataclass(slots=True)  # add kw_only=True if you like
class TableReader(BaseReader):
    """
    Reads tabular data from CSV, TSV, XLSX, or Parquet files.

    Args:
        BaseReader (_type_): _base class for all readers_
        text_cols (list[str] | str | None, optional): Column(s) to use for text content.
            If None, will guess based on column names or content.
            Defaults to None.
        metadata_cols (set[str] | str | None, optional): Columns to include in metadata.
            If None, all columns except text_cols will be included.
            If a string, it will be treated as a single column name.
            Defaults to None.
        id_col (str | None, optional): Column to use as document ID.
            If None, no ID will be set.
            Defaults to None.
        combine_with (str, optional): String to join multiple text columns.
            Defaults to "\n".
        clean_fn (CleanFn | None, optional): Function to clean text content.
            If None, basic cleaning will be applied.
            Defaults to basic_clean.
        row_filter (RowFilter | None, optional): Function to filter rows.
            If None, all rows will be included.
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
            If None, will use "," for CSV and "\t" for TSV.

    Raises:
        ValueError: If an unsupported file type is provided.

    Returns:
        _type_: A list of Document objects, each containing text and metadata.
    """

    # Accept either a single column name or a list; normalize in __post_init__
    text_cols: list[str] | str | None = None
    metadata_cols: set[str] | str | None = None
    id_col: str | None = None
    combine_with: str = "\n"
    clean_fn: CleanFn | None = basic_clean
    row_filter: RowFilter | None = None
    limit: int | None = None
    auto_text_guess: bool = True
    encoding: str = "utf-8"
    excel_sheet: str | int | None = None  # for XLSX
    csv_sep: str | None = None  # allow overriding delimiter

    def __post_init__(self) -> None:
        # Normalize config
        if isinstance(self.text_cols, str):
            self.text_cols = [self.text_cols]
        # convert iterables in a future refactor if needed; today we accept str|list only
        if isinstance(self.metadata_cols, str):
            self.metadata_cols = {self.metadata_cols}

        if self.combine_with is None:
            self.combine_with = "\n"  # guardrail

    # ---- helpers
    def _guess_text_cols(self, df: pd.DataFrame) -> list[str]:
        """
        Guess the text columns in a DataFrame based on common patterns.

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
        """
        Combine text from specified columns in a DataFrame row.

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

    def load_data(self, file: Path, extra_info: dict | None = None) -> list[Document]:
        """
        Load data from a file into a list of Document objects.

        Args:
            file (Path): The path to the file to load.
            extra_info (dict | None, optional): Additional information to include in the Document metadata. Defaults to None.

        Raises:
            ValueError: If the file type is unsupported.

        Returns:
            list[Document]: A list of Document objects representing the loaded data.
        """        
        file = Path(file)
        suffix = file.suffix.lower()

        # ---- Load to DataFrame
        if suffix in {".csv", ".tsv"}:
            sep = self.csv_sep if self.csv_sep else ("\t" if suffix == ".tsv" else ",")
            df = pd.read_csv(file, sep=sep, encoding=self.encoding)
            ft_extras = {"csv": {"sep": sep}}
            mimetype = "text/tab-separated-values" if sep == "\t" else "text/csv"
        elif suffix in {".xlsx", ".xls"}:
            # If excel_sheet is None, pandas would return a dict of DataFrames, which
            # breaks later when we call reset_index on it. Default to first sheet (0)
            # unless the user explicitly set a sheet name/index.
            effective_sheet = self.excel_sheet if self.excel_sheet is not None else 0
            df = pd.read_excel(file, sheet_name=effective_sheet)
            sheet = effective_sheet
            ft_extras = {"excel": {"sheet": sheet}}
            mimetype = (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif suffix == ".parquet":
            df = pd.read_parquet(file)
            ft_extras = {"parquet": {}}
            mimetype = "application/x-parquet"
        else:
            raise ValueError(f"Unsupported table type: {suffix}")

        df = df.reset_index(drop=True)
        text_cols = self.text_cols or self._guess_text_cols(df)
        meta_cols = (
            [c for c in df.columns if c not in set(text_cols)]
            if self.metadata_cols is None
            else [c for c in df.columns if c in self.metadata_cols]
        )

        docs: list[Document] = []
        n_rows, n_cols = len(df), len(df.columns)
        columns = list(df.columns)
        count = 0

        for i, row in df.iterrows():
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            if self.row_filter and not self.row_filter(row_dict):
                continue

            content = self._combine_text(row, text_cols)
            if self.clean_fn:
                content = self.clean_fn(content)
            if not content.strip():
                continue

            metadata = {
                "origin": {
                    "filename": file.name,
                    "filetype": mimetype or (mimetypes.guess_type(file.name)[0] or ""),
                },
                "source": "table",
                "table": {
                    "columns": columns,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "row_index": i,
                },
                "ft": ft_extras,
            }
            if extra_info:
                metadata.update(extra_info)

            for k in meta_cols:
                metadata[k] = row_dict.get(k)

            # Only set doc_id if present; passing None triggers Pydantic validation in some versions
            if self.id_col and row_dict.get(self.id_col) is not None:
                docs.append(
                    Document(
                        text=content,
                        metadata=metadata,
                        doc_id=str(row_dict[self.id_col]),
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
