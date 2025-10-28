from pathlib import Path

import pandas as pd
from docint.core.readers.tables import TableReader, basic_clean
from docint.utils.hashing import compute_file_hash


def test_basic_clean_collapse_whitespace() -> None:
    raw = "   Hello   \r\n\rWorld   \n\n\n"
    assert basic_clean(raw) == "Hello\nWorld"


def test_table_reader_loads_csv_with_selected_metadata(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Alpha", "Beta"],
            "body": [" Alpha body  ", "\nBeta body\n"],
            "author": ["Alice", "Bob"],
            "extra": [10, 20],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    reader = TableReader(
        text_cols=["title", "body"],
        metadata_cols={"author"},
        id_col="id",
    )
    docs = reader.load_data(csv_path)

    assert [doc.doc_id for doc in docs] == ["1", "2"]

    first = docs[0]
    expected_text = basic_clean("Alpha\n Alpha body  ")
    assert first.text == expected_text
    assert first.metadata["author"] == "Alice"
    assert first.metadata["table"]["n_rows"] == 2
    assert first.metadata["table"]["row_index"] == 0
    assert first.metadata["origin"]["filename"] == "sample.csv"
    assert first.metadata["source"] == "table"
    expected_hash = compute_file_hash(csv_path)
    assert first.metadata["file_hash"] == expected_hash
    assert first.metadata["origin"]["file_hash"] == expected_hash
    assert "extra" not in first.metadata


def test_table_reader_limit_and_row_filter(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "text": ["keep", "skip", "also keep"],
            "flag": [True, False, True],
        }
    )
    tsv_path = tmp_path / "sample.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    def only_true(row: dict) -> bool:
        return bool(row["flag"])

    reader = TableReader(limit=1, row_filter=only_true)
    docs = reader.load_data(tsv_path)

    assert len(docs) == 1
    only = docs[0]
    assert only.text == "keep"
    assert only.metadata["flag"] is True
    assert only.metadata["table"]["row_index"] == 0
    assert only.metadata["origin"]["filetype"] == "text/tab-separated-values"
    expected_hash = compute_file_hash(tsv_path)
    assert only.metadata["file_hash"] == expected_hash
    assert only.metadata["origin"]["file_hash"] == expected_hash
