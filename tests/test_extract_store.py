"""Tests for the on-disk store of rendered extracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from docint.core.extract.store import ExtractStore

_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _store(tmp_path: Path) -> ExtractStore:
    return ExtractStore(tmp_path / "extracts")


def _write(store: ExtractStore, physical: str = "u0000__col", *, now: datetime = _NOW, **meta: object) -> str:
    record = store.write(
        physical,
        zip_bytes=b"PK-payload",
        meta={"collection": "col", "target": None, "counts": {}, "pdf_skipped": False, **meta},
        now=now,
    )
    return record["extract_id"]


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    """A stored extract can be listed and its bytes fetched back."""
    store = _store(tmp_path)
    extract_id = _write(store)
    (record,) = store.list("u0000__col")
    assert record["extract_id"] == extract_id
    assert record["size"] == len(b"PK-payload")
    assert record["collection"] == "col"
    assert store.path("u0000__col", extract_id).read_bytes() == b"PK-payload"


def test_filename_names_the_collection_and_the_moment(tmp_path: Path) -> None:
    """The downloaded file must be identifiable in a downloads folder."""
    store = _store(tmp_path)
    _write(store)
    (record,) = store.list("u0000__col")
    assert record["filename"] == "col-extract-20260102-0304.zip"


def test_list_returns_newest_first(tmp_path: Path) -> None:
    """The SPA shows the most recent build at the top."""
    store = _store(tmp_path)
    old = _write(store, now=_NOW)
    new = _write(store, now=_NOW + timedelta(hours=1))
    assert [record["extract_id"] for record in store.list("u0000__col")] == [new, old]


def test_list_is_empty_for_an_unknown_collection(tmp_path: Path) -> None:
    """A collection that was never extracted is empty, not an error."""
    assert _store(tmp_path).list("u0000__other") == []


def test_get_rejects_a_traversing_id(tmp_path: Path) -> None:
    """An extract id reaches the filesystem, so its shape is validated."""
    store = _store(tmp_path)
    _write(store)
    assert store.get("u0000__col", "../../etc/passwd") is None
    assert store.get("u0000__col", "20260102-030405-zzzzzzzz") is None


def test_delete_removes_the_archive_and_its_sidecar(tmp_path: Path) -> None:
    """Deleting an extract leaves nothing behind to list."""
    store = _store(tmp_path)
    extract_id = _write(store)
    assert store.delete("u0000__col", extract_id) is True
    assert store.list("u0000__col") == []
    assert store.delete("u0000__col", extract_id) is False


def test_prune_drops_the_oldest_beyond_the_cap(tmp_path: Path) -> None:
    """A collection keeps a bounded number of builds."""
    store = _store(tmp_path)
    ids = [_write(store, now=_NOW + timedelta(hours=hour)) for hour in range(4)]
    store.prune("u0000__col", retention_days=365, max_per_collection=2, now=_NOW + timedelta(hours=4))
    assert [record["extract_id"] for record in store.list("u0000__col")] == [ids[3], ids[2]]


def test_prune_drops_anything_past_its_retention(tmp_path: Path) -> None:
    """An extract is a convenience copy, not an archive of record."""
    store = _store(tmp_path)
    _write(store, now=_NOW)
    fresh = _write(store, now=_NOW + timedelta(days=10))
    store.prune("u0000__col", retention_days=7, max_per_collection=50, now=_NOW + timedelta(days=10))
    assert [record["extract_id"] for record in store.list("u0000__col")] == [fresh]


def test_an_orphaned_sidecar_does_not_break_the_listing(tmp_path: Path) -> None:
    """A half-written build must not make the whole list unreadable."""
    store = _store(tmp_path)
    extract_id = _write(store)
    store.path("u0000__col", extract_id).unlink()
    (tmp_path / "extracts" / "u0000__col" / "20260102-030405-deadbeef.json").write_text("{not json")
    assert store.list("u0000__col") == []


def test_delete_collection_removes_the_whole_directory(tmp_path: Path) -> None:
    """Extracts share their collection's lifecycle, like the companions do."""
    store = _store(tmp_path)
    _write(store)
    store.delete_collection("u0000__col")
    assert store.list("u0000__col") == []
    assert not (tmp_path / "extracts" / "u0000__col").exists()


def test_a_physical_name_with_separators_is_refused(tmp_path: Path) -> None:
    """The collection name is part of a path, so it is validated too."""
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        _write(store, physical="../escape")


def test_sidecar_records_what_the_build_produced(tmp_path: Path) -> None:
    """Counts and the skipped-PDF flag ride the record the SPA renders."""
    store = _store(tmp_path)
    extract_id = _write(store, counts={"documents": 2, "figures": 5}, pdf_skipped=True, target="abc")
    sidecar = json.loads((tmp_path / "extracts" / "u0000__col" / f"{extract_id}.json").read_text())
    assert sidecar["counts"] == {"documents": 2, "figures": 5}
    assert sidecar["pdf_skipped"] is True
    assert sidecar["target"] == "abc"
