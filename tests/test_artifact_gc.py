"""Tests for finding and removing a deleted collection's PDF pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docint.core.storage.artifact_gc import (
    collection_artifacts,
    recorded_sources,
    remove_artifacts,
    still_referenced,
)


def _artifact(root: Path, file_hash: str, source: Path | None) -> Path:
    """One artifact directory whose manifest names ``source``."""
    directory = root / file_hash
    (directory / "pages").mkdir(parents=True)
    manifest = {"doc_id": file_hash, "file_path": str(source)} if source is not None else {"doc_id": file_hash}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_recorded_sources_reads_each_manifests_source_file(tmp_path: Path) -> None:
    """A directory without a readable manifest names no source — it is not guessed at."""
    root = tmp_path / "artifacts"
    _artifact(root, "aaa", tmp_path / "src" / "c1" / "akte.pdf")
    _artifact(root, "bbb", None)
    (root / "ccc").mkdir()
    (root / "ddd").mkdir()
    (root / "ddd" / "manifest.json").write_text("{not json")

    assert recorded_sources(root) == {"aaa": str(tmp_path / "src" / "c1" / "akte.pdf")}
    assert recorded_sources(tmp_path / "no-such-root") == {}


def test_a_collection_accounts_for_its_manifest_rows_and_its_own_files(tmp_path: Path) -> None:
    """An upload preprocessed but never ingested has no manifest row, only a source path."""
    source_dir = tmp_path / "src" / "c1"
    recorded = {
        "staged": str(source_dir / "neu.pdf"),
        "nested": str(source_dir / "ordner" / "alt.pdf"),
        "elsewhere": str(tmp_path / "src" / "c2" / "fremd.pdf"),
        "prefix-lookalike": str(tmp_path / "src" / "c10" / "fremd.pdf"),
    }

    assert collection_artifacts(source_dir, {"ingested"}, recorded) == {"ingested", "staged", "nested"}


def test_anything_another_collection_or_a_worker_uses_is_kept(tmp_path: Path) -> None:
    """Shared by hash, so a directory goes only when nothing else refers to it."""
    other = tmp_path / "src" / "c2"
    recorded = {"theirs": str(other / "fremd.pdf"), "orphan": str(tmp_path / "src" / "gone" / "x.pdf")}

    keep = still_referenced(
        recorded=recorded, other_source_dirs=[other], other_manifest_hashes={"shared"}, inflight={"reading"}
    )

    assert keep == {"theirs", "shared", "reading"}


def test_remove_artifacts_removes_only_what_it_is_given(tmp_path: Path) -> None:
    """Removal counts what actually went."""
    root = tmp_path / "artifacts"
    doomed = _artifact(root, "aaa", None)
    kept = _artifact(root, "bbb", None)

    assert remove_artifacts(root, {"aaa", "not-there"}) == 1

    assert not doomed.exists()
    assert kept.exists()


@pytest.mark.parametrize("name", ["..", "../outside", "a/b", "", "."])
def test_remove_artifacts_never_leaves_the_root(tmp_path: Path, name: str) -> None:
    """A hash is a directory name, never a path."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (tmp_path / "outside").mkdir()

    assert remove_artifacts(root, {name}) == 0

    assert root.exists()
    assert (tmp_path / "outside").exists()
