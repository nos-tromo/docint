"""Tests for the on-disk topic store."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from docint.core.topics import store as store_module
from docint.core.topics.model import TopicInfo, TopicTable
from docint.core.topics.store import RUN_ID_PATTERN, TopicRun, TopicStore

PHYSICAL = "u0123456789ab__docs"


def _run(run_id: str, owner_created_at: str | None = "2026-01-01T00:00:00+00:00") -> TopicRun:
    """Build a small run.

    Args:
        run_id: The run id.
        owner_created_at: The ownership row's creation time.

    Returns:
        The run.
    """
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(2, 4))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    centres = raw.astype(np.float32).astype(np.float64)
    topics = tuple(
        TopicInfo(
            id=i,
            label=f"t{i}",
            description="d",
            keywords=(("w", 0.5),),
            origin="build",
            added_at="2026-01-01T00:00:00+00:00",
            fit_size=10,
            floor=0.25,
        )
        for i in range(2)
    )
    record = {
        "collection": "docs",
        "created_at": "2026-01-01T00:00:00+00:00",
        "owner_created_at": owner_created_at,
        "dense": {"name": "m", "dim": 4},
        "params": {"min_cluster_size": 5},
    }
    return TopicRun(run_id=run_id, record=record, table=TopicTable(topics=topics, centres=centres))


def _id(store: TopicStore, minute: int = 0) -> str:
    """Mint a run id."""
    return store.new_run_id(datetime(2026, 10, 25, 12, minute, 0, tzinfo=UTC))


def test_new_run_ids_match_the_pattern_and_differ(tmp_path: Path) -> None:
    """New run ids match the pattern and differ."""
    store = TopicStore(tmp_path)
    a, b = _id(store), _id(store)
    assert RUN_ID_PATTERN.fullmatch(a)
    assert a.startswith("20261025-120000-")
    assert a != b


@pytest.mark.parametrize("bad", ["../x", "20261025-000000-abcdef\n", "20261025-000000-ABCDEF"])
def test_unsafe_run_ids_are_refused(tmp_path: Path, bad: str) -> None:
    """Unsafe run ids are refused."""
    store = TopicStore(tmp_path)
    with pytest.raises(ValueError):
        store.load(PHYSICAL, bad)
    with pytest.raises(ValueError):
        store.delete_run(PHYSICAL, bad)
    with pytest.raises(ValueError):
        store.activate(PHYSICAL, bad)
    with pytest.raises(ValueError):
        store.prune(PHYSICAL, keep=bad)


def test_save_then_load_round_trips_the_run_exactly(tmp_path: Path) -> None:
    """Save then load round trips the run exactly."""
    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    loaded = store.load(PHYSICAL, run.run_id)
    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.table.topics == run.table.topics
    assert np.array_equal(loaded.table.centres, run.table.centres)
    assert dict(loaded.record) == {**run.record, "run_id": run.run_id}
    assert "topics" not in loaded.record


def test_centres_are_stored_as_float32_without_pickle(tmp_path: Path) -> None:
    """Centres are stored as float32 without pickle."""
    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    path = tmp_path / PHYSICAL / run.run_id / "centres.npy"
    assert np.load(path, allow_pickle=False).dtype == np.float32


def test_a_failed_write_leaves_the_previous_run_json_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed write leaves the previous run json intact."""
    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    run_json = tmp_path / PHYSICAL / run.run_id / "run.json"
    before = run_json.read_bytes()

    real = os.replace
    calls = {"n": 0}

    def flaky(src: object, dst: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        real(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(store_module.os, "replace", flaky)
    changed = _run(run.run_id, owner_created_at="other")
    with pytest.raises(OSError):
        store.save(PHYSICAL, changed)
    assert run_json.read_bytes() == before
    assert [p.name for p in run_json.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_a_corrupt_run_loads_as_none_and_logs_no_content(tmp_path: Path) -> None:
    """A corrupt run loads as none and logs no content."""
    from loguru import logger

    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    (tmp_path / PHYSICAL / run.run_id / "run.json").write_text("SECRET-CONTENT {", encoding="utf-8")
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        assert store.load(PHYSICAL, run.run_id) is None
    finally:
        logger.remove(sink)
    assert messages
    assert not any("SECRET-CONTENT" in m for m in messages)


def test_a_missing_or_misshapen_run_loads_as_none(tmp_path: Path) -> None:
    """A missing or misshapen run loads as none."""
    store = TopicStore(tmp_path)
    assert store.load(PHYSICAL, _id(store)) is None
    run = _run(_id(store, 1))
    store.save(PHYSICAL, run)
    np.save(tmp_path / PHYSICAL / run.run_id / "centres.npy", np.zeros((2, 3), dtype=np.float32))
    assert store.load(PHYSICAL, run.run_id) is None


def test_a_torn_read_mid_save_is_read_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shape mismatch on the first read is retried once, quietly."""
    from loguru import logger

    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    real = np.load
    calls = {"n": 0}

    def torn(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return np.zeros((3, 4), dtype=np.float32)
        return real(*args, **kwargs)

    sleeps: list[float] = []
    monkeypatch.setattr(store_module.np, "load", torn)
    monkeypatch.setattr(store_module.time, "sleep", sleeps.append)
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        loaded = store.load(PHYSICAL, run.run_id)
    finally:
        logger.remove(sink)
    assert loaded is not None and np.array_equal(loaded.table.centres, run.table.centres)
    assert calls["n"] == 2 and len(sleeps) == 1
    assert messages == []


def test_a_run_resaved_with_appended_rows_loads_the_larger_table(tmp_path: Path) -> None:
    """Appending topics under the same id is a supported save."""
    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    extra = replace(run.table.topics[0], id=2, label="t2")
    grown = TopicRun(run.run_id, run.record, run.table.appended([extra], np.eye(1, 4)))
    store.save(PHYSICAL, grown)
    loaded = store.load(PHYSICAL, run.run_id)
    assert loaded is not None
    assert [t.id for t in loaded.table.topics] == [0, 1, 2]
    assert np.array_equal(loaded.table.centres[:2], run.table.centres)


def test_active_is_none_without_active_json(tmp_path: Path) -> None:
    """Active is none without active json."""
    store = TopicStore(tmp_path)
    assert store.active_id(PHYSICAL) is None
    assert store.active(PHYSICAL, owner_created_at=None) is None


def test_an_active_run_is_returned_and_matches_none_to_none(tmp_path: Path) -> None:
    """An active run is returned and matches none to none."""
    store = TopicStore(tmp_path)
    run = _run(_id(store), owner_created_at=None)
    store.save(PHYSICAL, run)
    store.activate(PHYSICAL, run.run_id)
    assert store.active_id(PHYSICAL) == run.run_id
    active = store.active(PHYSICAL, owner_created_at=None)
    assert active is not None and active.run_id == run.run_id
    assert store.active(PHYSICAL, owner_created_at="2026-02-01T00:00:00+00:00") is None


def test_an_active_run_with_another_owner_created_at_is_absent(tmp_path: Path) -> None:
    """An active run with another owner created at is absent."""
    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    store.activate(PHYSICAL, run.run_id)
    assert store.active(PHYSICAL, owner_created_at="2027-01-01T00:00:00+00:00") is None
    assert (tmp_path / PHYSICAL / run.run_id).is_dir()


def test_discard_stale_deletes_it(tmp_path: Path) -> None:
    """Discard stale deletes it."""
    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    store.activate(PHYSICAL, run.run_id)
    assert store.discard_stale(PHYSICAL, owner_created_at=run.record["owner_created_at"]) is False
    assert store.discard_stale(PHYSICAL, owner_created_at="2027-01-01T00:00:00+00:00") is True
    assert not (tmp_path / PHYSICAL / run.run_id).exists()
    assert store.active_id(PHYSICAL) is None


def test_try_lock_excludes_a_second_holder_in_the_same_process(tmp_path: Path) -> None:
    """Try lock excludes a second holder in the same process."""
    first, second = TopicStore(tmp_path), TopicStore(tmp_path)
    held = first.try_lock(PHYSICAL)
    assert held is not None
    assert second.try_lock(PHYSICAL) is None
    held.release()


def test_the_lock_is_free_again_after_release_and_after_the_with_block(tmp_path: Path) -> None:
    """The lock is free again after release and after the with block."""
    first, second = TopicStore(tmp_path), TopicStore(tmp_path)
    held = first.try_lock(PHYSICAL)
    assert held is not None
    held.release()
    held.release()
    lock = second.try_lock(PHYSICAL)
    assert lock is not None
    with lock:
        assert first.try_lock(PHYSICAL) is None
    again = first.try_lock(PHYSICAL)
    assert again is not None
    again.release()


@pytest.mark.parametrize("name", ["has space", "a:b", "ünï"])
def test_collection_names_that_are_not_plain_are_hashed(tmp_path: Path, name: str) -> None:
    """Collection names that are not plain are hashed."""
    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(name, run)
    dirs = [p.name for p in tmp_path.iterdir()]
    assert len(dirs) == 1
    assert dirs[0].startswith("~") and len(dirs[0]) == 65


def test_paths_cannot_escape_the_root(tmp_path: Path) -> None:
    """Paths cannot escape the root."""
    root = tmp_path / "root"
    root.mkdir()
    store = TopicStore(root)
    run = _run(_id(store))
    for name in ("..", ".", "../escape"):
        store.save(name, run)
    assert not (tmp_path / "escape").exists()
    assert all(p.name.startswith("~") for p in root.iterdir())


def test_prune_keeps_only_the_given_run(tmp_path: Path) -> None:
    """Prune keeps only the given run."""
    store = TopicStore(tmp_path)
    a, b = _run(_id(store, 1)), _run(_id(store, 2))
    store.save(PHYSICAL, a)
    store.save(PHYSICAL, b)
    store.activate(PHYSICAL, b.run_id)
    store.prune(PHYSICAL, keep=b.run_id)
    assert store.load(PHYSICAL, a.run_id) is None
    assert store.load(PHYSICAL, b.run_id) is not None
    assert json.loads((tmp_path / PHYSICAL / "active.json").read_text())["run_id"] == b.run_id


def test_delete_collection_removes_every_run_and_keeps_the_lock(tmp_path: Path) -> None:
    """A deleted collection's lock still excludes a job on a collection created again under its name."""
    first, second = TopicStore(tmp_path), TopicStore(tmp_path)
    run = _run(_id(first))
    held = first.try_lock(PHYSICAL)
    assert held is not None
    first.save(PHYSICAL, run)
    first.activate(PHYSICAL, run.run_id)
    directory = tmp_path / PHYSICAL
    (directory / ".active.json.abc123.tmp").write_bytes(b"{}")
    inode = (directory / ".lock").stat().st_ino

    first.delete_collection(PHYSICAL)

    assert [p.name for p in directory.iterdir()] == [".lock"]
    assert (directory / ".lock").stat().st_ino == inode
    assert second.load(PHYSICAL, run.run_id) is None
    assert second.active_id(PHYSICAL) is None
    assert second.try_lock(PHYSICAL) is None
    held.release()
    again = second.try_lock(PHYSICAL)
    assert again is not None
    again.release()
    first.delete_collection(PHYSICAL)
    assert [p.name for p in directory.iterdir()] == [".lock"]
    first.delete_collection("never-stored")
    assert not (tmp_path / "never-stored").exists()


def test_widen_seen_rewrites_only_the_record_and_never_recreates_a_deleted_run(tmp_path: Path) -> None:
    """Widening seen times touches run.json alone, only ever widens, and leaves a deleted run deleted."""
    store = TopicStore(tmp_path)
    run = _run(_id(store))
    store.save(PHYSICAL, run)
    directory = tmp_path / PHYSICAL / run.run_id
    centres = (directory / "centres.npy").read_bytes()
    jan, feb = "2025-01-01T00:00:00+00:00", "2025-02-01T00:00:00+00:00"

    assert store.widen_seen(PHYSICAL, run.run_id, {0: (jan, feb)}, updated_at="2026-03-01T00:00:00+00:00")

    stored = store.load(PHYSICAL, run.run_id)
    assert stored is not None
    assert [(t.first_seen, t.last_seen) for t in stored.table.topics] == [(jan, feb), (None, None)]
    assert stored.record["updated_at"] == "2026-03-01T00:00:00+00:00"
    assert stored.record["owner_created_at"] == run.record["owner_created_at"]
    assert (directory / "centres.npy").read_bytes() == centres
    before = (directory / "run.json").read_bytes()
    narrower = ("2025-01-10T00:00:00+00:00", "2025-01-20T00:00:00+00:00")
    assert store.widen_seen(PHYSICAL, run.run_id, {0: narrower}, updated_at="2026-04-01T00:00:00+00:00")
    assert (directory / "run.json").read_bytes() == before

    store.delete_run(PHYSICAL, run.run_id)
    assert not store.widen_seen(PHYSICAL, run.run_id, {0: (jan, feb)}, updated_at="2026-05-01T00:00:00+00:00")
    assert not directory.exists()


def test_prune_removes_a_run_even_when_it_is_the_active_one(tmp_path: Path) -> None:
    """Prune ignores the active pointer."""
    store = TopicStore(tmp_path)
    a, b = _run(_id(store, 1)), _run(_id(store, 2))
    store.save(PHYSICAL, a)
    store.save(PHYSICAL, b)
    store.activate(PHYSICAL, a.run_id)
    store.prune(PHYSICAL, keep=b.run_id)
    assert not (tmp_path / PHYSICAL / a.run_id).exists()
    assert (tmp_path / PHYSICAL / "active.json").is_file()
    assert store.active_id(PHYSICAL) == a.run_id
    assert store.active(PHYSICAL, owner_created_at=a.record["owner_created_at"]) is None
