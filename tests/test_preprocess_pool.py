"""Tests for the per-file preprocessing pool (``core/ingest/preprocess.py``).

Pins the pool's three rules — one in-flight task per key, join-or-run, and
eviction on completion so a failed task can be retried — plus the router that
decides which heavy stage a staged file gets.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typing_extensions import override

from docint.core.ingest import preprocess
from docint.core.ingest.preprocess import PreprocessPool, preprocess_key, submit_file


@pytest.fixture
def pool() -> Any:
    """A two-worker pool, shut down after the test."""
    instance = PreprocessPool(max_workers=2)
    yield instance
    instance.shutdown()


def test_same_key_submitted_twice_runs_once(pool: PreprocessPool) -> None:
    """A second submit for a key still in flight returns the existing future."""
    release = threading.Event()
    calls: list[int] = []

    def task() -> str:
        calls.append(1)
        release.wait(timeout=5)
        return "done"

    first = pool.submit("k", task)
    second = pool.submit("k", task)
    release.set()

    assert first is second
    assert first.result(timeout=5) == "done"
    assert calls == [1]


def test_run_joins_an_in_flight_task(pool: PreprocessPool) -> None:
    """``run`` waits for the task another caller started instead of running its own."""
    release = threading.Event()

    def slow() -> str:
        release.wait(timeout=5)
        return "from-submit"

    def never() -> str:
        raise AssertionError("run() must join, not re-run")

    pool.submit("k", slow)
    threading.Timer(0.05, release.set).start()

    assert pool.run("k", never) == "from-submit"


def test_completed_key_is_evicted_so_a_later_submit_runs_again(pool: PreprocessPool) -> None:
    """The caches on disk and in Qdrant are the memory; the pool keeps no results."""
    calls: list[int] = []

    def task() -> int:
        calls.append(1)
        return len(calls)

    assert pool.submit("k", task).result(timeout=5) == 1
    assert pool.run("k", task) == 2
    assert calls == [1, 1]


def test_run_reraises_and_evicts_a_failed_task(pool: PreprocessPool) -> None:
    """A failure reaches the caller, and the key is free for a retry."""
    attempts: list[int] = []

    def task() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return "ok"

    with pytest.raises(RuntimeError, match="boom"):
        pool.run("k", task)
    assert pool.run("k", task) == "ok"


def test_submit_failure_is_logged_not_raised(pool: PreprocessPool, loguru_caplog: Any) -> None:
    """A fire-and-forget task that fails leaves one warning behind."""

    def task() -> None:
        raise RuntimeError("upload-time failure")

    future = pool.submit("k", task)
    with pytest.raises(RuntimeError):
        future.result(timeout=5)
    pool.wait_idle(timeout=5)

    assert "upload-time failure" in loguru_caplog.text


def test_key_carries_kind_collection_and_hash() -> None:
    """Keys are namespaced so a PDF and an image of the same bytes never collide."""
    assert preprocess_key("pdf", "u1__docs", "abc") == "pdf:u1__docs:abc"


# ---------------------------------------------------------------------------
# submit_file routing
# ---------------------------------------------------------------------------


class _RecordingPool(PreprocessPool):
    """Pool double that records keys and runs tasks inline."""

    def __init__(self) -> None:
        """Start with no submissions and no executor."""
        self.keys: list[str] = []

    @override
    def submit(self, key: str, fn: Any) -> Any:
        """Record the key and run the task synchronously.

        Args:
            key: The dedupe key the router chose.
            fn: The task closure.

        Returns:
            Whatever the task returned, wrapped like a future.
        """
        self.keys.append(key)
        value = fn()
        return SimpleNamespace(result=lambda timeout=None: value)


@pytest.fixture
def routed(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Path]]:
    """Replace the three task builders with recorders and enable Nextext."""
    seen: dict[str, list[Path]] = {"pdf": [], "image": [], "media": []}
    for kind in seen:
        monkeypatch.setattr(
            preprocess,
            f"preprocess_{kind}",
            lambda path, collection, *, image_service=None, kind=kind: seen[kind].append(path),
        )
    monkeypatch.setattr(preprocess, "load_nextext_env", lambda: SimpleNamespace(enabled=True))
    monkeypatch.setattr(preprocess, "load_ingestion_env", lambda: SimpleNamespace(media_filetypes=[".mp4"]))
    return seen


def test_submit_file_routes_by_extension(tmp_path: Path, routed: dict[str, list[Path]]) -> None:
    """A PDF, an image and a clip each reach their own stage; anything else is skipped."""
    pool = _RecordingPool()
    files = {name: tmp_path / name for name in ("a.pdf", "b.PNG", "c.mp4", "d.txt")}
    for path in files.values():
        path.write_bytes(path.name.encode())

    futures = {name: submit_file(path, "col", pool=pool, file_hash=f"h-{name}") for name, path in files.items()}

    assert futures["d.txt"] is None
    assert routed["pdf"] == [files["a.pdf"]]
    assert routed["image"] == [files["b.PNG"]]
    assert routed["media"] == [files["c.mp4"]]
    assert pool.keys == ["pdf:col:h-a.pdf", "image:col:h-b.PNG", "media:col:h-c.mp4"]


def test_submit_file_hashes_when_not_given(tmp_path: Path, routed: dict[str, list[Path]]) -> None:
    """The upload handler passes the hash it already computed; other callers get it read."""
    path = tmp_path / "a.pdf"
    path.write_bytes(b"pdf")
    pool = _RecordingPool()

    submit_file(path, "col", pool=pool)

    assert pool.keys[0].startswith("pdf:col:") and len(pool.keys[0]) == len("pdf:col:") + 64


def test_submit_file_skips_media_when_nextext_is_off(
    tmp_path: Path, routed: dict[str, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without Nextext a clip has no stage to run; the job warns about it later."""
    monkeypatch.setattr(preprocess, "load_nextext_env", lambda: SimpleNamespace(enabled=False))
    path = tmp_path / "c.mp4"
    path.write_bytes(b"v")

    assert submit_file(path, "col", pool=_RecordingPool(), file_hash="h") is None
    assert routed["media"] == []
