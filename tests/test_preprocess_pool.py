"""Tests for the per-file preprocessing pool (``core/ingest/preprocess.py``).

Pins the pool's three rules — one in-flight task per key, join-or-run, and
eviction on completion so a failed task can be retried — plus the router that
decides which heavy stage a staged file gets.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typing_extensions import override

from docint.core.ingest import preprocess
from docint.core.ingest.preprocess import (
    PreprocessPool,
    collection_of_key,
    preprocess_key,
    submit_file,
)


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


def test_join_waits_for_an_in_flight_task_and_swallows_its_failure(pool: PreprocessPool) -> None:
    """A caller about to read a cache waits for whoever is filling it; a miss is theirs to handle."""
    release = threading.Event()
    finished: list[int] = []

    def slow() -> None:
        release.wait(timeout=5)
        finished.append(1)
        raise RuntimeError("cache stays cold")

    pool.submit("k", slow)
    threading.Timer(0.05, release.set).start()

    pool.join("k")
    assert finished == [1]
    pool.join("never-submitted")


def test_run_all_never_runs_a_finished_task_twice(pool: PreprocessPool) -> None:
    """A fast task evicted before its turn is read from its future, not re-keyed and re-run."""
    calls: list[str] = []

    def task(name: str) -> Any:
        return lambda: calls.append(name) or name

    results = preprocess.run_all(pool, [("a", task("a")), ("b", task("b"))])
    pool.wait_idle(timeout=5)

    assert results == ["a", "b"]
    assert sorted(calls) == ["a", "b"]


def test_run_all_reports_each_job_as_it_finishes(pool: PreprocessPool) -> None:
    """A stage that runs for hours has to be able to say how far it has got."""
    seen: list[tuple[int, int]] = []

    results = preprocess.run_all(
        pool,
        [(name, lambda name=name: name) for name in ("a", "b", "c")],
        on_done=lambda done, total: seen.append((done, total)),
    )

    assert results == ["a", "b", "c"]
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_run_all_reports_inline_jobs_too(pool: PreprocessPool) -> None:
    """A task already on a pool worker runs its own jobs inline, and still reports them."""
    seen: list[tuple[int, int]] = []

    preprocess.run_all(None, [("a", lambda: 1), ("b", lambda: 2)], on_done=lambda d, t: seen.append((d, t)))

    assert seen == [(1, 2), (2, 2)]


def test_cancel_collection_drops_the_queue_but_not_the_running_task() -> None:
    """What an aborted run reclaims: nothing can take back a call already in flight."""
    pool = PreprocessPool(max_workers=1, media_workers=1)
    release = threading.Event()
    ran: list[str] = []
    try:
        pool.submit("image#c#1", lambda: ran.append("first") or release.wait(timeout=5))
        while not pool._futures["image#c#1"].running():
            time.sleep(0.01)
        queued = pool.submit("image#c#2", lambda: ran.append("second"))
        other = pool.submit("image#other#3", lambda: ran.append("other"))

        assert pool.cancel_collection("c") == 1
        assert queued.cancelled()
        assert not other.cancelled()
        release.set()
        pool.wait_idle(timeout=5)
    finally:
        pool.shutdown()

    assert "second" not in ran
    assert "first" in ran


def test_a_cancelled_task_is_reported_as_a_cancelled_job() -> None:
    """`CancelledError` is a BaseException, so every `except Exception` on the way out misses it."""
    from docint.core.jobs import JobCancelled

    pool = PreprocessPool(max_workers=1)
    release = threading.Event()
    try:
        pool.submit("image#c#1", lambda: release.wait(timeout=5))
        while not pool._futures["image#c#1"].running():
            time.sleep(0.01)
        future = pool.submit("image#c#2", lambda: None)
        assert pool.cancel_collection("c") == 1

        with pytest.raises(JobCancelled):
            preprocess.join_future(future)
    finally:
        release.set()
        pool.shutdown()


def test_a_clip_is_not_queued_behind_the_image_backlog() -> None:
    """Media keys run on their own executor, so a blocked image queue never starves Nextext."""
    pool = PreprocessPool(max_workers=1, media_workers=1)
    release = threading.Event()
    try:
        pool.submit("image#c#1", release.wait)
        pool.submit("image#c#2", release.wait)  # queued behind the first on the only vision worker

        assert pool.submit("media#c#3", lambda: "transcribed").result(timeout=5) == "transcribed"
    finally:
        release.set()
        pool.shutdown()


def test_key_carries_kind_collection_and_hash() -> None:
    """Keys are namespaced so a PDF and an image of the same bytes never collide."""
    assert preprocess_key("pdf", "u1__docs", "abc") == "pdf#u1__docs#abc"


def test_a_key_names_its_collection_even_when_the_name_carries_a_colon() -> None:
    """The separator has to be one no collection name can contain, or the count is wrong.

    A colon is legal in a collection name and the linker puts one inside its
    own last key component, so ``#`` — which ``validate_collection_name``
    refuses — is what makes a key parseable.
    """
    assert collection_of_key(preprocess_key("pdf", "u1__a:b", "abc")) == "u1__a:b"
    assert collection_of_key(preprocess_key("image-link", "u1__docs", "abc:posting-1")) == "u1__docs"
    assert collection_of_key("not-a-pool-key") is None


def test_inflight_counts_only_the_collection_asked_about(pool: PreprocessPool) -> None:
    """The staged card is per-collection, so one owner's work must never show on another's."""
    release = threading.Event()
    pool.submit(preprocess_key("image", "mine", "1"), release.wait)
    pool.submit(preprocess_key("image", "mine", "2"), release.wait)
    pool.submit(preprocess_key("image", "theirs", "3"), release.wait)
    try:
        # Two workers, so one of "mine" runs and one waits.
        running, queued = pool.inflight("mine")
        assert running + queued == 2
        assert pool.inflight("nobody") == (0, 0)
    finally:
        release.set()
    pool.wait_idle(timeout=5)
    assert pool.inflight("mine") == (0, 0)


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
    assert pool.keys == ["pdf#col#h-a.pdf", "image#col#h-b.PNG", "media#col#h-c.mp4"]


def test_submit_file_hashes_when_not_given(tmp_path: Path, routed: dict[str, list[Path]]) -> None:
    """The upload handler passes the hash it already computed; other callers get it read."""
    path = tmp_path / "a.pdf"
    path.write_bytes(b"pdf")
    pool = _RecordingPool()

    submit_file(path, "col", pool=pool)

    assert pool.keys[0].startswith("pdf#col#") and len(pool.keys[0]) == len("pdf#col#") + 64


def test_prefetch_batch_submits_every_heavy_file_the_collection_lacks(
    tmp_path: Path, routed: dict[str, list[Path]]
) -> None:
    """A PDF already ingested is skipped; a table is never even hashed; the rest is submitted."""
    from docint.utils.hashing import compute_file_hash

    files = {name: tmp_path / name for name in ("a.pdf", "b.png", "c.mp4", "d.csv")}
    for path in files.values():
        path.write_bytes(path.name.encode())
    pool = _RecordingPool()

    submitted = preprocess.prefetch_batch(tmp_path, "col", skip_hashes={compute_file_hash(files["a.pdf"])}, pool=pool)

    assert submitted == 2
    assert routed["pdf"] == []
    assert routed["image"] == [files["b.png"]]
    assert routed["media"] == [files["c.mp4"]]


def test_the_shared_pool_is_rebuilt_after_shutdown() -> None:
    """The lifespan stops the pool; the next caller must get a working one, not a stopped one."""
    first = preprocess.get_preprocess_pool()
    assert preprocess.get_preprocess_pool() is first

    preprocess.shutdown_preprocess_pool()
    second = preprocess.get_preprocess_pool()
    try:
        assert second is not first
        assert second.run("k", lambda: 1) == 1
    finally:
        preprocess.shutdown_preprocess_pool()


def test_submit_file_skips_media_when_nextext_is_off(
    tmp_path: Path, routed: dict[str, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without Nextext a clip has no stage to run; the job warns about it later."""
    monkeypatch.setattr(preprocess, "load_nextext_env", lambda: SimpleNamespace(enabled=False))
    path = tmp_path / "c.mp4"
    path.write_bytes(b"v")

    assert submit_file(path, "col", pool=_RecordingPool(), file_hash="h") is None
    assert routed["media"] == []
