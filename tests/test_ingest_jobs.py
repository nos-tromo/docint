"""Unit tests for the in-memory ingest job registry."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from docint.core.jobs import IngestJobManager, IngestJobState, JobStatus, format_sse


def _state() -> IngestJobState:
    return IngestJobState(
        job_id="job-1",
        owner="alice",
        logical_name="mydocs",
        physical="u000000000000__mydocs",
        batch_dir=Path("/nonexistent/batch"),
        hybrid=True,
        ner=None,
        hate_speech=None,
        resolve=False,
    )


def _events(frames: list[str]) -> list[str]:
    return [line[len("event: ") :] for f in frames for line in f.splitlines() if line.startswith("event: ")]


def test_history_keeps_only_the_latest_progress_frame() -> None:
    """Test that history collapses progress frames to the latest only."""
    state = _state()
    state.record("ingestion_started", format_sse("ingestion_started", {"collection": "mydocs"}))
    for i in range(1, 501):
        state.record("ingestion_progress", format_sse("ingestion_progress", {"message": f"chunk {i}"}))

    assert _events(state.history()) == ["ingestion_started", "ingestion_progress"]
    assert "chunk 500" in state.history()[-1]


def test_history_keeps_every_warning() -> None:
    """Test that all warning frames are retained in history."""
    state = _state()
    state.record("ingestion_started", format_sse("ingestion_started", {"collection": "mydocs"}))
    state.record("warning", format_sse("warning", {"message": "first"}))
    state.record("ingestion_progress", format_sse("ingestion_progress", {"message": "working"}))
    state.record("warning", format_sse("warning", {"message": "second"}))

    assert _events(state.history()) == [
        "ingestion_started",
        "warning",
        "warning",
        "ingestion_progress",
    ]


def test_history_ends_with_the_terminal_frame() -> None:
    """Test that terminal frames always appear at the end of history."""
    state = _state()
    state.record("ingestion_started", format_sse("ingestion_started", {"collection": "mydocs"}))
    state.record("ingestion_progress", format_sse("ingestion_progress", {"message": "working"}))
    state.record("ingestion_complete", format_sse("ingestion_complete", {"collection": "mydocs"}))

    assert _events(state.history())[-1] == "ingestion_complete"


def test_record_tracks_latest_message() -> None:
    """Test that record method extracts and updates message from frames."""
    state = _state()
    state.record("ingestion_progress", format_sse("ingestion_progress", {"message": "working"}))
    assert state.message == "working"


def test_snapshot_is_json_serializable() -> None:
    """Test that snapshot produces JSON-serializable output without physical name."""
    state = _state()
    state.status = JobStatus.RUNNING
    payload = json.loads(json.dumps(state.snapshot()))
    assert payload["job_id"] == "job-1"
    assert payload["collection"] == "mydocs"
    assert payload["status"] == "running"
    # The physical (owner-namespaced) name is internal and must never be echoed.
    assert "u000000000000__mydocs" not in json.dumps(payload)


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only (trio is not a dependency)."""
    return "asyncio"


def _noop_runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
    push("ingestion_progress", {"message": "working"})
    return {"empty": False, "resolution": None}


async def _create(manager: IngestJobManager, *, owner: str = "alice", physical: str = "p1") -> IngestJobState:
    return await manager.create(
        owner=owner,
        logical_name="mydocs",
        physical=physical,
        batch_dir=Path("/nonexistent/batch"),
        hybrid=True,
        ner=None,
        hate_speech=None,
        resolve=False,
    )


async def _drain(manager: IngestJobManager, state: IngestJobState) -> None:
    """Wait for a job to reach a terminal status."""
    for _ in range(200):
        if state.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job did not finish; status={state.status}")


@pytest.mark.anyio
async def test_worker_runs_to_completion_with_no_subscriber() -> None:
    """The bug this replaces: work was abandoned when nobody was attached."""
    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager)
    await _drain(manager, state)

    assert state.status is JobStatus.COMPLETED
    assert state.message == "working"
    assert state.finished_at is not None
    await manager.stop()


@pytest.mark.anyio
async def test_resolution_summary_is_recorded() -> None:
    """Test that a runner's entity-resolution summary lands on the job state."""

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        return {"empty": False, "resolution": {"processed": 3, "minted": 1}}

    manager = IngestJobManager(runner=runner)
    state = await _create(manager)
    await _drain(manager, state)

    assert state.resolution == {"processed": 3, "minted": 1}
    await manager.stop()


@pytest.mark.anyio
async def test_failed_runner_marks_job_failed_without_leaking_detail() -> None:
    """Test that a raised exception fails the job with static copy only."""

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        raise RuntimeError("connection string user:hunter2@host")

    manager = IngestJobManager(runner=runner)
    state = await _create(manager)
    await _drain(manager, state)

    assert state.status is JobStatus.FAILED
    assert state.error == "Ingestion failed."
    assert "hunter2" not in json.dumps(state.snapshot())
    await manager.stop()


@pytest.mark.anyio
async def test_get_is_owner_scoped() -> None:
    """Test that get() returns a job for its owner and None for anyone else."""
    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager, owner="alice")
    await _drain(manager, state)

    assert await manager.get(state.job_id, owner="alice") is state
    assert await manager.get(state.job_id, owner="bob") is None
    await manager.stop()


@pytest.mark.anyio
async def test_active_for_finds_only_unfinished_jobs() -> None:
    """Test that active_for() only matches queued/running jobs, not finished ones."""
    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager, physical="p1")

    assert await manager.active_for("alice", "p1") is not None
    await _drain(manager, state)
    assert await manager.active_for("alice", "p1") is None
    await manager.stop()


@pytest.mark.anyio
async def test_subscribe_owner_replays_history_then_live_tails() -> None:
    """Test that a late subscriber gets the replay, then live frames, on one stream."""
    gate = asyncio.Event()

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        push("ingestion_progress", {"message": "step one"})
        asyncio.run_coroutine_threadsafe(_wait(gate), state_loop).result(timeout=5)
        push("ingestion_progress", {"message": "step two"})
        return {"empty": False, "resolution": None}

    async def _wait(event: asyncio.Event) -> None:
        await event.wait()

    state_loop = asyncio.get_running_loop()
    manager = IngestJobManager(runner=runner)
    await _create(manager)

    # Let the first push land, then attach — the replay must carry it. The
    # replay also carries the (separate, always-first) ingestion_started
    # frame ahead of it — see IngestJobState.history() — so drain until
    # "step one" appears rather than asserting on the very first frame.
    await asyncio.sleep(0.1)
    stream = manager.subscribe_owner("alice")
    replayed: list[str] = []
    while "step one" not in "".join(replayed):
        replayed.append(await asyncio.wait_for(stream.__anext__(), timeout=5))

    gate.set()
    seen: list[str] = []
    while "ingestion_complete" not in "".join(seen):
        seen.append(await asyncio.wait_for(stream.__anext__(), timeout=5))
    assert any("step two" in frame for frame in seen)

    await stream.aclose()
    await manager.stop()


@pytest.mark.anyio
async def test_concurrency_of_one_serializes_jobs() -> None:
    """Test that concurrency=1 never runs two jobs' runners overlapping."""
    running = 0
    peak = 0

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        import time

        time.sleep(0.05)
        running -= 1
        return {"empty": False, "resolution": None}

    manager = IngestJobManager(runner=runner, concurrency=1)
    states = [await _create(manager, physical=f"p{i}") for i in range(3)]
    for state in states:
        await _drain(manager, state)

    assert peak == 1
    await manager.stop()


@pytest.mark.anyio
async def test_remove_is_owner_scoped() -> None:
    """Test that remove() refuses a non-owner and drops the job for its owner."""
    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager, owner="alice")
    await _drain(manager, state)

    assert await manager.remove(state.job_id, owner="bob") is False
    assert await manager.remove(state.job_id, owner="alice") is True
    assert await manager.get(state.job_id, owner="alice") is None
    await manager.stop()


@pytest.mark.anyio
async def test_remove_refuses_a_running_job() -> None:
    """Test that remove() refuses a queued/running job even for its owner.

    A removed-but-running job would keep writing with nobody watching, and
    would silently defeat active_for()'s double-write guard — both only ever
    see self._jobs, so a removed running job drops out of both at once.
    """
    gate = asyncio.Event()

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        asyncio.run_coroutine_threadsafe(_wait(gate), state_loop).result(timeout=5)
        return {"empty": False, "resolution": None}

    async def _wait(event: asyncio.Event) -> None:
        await event.wait()

    state_loop = asyncio.get_running_loop()
    manager = IngestJobManager(runner=runner)
    state = await _create(manager, physical="p1")

    for _ in range(200):
        if state.status is JobStatus.RUNNING:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("job never reached RUNNING")

    assert await manager.remove(state.job_id, owner="alice") is False
    assert await manager.get(state.job_id, owner="alice") is state
    assert await manager.active_for("alice", "p1") is state

    gate.set()
    await _drain(manager, state)
    await manager.stop()
