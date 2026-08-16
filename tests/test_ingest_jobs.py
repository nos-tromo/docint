"""Unit tests for the in-memory ingest job registry."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture

from docint.core.jobs import (
    MAX_UPLOAD_LEAD_S,
    IngestJobManager,
    IngestJobState,
    JobStatus,
    _clamp_lead,
    format_sse,
)
from docint.utils.logfmt import ProgressLogThrottle


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


def _data_of(frame: str) -> dict[str, Any]:
    """Return one SSE frame's decoded payload."""
    for line in frame.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"frame carries no data line: {frame!r}")


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
async def test_completed_job_logs_how_long_the_whole_run_took(
    loguru_caplog_info: LogCaptureFixture,
) -> None:
    """The job's terminal line reports the run an operator waited for.

    The window logged here is the one ``snapshot()`` reports and the one the
    SPA's ingest card freezes on. Timing any single stage instead — the
    pipeline call alone, say — logs a number roughly half the card's, which
    is the mismatch this pins.
    """
    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager)
    await _drain(manager, state)

    completed = [r for r in loguru_caplog_info.messages if "completed in" in r]
    assert completed, f"no completion line logged; got {loguru_caplog_info.messages}"
    assert re.search(rf"Job {state.job_id} \(ingest\) completed in \d{{2}}:\d{{2}}\.", completed[-1]), completed[-1]
    await manager.stop()


@pytest.mark.anyio
async def test_failed_job_logs_how_long_it_ran_before_failing(
    loguru_caplog_info: LogCaptureFixture,
) -> None:
    """A failure reports its duration too — a slow failure is a symptom.

    An ingest that dies after twenty minutes on an unreachable endpoint and
    one that dies immediately on a bad payload read identically without it.
    """

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        raise RuntimeError("boom")

    manager = IngestJobManager(runner=runner)
    state = await _create(manager)
    await _drain(manager, state)

    failed = [r for r in loguru_caplog_info.messages if "failed after" in r]
    assert failed, f"no failure line logged; got {loguru_caplog_info.messages}"
    assert re.search(rf"Job {state.job_id} \(ingest\) failed after \d{{2}}:\d{{2}}\.", failed[-1]), failed[-1]
    await manager.stop()


@pytest.mark.anyio
async def test_runner_progress_reaches_the_log(
    loguru_caplog_info: LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress went to the client stream and nowhere else.

    The CLI passed ``logger.info`` as the pipeline's progress sink and got a
    readable run; the API passed an SSE publisher and got six-minute silences
    in the container log. Same code, same run.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("LOG_PROGRESS_INTERVAL_S", "0")
    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager)
    await _drain(manager, state)

    assert any(f"Job {state.job_id} (ingest) progress: working" in m for m in loguru_caplog_info.messages), (
        f"progress never reached the log; got {loguru_caplog_info.messages}"
    )
    await manager.stop()


@pytest.mark.anyio
async def test_runner_warnings_reach_the_log(
    loguru_caplog_info: LogCaptureFixture,
) -> None:
    """A pushed warning used to produce no log line at all.

    "No staged files found" is exactly the kind of thing an operator goes
    to the log for, and it was visible only to an attached browser.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        push("warning", {"message": "No ingestable files found."})
        return {"empty": True, "resolution": None}

    manager = IngestJobManager(runner=runner)
    state = await _create(manager)
    await _drain(manager, state)

    assert any("warning: No ingestable files found." in m for m in loguru_caplog_info.messages), (
        f"warning never reached the log; got {loguru_caplog_info.messages}"
    )
    await manager.stop()


@pytest.mark.anyio
async def test_per_chunk_ticks_are_throttled_but_the_last_one_survives(
    loguru_caplog_info: LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hundreds of ticks collapse to a heartbeat without losing the tail.

    A raw tee of the progress stream would bury the log; dropping the held
    tick would end the stage mid-count. Neither is acceptable.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("LOG_PROGRESS_INTERVAL_S", "3600")

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        for n in range(1, 201):
            push("ingestion_progress", {"message": f"Extracting entities: {n}/500 chunks processed"})
        return {"empty": False, "resolution": None}

    manager = IngestJobManager(runner=runner)
    state = await _create(manager)
    await _drain(manager, state)

    ticks = [m for m in loguru_caplog_info.messages if "Extracting entities" in m]
    assert len(ticks) == 2, f"expected an opening tick and a flushed tail, got {ticks}"
    assert "1/500" in ticks[0]
    assert "200/500" in ticks[1], "the last thing the stage said must survive the throttle"
    await manager.stop()


@pytest.mark.anyio
async def test_a_throwing_log_tee_never_fails_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """The log tee describes a run; it must not be able to end one.

    ``ingestion_pipeline`` re-raises whatever a progress callback threw via
    ``future.result()``, so an exception raised while deciding whether to
    log would fail the batch it was only meant to narrate. The client
    stream must be unaffected too.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    def _boom(self: ProgressLogThrottle, message: str) -> list[str]:
        """Fail the way a future refactor might.

        Args:
            self (ProgressLogThrottle): The throttle instance.
            message (str): The progress message.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("throttle exploded")

    monkeypatch.setattr(ProgressLogThrottle, "observe", _boom)

    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager)
    await _drain(manager, state)

    assert state.status is JobStatus.COMPLETED
    # The client stream still carries the progress the log could not render.
    assert state.message == "working"
    await manager.stop()


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        (-1.0, 0.0),
        (0.0, 0.0),
        (12.5, 12.5),
        (MAX_UPLOAD_LEAD_S + 1, MAX_UPLOAD_LEAD_S),
        (float("inf"), 0.0),
        (float("nan"), 0.0),
    ],
)
def test_upload_lead_is_clamped_to_a_plausible_run(reported: float, expected: float) -> None:
    """A caller-reported lead bounds a log line, so it must stay sane.

    Args:
        reported (float): What the client claimed it spent uploading.
        expected (float): The value the job is allowed to run with.
    """
    assert _clamp_lead(reported) == expected


@pytest.mark.anyio
async def test_terminal_frame_carries_the_run_duration() -> None:
    """The SPA renders the server's number instead of deriving its own.

    Two nearly equal durations floored on either side of the wire disagree by
    a whole second whenever their difference straddles a boundary, so the card
    must read the value the completion line logged, not one of its own.
    """
    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager)
    await _drain(manager, state)

    payload = _data_of(state.history()[-1])
    assert payload["duration_ms"] == state.duration_ms
    assert isinstance(payload["duration_ms"], int)
    await manager.stop()


@pytest.mark.anyio
async def test_failure_frame_carries_the_run_duration() -> None:
    """A failed run reports its duration too — a slow failure is a symptom."""

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        raise RuntimeError("boom")

    manager = IngestJobManager(runner=runner)
    state = await _create(manager)
    await _drain(manager, state)

    payload = _data_of(state.history()[-1])
    assert payload["code"] == "ingestion_failed"
    assert payload["duration_ms"] == state.duration_ms
    await manager.stop()


@pytest.mark.anyio
async def test_upload_lead_is_folded_into_the_run_duration(
    loguru_caplog_info: LogCaptureFixture,
) -> None:
    """The run starts when the user did, not when the job was created.

    The upload leg happens before any job exists, so the client reports it and
    the job carries it. Without this the log undercounts exactly the stretch
    the ingest card was already ticking through.
    """
    manager = IngestJobManager(runner=_noop_runner)
    state = await manager.create(
        owner="alice",
        logical_name="mydocs",
        physical="p1",
        batch_dir=Path("/nonexistent/batch"),
        upload_lead_s=30.0,
    )
    await _drain(manager, state)

    assert state.duration_s is not None
    assert state.duration_s >= 30.0
    assert state.run_started_at < state.created_at
    completed = [r for r in loguru_caplog_info.messages if "completed in" in r]
    assert re.search(r"completed in 00:3\d\.", completed[-1]), completed[-1]
    await manager.stop()


@pytest.mark.anyio
async def test_queue_wait_counts_toward_the_run_duration() -> None:
    """A job that waited for a worker slot waited as far as the user is concerned.

    The clock starts at creation, not at slot acquisition: with concurrency 1
    the second run's card ticks through the queue, so its logged duration has
    to as well.
    """

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        time.sleep(0.05)
        return {"empty": False, "resolution": None}

    manager = IngestJobManager(runner=runner, concurrency=1)
    first = await _create(manager, physical="p1")
    second = await _create(manager, physical="p2")
    await _drain(manager, first)
    await _drain(manager, second)

    assert first.duration_s is not None
    assert second.duration_s is not None
    # The second job ran the same 50 ms as the first, plus the first's whole
    # run spent queued — so it is strictly longer, with a wide margin.
    assert second.duration_s > first.duration_s
    await manager.stop()


@pytest.mark.anyio
async def test_snapshot_reports_the_run_window() -> None:
    """A reattaching client gets the same anchor and the same total."""
    manager = IngestJobManager(runner=_noop_runner)
    state = await _create(manager)
    assert state.snapshot()["duration_ms"] is None  # unfinished: no total yet
    await _drain(manager, state)

    payload = state.snapshot()
    assert payload["duration_ms"] == state.duration_ms
    assert payload["run_started_at"] == state.run_started_at.isoformat()
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
async def test_failed_runner_error_frame_carries_machine_readable_code() -> None:
    """The terminal SSE ``error`` frame carries ``code: "ingestion_failed"``.

    This is the machine-readable contract API/frontend consumers switch on
    (as opposed to ``state.error``, which is prose for display). Pinned here
    at the source (``IngestJobManager._worker``'s failure path) rather than
    at the API layer, where driving ``/ingest/jobs/events`` — which never
    terminates on its own — through an HTTP test transport isn't viable; see
    ``tests/test_api_generic_errors.py::test_ingest_finalize_job_error_is_generic_and_logged``.
    """

    def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        raise RuntimeError("boom")

    manager = IngestJobManager(runner=runner)
    state = await _create(manager)
    await _drain(manager, state)

    terminal = state.history()[-1]
    assert "event: error" in terminal
    assert '"code": "ingestion_failed"' in terminal
    await manager.stop()


@pytest.mark.anyio
async def test_create_if_idle_is_atomic_under_concurrent_calls() -> None:
    """Two concurrent ``create_if_idle`` calls for one collection must not both create a job.

    Regression guard for the TOCTOU a separate ``active_for()`` check +
    ``create()`` call would have: two ``/ingest/finalize`` requests
    interleaving between those two lock acquisitions could both observe "no
    job in flight" and both create one, defeating the double-write guard the
    409 exists to enforce. ``asyncio.gather`` races two calls through the
    *same* manager; because both go through ``create_if_idle``'s single lock
    acquisition, the outcome is deterministic regardless of which one the
    event loop happens to run first: exactly one creates a job, and the other
    is handed that same job back.
    """
    manager = IngestJobManager(runner=_noop_runner)
    kwargs: dict[str, Any] = {
        "owner": "alice",
        "logical_name": "mydocs",
        "physical": "p1",
        "batch_dir": Path("/nonexistent/batch"),
        "hybrid": True,
        "ner": None,
        "hate_speech": None,
        "resolve": False,
    }

    (state_a, created_a), (state_b, created_b) = await asyncio.gather(
        manager.create_if_idle(**kwargs), manager.create_if_idle(**kwargs)
    )

    assert {created_a, created_b} == {True, False}
    assert state_a.job_id == state_b.job_id
    await _drain(manager, state_a)
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


def test_warning_history_is_capped_with_a_dropped_count() -> None:
    """A pathological run must not retain every warning frame forever.

    Warnings are the one event class kept in full for replay, so a run where
    most files warn grows the history without bound and replays all of it to
    every reattaching tab.
    """
    state = IngestJobState(
        job_id="j1",
        owner="alice",
        logical_name="mydocs",
        physical="p1",
        batch_dir=Path("/nonexistent/batch"),
        hybrid=True,
        ner=None,
        hate_speech=None,
        resolve=False,
    )
    overflow = IngestJobState.MAX_RETAINED_WARNINGS + 5
    for i in range(overflow):
        state.record("warning", format_sse("warning", {"message": f"skipped file {i}"}))

    history = state.history()
    assert len(history) == IngestJobState.MAX_RETAINED_WARNINGS + 1

    # The earliest warnings are kept - they explain what started going wrong.
    assert "skipped file 0" in history[0]
    # The client is told what it is not seeing rather than silently losing it.
    assert "5" in history[-1]


def test_warning_history_uncapped_below_the_limit() -> None:
    """A normal run keeps every warning and gains no sentinel frame."""
    state = IngestJobState(
        job_id="j1",
        owner="alice",
        logical_name="mydocs",
        physical="p1",
        batch_dir=Path("/nonexistent/batch"),
        hybrid=True,
        ner=None,
        hate_speech=None,
        resolve=False,
    )
    for i in range(3):
        state.record("warning", format_sse("warning", {"message": f"skipped file {i}"}))

    assert len(state.history()) == 3


@pytest.mark.anyio
async def test_terminal_jobs_are_capped_per_owner() -> None:
    """Finished jobs must not accumulate for a client that never dismisses.

    The registry is in-memory and only pruned by an explicit client dismiss,
    so a long-lived backend otherwise retains every job an owner ever ran.
    """
    manager = IngestJobManager(runner=_noop_runner)
    cap = IngestJobManager.MAX_TERMINAL_JOBS_PER_OWNER

    created = []
    for i in range(cap + 3):
        state = await _create(manager, physical=f"p{i}")
        await _drain(manager, state)
        created.append(state)

    retained = await manager.list_for_owner("alice")
    assert len(retained) == cap

    # The oldest terminal jobs are the ones evicted.
    retained_ids = {s.job_id for s in retained}
    assert created[0].job_id not in retained_ids
    assert created[-1].job_id in retained_ids


class TestSummaryJobs:
    """``kind="summary"`` jobs share the registry but frame their own events."""

    @pytest.mark.anyio
    async def test_summary_job_emits_summary_events(self) -> None:
        """A ``kind="summary"`` job frames ``summary_started``/``summary_completed``."""

        def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
            push("summary_progress", {"message": "Summarizing 1/2 documents", "mapped": 1, "total_units": 2})
            return {"empty": False, "resolution": None}

        manager = IngestJobManager(runner=runner, concurrency=1, summary_concurrency=1)
        state = await manager.create(owner="owner-a", logical_name="col", physical="phys", kind="summary")
        await _drain(manager, state)

        history = "".join(state.history())
        assert "event: summary_started" in history
        assert "event: summary_progress" in history
        assert "event: summary_completed" in history
        assert state.snapshot()["kind"] == "summary"
        await manager.stop()

    @pytest.mark.anyio
    async def test_summary_failure_uses_summary_code(self) -> None:
        """A failed summary job carries ``summary_failed``/its own error copy."""

        def runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
            raise RuntimeError("boom")

        manager = IngestJobManager(runner=runner, summary_concurrency=1)
        state = await manager.create(owner="o", logical_name="c", physical="p", kind="summary")
        await _drain(manager, state)

        assert state.status is JobStatus.FAILED
        assert state.error == "Summary generation failed."
        assert '"code": "summary_failed"' in "".join(state.history())
        await manager.stop()

    @pytest.mark.anyio
    async def test_ingest_and_summary_jobs_do_not_block_each_other(self) -> None:
        """``create_if_idle`` keys idleness on ``(owner, physical, kind)``."""
        gate = asyncio.Event()

        def blocking_runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
            asyncio.run_coroutine_threadsafe(gate.wait(), state_loop).result(timeout=5)
            return {"empty": False, "resolution": None}

        state_loop = asyncio.get_running_loop()
        manager = IngestJobManager(runner=blocking_runner, concurrency=1, summary_concurrency=1)
        ingest, created_i = await manager.create_if_idle(
            owner="o",
            logical_name="c",
            physical="p",
            batch_dir=Path("/nonexistent"),
            hybrid=None,
            ner=None,
            hate_speech=None,
            resolve=False,
            kind="ingest",
        )
        summary, created_s = await manager.create_if_idle(owner="o", logical_name="c", physical="p", kind="summary")
        assert created_i and created_s

        dup, created_dup = await manager.create_if_idle(owner="o", logical_name="c", physical="p", kind="summary")
        assert not created_dup
        assert dup.job_id == summary.job_id

        gate.set()
        await _drain(manager, ingest)
        await _drain(manager, summary)
        await manager.stop()

    @pytest.mark.anyio
    async def test_summary_jobs_run_on_their_own_semaphore(self) -> None:
        """A summary job reaches RUNNING while an ingest job still holds its slot.

        Regression guard for a single shared semaphore: if ``__init__``
        aliased ``kind="ingest"`` and ``kind="summary"`` to the same
        ``asyncio.Semaphore`` (concurrency=1 each), the summary job below
        would stay QUEUED behind the still-running ingest job instead of
        starting immediately — the exact "a summary rebuild cannot consume an
        ingest worker slot" guarantee this task exists to provide.
        """
        gate = asyncio.Event()

        def blocking_runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
            asyncio.run_coroutine_threadsafe(gate.wait(), state_loop).result(timeout=5)
            return {"empty": False, "resolution": None}

        async def _wait_running(state: IngestJobState) -> None:
            for _ in range(200):
                if state.status is JobStatus.RUNNING:
                    return
                await asyncio.sleep(0.01)
            raise AssertionError(f"job never reached RUNNING; status={state.status}")

        state_loop = asyncio.get_running_loop()
        manager = IngestJobManager(runner=blocking_runner, concurrency=1, summary_concurrency=1)

        ingest = await manager.create(owner="o", logical_name="c1", physical="p1", kind="ingest")
        await _wait_running(ingest)

        summary = await manager.create(owner="o", logical_name="c2", physical="p2", kind="summary")
        # If both kinds shared one semaphore, this would time out with
        # summary still QUEUED, since the ingest job above holds the only slot.
        await _wait_running(summary)

        assert ingest.status is JobStatus.RUNNING
        assert summary.status is JobStatus.RUNNING

        gate.set()
        await _drain(manager, ingest)
        await _drain(manager, summary)
        await manager.stop()

    @pytest.mark.anyio
    async def test_default_kind_is_ingest_and_ingest_events_unchanged(self) -> None:
        """Regression: default-kind jobs still emit ``ingestion_*`` frames."""
        manager = IngestJobManager(runner=_noop_runner)
        state = await _create(manager)
        await _drain(manager, state)

        history = "".join(state.history())
        assert "event: ingestion_started" in history
        assert "event: ingestion_progress" in history
        assert "event: ingestion_complete" in history
        assert state.kind == "ingest"
        assert state.snapshot()["kind"] == "ingest"
        await manager.stop()


@pytest.mark.anyio
async def test_eviction_never_drops_an_unfinished_job() -> None:
    """A still-running job survives eviction regardless of the cap.

    Dropping one would strand a worker writing to Qdrant with no way for the
    owner to observe it.
    """
    import threading

    gate = threading.Event()

    def _hold_one(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        """Block forever on the "held" job; finish every other immediately."""
        if state.physical == "held":
            gate.wait(timeout=5)
        return {"empty": False, "resolution": None}

    manager = IngestJobManager(runner=_hold_one, concurrency=8)
    cap = IngestJobManager.MAX_TERMINAL_JOBS_PER_OWNER

    in_flight = await _create(manager, physical="held")
    for i in range(cap + 2):
        await _drain(manager, await _create(manager, physical=f"p{i}"))

    retained_ids = {s.job_id for s in await manager.list_for_owner("alice")}
    assert in_flight.job_id in retained_ids
    assert in_flight.status is not JobStatus.COMPLETED
    gate.set()
