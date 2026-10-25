"""Tests for the ``topics`` job kind in the in-memory job registry."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any

import pytest

from docint.core.jobs import KIND_EVENTS, TERMINAL_STATUSES, IngestJobManager, IngestJobState, JobStatus


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only.

    Returns:
        The backend name.
    """
    return "asyncio"


def _events(frames: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Decode SSE frames.

    Args:
        frames: Rendered frames.

    Returns:
        ``(event, data)`` per frame.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for frame in frames:
        lines = frame.splitlines()
        event = next(line[len("event: ") :] for line in lines if line.startswith("event: "))
        data = next(line[len("data: ") :] for line in lines if line.startswith("data: "))
        out.append((event, json.loads(data)))
    return out


def _runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
    """Report one progress frame and finish.

    Args:
        state: The job.
        push: The event publisher.

    Returns:
        A non-empty result.
    """
    push("topics_progress", {"message": "Starting topics run", "stage": "starting", "done": 0, "total": 0})
    return {"empty": False, "resolution": None}


async def _drain(state: IngestJobState) -> None:
    """Wait for a job to finish.

    Args:
        state: The job.

    Raises:
        AssertionError: When it does not finish in time.
    """
    for _ in range(300):
        if state.status in TERMINAL_STATUSES:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job did not finish; status={state.status}")


def test_the_topics_kind_has_its_events_and_a_worker_slot() -> None:
    """The kind frames its lifecycle with its own names and has its own slot."""
    assert KIND_EVENTS["topics"] == {
        "started": "topics_started",
        "progress": "topics_progress",
        "complete": "topics_completed",
        "cancelled": "topics_cancelled",
        "failed_code": "topics_failed",
        "failed_message": "Topics run failed.",
    }
    manager = IngestJobManager(runner=_runner)
    assert "topics" in manager._semaphores
    assert manager._semaphores["topics"] is not manager._semaphores["ingest"]


@pytest.mark.anyio
async def test_a_topics_job_carries_its_action_on_the_started_frame_and_its_snapshot() -> None:
    """The action rides the started frame and the snapshot; other kinds carry none."""
    manager = IngestJobManager(runner=_runner)
    state = await manager.create(owner="alice", logical_name="docs", physical="p1", kind="topics", action="discover")
    await _drain(state)

    events = _events(state.history())
    assert [name for name, _ in events] == ["topics_started", "topics_progress", "topics_completed"]
    assert events[0][1]["action"] == "discover"
    assert state.snapshot()["action"] == "discover"
    assert state.snapshot()["kind"] == "topics"

    ingest = await manager.create(owner="alice", logical_name="docs", physical="p1")
    await _drain(ingest)
    assert "action" not in _events(ingest.history())[0][1]
    assert ingest.snapshot()["action"] is None
    await manager.stop()


@pytest.mark.anyio
async def test_a_second_topics_job_is_refused_but_an_ingest_beside_it_is_not() -> None:
    """Only one topics job runs per collection; an ingest is a different kind."""
    gate = threading.Event()

    def blocking(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        """Hold the job until released.

        Args:
            state: The job.
            push: The event publisher.

        Returns:
            A non-empty result.
        """
        gate.wait(timeout=5)
        return {"empty": False, "resolution": None}

    manager = IngestJobManager(runner=blocking)
    first, created = await manager.create_if_idle(
        owner="alice", logical_name="docs", physical="p1", kind="topics", action="build"
    )
    assert created
    again, created_again = await manager.create_if_idle(
        owner="alice", logical_name="docs", physical="p1", kind="topics", action="align"
    )
    assert not created_again
    assert again.job_id == first.job_id
    assert again.action == "build"
    _, created_ingest = await manager.create_if_idle(owner="alice", logical_name="docs", physical="p1")
    assert created_ingest

    gate.set()
    await _drain(first)
    await manager.stop()


def test_topics_concurrency_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DOCINT_TOPICS_CONCURRENCY`` sizes the slot unless the caller passes one.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setenv("DOCINT_TOPICS_CONCURRENCY", "3")
    assert IngestJobManager(runner=_runner)._semaphores["topics"]._value == 3
    assert IngestJobManager(runner=_runner, topics_concurrency=2)._semaphores["topics"]._value == 2
    monkeypatch.delenv("DOCINT_TOPICS_CONCURRENCY")
    assert IngestJobManager(runner=_runner)._semaphores["topics"]._value == 1


@pytest.mark.anyio
async def test_a_topics_failure_is_an_error_frame_with_topics_failed() -> None:
    """A failed run ends on the shared ``error`` frame with static copy only."""

    def failing(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        """Fail with a message that must not reach a client.

        Args:
            state: The job.
            push: The event publisher.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("http://internal-host:4000 refused")

    manager = IngestJobManager(runner=failing)
    state = await manager.create(owner="alice", logical_name="docs", physical="p1", kind="topics", action="build")
    await _drain(state)

    assert state.status is JobStatus.FAILED
    name, data = _events(state.history())[-1]
    assert name == "error"
    assert data["code"] == "topics_failed"
    assert data["message"] == "Topics run failed."
    assert "internal-host" not in json.dumps(state.history())
    await manager.stop()
