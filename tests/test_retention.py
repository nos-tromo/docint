"""Tests for the retention sweeper loop in ``docint.core.retention``."""

from __future__ import annotations

import asyncio

import pytest

from docint.core.retention import run_retention_loop


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only (trio is not a dependency)."""
    return "asyncio"


@pytest.mark.anyio
async def test_the_loop_waits_first_then_runs_on_its_interval() -> None:
    """The first sweep runs a little after boot, then once per interval."""
    waits: list[float] = []
    runs: list[int] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) > 3:
            raise asyncio.CancelledError

    async def sweep() -> None:
        runs.append(len(waits))

    with pytest.raises(asyncio.CancelledError):
        await run_retention_loop(sweep, first_delay=300, interval=86_400, sleep=sleep)

    assert waits == [300, 86_400, 86_400, 86_400]
    assert runs == [1, 2, 3]


@pytest.mark.anyio
async def test_the_loop_survives_a_failed_sweep() -> None:
    """One bad run must not end retention for the rest of the process's life."""
    waits: list[float] = []
    runs: list[int] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) > 2:
            raise asyncio.CancelledError

    async def sweep() -> None:
        runs.append(len(waits))
        if len(runs) == 1:
            raise RuntimeError("sessions store locked")

    with pytest.raises(asyncio.CancelledError):
        await run_retention_loop(sweep, first_delay=1, interval=2, sleep=sleep)

    assert runs == [1, 2]


@pytest.mark.anyio
async def test_cancelling_during_a_sweep_stops_the_loop() -> None:
    """Shutdown cancels mid-sweep; the failure handler must not swallow that."""
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) > 3:
            raise AssertionError("the loop swallowed the cancellation")

    async def sweep() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_retention_loop(sweep, first_delay=0, interval=0, sleep=sleep)
