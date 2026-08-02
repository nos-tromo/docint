"""In-memory ingest job registry for the docint FastAPI backend.

Ingestion used to stream its progress on the request that started it, so a
client disconnect (navigation, reload, a closed tab) severed the only view of
a run that kept going regardless. This module owns runs instead: a job is
registered, a worker thread runs the pipeline, and clients attach to an
owner-multiplexed SSE stream that replays a collapsed history on connect.

Jobs live only in memory. They survive a browser reload — the frontend
re-discovers them by owner — but not a backend restart. The design mirrors
``Nextext/nextext/api/jobs.py``; the one deliberate deviation is the collapsed
event history (see :meth:`IngestJobState.record`).

The module holds no docint domain imports: the pipeline call is injected as a
``runner`` callable, so the manager is testable without Qdrant, models, or a
network.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from anyio import to_thread
from loguru import logger

from docint.utils.env_cfg import load_ingest_concurrency

TERMINAL_EVENTS: frozenset[str] = frozenset({"ingestion_complete", "error"})


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp.

    Returns:
        datetime: Current time with an explicit UTC offset.
    """
    return datetime.now(tz=UTC)


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Render a payload as an SSE frame.

    Args:
        event (str): SSE event name (e.g. ``ingestion_progress``).
        data (dict[str, Any]): JSON-serializable payload.

    Returns:
        str: A complete ``event:``/``data:`` frame terminated by a blank line.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class JobStatus(StrEnum):
    """Lifecycle state of an ingest job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})


@dataclass
class IngestJobState:
    """Mutable state for one queued, running, or finished ingest job.

    ``owner`` is the principal resolved per request by
    :func:`docint.core.auth.principal.resolve_principal`; routes consult it to
    enforce per-owner access (cross-owner reads 404 so existence never leaks).
    ``physical`` is the owner-namespaced Qdrant name and stays internal —
    :meth:`snapshot` echoes only the caller's logical name.
    """

    job_id: str
    owner: str
    logical_name: str
    physical: str
    batch_dir: Path
    hybrid: bool | None
    ner: bool | None
    hate_speech: bool | None
    resolve: bool
    status: JobStatus = JobStatus.QUEUED
    message: str | None = None
    error: str | None = None
    empty: bool = False
    resolution: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    _started_frame: str | None = field(default=None, repr=False)
    _warning_frames: list[str] = field(default_factory=list, repr=False)
    _dropped_warnings: int = field(default=0, repr=False)
    _progress_frame: str | None = field(default=None, repr=False)
    _terminal_frame: str | None = field(default=None, repr=False)

    # Warnings are the only event class replayed in full, so a run where most
    # files warn would otherwise grow the history without bound and replay all
    # of it to every reattaching tab. The earliest are kept: they explain what
    # started going wrong, and later ones are usually the same cause repeating.
    MAX_RETAINED_WARNINGS: ClassVar[int] = 100

    def record(self, event_name: str, frame: str) -> None:
        """Fold a frame into the collapsed replay history.

        docint's progress is a stream of free-form messages — a long run emits
        thousands ("Extracting entities: 847/2000 chunks"). Replaying all of
        them on every reattach would be untenable and pointless: only the
        newest one describes the current state. Warnings are different — each
        carries unique information — so all of them are kept.

        Args:
            event_name (str): SSE event name.
            frame (str): The pre-rendered SSE frame.
        """
        if event_name == "ingestion_started":
            self._started_frame = frame
        elif event_name == "warning":
            if len(self._warning_frames) < self.MAX_RETAINED_WARNINGS:
                self._warning_frames.append(frame)
            else:
                self._dropped_warnings += 1
        elif event_name == "ingestion_progress":
            self._progress_frame = frame
        elif event_name in TERMINAL_EVENTS:
            self._terminal_frame = frame

        if event_name in {"ingestion_progress", "warning"}:
            try:
                payload = json.loads(frame.split("data: ", 1)[1])
            except (IndexError, json.JSONDecodeError):
                return
            message = payload.get("message")
            if isinstance(message, str):
                self.message = message

    def history(self) -> list[str]:
        """Return the collapsed frames a reattaching client should replay.

        Returns:
            list[str]: ``ingestion_started``, then the retained ``warning``
            frames (plus a sentinel naming how many were dropped, when the
            cap was hit), then the latest ``ingestion_progress``, then the
            terminal frame — each omitted if it has not occurred yet.
        """
        frames: list[str] = []
        if self._started_frame is not None:
            frames.append(self._started_frame)
        frames.extend(self._warning_frames)
        if self._dropped_warnings:
            frames.append(
                format_sse(
                    "warning",
                    {"message": f"{self._dropped_warnings} further warnings omitted."},
                )
            )
        if self._progress_frame is not None:
            frames.append(self._progress_frame)
        if self._terminal_frame is not None:
            frames.append(self._terminal_frame)
        return frames

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable, caller-safe view of the job.

        Returns:
            dict[str, Any]: Public job fields. The physical collection name is
            deliberately excluded — callers only ever see their logical name.
        """
        return {
            "job_id": self.job_id,
            "collection": self.logical_name,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            "empty": self.empty,
            "resolution": self.resolution,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


PushEvent = Callable[[str, dict[str, Any]], None]
JobRunner = Callable[[IngestJobState, PushEvent], dict[str, Any]]

_PING_INTERVAL_S = 15.0


class IngestJobManager:
    """In-memory ingest job store and worker dispatcher.

    Jobs are held in a dict keyed by ``job_id`` and scoped by ``owner``. Async
    workers bounded by an ``asyncio.Semaphore`` (``DOCINT_INGEST_CONCURRENCY``,
    default 1) run the injected ``runner`` on a worker thread. Clients attach
    with :meth:`subscribe_owner`, which replays each owned job's collapsed
    history before live-tailing — so a browser that reloads mid-run re-attaches
    and resumes the live view.

    There is no durable storage: jobs do not survive a process restart. A job
    is retained until its owner removes it, except that finishing a run evicts
    that owner's oldest *terminal* jobs beyond
    :attr:`MAX_TERMINAL_JOBS_PER_OWNER` — a client that never dismisses would
    otherwise grow the process without bound. Queued and running jobs are
    never evicted.
    """

    # Finished jobs an owner may accumulate before the oldest are evicted.
    # A client that never dismisses would otherwise grow the process without
    # bound; the SPA only ever needs the newest, so this is generous.
    MAX_TERMINAL_JOBS_PER_OWNER: ClassVar[int] = 50

    def __init__(self, runner: JobRunner, concurrency: int | None = None) -> None:
        """Initialize the manager.

        Args:
            runner (JobRunner): Blocking callable executing one job. Receives
                the job state and a thread-safe ``push(event, payload)``.
                Returns ``{"empty": bool, "resolution": dict | None}``.
            concurrency (int | None): Worker semaphore size. Defaults to
                :func:`docint.utils.env_cfg.load_ingest_concurrency`.
        """
        self._runner = runner
        self._jobs: dict[str, IngestJobState] = {}
        self._subscribers: dict[str, list[asyncio.Queue[str | None]]] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(concurrency if concurrency is not None else load_ingest_concurrency())
        self._tasks: set[asyncio.Task[None]] = set()

    async def create(
        self,
        *,
        owner: str,
        logical_name: str,
        physical: str,
        batch_dir: Path,
        hybrid: bool | None,
        ner: bool | None,
        hate_speech: bool | None,
        resolve: bool,
    ) -> IngestJobState:
        """Register a job and dispatch its worker, unconditionally.

        Callers that must refuse a second concurrent job for the same
        ``(owner, physical)`` pair — i.e. every route — should use
        :meth:`create_if_idle` instead: a separate ``active_for()`` check
        before calling this method is a TOCTOU (two interleaved callers can
        both observe no in-flight job and both create one).

        Args:
            owner (str): Resolved principal owning the job.
            logical_name (str): The caller's collection name.
            physical (str): Owner-namespaced Qdrant collection name.
            batch_dir (Path): Directory of staged source files.
            hybrid (bool | None): Whether hybrid search is enabled for the
                run; ``None`` keeps the RAG engine's derived default instead
                of forcing it.
            ner (bool | None): Per-request NER override.
            hate_speech (bool | None): Per-request hate-speech override.
            resolve (bool): Whether entity resolution follows the ingest.

        Returns:
            IngestJobState: The newly registered job.
        """
        state = self._new_state(
            owner=owner,
            logical_name=logical_name,
            physical=physical,
            batch_dir=batch_dir,
            hybrid=hybrid,
            ner=ner,
            hate_speech=hate_speech,
            resolve=resolve,
        )
        async with self._lock:
            self._jobs[state.job_id] = state
        self._dispatch_worker(state)
        return state

    async def create_if_idle(
        self,
        *,
        owner: str,
        logical_name: str,
        physical: str,
        batch_dir: Path,
        hybrid: bool | None,
        ner: bool | None,
        hate_speech: bool | None,
        resolve: bool,
    ) -> tuple[IngestJobState, bool]:
        """Atomically check for an in-flight job and create one only if idle.

        The check (:meth:`active_for`'s logic) and the insert must share a
        single ``self._lock`` acquisition, not two separate ones. Two
        ``POST /ingest/finalize`` requests for the same collection can
        interleave at the ``await`` between a check and a later create call;
        if the check and the create were separate lock acquisitions, both
        requests could observe "no job in flight" and both create one —
        defeating the very guard this exists to enforce (overlapping runs can
        double-write, since file hashes are only recorded as ingested after a
        run's final node batch). Doing both under one lock makes the two
        outcomes ("this call created a job" / "another job is already there")
        mutually exclusive and exhaustive: at most one concurrent caller for a
        given ``(owner, physical)`` ever sees ``created=True``.

        Args:
            owner (str): Resolved principal owning the job.
            logical_name (str): The caller's collection name.
            physical (str): Owner-namespaced Qdrant collection name.
            batch_dir (Path): Directory of staged source files.
            hybrid (bool | None): Whether hybrid search is enabled for the
                run; ``None`` keeps the RAG engine's derived default instead
                of forcing it.
            ner (bool | None): Per-request NER override.
            hate_speech (bool | None): Per-request hate-speech override.
            resolve (bool): Whether entity resolution follows the ingest.

        Returns:
            tuple[IngestJobState, bool]: ``(state, created)``. When
            ``created`` is ``True``, ``state`` is the newly dispatched job
            (caller should respond 202). When ``False``, ``state`` is the
            pre-existing queued/running job for this ``(owner, physical)``
            (caller should respond 409 carrying its ``job_id``).
        """
        async with self._lock:
            existing = self._active_locked(owner, physical)
            if existing is not None:
                return existing, False
            state = self._new_state(
                owner=owner,
                logical_name=logical_name,
                physical=physical,
                batch_dir=batch_dir,
                hybrid=hybrid,
                ner=ner,
                hate_speech=hate_speech,
                resolve=resolve,
            )
            self._jobs[state.job_id] = state
        self._dispatch_worker(state)
        return state, True

    def _new_state(
        self,
        *,
        owner: str,
        logical_name: str,
        physical: str,
        batch_dir: Path,
        hybrid: bool | None,
        ner: bool | None,
        hate_speech: bool | None,
        resolve: bool,
    ) -> IngestJobState:
        """Build a fresh, unregistered job state.

        Args:
            owner (str): Resolved principal owning the job.
            logical_name (str): The caller's collection name.
            physical (str): Owner-namespaced Qdrant collection name.
            batch_dir (Path): Directory of staged source files.
            hybrid (bool | None): Whether hybrid search is enabled for the
                run; ``None`` keeps the RAG engine's derived default instead
                of forcing it.
            ner (bool | None): Per-request NER override.
            hate_speech (bool | None): Per-request hate-speech override.
            resolve (bool): Whether entity resolution follows the ingest.

        Returns:
            IngestJobState: A new, not-yet-registered job state.
        """
        return IngestJobState(
            job_id=uuid.uuid4().hex,
            owner=owner,
            logical_name=logical_name,
            physical=physical,
            batch_dir=batch_dir,
            hybrid=hybrid,
            ner=ner,
            hate_speech=hate_speech,
            resolve=resolve,
        )

    def _dispatch_worker(self, state: IngestJobState) -> None:
        """Schedule a job's worker task and track it for cleanup.

        Args:
            state (IngestJobState): The job to run.
        """
        task = asyncio.create_task(self._worker(state))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def get(self, job_id: str, owner: str) -> IngestJobState | None:
        """Return an owned job, or ``None``.

        Args:
            job_id (str): Job identifier.
            owner (str): Resolved principal; a mismatch reads as absent so
                routes can 404 without leaking existence.

        Returns:
            IngestJobState | None: The job when owned by ``owner``.
        """
        async with self._lock:
            state = self._jobs.get(job_id)
        return state if state is not None and state.owner == owner else None

    async def active_for(self, owner: str, physical: str) -> IngestJobState | None:
        """Return the owner's unfinished job for a collection, if any.

        Used to reject a second concurrent ingest into the same collection:
        overlapping runs can double-write, because file hashes are only
        recorded as ingested after a run's final node batch.

        This is a point-in-time read only. A caller that means to act on the
        result by creating a job if none is found must use
        :meth:`create_if_idle` instead of calling this and then ``create()``
        separately — those would be two lock acquisitions with a TOCTOU gap
        between them.

        Args:
            owner (str): Resolved principal.
            physical (str): Owner-namespaced Qdrant collection name.

        Returns:
            IngestJobState | None: The queued/running job, if one exists.
        """
        async with self._lock:
            return self._active_locked(owner, physical)

    def _active_locked(self, owner: str, physical: str) -> IngestJobState | None:
        """Find the owner's unfinished job for a collection; caller must hold ``self._lock``.

        Args:
            owner (str): Resolved principal.
            physical (str): Owner-namespaced Qdrant collection name.

        Returns:
            IngestJobState | None: The queued/running job, if one exists.
        """
        for state in self._jobs.values():
            if (
                state.owner == owner
                and state.physical == physical
                and state.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            ):
                return state
        return None

    async def list_for_owner(self, owner: str) -> list[IngestJobState]:
        """Return the owner's jobs, newest first.

        Not used for the frontend's reload re-discovery — that goes through
        the persisted ``activeJobId`` plus the SSE replay instead. Available
        for other/future clients that want to enumerate a caller's jobs.

        Args:
            owner (str): Resolved principal.

        Returns:
            list[IngestJobState]: Owned jobs, newest first.
        """
        async with self._lock:
            states = [s for s in self._jobs.values() if s.owner == owner]
        states.sort(key=lambda s: s.created_at, reverse=True)
        return states

    async def remove(self, job_id: str, owner: str) -> bool:
        """Drop an owned, finished job from the registry.

        Refuses a queued/running job even from its owner — the worker thread
        cannot be killed, so a removed-but-running job would keep writing
        with nobody watching, and would drop out of ``active_for()``'s view
        (it only sees ``self._jobs``), silently defeating the double-write
        guard that method exists to enforce. It would also become invisible
        to any new ``subscribe_owner()`` replay (e.g. a reloaded tab). Route
        callers are expected to separately reject a running job with a 409
        before ever calling this; the check here is defense in depth, not a
        replacement for that.

        Args:
            job_id (str): Job identifier.
            owner (str): Resolved principal.

        Returns:
            bool: ``True`` when a finished job was removed. ``False`` both
            when the job isn't owned by ``owner`` and when it is still
            queued or running — the two reasons are deliberately
            indistinguishable to the caller, so cross-owner probing can't be
            told apart from a legitimate "still running" refusal.
        """
        async with self._lock:
            state = self._jobs.get(job_id)
            if state is None or state.owner != owner:
                return False
            if state.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                return False
            del self._jobs[job_id]
        return True

    async def subscribe_owner(self, owner: str) -> AsyncGenerator[str, None]:
        """Yield SSE frames for every job this owner has, over one connection.

        Replays each owned job's collapsed history on connect (oldest job
        first), then live-tails every subsequent frame — including jobs created
        after the stream opened. A job reaching a terminal state does not close
        the stream; it serves the owner's other and future jobs until the client
        disconnects.

        Args:
            owner (str): Resolved principal.

        Yields:
            str: SSE-formatted frames, each payload tagged with ``job_id``.
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        async with self._lock:
            # Snapshot + register atomically: put_nowait never awaits, so no
            # concurrently dispatched frame can slip between replay and
            # registration (dispatch runs on this same loop thread).
            owned = sorted(
                (s for s in self._jobs.values() if s.owner == owner),
                key=lambda s: s.created_at,
            )
            for state in owned:
                for frame in state.history():
                    queue.put_nowait(frame)
            self._subscribers.setdefault(owner, []).append(queue)
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(), timeout=_PING_INTERVAL_S)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                if frame is None:
                    return
                yield frame
        finally:
            async with self._lock:
                queues = self._subscribers.get(owner)
                if queues is not None and queue in queues:
                    queues.remove(queue)
                if queues is not None and not queues:
                    del self._subscribers[owner]

    async def stop(self) -> None:
        """Close every open subscriber stream on shutdown."""
        async with self._lock:
            for queues in self._subscribers.values():
                for queue in queues:
                    queue.put_nowait(None)
            self._subscribers.clear()

    async def _worker(self, state: IngestJobState) -> None:
        """Run one job's pipeline once a worker slot is free.

        Args:
            state (IngestJobState): The job to process.
        """
        try:
            await self._run(state)
        finally:
            # Both terminal paths (completed and failed) land here, so the
            # owner's finished-job backlog is trimmed exactly once per run.
            await self._prune_terminal(state.owner)

    async def _run(self, state: IngestJobState) -> None:
        """Execute the job body, holding a worker slot for its duration.

        Args:
            state (IngestJobState): The job to process.
        """
        async with self._semaphore:
            state.status = JobStatus.RUNNING
            state.started_at = _utcnow()
            loop = asyncio.get_running_loop()

            def _frame(event_name: str, payload: dict[str, Any]) -> str:
                """Tag a payload with the job id and render it as an SSE frame.

                Single choke point for both dispatch paths below, so every
                frame is self-identifying on the multiplexed owner stream
                regardless of which thread produced it.

                Args:
                    event_name (str): SSE event name.
                    payload (dict[str, Any]): JSON-serializable payload.

                Returns:
                    str: The rendered SSE frame.
                """
                tagged = {"job_id": state.job_id, **payload}
                return format_sse(event_name, tagged)

            def _emit(event_name: str, payload: dict[str, Any]) -> None:
                """Dispatch a frame synchronously. Loop thread only.

                Used for the frames ``_worker`` produces itself
                (``ingestion_started`` and both terminal events), which
                already run on the loop thread — right beside the status
                change each announces. Routing those through
                ``call_soon_threadsafe`` too (as ``_push`` below must) would
                defer recording the frame to a *later* loop iteration,
                leaving a window where ``state.status`` already reads
                ``FAILED``/``COMPLETED`` but ``state.history()`` does not yet
                contain the matching terminal frame — observable by a caller
                that polls status then immediately reads history (or
                attaches to the events stream) in that window.

                Args:
                    event_name (str): SSE event name.
                    payload (dict[str, Any]): JSON-serializable payload.
                """
                self._dispatch(state, event_name, _frame(event_name, payload))

            def _push(event_name: str, payload: dict[str, Any]) -> None:
                """Publish an event from the worker thread (thread-safe).

                Handed to the injected runner as its ``push`` callable — the
                runner executes on a worker thread (via
                ``to_thread.run_sync``), so this genuinely needs the
                threadsafe hop back onto the loop thread. ``_worker``'s own
                frames use ``_emit`` instead; see its docstring for why.

                Args:
                    event_name (str): SSE event name.
                    payload (dict[str, Any]): JSON-serializable payload.
                """
                frame = _frame(event_name, payload)
                loop.call_soon_threadsafe(self._dispatch, state, event_name, frame)

            _emit("ingestion_started", {"collection": state.logical_name})
            try:
                result = await to_thread.run_sync(self._runner, state, _push)
            except Exception:
                logger.exception("Ingest job {} failed.", state.job_id)
                state.status = JobStatus.FAILED
                state.error = "Ingestion failed."
                state.finished_at = _utcnow()
                # Static protocol copy only: the exception text can carry
                # connection strings or file paths and never reaches a client.
                _emit("error", {"message": "Ingestion failed.", "code": "ingestion_failed"})
                return
            state.empty = bool(result.get("empty", False))
            state.resolution = result.get("resolution")
            state.status = JobStatus.COMPLETED
            state.finished_at = _utcnow()
            terminal: dict[str, Any] = {
                "collection": state.logical_name,
                "empty": state.empty,
            }
            if state.resolution is not None:
                terminal["resolution"] = state.resolution
            _emit("ingestion_complete", terminal)

    async def _prune_terminal(self, owner: str) -> None:
        """Drop an owner's oldest finished jobs beyond the retention cap.

        The registry is in-memory with no durable storage, and a job is
        otherwise retained until its owner explicitly dismisses it — so a
        client that never dismisses makes the process grow without bound.
        Only terminal jobs are eligible: dropping a queued or running one
        would strand a worker with no way for the owner to observe it.

        Args:
            owner (str): The principal whose backlog to trim.
        """
        async with self._lock:
            terminal = sorted(
                (s for s in self._jobs.values() if s.owner == owner and s.status in TERMINAL_STATUSES),
                key=lambda s: s.finished_at or s.created_at,
            )
            for state in terminal[: max(len(terminal) - self.MAX_TERMINAL_JOBS_PER_OWNER, 0)]:
                del self._jobs[state.job_id]

    def _dispatch(self, state: IngestJobState, event_name: str, frame: str) -> None:
        """Record a frame in history and fan it out.

        Runs on the event-loop thread (scheduled via ``call_soon_threadsafe``).

        Args:
            state (IngestJobState): The job emitting the frame.
            event_name (str): SSE event name.
            frame (str): Pre-rendered SSE frame.
        """
        state.record(event_name, frame)
        for queue in self._subscribers.get(state.owner, ()):
            queue.put_nowait(frame)
