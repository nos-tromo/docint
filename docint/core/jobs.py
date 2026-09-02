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

The registry also carries collection-summary rebuild jobs (``kind="summary"``)
alongside ingest jobs (``kind="ingest"``, the default). Both kinds share the
same registry, worker dispatch, and owner-multiplexed SSE stream; each is
framed with its own event names (see :data:`KIND_EVENTS`) and runs under its
own concurrency semaphore, so a summary rebuild cannot consume an ingest
worker slot (or vice versa). An ingest job and a summary job for the same
collection may run at once — :meth:`IngestJobManager.create_if_idle` refuses
overlap only within the same ``(owner, physical, kind)``.

The module holds no docint domain imports: the pipeline call is injected as a
``runner`` callable, so the manager is testable without Qdrant, models, or a
network.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from anyio import to_thread
from loguru import logger

from docint.utils.duration import format_elapsed
from docint.utils.env_cfg import (
    load_extract_concurrency,
    load_ingest_concurrency,
    load_logging_env,
    load_summary_concurrency,
)
from docint.utils.logfmt import (
    ProgressLogThrottle,
    describe_inputs,
    format_by_type,
    format_bytes,
    format_override,
)

#: Per-kind SSE event names and failure copy. Keyed by :attr:`IngestJobState.kind`.
#: ``"ingest"`` preserves the pre-existing event names exactly (backward
#: compatibility for callers that pass no ``kind``); ``"summary"`` frames the
#: same lifecycle for collection-summary rebuild jobs.
KIND_EVENTS: dict[str, dict[str, str]] = {
    "ingest": {
        "started": "ingestion_started",
        "progress": "ingestion_progress",
        "complete": "ingestion_complete",
        "failed_code": "ingestion_failed",
        "failed_message": "Ingestion failed.",
    },
    "summary": {
        "started": "summary_started",
        "progress": "summary_progress",
        "complete": "summary_completed",
        "failed_code": "summary_failed",
        "failed_message": "Summary generation failed.",
    },
    "extract": {
        "started": "extract_started",
        "progress": "extract_progress",
        "complete": "extract_completed",
        "failed_code": "extract_failed",
        "failed_message": "Extract failed.",
    },
}

#: SSE event names, across all kinds, that open a run's history.
STARTED_EVENTS: frozenset[str] = frozenset({"ingestion_started", "summary_started", "extract_started"})
#: SSE event names, across all kinds, carrying a collapsed-to-latest progress update.
PROGRESS_EVENTS: frozenset[str] = frozenset({"ingestion_progress", "summary_progress", "extract_progress"})
#: SSE event names, across all kinds, that end a run.
TERMINAL_EVENTS: frozenset[str] = frozenset({"ingestion_complete", "summary_completed", "extract_completed", "error"})


#: Upper bound on a caller-reported upload lead. A day is far longer than any
#: real upload and short enough that a bogus value cannot claim a nonsense run.
MAX_UPLOAD_LEAD_S: float = 86_400.0

#: Most files the run-start banner lists individually. Beyond this it prints a
#: rollup naming how many it left out — a silently truncated listing would read
#: as the whole batch.
INPUT_LIST_LIMIT: int = 50


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp.

    Returns:
        datetime: Current time with an explicit UTC offset.
    """
    return datetime.now(tz=UTC)


def _clamp_lead(seconds: float) -> float:
    """Bound a caller-reported upload lead to a plausible range.

    Args:
        seconds (float): Reported seconds spent before the job was created.

    Returns:
        float: The value clamped to ``[0, MAX_UPLOAD_LEAD_S]``; non-finite
        input yields ``0.0``, since a run with no measurable lead is the
        honest fallback.
    """
    if not math.isfinite(seconds):
        return 0.0
    return min(max(seconds, 0.0), MAX_UPLOAD_LEAD_S)


def _summary_fields(state: IngestJobState, stats: dict[str, Any] | None, *, failed: bool = False) -> str:
    """Render a finished run's counters as one greppable field list.

    The old completion line said only ``Job <id> (ingest) completed in
    MM:SS.`` — an opaque id and a duration, with no collection and no
    counts. Everything a run knew about itself was either discarded or
    locked behind ``INGEST_BENCHMARK_ENABLED``, which is tuning telemetry
    rather than operator information.

    ``stats`` is rendered generically, key by key, so this module keeps no
    knowledge of docint's ingest domain and the summary job can supply its
    own counters through the same path.

    Both ``duration`` and ``duration_ms`` are emitted: the first is what a
    human reads, the second is the exact integer the SPA's ingest card
    renders. Printing both is what makes the log and the card provably
    agree rather than nearly agree.

    Args:
        state (IngestJobState): The finished job.
        stats (dict[str, Any] | None): Counters from the runner's result,
            or ``None`` on a failure (the run never produced any).
        failed (bool, optional): Whether this is the failure path. Passed
            explicitly rather than read from ``state.status``, which the
            caller has not updated yet at the point it logs. A failed run
            has no ``empty`` to report — it never got far enough to know.

    Returns:
        str: Space-separated ``key=value`` pairs.
    """
    fields = [
        f"job_id={state.job_id}",
        f"collection={state.logical_name!r}",
        f"duration={format_elapsed(state.duration_s or 0.0)}",
        f"duration_ms={state.duration_ms}",
    ]
    for key, value in (stats or {}).items():
        fields.append(f"{key}={value}")
    for key in ("minted", "attached"):
        if state.resolution and key in state.resolution:
            fields.append(f"entities_{key}={state.resolution[key]}")
    if not failed:
        fields.append(f"empty={str(state.empty).lower()}")
    return " ".join(fields)


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
    """Lifecycle state of a job (ingest or summary)."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})


@dataclass
class IngestJobState:
    """Mutable state for one queued, running, or finished job (ingest or summary).

    ``owner`` is the principal resolved per request by
    :func:`docint.core.auth.principal.resolve_principal`; routes consult it to
    enforce per-owner access (cross-owner reads 404 so existence never leaks).
    ``physical`` is the owner-namespaced Qdrant name and stays internal —
    :meth:`snapshot` echoes only the caller's logical name.

    ``kind`` distinguishes an ingest run from a collection-summary rebuild;
    only the four identity fields are required — the ingest-only options
    (``batch_dir``, ``hybrid``, ``ner``, ``hate_speech``, ``resolve``) default
    to values a summary job can safely omit.
    """

    job_id: str
    owner: str
    logical_name: str
    physical: str
    kind: str = "ingest"
    #: The one source an extract job covers; ``None`` for a whole collection.
    target: str | None = None
    #: Case file an extract is filed under, printed on every page of its PDF.
    reference_number: str | None = None
    #: Who asked for the extract, printed under its title.
    operator: str | None = None
    #: The stored artifact a finished extract job produced, if any.
    artifact: dict[str, Any] | None = None
    batch_dir: Path | None = None
    hybrid: bool | None = None
    ner: bool | None = None
    hate_speech: bool | None = None
    resolve: bool = False
    status: JobStatus = JobStatus.QUEUED
    message: str | None = None
    error: str | None = None
    empty: bool = False
    resolution: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: Seconds this run had already spent before the job existed — for an
    #: ingest, the client's upload leg, reported on ``POST /ingest/finalize``.
    #: The run the user waited for starts there, not here.
    upload_lead_s: float = 0.0
    #: Total run duration, computed once at the terminal path. It is what the
    #: completion line logs and what the terminal frame carries, so the log and
    #: the SPA's ingest card render one number rather than two nearly-equal
    #: ones that floor apart at a second boundary.
    duration_s: float | None = None
    #: Monotonic twin of ``created_at``. Monotonic because an NTP step mid-run
    #: would skew a wall-clock subtraction, and a long ingest is exactly when
    #: one can land.
    _created_ticks: float = field(default_factory=time.monotonic, repr=False)
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

    @property
    def run_started_at(self) -> datetime:
        """When the run began, upload leg included.

        Earlier than ``created_at`` by the upload lead, and earlier than
        ``started_at`` by the queue wait as well. This is the anchor a
        reattaching client ticks from; ``started_at`` keeps its own meaning
        (the worker slot was acquired), which is what queue-depth analysis
        needs.

        Returns:
            datetime: The run's start instant, in UTC.
        """
        return self.created_at - timedelta(seconds=self.upload_lead_s)

    @property
    def duration_ms(self) -> int | None:
        """The run's total duration in whole milliseconds, once it has one.

        Returns:
            int | None: Milliseconds, or ``None`` while the job is unfinished.
        """
        return None if self.duration_s is None else round(self.duration_s * 1000)

    def elapsed_s(self) -> float:
        """Seconds since the run began, upload leg and queue wait included.

        Returns:
            float: Elapsed seconds, measured monotonically from creation.
        """
        return self.upload_lead_s + (time.monotonic() - self._created_ticks)

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
        if event_name in STARTED_EVENTS:
            self._started_frame = frame
        elif event_name == "warning":
            if len(self._warning_frames) < self.MAX_RETAINED_WARNINGS:
                self._warning_frames.append(frame)
            else:
                self._dropped_warnings += 1
        elif event_name in PROGRESS_EVENTS:
            self._progress_frame = frame
        elif event_name in TERMINAL_EVENTS:
            self._terminal_frame = frame

        if event_name in PROGRESS_EVENTS or event_name == "warning":
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
            "kind": self.kind,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            "empty": self.empty,
            "resolution": self.resolution,
            "target": self.target,
            "artifact": self.artifact,
            "created_at": self.created_at.isoformat(),
            "run_started_at": self.run_started_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
        }


PushEvent = Callable[[str, dict[str, Any]], None]
JobRunner = Callable[[IngestJobState, PushEvent], dict[str, Any]]

_PING_INTERVAL_S = 15.0


class IngestJobManager:
    """In-memory job store and worker dispatcher for ingest and summary jobs.

    Jobs are held in a dict keyed by ``job_id`` and scoped by ``owner``. Async
    workers run the injected ``runner`` on a worker thread, bounded by a
    per-``kind`` ``asyncio.Semaphore`` — ``DOCINT_INGEST_CONCURRENCY`` (default
    1) for ``kind="ingest"`` jobs, ``DOCINT_SUMMARY_CONCURRENCY`` (default 1)
    for ``kind="summary"`` jobs — so the two kinds never contend for the same
    worker slot. Clients attach with :meth:`subscribe_owner`, which replays
    each owned job's collapsed history before live-tailing — so a browser
    that reloads mid-run re-attaches and resumes the live view, regardless of
    job kind.

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

    def __init__(
        self,
        runner: JobRunner,
        concurrency: int | None = None,
        summary_concurrency: int | None = None,
        extract_concurrency: int | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            runner (JobRunner): Blocking callable executing one job. Receives
                the job state and a thread-safe ``push(event, payload)``.
                Returns ``{"empty": bool, "resolution": dict | None}``.
            concurrency (int | None): Worker semaphore size for ``kind="ingest"``
                jobs. Defaults to :func:`docint.utils.env_cfg.load_ingest_concurrency`.
            summary_concurrency (int | None): Worker semaphore size for
                ``kind="summary"`` jobs. Defaults to
                :func:`docint.utils.env_cfg.load_summary_concurrency`.
            extract_concurrency (int | None): Worker semaphore size for
                ``kind="extract"`` jobs. Defaults to
                :func:`docint.utils.env_cfg.load_extract_concurrency`.
        """
        self._runner = runner
        self._jobs: dict[str, IngestJobState] = {}
        self._subscribers: dict[str, list[asyncio.Queue[str | None]]] = {}
        self._lock = asyncio.Lock()
        self._semaphores: dict[str, asyncio.Semaphore] = {
            "ingest": asyncio.Semaphore(concurrency if concurrency is not None else load_ingest_concurrency()),
            "summary": asyncio.Semaphore(
                summary_concurrency if summary_concurrency is not None else load_summary_concurrency()
            ),
            "extract": asyncio.Semaphore(
                extract_concurrency if extract_concurrency is not None else load_extract_concurrency()
            ),
        }
        self._tasks: set[asyncio.Task[None]] = set()

    async def create(
        self,
        *,
        owner: str,
        logical_name: str,
        physical: str,
        batch_dir: Path | None = None,
        hybrid: bool | None = None,
        ner: bool | None = None,
        hate_speech: bool | None = None,
        resolve: bool = False,
        kind: str = "ingest",
        target: str | None = None,
        reference_number: str | None = None,
        operator: str | None = None,
        upload_lead_s: float = 0.0,
    ) -> IngestJobState:
        """Register a job and dispatch its worker, unconditionally.

        Callers that must refuse a second concurrent job for the same
        ``(owner, physical, kind)`` triple — i.e. every route — should use
        :meth:`create_if_idle` instead: a separate ``active_for()`` check
        before calling this method is a TOCTOU (two interleaved callers can
        both observe no in-flight job and both create one).

        Args:
            owner (str): Resolved principal owning the job.
            logical_name (str): The caller's collection name.
            physical (str): Owner-namespaced Qdrant collection name.
            batch_dir (Path | None): Directory of staged source files.
                Ingest-only; a summary job omits it.
            hybrid (bool | None): Whether hybrid search is enabled for the
                run; ``None`` keeps the RAG engine's derived default instead
                of forcing it. Ingest-only.
            ner (bool | None): Per-request NER override. Ingest-only.
            hate_speech (bool | None): Per-request hate-speech override.
                Ingest-only.
            resolve (bool): Whether entity resolution follows the ingest.
                Ingest-only.
            kind (str): ``"ingest"``, ``"summary"`` or ``"extract"``. Selects
                the SSE event names (:data:`KIND_EVENTS`) and the worker
                semaphore this job waits on.
            target (str | None): The one source an extract covers.
            reference_number (str | None): Case file an extract is filed
                under. Extract-only.
            operator (str | None): Who asked for the extract. Extract-only.
            upload_lead_s (float): Seconds the run had already spent before
                this job existed (an ingest's upload leg). Folded into the
                duration the job logs and reports.

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
            kind=kind,
            target=target,
            reference_number=reference_number,
            operator=operator,
            upload_lead_s=upload_lead_s,
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
        batch_dir: Path | None = None,
        hybrid: bool | None = None,
        ner: bool | None = None,
        hate_speech: bool | None = None,
        resolve: bool = False,
        kind: str = "ingest",
        target: str | None = None,
        reference_number: str | None = None,
        operator: str | None = None,
        upload_lead_s: float = 0.0,
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
        given ``(owner, physical, kind)`` ever sees ``created=True``.

        Idleness is scoped to ``(owner, physical, kind)``, not just
        ``(owner, physical)``: an ingest job and a summary job for the same
        collection are independent runs against independent worker pools, so
        they may legitimately run at once. Only a second job of the *same*
        kind for the same collection is refused.

        Args:
            owner (str): Resolved principal owning the job.
            logical_name (str): The caller's collection name.
            physical (str): Owner-namespaced Qdrant collection name.
            batch_dir (Path | None): Directory of staged source files.
                Ingest-only; a summary job omits it.
            hybrid (bool | None): Whether hybrid search is enabled for the
                run; ``None`` keeps the RAG engine's derived default instead
                of forcing it. Ingest-only.
            ner (bool | None): Per-request NER override. Ingest-only.
            hate_speech (bool | None): Per-request hate-speech override.
                Ingest-only.
            resolve (bool): Whether entity resolution follows the ingest.
                Ingest-only.
            kind (str): ``"ingest"``, ``"summary"`` or ``"extract"``. Selects
                the SSE event names (:data:`KIND_EVENTS`), the worker semaphore
                this job waits on, and the idleness scope checked before
                creating.
            target (str | None): The one source an extract covers.
            reference_number (str | None): Case file an extract is filed
                under. Extract-only.
            operator (str | None): Who asked for the extract. Extract-only.
            upload_lead_s (float): Seconds the run had already spent before
                this job existed (an ingest's upload leg). Folded into the
                duration the job logs and reports. Ignored when an in-flight
                job is adopted instead of created — that run's own lead
                already stands.

        Returns:
            tuple[IngestJobState, bool]: ``(state, created)``. When
            ``created`` is ``True``, ``state`` is the newly dispatched job
            (caller should respond 202). When ``False``, ``state`` is the
            pre-existing queued/running job for this ``(owner, physical, kind)``
            (caller should respond 409 carrying its ``job_id``).
        """
        async with self._lock:
            existing = self._active_locked(owner, physical, kind)
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
                kind=kind,
                target=target,
                reference_number=reference_number,
                operator=operator,
                upload_lead_s=upload_lead_s,
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
        batch_dir: Path | None = None,
        hybrid: bool | None = None,
        ner: bool | None = None,
        hate_speech: bool | None = None,
        resolve: bool = False,
        kind: str = "ingest",
        target: str | None = None,
        reference_number: str | None = None,
        operator: str | None = None,
        upload_lead_s: float = 0.0,
    ) -> IngestJobState:
        """Build a fresh, unregistered job state.

        Args:
            owner (str): Resolved principal owning the job.
            logical_name (str): The caller's collection name.
            physical (str): Owner-namespaced Qdrant collection name.
            batch_dir (Path | None): Directory of staged source files.
                Ingest-only; a summary job omits it.
            hybrid (bool | None): Whether hybrid search is enabled for the
                run; ``None`` keeps the RAG engine's derived default instead
                of forcing it. Ingest-only.
            ner (bool | None): Per-request NER override. Ingest-only.
            hate_speech (bool | None): Per-request hate-speech override.
                Ingest-only.
            resolve (bool): Whether entity resolution follows the ingest.
                Ingest-only.
            kind (str): ``"ingest"``, ``"summary"`` or ``"extract"``.
            target (str | None): The one source an extract covers.
            reference_number (str | None): Case file an extract is filed
                under. Extract-only.
            operator (str | None): Who asked for the extract. Extract-only.
            upload_lead_s (float): Seconds the run had already spent before
                this job existed. Clamped here — it reaches the server as a
                client-reported number, and it bounds a log line, so it must
                not be able to express a negative or absurd run.

        Returns:
            IngestJobState: A new, not-yet-registered job state.
        """
        return IngestJobState(
            job_id=uuid.uuid4().hex,
            owner=owner,
            logical_name=logical_name,
            physical=physical,
            kind=kind,
            target=target,
            reference_number=reference_number,
            operator=operator,
            batch_dir=batch_dir,
            hybrid=hybrid,
            ner=ner,
            hate_speech=hate_speech,
            resolve=resolve,
            upload_lead_s=_clamp_lead(upload_lead_s),
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

    async def active_for(self, owner: str, physical: str, kind: str = "ingest") -> IngestJobState | None:
        """Return the owner's unfinished job for a collection, if any.

        Used to reject a second concurrent job of the same kind into the same
        collection: overlapping ingest runs can double-write, because file
        hashes are only recorded as ingested after a run's final node batch.

        This is a point-in-time read only. A caller that means to act on the
        result by creating a job if none is found must use
        :meth:`create_if_idle` instead of calling this and then ``create()``
        separately — those would be two lock acquisitions with a TOCTOU gap
        between them.

        Args:
            owner (str): Resolved principal.
            physical (str): Owner-namespaced Qdrant collection name.
            kind (str): ``"ingest"`` or ``"summary"``. Idleness is scoped to
                ``(owner, physical, kind)`` — a job of the other kind for the
                same collection does not count as active here.

        Returns:
            IngestJobState | None: The queued/running job, if one exists.
        """
        async with self._lock:
            return self._active_locked(owner, physical, kind)

    def _active_locked(self, owner: str, physical: str, kind: str = "ingest") -> IngestJobState | None:
        """Find the owner's unfinished job for a collection; caller must hold ``self._lock``.

        Args:
            owner (str): Resolved principal.
            physical (str): Owner-namespaced Qdrant collection name.
            kind (str): ``"ingest"`` or ``"summary"``. Only a job of this
                kind counts as active.

        Returns:
            IngestJobState | None: The queued/running job, if one exists.
        """
        for state in self._jobs.values():
            if (
                state.owner == owner
                and state.physical == physical
                and state.kind == kind
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

    async def _log_run_banner(self, state: IngestJobState) -> None:
        """Log what a run is about to do, before it starts doing it.

        Until now nothing marked a run's beginning in the log at all — the
        ``started`` frame went only to attached clients — and no line
        anywhere named the staged files, their sizes, or their types. The
        first thing an operator saw was a per-document line from whichever
        reader happened to log one.

        Every line carries the full ``job_id``, so one ``grep`` reconstructs
        a run even when ``DOCINT_INGEST_CONCURRENCY`` lets two interleave.

        The inventory walk runs on a worker thread: ``_run`` is on the event
        loop, and a network-backed volume must not stall it. Failure is
        swallowed — a banner is a log line and must not be able to fail a
        run.

        Args:
            state (IngestJobState): The job about to execute.
        """
        label = state.kind.capitalize()
        try:
            inventory = (
                None
                if state.batch_dir is None
                else await to_thread.run_sync(describe_inputs, state.batch_dir, INPUT_LIST_LIMIT)
            )
        except Exception:
            inventory = None

        if inventory is None:
            logger.info(
                "{} job started | job_id={} collection={!r}",
                label,
                state.job_id,
                state.logical_name,
            )
            return

        logger.info(
            "{} job started | job_id={} collection={!r} files={} bytes={} by_type={} "
            "hybrid={} ner={} hate_speech={} resolve={}",
            label,
            state.job_id,
            state.logical_name,
            inventory.total_files,
            format_bytes(inventory.total_bytes),
            format_by_type(inventory.by_type),
            format_override(state.hybrid),
            format_override(state.ner),
            format_override(state.hate_speech),
            str(state.resolve).lower(),
        )
        for index, item in enumerate(inventory.files, start=1):
            logger.info(
                "{} input {}/{} | job_id={} file={!r} type={} bytes={}",
                label,
                index,
                inventory.total_files,
                state.job_id,
                item.name,
                item.kind,
                format_bytes(item.size_bytes),
            )
        if inventory.omitted:
            logger.info(
                "{} inputs truncated | job_id={} listed={} omitted={}",
                label,
                state.job_id,
                len(inventory.files),
                inventory.omitted,
            )

    async def _run(self, state: IngestJobState) -> None:
        """Execute the job body, holding a worker slot for its duration.

        Both terminal paths compute the run's duration exactly once
        (:meth:`IngestJobState.elapsed_s`), log it, and carry it on the
        terminal frame as ``duration_ms``. Two things follow from computing it
        here and only here. First, it is the only boundary that spans a whole
        run — the injected runner's own stages (for an ingest: the pipeline
        call, then entity resolution, then the collection summary) each cover
        a fraction of it, and the clock starts at *creation*, so a job that
        waited on a busy semaphore counts its queue wait too. Second, the SPA
        renders this same number rather than deriving its own: two nearly
        equal durations floored independently disagree by a whole second
        whenever their difference straddles a boundary, which no amount of
        narrowing the two windows can fix.

        Args:
            state (IngestJobState): The job to process.
        """
        async with self._semaphores[state.kind]:
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

                Used for the frames ``_worker`` produces itself (the kind's
                ``started`` frame and both terminal events, per
                :data:`KIND_EVENTS`), which already run on the loop thread —
                right beside the status change each announces. Routing those
                through
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

            def _log(message: str, *, warning: bool = False) -> None:
                """Write one runner message to the log, tagged with the job.

                Every line a job produces carries the full ``job_id``, so
                ``docker logs | grep <job_id>`` reconstructs one run even when
                ``DOCINT_INGEST_CONCURRENCY`` lets two interleave.

                Args:
                    message (str): The runner's own message, verbatim.
                    warning (bool, optional): Log at WARNING instead of INFO.
                """
                write = logger.warning if warning else logger.info
                write(
                    "Job {} ({}) {}: {}",
                    state.job_id,
                    state.kind,
                    "warning" if warning else "progress",
                    message,
                )

            def _tee(event_name: str, payload: dict[str, Any]) -> None:
                """Mirror a runner event into the log.

                The runner's progress messages were written for a client that
                renders the latest and discards the rest, so they arrive per
                chunk — thousands per run. ``throttle`` decides which an
                operator sees; warnings are never throttled, and until now
                reached no log at all.

                Never raises: ``ingestion_pipeline`` re-raises whatever a
                progress callback threw via ``future.result()``, so an
                exception here would fail the batch it was only meant to
                describe.

                Args:
                    event_name (str): SSE event name.
                    payload (dict[str, Any]): JSON-serializable payload.
                """
                try:
                    message = str(payload.get("message") or "").strip()
                    if not message:
                        return
                    if event_name in PROGRESS_EVENTS:
                        for line in throttle.observe(message):
                            _log(line)
                    elif event_name == "warning":
                        _log(message, warning=True)
                except Exception:
                    pass

            def _push(event_name: str, payload: dict[str, Any]) -> None:
                """Publish an event from the worker thread (thread-safe).

                Handed to the injected runner as its ``push`` callable — the
                runner executes on a worker thread (via
                ``to_thread.run_sync``), so this genuinely needs the
                threadsafe hop back onto the loop thread. ``_worker``'s own
                frames use ``_emit`` instead; see its docstring for why.

                The log tee runs before the hop, on the worker thread, so the
                line lands when the event happened rather than a loop
                iteration later. loguru's sink is thread-safe.

                Args:
                    event_name (str): SSE event name.
                    payload (dict[str, Any]): JSON-serializable payload.
                """
                _tee(event_name, payload)
                frame = _frame(event_name, payload)
                loop.call_soon_threadsafe(self._dispatch, state, event_name, frame)

            throttle = ProgressLogThrottle(load_logging_env().progress_interval_s)
            names = KIND_EVENTS[state.kind]
            _emit(names["started"], {"collection": state.logical_name})
            await self._log_run_banner(state)
            try:
                result = await to_thread.run_sync(self._runner, state, _push)
            except Exception:
                # Release a held tick first: how far a stage got before it
                # died is the most useful line in a failed run.
                for line in throttle.flush():
                    _log(line)
                state.duration_s = state.elapsed_s()
                logger.exception(
                    "{} job failed | {}",
                    state.kind.capitalize(),
                    _summary_fields(state, None, failed=True),
                )
                state.status = JobStatus.FAILED
                state.error = names["failed_message"]
                state.finished_at = _utcnow()
                # Static protocol copy only: the exception text can carry
                # connection strings or file paths and never reaches a client.
                _emit(
                    "error",
                    {
                        "message": names["failed_message"],
                        "code": names["failed_code"],
                        "duration_ms": state.duration_ms,
                    },
                )
                return
            for line in throttle.flush():
                _log(line)
            state.empty = bool(result.get("empty", False))
            state.resolution = result.get("resolution")
            state.artifact = result.get("artifact")
            state.status = JobStatus.COMPLETED
            state.finished_at = _utcnow()
            state.duration_s = state.elapsed_s()
            logger.info(
                "{} job completed | {}",
                state.kind.capitalize(),
                _summary_fields(state, result.get("stats")),
            )
            terminal: dict[str, Any] = {
                "collection": state.logical_name,
                "empty": state.empty,
                "duration_ms": state.duration_ms,
            }
            if state.resolution is not None:
                terminal["resolution"] = state.resolution
            if state.artifact is not None:
                terminal["artifact"] = state.artifact
            _emit(names["complete"], terminal)

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
