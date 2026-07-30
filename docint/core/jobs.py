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

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

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
    hybrid: bool
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
    _progress_frame: str | None = field(default=None, repr=False)
    _terminal_frame: str | None = field(default=None, repr=False)

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
            self._warning_frames.append(frame)
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
            list[str]: ``ingestion_started``, then every ``warning``, then the
            latest ``ingestion_progress``, then the terminal frame — each
            omitted if it has not occurred yet.
        """
        frames: list[str] = []
        if self._started_frame is not None:
            frames.append(self._started_frame)
        frames.extend(self._warning_frames)
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
