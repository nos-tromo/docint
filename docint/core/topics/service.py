"""The topics service: build, discover and align runs over a collection.

Orchestration only: Qdrant access lives in :mod:`gather`, the maths in
:mod:`model` and :mod:`sample`, persistence in :mod:`store`. Every job holds
the collection's lock from start to finish, the fit included, and reports
through ``progress``, which is also its cancel channel: whatever it raises
ends the job once the lock is released and any half-built run is removed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from loguru import logger
from qdrant_client import models

from docint.core.topics.gather import (
    PAGE_SIZE,
    TIMESTAMP_KEYS,
    DenseVector,
    PointRow,
    and_filters,
    count,
    dense_vector,
    eligible_filter,
    ensure_topic_indexes,
    iter_rows,
    pending_filter,
    retrieve_candidates,
    scan_ids,
    topic_filter,
    write_assignments,
)
from docint.core.topics.labels import PASSAGES_PER_TOPIC, LabelRequest, TopicLabeller, fallback_label
from docint.core.topics.model import (
    Origin,
    TopicInfo,
    TopicTable,
    central_members,
    centres_for,
    floors_for,
    kept_members,
    merge_new,
)
from docint.core.topics.sample import (
    MIN_DF,
    TOP_N_WORDS,
    FitSample,
    auto_min_topic_size,
    build_sample,
    min_eligible,
    nr_topics_for,
    texts_within,
    top_ranked,
)
from docint.core.topics.store import TopicLock, TopicRun, TopicStore
from docint.utils.env_cfg import TopicsConfig
from docint.utils.topics_client import FitParams, FitResult, TopicsBusy, max_fit_points, texts_limit

Action = Literal["build", "discover", "align"]
STAGES = ("starting", "waiting", "scanning", "sampling", "fitting", "labelling", "assigning", "aligning")
HEARTBEAT_S = 5.0
LOCK_POLL_S = 1.0
BUSY_ATTEMPTS = 10
PENDING_PASSES = 5

Progress = Callable[[str, int, int], None]
Warn = Callable[[str, str], None]
FitFn = Callable[[Sequence[str], np.ndarray, FitParams], FitResult]

_ACTIONS = ("build", "discover", "align")

_PROGRESS_COPY = {
    "starting": "Starting topics run",
    "waiting": "Waiting for another topics run on this collection: {done} s",
    "scanning": "Scanning chunks: {done} chunks scanned",
    "sampling": "Reading the sample: {done}/{total} chunks read",
    "fitting": "Fitting topics: {done} s",
    "labelling": "Labelling topics: {done}/{total} topics labelled",
    "assigning": "Assigning topics: {done}/{total} chunks processed",
    "aligning": "Aligning topics: {done}/{total} chunks processed",
}

REFUSALS = {
    "too_few_chunks": "The collection has too few chunks with enough text to build topics.",
    "too_few_unassigned": "Too few unassigned chunks to look for new topics.",
    "no_run": "Build topics first.",
    "dimension_changed": "The collection's vectors changed size since the topics were built; rebuild them.",
    "collection_changed": "The collection was replaced while its topics were built.",
}

WARNINGS = {
    "none_found": "The fit found no topics; the collection keeps its current topics.",
    "none_new": "No new topics were found; the unassigned chunks stay unassigned.",
}

# Topic id to the (earliest, latest) member timestamp seen, UTC.
_Seen = dict[int, tuple[datetime, datetime]]


def progress_message(stage: str, done: int, total: int) -> str:
    """Render a stage's progress copy.

    Args:
        stage: One of :data:`STAGES`.
        done: Units done.
        total: Units in all, 0 when unknown.

    Returns:
        The static English message.

    Raises:
        ValueError: For an unknown stage.
    """
    template = _PROGRESS_COPY.get(stage)
    if template is None:
        raise ValueError(f"Unknown topics stage: {stage!r}")
    return template.format(done=done, total=total)


def progress_payload(stage: str, done: int, total: int) -> dict[str, Any]:
    """Build a progress frame's payload.

    Args:
        stage: One of :data:`STAGES`.
        done: Units done.
        total: Units in all, 0 when unknown.

    Returns:
        ``{"message", "stage", "done", "total"}``.
    """
    return {"message": progress_message(stage, done, total), "stage": stage, "done": done, "total": total}


class TopicsRefused(Exception):
    """A job that cannot run on the collection as it is; not a failure.

    Attributes:
        code: The refusal code, a key of :data:`REFUSALS`.
        message: Its static English message, naming nothing.
    """

    def __init__(self, code: str) -> None:
        """Look the message up.

        Args:
            code: The refusal code.
        """
        self.code = code
        self.message = REFUSALS[code]
        super().__init__(self.message)


@dataclass(frozen=True)
class TopicsOutcome:
    """What a job did.

    Attributes:
        action: The action run.
        run_id: The run after the job, ``None`` when none exists.
        topics: Topics in that run, 0 when none exists.
        added: Topics the job added.
        assigned: Points the job placed in a topic.
        empty: Whether the job found nothing to add.
        deferred: Whether the job did not run because the lock was held.
    """

    action: str
    run_id: str | None
    topics: int
    added: int
    assigned: int
    empty: bool
    deferred: bool = False

    def artifact(self) -> dict[str, Any]:
        """Return the job artifact.

        Returns:
            The action, run id and counts.
        """
        return {
            "action": self.action,
            "run_id": self.run_id,
            "topics": self.topics,
            "added": self.added,
            "assigned": self.assigned,
            "deferred": self.deferred,
        }


def _noop(_physical: str) -> None:
    """Do nothing.

    Args:
        _physical: Ignored.
    """


def _utcnow() -> datetime:
    """Return the current time.

    Returns:
        An aware UTC datetime.
    """
    return datetime.now(UTC)


@dataclass(frozen=True)
class TopicsDeps:
    """Everything the service talks to.

    Attributes:
        client: Qdrant client.
        fit: The fit call, ``TopicsClient.fit`` in production.
        store: The topic store.
        labeller: The label writer, ``None`` to name topics by their keywords.
        config: Topics configuration.
        embed_model: The embedding model id, recorded with each run.
        stop_words: Words kept out of keywords.
        stop_words_sha256: Their digest, recorded with each run.
        on_written: Called with the collection once per job that wrote assignments, failed or not.
        now: The clock.
        sleep: The sleep used between polls and busy retries.
    """

    client: Any
    fit: FitFn
    store: TopicStore
    labeller: TopicLabeller | None
    config: TopicsConfig
    embed_model: str
    stop_words: tuple[str, ...]
    stop_words_sha256: str
    on_written: Callable[[str], None] = _noop
    now: Callable[[], datetime] = _utcnow
    sleep: Callable[[float], None] = time.sleep


@dataclass(frozen=True)
class _Fitted:
    """A sample and the fit over it.

    Attributes:
        sample: The fitted points.
        vectors: Their vectors, one row each.
        labels: The fit's topic id per row.
        result: The fit response.
        min_topic_size: The minimum topic size used.
    """

    sample: FitSample
    vectors: np.ndarray
    labels: np.ndarray
    result: FitResult
    min_topic_size: int


@dataclass
class _Job:
    """What one job has written so far.

    Attributes:
        seen: Topic id to the earliest and latest member time written, UTC.
        wrote: Whether any assignment was written, so ``on_written`` must fire.
    """

    seen: _Seen = field(default_factory=dict)
    wrote: bool = False


def _lookup(payload: Mapping[str, Any], key: str) -> object:
    """Read a dotted key, flat or nested.

    Args:
        payload: A point payload.
        key: A dotted payload path.

    Returns:
        The value, or ``None``.
    """
    if key in payload:
        return payload[key]
    value: object = payload
    for part in key.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _timestamp(payload: Mapping[str, Any]) -> datetime | None:
    """Read a point's first parseable timestamp.

    Args:
        payload: A point payload.

    Returns:
        The time in UTC, a naive value read as UTC; ``None`` when none parses.
    """
    for key in TIMESTAMP_KEYS:
        value = _lookup(payload, key)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _widen(seen: _Seen, topic_id: int, when: datetime) -> None:
    """Widen one topic's seen range.

    Args:
        seen: The ranges, updated in place.
        topic_id: The topic.
        when: A member's time.
    """
    current = seen.get(topic_id)
    seen[topic_id] = (when, when) if current is None else (min(current[0], when), max(current[1], when))


def _iso_ranges(seen: _Seen) -> dict[int, tuple[str, str]]:
    """Render seen ranges as ISO-8601 UTC strings.

    Args:
        seen: The ranges.

    Returns:
        Topic id to ``(min, max)`` strings.
    """
    return {tid: (lo.astimezone(UTC).isoformat(), hi.astimezone(UTC).isoformat()) for tid, (lo, hi) in seen.items()}


def _inherit(sample: FitSample, overrides: dict[str, int]) -> dict[str, int]:
    """Give each duplicate copy its representative's topic.

    Args:
        sample: The fitted sample.
        overrides: Representative id to topic id.

    Returns:
        The overrides plus one entry per copy of an overridden representative.
    """
    copies = {copy: overrides[rep] for copy, rep in sample.copies.items() if rep in overrides}
    return {**overrides, **copies}


class TopicsService:
    """Builds, extends and maintains a collection's topics."""

    def __init__(self, deps: TopicsDeps) -> None:
        """Store the dependencies.

        Args:
            deps: What the service talks to.
        """
        self._deps = deps

    def run(
        self,
        action: Action,
        physical: str,
        *,
        logical: str,
        owner_created_at: Callable[[], str | None],
        progress: Progress,
        warn: Warn,
    ) -> TopicsOutcome:
        """Run one job under the collection's lock, waiting for it if needed.

        Args:
            action: ``build``, ``discover`` or ``align``.
            physical: Physical collection name.
            logical: Logical collection name, for the refusal log line.
            owner_created_at: Reads the ownership row's ``created_at`` now.
            progress: Receives ``(stage, done, total)``; a raise stops the job.
            warn: Receives ``(code, message)`` for a job that finds nothing.

        Returns:
            The outcome.

        Raises:
            ValueError: For an unknown action.
            TopicsRefused: When the job cannot run on the collection as it is.
        """
        if action not in _ACTIONS:
            raise ValueError(f"Unknown topics action: {action!r}")
        lock = self._wait_for_lock(physical, progress)
        job = _Job()
        try:
            outcome = self._run_locked(action, physical, owner_created_at, progress, warn, job)
        except TopicsRefused as exc:
            logger.info("Topics {} refused for '{}' | code={}", action, logical, exc.code)
            raise
        finally:
            # Writes from a cancelled or failed job are announced too.
            try:
                self._announce(physical, job)
            finally:
                lock.release()
        _log_outcome(outcome)
        return outcome

    def align_on_ingest(
        self,
        physical: str,
        *,
        owner_created_at: Callable[[], str | None],
        progress: Callable[[int, int], None],
    ) -> TopicsOutcome | None:
        """Place newly ingested points in the active run, without waiting.

        Args:
            physical: Physical collection name.
            owner_created_at: Reads the ownership row's ``created_at`` now.
            progress: Receives ``(done, total)``; a raise stops the stage.

        Returns:
            The outcome; a deferred one when another job holds the lock, and
            ``None`` when there is no usable run.
        """
        store = self._deps.store
        lock = store.try_lock(physical)
        if lock is None:
            outcome = TopicsOutcome("align", None, 0, 0, 0, empty=True, deferred=True)
            _log_outcome(outcome)
            return outcome
        job = _Job()
        try:
            created = owner_created_at()
            store.discard_stale(physical, owner_created_at=created)
            run = store.active(physical, owner_created_at=created)
            if run is None:
                return None
            ensure_topic_indexes(self._deps.client, physical)
            dense = dense_vector(self._deps.client, physical)
            if dense.dim != run.table.dim:
                logger.warning("Topic alignment after ingest skipped | code={}", "dimension_changed")
                return None
            outcome = self._align(physical, dense, run, lambda _stage, done, total: progress(done, total), job)
        finally:
            try:
                self._announce(physical, job)
            finally:
                lock.release()
        _log_outcome(outcome)
        return outcome

    def _announce(self, physical: str, job: _Job) -> None:
        """Tell ``on_written`` about a job's writes, never failing the job over it.

        Args:
            physical: Physical collection name.
            job: The job.
        """
        if not job.wrote:
            return
        try:
            self._deps.on_written(physical)
        except Exception:
            logger.exception("Announcing topic writes for '{}' failed", physical)

    def _wait_for_lock(self, physical: str, progress: Progress) -> TopicLock:
        """Poll the collection's lock until it is held.

        Args:
            physical: Physical collection name.
            progress: Receives a ``waiting`` frame before each poll's sleep.

        Returns:
            The held lock.
        """
        polls = 0
        while True:
            lock = self._deps.store.try_lock(physical)
            if lock is not None:
                return lock
            progress("waiting", int(polls * LOCK_POLL_S), 0)
            self._deps.sleep(LOCK_POLL_S)
            polls += 1

    def _run_locked(
        self,
        action: Action,
        physical: str,
        owner_created_at: Callable[[], str | None],
        progress: Progress,
        warn: Warn,
        job: _Job,
    ) -> TopicsOutcome:
        """Dispatch a job once the lock is held.

        Args:
            action: The action.
            physical: Physical collection name.
            owner_created_at: Reads the ownership row's ``created_at`` now.
            progress: The progress channel.
            warn: The warning channel.
            job: What the job has written.

        Returns:
            The outcome.

        Raises:
            TopicsRefused: ``no_run`` or ``dimension_changed`` for discover and align.
        """
        store, client = self._deps.store, self._deps.client
        created = owner_created_at()
        store.discard_stale(physical, owner_created_at=created)
        ensure_topic_indexes(client, physical)
        dense = dense_vector(client, physical)
        if action == "build":
            return self._build(physical, dense, created, owner_created_at, progress, warn, job)
        run = store.active(physical, owner_created_at=created)
        if run is None:
            raise TopicsRefused("no_run")
        if dense.dim != run.table.dim:
            raise TopicsRefused("dimension_changed")
        if action == "discover":
            return self._discover(physical, dense, run, progress, warn, job)
        return self._align(physical, dense, run, progress, job)

    def _build(
        self,
        physical: str,
        dense: DenseVector,
        created: str | None,
        owner_created_at: Callable[[], str | None],
        progress: Progress,
        warn: Warn,
        job: _Job,
    ) -> TopicsOutcome:
        """Fit a new run over the collection and make it the active one.

        Args:
            physical: Physical collection name.
            dense: The collection's dense vector.
            created: The ownership row's ``created_at`` when the job started.
            owner_created_at: Reads it again before activation.
            progress: The progress channel.
            warn: The warning channel.
            job: What the job has written.

        Returns:
            The outcome.

        Raises:
            TopicsRefused: ``too_few_chunks``, or ``collection_changed`` before activation.
        """
        deps, store = self._deps, self._deps.store
        fitted = self._fit_pool(physical, dense, eligible_filter(), "too_few_chunks", progress)
        if not any(t.size > 0 for t in fitted.result.topics):
            warn("none_found", WARNINGS["none_found"])
            current = store.active(physical, owner_created_at=created)
            topics = len(current.table.topics) if current is not None else 0
            return TopicsOutcome("build", current.run_id if current else None, topics, 0, 0, empty=True)
        table = self._fitted_table(fitted, "build", progress)
        at = self._now_iso()
        run_id = store.new_run_id(deps.now())
        record: dict[str, Any] = {
            "params": self._params_record(fitted.min_topic_size, dense),
            "dense": {"name": dense.name, "dim": dense.dim},
            "embed_model": deps.embed_model,
            "stop_words": {"count": len(deps.stop_words), "sha256": deps.stop_words_sha256},
            "owner_created_at": created,
            "created_at": at,
            "updated_at": at,
            "fits": [self._fit_entry("build", fitted)],
        }
        run = TopicRun(run_id, record, table)
        try:
            store.save(physical, run)
            overrides = {
                p.point_id: int(label)
                for p, label in zip(fitted.sample.points, fitted.labels, strict=True)
                if label >= 0
            }
            assigned = self._pending_loop(
                physical, dense, run_id, table, _inherit(fitted.sample, overrides), "assigning", progress, job
            )
            if owner_created_at() != created:
                raise TopicsRefused("collection_changed")
            store.activate(physical, run_id)
        except BaseException:
            store.delete_run(physical, run_id)
            raise
        store.prune(physical, keep=run_id)
        assigned += self._pending_loop(physical, dense, run_id, table, {}, "aligning", progress, job)
        saved = self._save_seen(physical, run, table, job)
        return TopicsOutcome("build", run_id, len(saved.table.topics), len(table.topics), assigned, empty=False)

    def _discover(
        self, physical: str, dense: DenseVector, run: TopicRun, progress: Progress, warn: Warn, job: _Job
    ) -> TopicsOutcome:
        """Fit the unassigned pool and append the topics that are new.

        Args:
            physical: Physical collection name.
            dense: The collection's dense vector.
            run: The active run.
            progress: The progress channel.
            warn: The warning channel.
            job: What the job has written.

        Returns:
            The outcome.

        Raises:
            TopicsRefused: ``too_few_unassigned``.
        """
        deps, run_id = self._deps, run.run_id
        assigned = self._pending_loop(physical, dense, run_id, run.table, {}, "aligning", progress, job)
        # Heal a pool an interrupted discover left behind; writes nothing in steady state.
        assigned += self._realign_pool(physical, dense, run_id, run.table, {}, "aligning", progress, job)
        run = self._save_seen(physical, run, run.table, job)
        pool = and_filters(topic_filter(run_id, -1), eligible_filter())
        fitted = self._fit_pool(physical, dense, pool, "too_few_unassigned", progress)
        # A member nearer another topic, old or new, is aligned with the rest of the pool instead.
        kept = kept_members(fitted.vectors, fitted.labels, run.table.centres, min_topic_size=fitted.min_topic_size)
        fitted = replace(fitted, labels=kept)
        candidates = self._candidates(fitted, "discover")
        table, mapping = merge_new(
            run.table,
            [info for info, _ in candidates],
            np.array([centre for _, centre in candidates]).reshape(len(candidates), dense.dim),
            min_similarity=deps.config.merge_min_similarity,
        )
        if not mapping:
            warn("none_new", WARNINGS["none_new"])
            return TopicsOutcome("discover", run_id, len(run.table.topics), 0, assigned, empty=True)
        table = self._labelled(table, fitted, {new: old for old, new in mapping.items()}, progress)
        fits = [*run.record.get("fits", []), {**self._fit_entry("discover", fitted), "kept": len(mapping)}]
        run = TopicRun(run_id, {**run.record, "fits": fits, "updated_at": self._now_iso()}, table)
        deps.store.save(physical, run)
        overrides = {
            p.point_id: mapping[int(label)]
            for p, label in zip(fitted.sample.points, fitted.labels, strict=True)
            if int(label) in mapping
        }
        assigned += self._realign_pool(
            physical, dense, run_id, table, _inherit(fitted.sample, overrides), "assigning", progress, job
        )
        assigned += self._pending_loop(physical, dense, run_id, table, {}, "aligning", progress, job)
        saved = self._save_seen(physical, run, table, job)
        return TopicsOutcome("discover", run_id, len(saved.table.topics), len(mapping), assigned, empty=False)

    def _align(self, physical: str, dense: DenseVector, run: TopicRun, progress: Progress, job: _Job) -> TopicsOutcome:
        """Place every pending point in the active run and store the seen ranges.

        Args:
            physical: Physical collection name.
            dense: The collection's dense vector.
            run: The active run.
            progress: The progress channel.
            job: What the job has written.

        Returns:
            The outcome.
        """
        assigned = self._pending_loop(physical, dense, run.run_id, run.table, {}, "aligning", progress, job)
        saved = self._save_seen(physical, run, run.table, job)
        return TopicsOutcome("align", run.run_id, len(saved.table.topics), 0, assigned, empty=False)

    def _fit_pool(
        self, physical: str, dense: DenseVector, flt: models.Filter, refusal: str, progress: Progress
    ) -> _Fitted:
        """Sample the points a filter selects and fit them.

        Args:
            physical: Physical collection name.
            dense: The collection's dense vector.
            flt: Which points may join the sample.
            refusal: The refusal code for a sample too small to fit.
            progress: The progress channel.

        Returns:
            The sample and its fit.

        Raises:
            TopicsRefused: With ``refusal`` when the sample is too small.
        """
        cfg, client = self._deps.config, self._deps.client
        limit = min(cfg.fit_max_points, max_fit_points(dense.dim))
        ids = top_ranked(_scanned(scan_ids(client, physical, scroll_filter=flt), progress), seed=cfg.seed, limit=limit)
        candidates = retrieve_candidates(
            client, physical, ids, dense, progress=lambda read: progress("sampling", read, len(ids))
        )
        sample = build_sample(candidates, min_text_chars=cfg.min_text_chars)
        mts = cfg.min_topic_size or auto_min_topic_size(sample.n)
        if sample.n < min_eligible(mts):
            raise TopicsRefused(refusal)
        params = FitParams(cfg.seed, mts, nr_topics_for(cfg.max_topics), TOP_N_WORDS, MIN_DF, self._deps.stop_words)
        texts = texts_within([p.text for p in sample.points], texts_limit(sample.n))
        vectors = np.stack([p.vector for p in sample.points])
        result = self._fit(texts, vectors, params, progress)
        return _Fitted(sample, vectors, np.asarray(result.labels, dtype=np.int64), result, mts)

    def _fit(self, texts: Sequence[str], vectors: np.ndarray, params: FitParams, progress: Progress) -> FitResult:
        """Call the fit on a daemon thread, heartbeating and retrying while busy.

        A cancel raised from a heartbeat abandons the worker: its HTTP call
        finishes on its own and its answer is dropped. Being a daemon, it never
        holds up interpreter exit.

        Args:
            texts: One text per row.
            vectors: Row matrix.
            params: Fit settings.
            progress: Receives ``fitting`` frames with the seconds elapsed.

        Returns:
            The fit response.

        Raises:
            TopicsBusy: When the service is still busy after :data:`BUSY_ATTEMPTS` tries.
        """
        elapsed = 0.0
        progress("fitting", 0, 0)
        for attempt in range(1, BUSY_ATTEMPTS + 1):
            future: Future[FitResult] = Future()
            worker = threading.Thread(
                target=_fill, args=(future, self._deps.fit, texts, vectors, params), name="topics-fit", daemon=True
            )
            worker.start()
            while not wait([future], timeout=HEARTBEAT_S).done:
                elapsed += HEARTBEAT_S
                progress("fitting", int(elapsed), 0)
            try:
                return future.result()
            except TopicsBusy as exc:
                if attempt == BUSY_ATTEMPTS:
                    raise
                remaining = max(0.0, exc.retry_after)
                while remaining > 0:
                    step = min(HEARTBEAT_S, remaining)
                    self._deps.sleep(step)
                    remaining -= step
                    elapsed += step
                    progress("fitting", int(elapsed), 0)
        raise AssertionError("unreachable")

    def _candidates(self, fitted: _Fitted, origin: Origin) -> list[tuple[TopicInfo, np.ndarray]]:
        """Turn a fit's topics into records named by their keywords.

        Args:
            fitted: The sample and its fit.
            origin: Who made the topics.

        Returns:
            One ``(record, centre)`` per topic with members, ascending id.
        """
        members = set(fitted.labels.tolist())
        topic_ids = sorted(t.id for t in fitted.result.topics if t.id in members)
        if not topic_ids:
            return []
        keywords = {t.id: t.keywords for t in fitted.result.topics}
        centres = centres_for(fitted.vectors, fitted.labels, topic_ids)
        floors = floors_for(fitted.vectors, fitted.labels, topic_ids, centres, self._deps.config.admit_percentile)
        at = self._now_iso()
        out: list[tuple[TopicInfo, np.ndarray]] = []
        for i, tid in enumerate(topic_ids):
            info = TopicInfo(
                id=tid,
                label=fallback_label(tid, [w for w, _ in keywords[tid]]),
                description="",
                keywords=keywords[tid],
                origin=origin,
                added_at=at,
                fit_size=int(np.count_nonzero(fitted.labels == tid)),
                floor=float(floors[i]),
            )
            out.append((info, centres[i]))
        return out

    def _fitted_table(self, fitted: _Fitted, origin: Origin, progress: Progress) -> TopicTable:
        """Build a labelled table from a fit.

        Args:
            fitted: The sample and its fit.
            origin: Who made the topics.
            progress: Receives ``labelling`` frames.

        Returns:
            The table, ids as the fit numbered them.
        """
        candidates = self._candidates(fitted, origin)
        table = TopicTable(
            topics=tuple(info for info, _ in candidates),
            centres=np.array([centre for _, centre in candidates]).reshape(len(candidates), fitted.vectors.shape[1]),
        )
        return self._labelled(table, fitted, {t.id: t.id for t in table.topics}, progress)

    def _labelled(
        self, table: TopicTable, fitted: _Fitted, fit_ids: Mapping[int, int], progress: Progress
    ) -> TopicTable:
        """Write labels for some of a table's topics.

        Args:
            table: The table.
            fitted: The fit the topics came from.
            fit_ids: Table id to the fit's id, for each topic to label.
            progress: Receives ``labelling`` frames.

        Returns:
            The table with those topics' labels and descriptions replaced.
        """
        labeller = self._deps.labeller
        if labeller is None or not fit_ids:
            return table
        requests = []
        for tid, fid in fit_ids.items():
            info, centre = table.topic(tid), table.centre(tid)
            if info is None or centre is None:
                continue
            rows = central_members(fitted.vectors, fitted.labels, fid, centre, PASSAGES_PER_TOPIC)
            passages = tuple(fitted.sample.points[r].text for r in rows)
            requests.append(LabelRequest(tid, tuple(w for w, _ in info.keywords), passages))
        names = labeller.label_all(requests, progress=lambda done, total: progress("labelling", done, total))
        topics = tuple(
            replace(t, label=names[t.id].label, description=names[t.id].description) if t.id in names else t
            for t in table.topics
        )
        return TopicTable(topics=topics, centres=table.centres)

    def _pending_loop(
        self,
        physical: str,
        dense: DenseVector,
        run_id: str,
        table: TopicTable,
        overrides: Mapping[str, int],
        stage: str,
        progress: Progress,
        job: _Job,
    ) -> int:
        """Place every point the run has not placed yet, in a few forward passes.

        Args:
            physical: Physical collection name.
            dense: The collection's dense vector.
            run_id: The run.
            table: Its topics.
            overrides: Point id to the topic it keeps regardless of alignment.
            stage: The stage its frames carry.
            progress: The progress channel.
            job: What the job has written, updated in place.

        Returns:
            Points placed in a topic (unassigned ones are written but not counted).
        """
        client, flt = self._deps.client, pending_filter(run_id)
        total = count(client, physical, flt)
        done = assigned = 0
        for _ in range(PENDING_PASSES):
            saw = False
            for page in iter_rows(client, physical, scroll_filter=flt, dense=dense):
                if not page:
                    continue
                saw = True
                topics = _topics_for(page, table, overrides)
                assigned += self._write(physical, run_id, page, topics, job)
                done += len(page)
                progress(stage, done, max(total, done))
            if not saw:
                break
        return assigned

    def _realign_pool(
        self,
        physical: str,
        dense: DenseVector,
        run_id: str,
        table: TopicTable,
        overrides: Mapping[str, int],
        stage: str,
        progress: Progress,
        job: _Job,
    ) -> int:
        """Move unassigned points into a topic of the table, in one forward pass.

        Args:
            physical: Physical collection name.
            dense: The collection's dense vector.
            run_id: The run.
            table: The table to align against.
            overrides: Pool sample members whose topic was kept, with its new id.
            stage: The stage its frames carry.
            progress: The progress channel.
            job: What the job has written, updated in place.

        Returns:
            Points moved into a topic; points left unassigned are not written.
        """
        client = self._deps.client
        pool = and_filters(topic_filter(run_id, -1), eligible_filter())
        total = count(client, physical, pool)
        done = moved = 0
        for page in iter_rows(client, physical, scroll_filter=pool, dense=dense):
            topics = _topics_for(page, table, overrides)
            kept = [(row, tid) for row, tid in zip(page, topics, strict=True) if tid >= 0]
            moved += self._write(physical, run_id, [r for r, _ in kept], [t for _, t in kept], job)
            done += len(page)
            progress(stage, done, max(total, done))
        return moved

    def _write(self, physical: str, run_id: str, rows: Sequence[PointRow], topics: Sequence[int], job: _Job) -> int:
        """Write one page's assignments and widen the seen ranges.

        Args:
            physical: Physical collection name.
            run_id: The run.
            rows: The rows.
            topics: Each row's topic.
            job: What the job has written, updated in place.

        Returns:
            Rows placed in a topic.
        """
        groups: dict[int, list[str]] = {}
        for row, tid in zip(rows, topics, strict=True):
            groups.setdefault(tid, []).append(row.point_id)
        if not groups:
            return 0
        # Set before the call: a write that fails midway may still have landed.
        job.wrote = True
        write_assignments(self._deps.client, physical, run_id, groups)
        placed = 0
        for row, tid in zip(rows, topics, strict=True):
            if tid < 0:
                continue
            placed += 1
            when = _timestamp(row.payload)
            if when is not None:
                _widen(job.seen, tid, when)
        return placed

    def _save_seen(self, physical: str, run: TopicRun, table: TopicTable, job: _Job) -> TopicRun:
        """Store a run with the job's seen ranges widened into it.

        Ranges only widen, so saving the same job's ranges twice changes nothing.

        Args:
            physical: Physical collection name.
            run: The run whose record is kept.
            table: The table to widen and store.
            job: What the job has written.

        Returns:
            The stored run.
        """
        widened = table.with_seen(_iso_ranges(job.seen))
        saved = TopicRun(run.run_id, {**run.record, "updated_at": self._now_iso()}, widened)
        self._deps.store.save(physical, saved)
        return saved

    def _params_record(self, min_topic_size: int, dense: DenseVector) -> dict[str, Any]:
        """Describe the settings a build ran with.

        Args:
            min_topic_size: The minimum topic size used.
            dense: The collection's dense vector.

        Returns:
            The ``params`` record.
        """
        cfg = self._deps.config
        return {
            "seed": cfg.seed,
            "min_topic_size": min_topic_size,
            "nr_topics": nr_topics_for(cfg.max_topics),
            "top_n_words": TOP_N_WORDS,
            "min_df": MIN_DF,
            "fit_max_points": min(cfg.fit_max_points, max_fit_points(dense.dim)),
            "min_text_chars": cfg.min_text_chars,
            "admit_percentile": cfg.admit_percentile,
            "merge_min_similarity": cfg.merge_min_similarity,
        }

    def _fit_entry(self, kind: str, fitted: _Fitted) -> dict[str, Any]:
        """Describe one fit for the run record.

        Args:
            kind: ``build`` or ``discover``.
            fitted: The sample and its fit.

        Returns:
            ``{kind, at, n, topics, outliers, versions, runtime, timings}``.
        """
        result = fitted.result
        return {
            "kind": kind,
            "at": self._now_iso(),
            "n": fitted.sample.n,
            "topics": len(result.topics),
            "outliers": result.outliers,
            "versions": dict(result.versions),
            "runtime": dict(result.runtime),
            "timings": dict(result.timings),
        }

    def _now_iso(self) -> str:
        """Return the clock's time.

        Returns:
            ISO-8601 UTC.
        """
        return self._deps.now().astimezone(UTC).isoformat()


def _fill(future: Future[FitResult], fit: FitFn, texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> None:
    """Run the fit and settle its future with the answer or the error.

    Args:
        future: The future the caller waits on.
        fit: The fit call.
        texts: One text per row.
        vectors: Row matrix.
        params: Fit settings.
    """
    if not future.set_running_or_notify_cancel():
        return
    try:
        future.set_result(fit(texts, vectors, params))
    except BaseException as exc:
        future.set_exception(exc)


def _scanned(ids: Iterable[str], progress: Progress) -> Iterable[str]:
    """Pass ids through, reporting the running count.

    Args:
        ids: Point ids.
        progress: Receives ``scanning`` frames.

    Yields:
        The ids unchanged.
    """
    n = 0
    for point_id in ids:
        n += 1
        if n % PAGE_SIZE == 0:
            progress("scanning", n, 0)
        yield point_id
    if n % PAGE_SIZE:
        progress("scanning", n, 0)


def _topics_for(rows: Sequence[PointRow], table: TopicTable, overrides: Mapping[str, int]) -> list[int]:
    """Choose each row's topic.

    Args:
        rows: The rows.
        table: The topics.
        overrides: Point id to the topic it keeps.

    Returns:
        One topic id per row: the override, else the alignment, else ``-1`` without a vector.
    """
    out = [-1] * len(rows)
    need = [(i, row.vector) for i, row in enumerate(rows) if row.point_id not in overrides and row.vector is not None]
    if need:
        aligned = table.align(np.stack([vector for _, vector in need]))
        for (i, _), tid in zip(need, aligned, strict=True):
            out[i] = int(tid)
    for i, row in enumerate(rows):
        if row.point_id in overrides:
            out[i] = overrides[row.point_id]
    return out


def _log_outcome(outcome: TopicsOutcome) -> None:
    """Log the one line a job leaves.

    Args:
        outcome: The outcome.
    """
    logger.info(
        "Topics {} | run={} topics={} added={} assigned={} deferred={}",
        outcome.action,
        outcome.run_id,
        outcome.topics,
        outcome.added,
        outcome.assigned,
        outcome.deferred,
    )
