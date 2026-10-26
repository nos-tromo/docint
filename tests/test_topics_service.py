"""Tests for the topics service: build, discover and align runs over a collection."""

from __future__ import annotations

import hashlib
import json
import pkgutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from loguru import logger
from qdrant_client import QdrantClient, models
from topics_fakes import (
    DENSE_NAME,
    DIM,
    SPARSE_NAME,
    FakeFit,
    RecordingClient,
    add_points,
    blob_vectors,
    make_collection,
)
from typing_extensions import override

import docint.core.topics
from docint.core.topics import gather, service
from docint.core.topics.gather import (
    RUN_FIELD,
    TOPIC_FIELD,
    DenseVector,
    count,
    parse_topic_value,
    pending_filter,
)
from docint.core.topics.labels import DEFAULT_TEMPLATE, TopicLabeller
from docint.core.topics.model import centres_for, floors_for
from docint.core.topics.sample import top_ranked
from docint.core.topics.service import (
    BUSY_ATTEMPTS,
    TopicsDeps,
    TopicsOutcome,
    TopicsRefused,
    TopicsService,
    progress_message,
    progress_payload,
)
from docint.core.topics.store import TopicRun, TopicStore
from docint.utils import topics_client
from docint.utils.env_cfg import TopicsConfig
from docint.utils.topics_client import FitParams, FitResult, TopicsBusy, encode_texts

COLL = "topics-coll"
CREATED = "2026-01-01T00:00:00+00:00"
BLOBS = np.eye(3, DIM, dtype=np.float64)
SECRET_TEXT = "SECRETPASSAGE"
SECRET_LABEL = "SECRETLABEL"


class Stop(Exception):
    """Raised by a test's progress callback to cancel a run."""


class Frames:
    """Records progress frames and warnings."""

    def __init__(self, hook: Callable[[str, int, int], None] | None = None) -> None:
        """Start empty.

        Args:
            hook: Called with each frame after it is recorded.
        """
        self.frames: list[tuple[str, int, int]] = []
        self.warnings: list[tuple[str, str]] = []
        self._hook = hook

    def progress(self, stage: str, done: int, total: int) -> None:
        """Record a progress frame.

        Args:
            stage: Stage name.
            done: Units done.
            total: Units in all.
        """
        self.frames.append((stage, done, total))
        if self._hook is not None:
            self._hook(stage, done, total)

    def warn(self, code: str, message: str) -> None:
        """Record a warning.

        Args:
            code: Warning code.
            message: Static message.
        """
        self.warnings.append((code, message))

    def stages(self) -> list[str]:
        """List the stages seen, in first-seen order.

        Returns:
            The distinct stage names.
        """
        out: list[str] = []
        for stage, _, _ in self.frames:
            if stage not in out:
                out.append(stage)
        return out


class WriteSpy:
    """Delegates to a client, recording the ids every payload write selects."""

    def __init__(self, inner: Any, *, fail: bool = False) -> None:
        """Wrap a client.

        Args:
            inner: The client to delegate to.
            fail: Whether every payload write raises.
        """
        self._inner = inner
        self._fail = fail
        self.written: list[str] = []

    def __getattr__(self, name: str) -> Any:
        """Delegate an attribute.

        Args:
            name: Attribute name.

        Returns:
            The inner client's attribute.
        """
        return getattr(self._inner, name)

    def batch_update_points(self, collection_name: str, update_operations: Sequence[Any], wait: bool = True) -> Any:
        """Record, then delegate or fail.

        Args:
            collection_name: Collection name.
            update_operations: The operations.
            wait: Whether to wait.

        Returns:
            The inner client's result.

        Raises:
            RuntimeError: When scripted to fail.
        """
        if self._fail:
            raise RuntimeError("write refused")
        for op in update_operations:
            for cond in op.set_payload.filter.must:
                self.written.extend(str(i) for i in cond.has_id)
        return self._inner.batch_update_points(
            collection_name=collection_name, update_operations=update_operations, wait=wait
        )


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    """Yield an in-memory Qdrant client holding an empty collection.

    Yields:
        The client.
    """
    c = QdrantClient(location=":memory:")
    make_collection(c, COLL)
    yield c
    c.close()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Return the topic store root.

    Args:
        tmp_path: Per-test directory.

    Returns:
        The root path.
    """
    return tmp_path / "topics"


def _cfg(**overrides: Any) -> TopicsConfig:
    """Build a test configuration.

    Args:
        **overrides: Fields to change.

    Returns:
        The configuration.
    """
    base = TopicsConfig(
        enabled=True,
        align_on_ingest=True,
        labels=True,
        fit_max_points=100_000,
        min_text_chars=10,
        seed=42,
        min_topic_size=0,
        max_topics=0,
        admit_percentile=5.0,
        merge_min_similarity=0.7,
    )
    return replace(base, **overrides)


def _labeller(label: str = "Synthetic label") -> TopicLabeller:
    """Build a labeller over a fake chat model.

    Args:
        label: The label every reply carries.

    Returns:
        The labeller.
    """

    def complete(prompt: str, response_format: dict[str, Any] | None) -> str:
        """Answer every prompt with the same label.

        Args:
            prompt: Ignored.
            response_format: Ignored.

        Returns:
            A JSON reply.
        """
        return json.dumps({"label": label, "description": "A synthetic description."})

    return TopicLabeller(complete, template=DEFAULT_TEMPLATE)


def _service(
    client: Any,
    root: Path,
    fit: Callable[[Sequence[str], np.ndarray, FitParams], FitResult] | None = None,
    *,
    cfg: TopicsConfig | None = None,
    labeller: TopicLabeller | None = None,
    sleep: Callable[[float], None] | None = None,
    on_written: Callable[[str], None] | None = None,
) -> TopicsService:
    """Build a service.

    Args:
        client: Qdrant client.
        root: Store root.
        fit: The fit callable; three-blob ``FakeFit`` when omitted.
        cfg: Configuration; :func:`_cfg` when omitted.
        labeller: The labeller; the fake one when omitted.
        sleep: The sleep callable; a no-op when omitted.
        on_written: The write hook; a no-op when omitted.

    Returns:
        The service.
    """
    return TopicsService(
        TopicsDeps(
            client=client,
            fit=fit if fit is not None else FakeFit(BLOBS),
            store=TopicStore(root),
            labeller=labeller if labeller is not None else _labeller(),
            config=cfg or _cfg(),
            embed_model="synthetic-embed",
            stop_words=("the", "and"),
            stop_words_sha256="0" * 64,
            on_written=on_written or (lambda _name: None),
            sleep=sleep or (lambda _s: None),
        )
    )


def _ids(n: int, offset: int = 0) -> list[str]:
    """Return ``n`` deterministic UUID strings.

    Args:
        n: How many.
        offset: First integer value minus one.

    Returns:
        The ids.
    """
    return [str(uuid.UUID(int=offset + i + 1)) for i in range(n)]


def _add_blobs(
    client: QdrantClient,
    *,
    per: int = 30,
    offset: int = 0,
    seed: int = 0,
    blobs: np.ndarray = BLOBS,
    noise: float = 0.05,
    payload: Callable[[int, str], dict[str, Any]] | None = None,
) -> list[str]:
    """Add ``per`` points around each blob centre.

    Args:
        client: Qdrant client.
        per: Points per blob.
        offset: Id offset.
        seed: Noise seed.
        blobs: Blob centres.
        noise: Noise scale.
        payload: Extra payload per ``(index, id)``.

    Returns:
        The ids, blob by blob.
    """
    vectors = blob_vectors(seed, blobs, per, noise)
    ids = _ids(len(vectors), offset)
    rows = []
    for i, (pid, vec) in enumerate(zip(ids, vectors, strict=True)):
        body = {"search_text": f"passage {pid} blob{i // per} shared words", "file_hash": f"h{i // per}"}
        if payload is not None:
            body.update(payload(i, pid))
        rows.append((pid, vec, body))
    add_points(client, COLL, rows)
    return ids


def _run(
    svc: TopicsService,
    action: service.Action,
    frames: Frames | None = None,
    *,
    created: Callable[[], str | None] = lambda: CREATED,
) -> TopicsOutcome:
    """Run one action.

    Args:
        svc: The service.
        action: The action.
        frames: Frame recorder; a fresh one when omitted.
        created: The ownership-row clock.

    Returns:
        The outcome.
    """
    rec = frames or Frames()
    return svc.run(action, COLL, logical="coll", owner_created_at=created, progress=rec.progress, warn=rec.warn)


def _vectors(client: QdrantClient, ids: Sequence[str]) -> np.ndarray:
    """Read points' stored dense vectors.

    Args:
        client: Qdrant client.
        ids: Point ids.

    Returns:
        One row per id, in order.
    """
    dense = DenseVector(DENSE_NAME, DIM)
    by_id = {str(p.id): dense.of(p) for p in client.retrieve(COLL, ids=list(ids), with_vectors=True)}
    rows = [by_id[pid] for pid in ids]
    return np.stack([row for row in rows if row is not None])


def _payloads(client: QdrantClient) -> dict[str, dict[str, Any]]:
    """Read every point's payload.

    Args:
        client: Qdrant client.

    Returns:
        Point id to payload.
    """
    points, _ = client.scroll(COLL, limit=10_000, with_payload=True)
    return {str(p.id): dict(p.payload or {}) for p in points}


def _topic_of(payload: dict[str, Any]) -> int | None:
    """Read a point's topic id.

    Args:
        payload: The point's payload.

    Returns:
        The topic id, or ``None`` when unassigned by any run.
    """
    parsed = parse_topic_value(payload.get(TOPIC_FIELD))
    return None if parsed is None else parsed[1]


def _topics(client: QdrantClient) -> dict[str, int | None]:
    """Read every point's topic id.

    Args:
        client: Qdrant client.

    Returns:
        Point id to topic id.
    """
    return {pid: _topic_of(p) for pid, p in _payloads(client).items()}


def _run_dirs(root: Path) -> set[str]:
    """List the stored run directories.

    Args:
        root: Store root.

    Returns:
        The run ids on disk.
    """
    directory = root / COLL
    return {p.name for p in directory.iterdir() if p.is_dir()} if directory.is_dir() else set()


def _fit_ids(fit: FakeFit, call: int = 0) -> list[str]:
    """Read the point ids sent to one fit call from its texts.

    Args:
        fit: The fake fit.
        call: Which call.

    Returns:
        The ids, in the order sent.
    """
    return [text.split()[1] for text in fit.requests[call][0]]


# Build


def test_build_fits_a_hash_chosen_sample_sorted_by_id(client: QdrantClient, root: Path) -> None:
    """The fit sees the seed's lowest-ranked ids, in id order.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    ids = _add_blobs(client)
    fit = FakeFit(BLOBS)
    _run(_service(client, root, fit, cfg=_cfg(fit_max_points=60)), "build")
    sent = _fit_ids(fit)
    assert sent == sorted(top_ranked(ids, seed=42, limit=60))


def test_build_fits_duplicates_once_and_copies_inherit(client: QdrantClient, root: Path) -> None:
    """An exact-duplicate vector is fitted once and its copy lands in the same topic.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    ids = _add_blobs(client)
    copies = _ids(5, offset=1000)
    # The same source vectors, so the stored copies are bit-identical.
    sources = blob_vectors(0, BLOBS, 30, 0.05)[:5]
    add_points(
        client,
        COLL,
        [(cid, vec, {"search_text": f"passage {cid} copied words"}) for cid, vec in zip(copies, sources, strict=True)],
    )
    # A member below its topic's floor: only inheritance puts its copy in the topic.
    stray, stray_copy = _ids(2, offset=700)
    vec = 0.7 * BLOBS[0] + np.sqrt(1 - 0.49) * np.eye(1, DIM, 5)[0]
    add_points(client, COLL, [(pid, vec, {"search_text": f"passage {pid} stray words"}) for pid in (stray, stray_copy)])
    fit = FakeFit(BLOBS)
    _run(_service(client, root, fit), "build")
    assert len(fit.requests[0][0]) == len(ids) + 1
    topics = _topics(client)
    for cid, original in zip([*copies, stray_copy], [*ids[:5], stray], strict=True):
        assert topics[cid] == topics[original] == 0


def test_build_assigns_every_eligible_chunk_and_activates_the_run(client: QdrantClient, root: Path) -> None:
    """Every non-coarse point is stamped with the new run, which becomes active.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    ids = _add_blobs(client)
    coarse = _ids(3, offset=500)
    add_points(
        client,
        COLL,
        [(pid, BLOBS[0], {"search_text": "coarse parent passage", "docint_hier_type": "coarse"}) for pid in coarse],
    )
    short = _ids(1, offset=600)[0]
    add_points(client, COLL, [(short, BLOBS[1], {"search_text": "x"})])
    written: list[str] = []
    frames = Frames()
    outcome = _run(_service(client, root, on_written=written.append), "build", frames)

    payloads = _payloads(client)
    for pid in [*ids, short]:
        assert payloads[pid][RUN_FIELD] == outcome.run_id
    for pid in coarse:
        assert RUN_FIELD not in payloads[pid]
    assert _topic_of(payloads[short]) == 1
    assert TopicStore(root).active_id(COLL) == outcome.run_id
    assert outcome.topics == 3 and outcome.added == 3 and not outcome.empty
    assert outcome.assigned == len(ids) + 1
    assert written == [COLL]
    assert frames.stages() == ["scanning", "sampling", "fitting", "labelling", "assigning"]


def test_build_twice_gives_identical_assignments(client: QdrantClient, root: Path) -> None:
    """Two builds over the same collection and seed place every point identically.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client, per=40)

    def digest() -> str:
        """Hash the sorted (point, topic) dump.

        Returns:
            The hex sha256.
        """
        dump = sorted((pid, -2 if t is None else t) for pid, t in _topics(client).items())
        return hashlib.sha256(json.dumps(dump).encode()).hexdigest()

    first = _run(_service(client, root, cfg=_cfg(fit_max_points=60)), "build")
    one = digest()
    second = _run(_service(client, root, cfg=_cfg(fit_max_points=60)), "build")
    assert first.run_id != second.run_id
    assert digest() == one


def test_a_rebuild_prunes_the_old_run_and_its_values_match_nothing(client: QdrantClient, root: Path) -> None:
    """After a rebuild only the new run is stored and no point carries the old run.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    old = _run(_service(client, root), "build")
    new = _run(_service(client, root), "build")
    assert old.run_id is not None and new.run_id is not None
    assert _run_dirs(root) == {new.run_id}
    old_filter = models.Filter(must=[models.FieldCondition(key=RUN_FIELD, match=models.MatchValue(value=old.run_id))])
    assert count(client, COLL, old_filter) == 0
    for tid in (-1, 0, 1, 2):
        flt = models.Filter(
            must=[models.FieldCondition(key=TOPIC_FIELD, match=models.MatchValue(value=f"{old.run_id}:{tid}"))]
        )
        assert count(client, COLL, flt) == 0


def test_sample_members_keep_their_fit_topic_even_below_the_floor(client: QdrantClient, root: Path) -> None:
    """A fitted member far from its centre keeps its fit topic though align would refuse it.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    stray = _ids(1, offset=700)[0]
    vec = 0.7 * BLOBS[0] + np.sqrt(1 - 0.49) * np.eye(1, DIM, 5)[0]
    add_points(client, COLL, [(stray, vec, {"search_text": f"passage {stray} stray words"})])
    outcome = _run(_service(client, root), "build")
    assert outcome.run_id is not None
    run = TopicStore(root).load(COLL, outcome.run_id)
    assert run is not None
    assert int(run.table.align(vec[None, :])[0]) == -1
    assert _topics(client)[stray] == 0


def test_a_point_without_a_dense_vector_is_written_unassigned(client: QdrantClient, root: Path) -> None:
    """A point with no dense vector is stamped -1, so it never stays pending.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    bare = _ids(1, offset=650)[0]
    client.upsert(
        COLL,
        points=[
            models.PointStruct(
                id=bare,
                vector={SPARSE_NAME: models.SparseVector(indices=[1], values=[1.0])},
                payload={"search_text": "sparse only passage"},
            )
        ],
        wait=True,
    )
    outcome = _run(_service(client, root), "build")
    assert outcome.run_id is not None
    assert _topics(client)[bare] == -1
    assert count(client, COLL, pending_filter(outcome.run_id)) == 0


def test_a_build_with_too_few_chunks_is_refused_and_writes_nothing(client: QdrantClient, root: Path) -> None:
    """Mostly short chunks shrink the sample below the minimum: refused, nothing written.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client, per=20, payload=lambda i, _pid: {"search_text": "short"} if i % 4 else {})
    fit = FakeFit(BLOBS)
    with pytest.raises(TopicsRefused) as caught:
        _run(_service(client, root, fit), "build")
    assert caught.value.code == "too_few_chunks"
    assert caught.value.message == "The collection has too few chunks with enough text to build topics."
    assert fit.requests == []
    assert all(RUN_FIELD not in p for p in _payloads(client).values())
    assert _run_dirs(root) == set()
    assert TopicStore(root).try_lock(COLL) is not None


def test_a_fit_without_topics_warns_none_found_and_keeps_the_current_run(client: QdrantClient, root: Path) -> None:
    """A fit with no topics warns and leaves the active run and its assignments alone.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    first = _run(_service(client, root), "build")
    before = _payloads(client)
    frames = Frames()
    outcome = _run(_service(client, root, FakeFit(BLOBS, radius=-1.0)), "build", frames)
    assert frames.warnings == [("none_found", "The fit found no topics; the collection keeps its current topics.")]
    assert outcome.empty and outcome.run_id == first.run_id and outcome.topics == 3
    assert TopicStore(root).active_id(COLL) == first.run_id
    assert _payloads(client) == before


# Fit, busy, limits


def test_a_busy_service_is_retried_after_its_retry_after(client: QdrantClient, root: Path) -> None:
    """A busy answer waits its Retry-After in heartbeat slices, then fits again.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    fit = FakeFit(BLOBS, raise_first=TopicsBusy("busy", retry_after=12.0))
    sleeps: list[float] = []
    frames = Frames()
    outcome = _run(_service(client, root, fit, sleep=sleeps.append), "build", frames)
    assert sleeps == [5.0, 5.0, 2.0]
    assert len(fit.requests) == 2
    assert outcome.topics == 3
    assert [f for f in frames.frames if f[0] == "fitting"] == [
        ("fitting", 0, 0),
        ("fitting", 5, 0),
        ("fitting", 10, 0),
        ("fitting", 12, 0),
    ]


def test_a_service_that_stays_busy_fails_after_the_last_attempt(client: QdrantClient, root: Path) -> None:
    """The busy error is re-raised once every attempt was refused, and the lock is freed.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    calls: list[int] = []

    def busy(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        """Always refuse.

        Args:
            texts: Ignored.
            vectors: Ignored.
            params: Ignored.

        Raises:
            TopicsBusy: Always.
        """
        calls.append(1)
        raise TopicsBusy("busy", retry_after=0.0)

    with pytest.raises(TopicsBusy):
        _run(_service(client, root, busy), "build")
    assert len(calls) == BUSY_ATTEMPTS
    assert _run_dirs(root) == set()
    assert TopicStore(root).try_lock(COLL) is not None


def test_texts_sent_to_the_fit_are_cut_and_within_the_service_limit(
    client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long non-Latin texts are cut to 2,000 characters and to the texts-part limit.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch.
    """
    monkeypatch.setattr(service, "texts_limit", lambda n: n * 3000)
    _add_blobs(client, payload=lambda i, _pid: {"search_text": "語" * 3000 + str(i)})
    fit = FakeFit(BLOBS)
    _run(_service(client, root, fit), "build")
    texts = fit.requests[0][0]
    assert all(len(t) <= 2000 for t in texts)
    assert len(encode_texts(texts)) <= len(texts) * 3000


def test_the_sample_is_capped_by_the_vector_dimension(
    client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service's per-dimension point cap binds below the configured maximum.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch.
    """
    monkeypatch.setattr(topics_client, "SERVICE_MAX_POINTS", 480)
    monkeypatch.setattr(topics_client, "VALUES_PER_POINT", 1)
    _add_blobs(client)
    fit = FakeFit(BLOBS)
    _run(_service(client, root, fit, cfg=_cfg(fit_max_points=100_000)), "build")
    assert len(fit.requests[0][0]) == 480 // DIM


# Cancel, failure, recreate


def test_a_cancel_during_the_fit_stops_the_job_and_leaves_no_new_run(
    client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel raised from a fit heartbeat ends the job, frees the lock and stores nothing.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch.
    """
    _add_blobs(client)
    first = _run(_service(client, root), "build")
    monkeypatch.setattr(service, "HEARTBEAT_S", 0.01)
    started, release = threading.Event(), threading.Event()
    inner = FakeFit(BLOBS)

    def slow(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        """Block until released.

        Args:
            texts: Texts.
            vectors: Vectors.
            params: Settings.

        Returns:
            The fake result.
        """
        started.set()
        release.wait(10)
        return inner(texts, vectors, params)

    def cancel(stage: str, done: int, total: int) -> None:
        """Cancel once the fit is running.

        Args:
            stage: Stage name.
            done: Units done.
            total: Units in all.

        Raises:
            Stop: On a fit heartbeat after the fit started.
        """
        if stage == "fitting" and started.is_set():
            raise Stop

    try:
        with pytest.raises(Stop):
            _run(_service(client, root, slow), "build", Frames(cancel))
        assert TopicStore(root).active_id(COLL) == first.run_id
        assert _run_dirs(root) == {first.run_id}
        assert TopicStore(root).try_lock(COLL) is not None
    finally:
        release.set()


def test_the_fit_runs_on_a_daemon_thread(client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An abandoned fit worker can never hold up interpreter exit.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch.
    """
    started: list[tuple[str, bool]] = []

    class Recording(threading.Thread):
        """A thread that records its name and daemon flag on start."""

        @override
        def start(self) -> None:
            """Record, then start."""
            started.append((self.name, self.daemon))
            super().start()

    monkeypatch.setattr(threading, "Thread", Recording)
    _add_blobs(client)
    _run(_service(client, root), "build")
    assert [daemon for name, daemon in started if name == "topics-fit"] == [True]


def test_a_failure_in_the_write_pass_deletes_the_new_run_and_keeps_the_old_one_active(
    client: QdrantClient, root: Path
) -> None:
    """A write failure before activation removes the half-built run.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    first = _run(_service(client, root), "build")
    written: list[str] = []
    with pytest.raises(RuntimeError, match="write refused"):
        _run(_service(WriteSpy(client, fail=True), root, on_written=written.append), "build")
    # A refused write may still have landed in part, so it is announced.
    assert written == [COLL]
    assert TopicStore(root).active_id(COLL) == first.run_id
    assert _run_dirs(root) == {first.run_id}
    assert TopicStore(root).try_lock(COLL) is not None


def test_build_refuses_activation_when_the_collection_was_recreated(client: QdrantClient, root: Path) -> None:
    """An ownership row recreated mid-build stops activation and deletes the run.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    clock = {"value": CREATED}

    def recreate(stage: str, done: int, total: int) -> None:
        """Recreate the collection's ownership row during the write pass.

        Args:
            stage: Stage name.
            done: Units done.
            total: Units in all.
        """
        if stage == "assigning":
            clock["value"] = "2026-02-02T00:00:00+00:00"

    with pytest.raises(TopicsRefused) as caught:
        _run(_service(client, root), "build", Frames(recreate), created=lambda: clock["value"])
    assert caught.value.code == "collection_changed"
    assert caught.value.message == "The collection was replaced while its topics were built."
    assert _run_dirs(root) == set()
    assert TopicStore(root).active_id(COLL) is None


def _replacing(fit: FakeFit, clock: dict[str, str]) -> Callable[[Sequence[str], np.ndarray, FitParams], FitResult]:
    """Wrap a fit so the collection is replaced while it runs.

    Args:
        fit: The fit to delegate to.
        clock: The ownership-row clock, moved by the wrapper.

    Returns:
        The wrapped fit.
    """

    def replacing(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        """Replace the collection, then fit.

        Args:
            texts: Texts.
            vectors: Vectors.
            params: Settings.

        Returns:
            The fake result.
        """
        clock["value"] = "2026-02-02T00:00:00+00:00"
        return fit(texts, vectors, params)

    return replacing


def test_build_refuses_before_writing_when_the_collection_is_replaced_during_the_fit(
    client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A collection replaced while the fit runs gets neither a run nor a payload write.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch.
    """
    _add_blobs(client)
    saved: list[str] = []
    save = TopicStore.save

    def recording(self: TopicStore, physical: str, run: TopicRun) -> None:
        """Record the run's id, then save it.

        Args:
            self: The store.
            physical: Collection name.
            run: The run.
        """
        saved.append(run.run_id)
        save(self, physical, run)

    monkeypatch.setattr(TopicStore, "save", recording)
    clock = {"value": CREATED}
    spy = WriteSpy(client)
    with pytest.raises(TopicsRefused) as caught:
        _run(_service(spy, root, _replacing(FakeFit(BLOBS), clock)), "build", created=lambda: clock["value"])
    assert caught.value.code == "collection_changed"
    assert spy.written == []
    assert saved == []
    assert _run_dirs(root) == set()
    assert TopicStore(root).active_id(COLL) is None


def test_discover_refuses_before_writing_when_the_collection_is_replaced_during_the_fit(
    client: QdrantClient, root: Path
) -> None:
    """A collection replaced while the pool is fitted gets neither the new topics nor a payload write.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client, per=60)
    built = _run(_service(client, root, FakeFit(BLOBS[:2])), "build")
    assert built.run_id is not None
    clock = {"value": CREATED}
    spy = WriteSpy(client)
    with pytest.raises(TopicsRefused) as caught:
        _run(_service(spy, root, _replacing(FakeFit(BLOBS), clock)), "discover", created=lambda: clock["value"])
    assert caught.value.code == "collection_changed"
    assert spy.written == []
    run = TopicStore(root).load(COLL, built.run_id)
    assert run is not None
    assert len(run.table.topics) == 2
    assert [f["kind"] for f in run.record["fits"]] == ["build"]


def test_a_discover_on_a_collection_replaced_during_the_re_align_ends_refused(client: QdrantClient, root: Path) -> None:
    """A discover cannot restore a run removed by a collection deletion during re-alignment.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client, per=60)
    built = _run(_service(client, root, FakeFit(BLOBS[:2])), "build")
    assert built.run_id is not None
    clock = {"value": CREATED}

    def replace_collection(stage: str, done: int, total: int) -> None:
        """Replace the collection once the re-align pass has written a page.

        Args:
            stage: Stage name.
            done: Units done.
            total: Units in all.
        """
        if stage == "assigning":
            clock["value"] = "2026-02-02T00:00:00+00:00"
            TopicStore(root).delete_collection(COLL)

    written: list[str] = []
    svc = _service(client, root, FakeFit(BLOBS), on_written=written.append)
    with pytest.raises(TopicsRefused) as caught:
        _run(svc, "discover", Frames(replace_collection), created=lambda: clock["value"])
    assert caught.value.code == "collection_changed"
    assert written == [COLL]
    assert TopicStore(root).active(COLL, owner_created_at=clock["value"]) is None
    assert _run_dirs(root) == set()
    with pytest.raises(TopicsRefused) as later:
        _run(_service(client, root), "align", created=lambda: clock["value"])
    assert later.value.code == "no_run"
    assert _run_dirs(root) == set()


@pytest.mark.parametrize("action", ["build", "discover", "align", "ingest"])
def test_alignment_cannot_restore_a_deleted_collections_run(
    client: QdrantClient,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: Literal["build", "discover", "align", "ingest"],
) -> None:
    """Replacement during a catch-up or align pass refuses without restoring deleted run files.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch.
        action: Which alignment path runs.
    """
    _add_blobs(client)
    if action != "build":
        _run(_service(client, root), "build")
        _add_blobs(client, per=2, offset=1000)
    else:
        activate = TopicStore.activate

        def with_pending(self: TopicStore, physical: str, run_id: str) -> None:
            """Add points just after activation so the build's catch-up has work.

            Args:
                self: The store.
                physical: Collection name.
                run_id: The run being activated.
            """
            activate(self, physical, run_id)
            _add_blobs(client, per=2, offset=1000)

        monkeypatch.setattr(TopicStore, "activate", with_pending)
    clock = {"value": CREATED}

    def deleted(stage: str, done: int, total: int) -> None:
        """Run the deletion cascade after the alignment page is written.

        Args:
            stage: Stage name.
            done: Units done.
            total: Units in all.
        """
        if stage == "aligning":
            clock["value"] = "2026-02-02T00:00:00+00:00"
            TopicStore(root).delete_collection(COLL)

    svc = _service(client, root)
    if action == "ingest":
        result = svc.align_on_ingest(
            COLL,
            owner_created_at=lambda: clock["value"],
            progress=lambda done, total: deleted("aligning", done, total),
        )
        assert result is None
    else:
        with pytest.raises(TopicsRefused) as caught:
            _run(svc, action, Frames(deleted), created=lambda: clock["value"])
        assert caught.value.code == "collection_changed"
    assert _run_dirs(root) == set()
    assert TopicStore(root).active_id(COLL) is None


@pytest.mark.parametrize("action", ["build", "discover"])
def test_empty_fit_on_a_replaced_collection_is_refused(
    client: QdrantClient, root: Path, action: service.Action
) -> None:
    """An empty fit after replacement reports collection_changed instead of a normal empty result.

    Args:
        client: Qdrant client.
        root: Store root.
        action: The fitted action.
    """
    _add_blobs(client, per=60)
    if action == "discover":
        _run(_service(client, root, FakeFit(BLOBS[:2])), "build")
    clock = {"value": CREATED}
    frames = Frames()
    empty = FakeFit(BLOBS, radius=-1.0)
    with pytest.raises(TopicsRefused) as caught:
        _run(_service(client, root, _replacing(empty, clock)), action, frames, created=lambda: clock["value"])
    assert caught.value.code == "collection_changed"
    assert frames.warnings == []


# Concurrency


def test_points_written_during_a_build_are_aligned_before_the_lock_is_released(
    client: QdrantClient, root: Path
) -> None:
    """Points upserted while the write pass runs carry the new run when the job ends.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    late = _ids(6, offset=900)
    state = {"added": False}

    def upsert(stage: str, done: int, total: int) -> None:
        """Upsert new points on the first write frame.

        Args:
            stage: Stage name.
            done: Units done.
            total: Units in all.
        """
        if stage == "assigning" and not state["added"]:
            state["added"] = True
            add_points(client, COLL, [(pid, BLOBS[0], {"search_text": "late words"}) for pid in late])

    outcome = _run(_service(client, root), "build", Frames(upsert))
    payloads = _payloads(client)
    assert all(payloads[pid][RUN_FIELD] == outcome.run_id for pid in late)
    assert all(_topic_of(payloads[pid]) == 0 for pid in late)
    assert TopicStore(root).try_lock(COLL) is not None


def test_the_lock_is_held_across_the_fit(client: QdrantClient, root: Path) -> None:
    """Another store cannot take the collection's lock while the fit runs.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    inner = FakeFit(BLOBS)
    seen: list[bool] = []

    def probing(texts: Sequence[str], vectors: np.ndarray, params: FitParams) -> FitResult:
        """Probe the lock, then fit.

        Args:
            texts: Texts.
            vectors: Vectors.
            params: Settings.

        Returns:
            The fake result.
        """
        seen.append(TopicStore(root).try_lock(COLL) is None)
        return inner(texts, vectors, params)

    _run(_service(client, root, probing), "build")
    assert seen == [True]
    assert TopicStore(root).try_lock(COLL) is not None


def test_a_waiting_job_reports_progress_until_the_lock_frees(client: QdrantClient, root: Path) -> None:
    """A job polls a held lock, reporting the wait, and runs once it frees.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    held = TopicStore(root).try_lock(COLL)
    assert held is not None
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        """Release the other holder on the third poll.

        Args:
            seconds: Requested sleep.
        """
        sleeps.append(seconds)
        if len(sleeps) == 3:
            held.release()

    frames = Frames()
    outcome = _run(_service(client, root, sleep=sleep), "build", frames)
    assert [f for f in frames.frames if f[0] == "waiting"] == [("waiting", 0, 0), ("waiting", 1, 0), ("waiting", 2, 0)]
    assert sleeps == [1.0, 1.0, 1.0]
    assert outcome.topics == 3


# Align


def test_align_assigns_only_pending_points_and_never_touches_assigned_ones(client: QdrantClient, root: Path) -> None:
    """Align writes the new points only and leaves placed ones as they were.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    built = _run(_service(client, root), "build")
    before = _payloads(client)
    fresh = _add_blobs(client, per=2, offset=800, noise=0.0)
    spy = WriteSpy(client)
    outcome = _run(_service(spy, root), "align")
    assert sorted(spy.written) == sorted(fresh)
    after = _payloads(client)
    assert {pid: after[pid] for pid in before} == before
    assert outcome.run_id == built.run_id and outcome.assigned == len(fresh) and outcome.added == 0
    assert [_topic_of(after[pid]) for pid in fresh] == [0, 0, 1, 1, 2, 2]


def test_a_reupserted_point_is_pending_and_aligned_again(client: QdrantClient, root: Path) -> None:
    """A re-ingested point loses its keys, becomes pending and is placed again.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    ids = _add_blobs(client)
    built = _run(_service(client, root), "build")
    assert built.run_id is not None
    add_points(client, COLL, [(ids[40], BLOBS[1], {"search_text": "again"})])
    assert count(client, COLL, pending_filter(built.run_id)) == 1
    _run(_service(client, root), "align")
    assert count(client, COLL, pending_filter(built.run_id)) == 0
    assert _topics(client)[ids[40]] == 1


def test_align_without_a_run_is_refused_with_no_run(client: QdrantClient, root: Path) -> None:
    """Align needs a built run.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    with pytest.raises(TopicsRefused) as caught:
        _run(_service(client, root), "align")
    assert caught.value.code == "no_run" and caught.value.message == "Build topics first."


def test_align_with_a_changed_dimension_is_refused(client: QdrantClient, root: Path) -> None:
    """A collection whose vectors changed size cannot be aligned against old centres.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    _run(_service(client, root), "build")
    client.delete_collection(COLL)
    make_collection(client, COLL, dim=4)
    add_points(client, COLL, [(_ids(1)[0], [1.0, 0.0, 0.0, 0.0], {"search_text": "four wide"})])
    with pytest.raises(TopicsRefused) as caught:
        _run(_service(client, root), "align")
    assert caught.value.code == "dimension_changed"


# Discover


def test_discover_appends_only_novel_topics_and_never_changes_existing(client: QdrantClient, root: Path) -> None:
    """A new blob becomes one appended topic; the built topics stay exactly as they were.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    ids = _add_blobs(client, per=60)
    built = _run(_service(client, root, FakeFit(BLOBS[:2])), "build")
    assert built.run_id is not None
    store = TopicStore(root)
    old = store.load(COLL, built.run_id)
    assert old is not None
    outcome = _run(_service(client, root, FakeFit(BLOBS), labeller=_labeller("Appended")), "discover")
    new = store.load(COLL, built.run_id)
    assert new is not None
    assert outcome.run_id == built.run_id and outcome.added == 1 and outcome.topics == 3
    assert [t.to_json() for t in new.table.topics[:2]] == [t.to_json() for t in old.table.topics]
    assert np.array_equal(new.table.centres[:2], old.table.centres)
    added = new.table.topics[2]
    assert added.id == 2 and added.origin == "discover" and added.label == "Appended"
    assert [f["kind"] for f in new.record["fits"]] == ["build", "discover"]
    assert new.record["fits"][1]["kept"] == 1
    # Every pool member was fitted into the kept topic, so each takes it, floor or not.
    topics = _topics(client)
    assert all(topics[pid] == 2 for pid in ids[120:]) and outcome.assigned == 60


def test_discover_moves_only_unassigned_chunks(client: QdrantClient, root: Path) -> None:
    """Only points that were unassigned change, and only into a new topic.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client, per=60)
    cfg = _cfg(fit_max_points=50)
    _run(_service(client, root, FakeFit(BLOBS[:2]), cfg=cfg), "build")
    before = _payloads(client)
    spy = WriteSpy(client)
    outcome = _run(_service(spy, root, FakeFit(BLOBS), cfg=cfg), "discover")
    after = _payloads(client)
    moved = [pid for pid in before if after[pid] != before[pid]]
    assert moved and len(moved) < sum(1 for p in before.values() if _topic_of(p) == -1)
    assert all(_topic_of(before[pid]) == -1 and _topic_of(after[pid]) == 2 for pid in moved)
    assert sorted(spy.written) == sorted(moved)
    assert outcome.assigned == len(moved)


def test_discover_keeps_a_pool_member_only_where_its_new_topic_is_nearest(client: QdrantClient, root: Path) -> None:
    """Leftovers of an existing topic fitted into a new one stay unassigned and stay out of its floor.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    ids = _add_blobs(client, per=60)
    built = _run(_service(client, root, FakeFit(BLOBS[:2])), "build")
    assert built.run_id is not None
    store = TopicStore(root)
    old = store.load(COLL, built.run_id)
    assert old is not None
    # Later chunks nearest topic 0 but under its floor, so they wait in the unassigned pool.
    leftovers = _ids(6, offset=900)
    noise = np.random.default_rng(1).normal(0.0, 0.01, (6, DIM))
    vectors = 0.6 * BLOBS[0] + 0.8 * np.eye(1, DIM, 5)[0] + noise
    add_points(
        client,
        COLL,
        [
            (pid, vec, {"search_text": f"passage {pid} later words"})
            for pid, vec in zip(leftovers, vectors, strict=True)
        ],
    )
    _run(_service(client, root), "align")
    assert (_vectors(client, leftovers) @ old.table.centres.T).argmax(axis=1).tolist() == [0] * 6
    assert [_topics(client)[pid] for pid in leftovers] == [-1] * 6

    # One pool topic takes every pool chunk, the leftovers included.
    outcome = _run(_service(client, root, FakeFit(BLOBS[2:], radius=2.0)), "discover")

    new = store.load(COLL, built.run_id)
    assert new is not None
    topics = _topics(client)
    assert [topics[pid] for pid in leftovers] == [-1] * 6
    assert all(topics[pid] == 2 for pid in ids[120:])
    members = sorted(ids[120:])
    member_vectors = _vectors(client, members)
    labels = np.zeros(len(members), dtype=np.int64)
    centre = centres_for(member_vectors, labels, [0])
    added = new.table.topics[2]
    assert added.fit_size == len(members)
    assert added.floor == pytest.approx(floors_for(member_vectors, labels, [0], centre, 5.0)[0])
    assert np.array_equal(new.table.centres[2], centre[0])
    assert [t.to_json() for t in new.table.topics[:2]] == [t.to_json() for t in old.table.topics]
    assert new.table.centres[:2].tobytes() == old.table.centres.tobytes()
    assert new.table.floors[:2].tolist() == old.table.floors.tolist()
    assert outcome.added == 1 and outcome.assigned == len(members)


def test_discover_with_nothing_novel_warns_none_new(client: QdrantClient, root: Path) -> None:
    """A pool whose topics repeat existing ones warns and changes nothing.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client, per=60)
    cfg = _cfg(fit_max_points=60, admit_percentile=80.0)
    built = _run(_service(client, root, cfg=cfg), "build")
    before = _payloads(client)
    frames = Frames()
    outcome = _run(_service(client, root, cfg=cfg), "discover", frames)
    assert frames.warnings == [("none_new", "No new topics were found; the unassigned chunks stay unassigned.")]
    assert outcome.empty and outcome.added == 0 and outcome.topics == 3 and outcome.run_id == built.run_id
    assert _payloads(client) == before


def test_discover_with_too_few_unassigned_is_refused(client: QdrantClient, root: Path) -> None:
    """A pool smaller than a fit needs is refused with its own code.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    _run(_service(client, root), "build")
    with pytest.raises(TopicsRefused) as caught:
        _run(_service(client, root), "discover")
    assert caught.value.code == "too_few_unassigned"


def _strand_a_discover(
    client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch, cfg: TopicsConfig, written: list[str]
) -> tuple[str, list[str]]:
    """Build two topics, then cancel a discover after its first assigning page.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch, used to shrink pages.
        cfg: Configuration for both jobs.
        written: Collects the discover's ``on_written`` calls.

    Returns:
        The run id and the third blob's points the discover left unassigned.
    """
    monkeypatch.setattr(gather, "PAGE_SIZE", 16)
    ids = _add_blobs(client, per=150)
    built = _run(_service(client, root, FakeFit(BLOBS[:2]), cfg=cfg), "build")
    assert built.run_id is not None

    def stop(stage: str, done: int, total: int) -> None:
        """Cancel on the first assigning frame.

        Args:
            stage: Stage name.
            done: Units done.
            total: Units in all.

        Raises:
            Stop: On the first assigning frame.
        """
        if stage == "assigning":
            raise Stop

    with pytest.raises(Stop):
        _run(_service(client, root, FakeFit(BLOBS), cfg=cfg, on_written=written.append), "discover", Frames(stop))
    topics = _topics(client)
    return built.run_id, [pid for pid in ids[300:] if topics[pid] == -1]


@pytest.mark.parametrize(("percentile", "exit_code"), [(5.0, "too_few_unassigned"), (80.0, "none_new")])
def test_a_discover_interrupted_while_assigning_is_healed_by_the_next_discover(
    client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch, percentile: float, exit_code: str
) -> None:
    """The next discover aligns a stranded pool first, whichever way it then ends.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch.
        percentile: Admission percentile; it decides how many points the heal leaves.
        exit_code: How the second discover ends.
    """
    cfg = _cfg(admit_percentile=percentile)
    run_id, stranded = _strand_a_discover(client, root, monkeypatch, cfg, [])
    run = TopicStore(root).load(COLL, run_id)
    assert run is not None and len(run.table.topics) == 3
    assert len(stranded) > 100
    expected = [int(t) for t in run.table.align(_vectors(client, stranded))]
    assert expected.count(2) > 0

    frames = Frames()
    svc = _service(client, root, FakeFit(BLOBS), cfg=cfg)
    if exit_code == "none_new":
        outcome = _run(svc, "discover", frames)
        assert outcome.empty and [code for code, _ in frames.warnings] == ["none_new"]
    else:
        with pytest.raises(TopicsRefused) as caught:
            _run(svc, "discover", frames)
        assert caught.value.code == exit_code
    topics = _topics(client)
    assert [topics[pid] for pid in stranded] == expected
    assert "assigning" not in frames.stages()


def test_on_written_fires_after_a_cancelled_discover_that_wrote(
    client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discover cancelled after writing still announces its writes.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: Pytest monkeypatch.
    """
    written: list[str] = []
    _strand_a_discover(client, root, monkeypatch, _cfg(), written)
    assert written == [COLL]


def test_a_job_that_writes_nothing_announces_nothing(client: QdrantClient, root: Path) -> None:
    """An align with nothing pending does not call ``on_written``.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    _run(_service(client, root), "build")
    written: list[str] = []
    _run(_service(client, root, on_written=written.append), "align")
    assert written == []


# Ingest


def test_align_on_ingest_is_silent_without_a_run(client: QdrantClient, root: Path) -> None:
    """With no run the ingest stage does nothing and reports nothing.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    recording = RecordingClient(client)
    frames: list[tuple[int, int]] = []
    result = _service(recording, root).align_on_ingest(
        COLL, owner_created_at=lambda: CREATED, progress=lambda d, t: frames.append((d, t))
    )
    assert result is None and frames == []
    assert "batch_update_points" not in recording.calls


def test_align_on_ingest_defers_when_the_lock_is_held(client: QdrantClient, root: Path) -> None:
    """A held lock defers the stage; once free, the stage aligns the new points.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    built = _run(_service(client, root), "build")
    assert built.run_id is not None
    fresh = _add_blobs(client, per=2, offset=800, noise=0.0)
    held = TopicStore(root).try_lock(COLL)
    assert held is not None
    svc = _service(client, root)
    frames: list[tuple[int, int]] = []
    deferred = svc.align_on_ingest(COLL, owner_created_at=lambda: CREATED, progress=lambda d, t: frames.append((d, t)))
    assert deferred is not None and deferred.deferred
    assert deferred.artifact()["deferred"] is True
    assert count(client, COLL, pending_filter(built.run_id)) == len(fresh)
    held.release()
    done = svc.align_on_ingest(COLL, owner_created_at=lambda: CREATED, progress=lambda d, t: frames.append((d, t)))
    assert done is not None and not done.deferred and done.assigned == len(fresh)
    assert frames == [(len(fresh), len(fresh))]
    assert count(client, COLL, pending_filter(built.run_id)) == 0


# Order, seen ranges, logs, imports


def test_indexes_are_ensured_before_any_facet_or_write(client: QdrantClient, root: Path) -> None:
    """Every index creation precedes the first facet or payload write.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client)
    recording = RecordingClient(client)
    _run(_service(recording, root), "build")
    calls = recording.calls
    first = min(i for i, name in enumerate(calls) if name in {"facet", "batch_update_points"})
    creates = [i for i, name in enumerate(calls) if name == "create_payload_index"]
    assert creates and max(creates) < first


def test_first_and_last_seen_span_the_members_timestamps(client: QdrantClient, root: Path) -> None:
    """Seen ranges span the members' first parseable timestamps, normalised to UTC.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    stamps: dict[int, dict[str, Any]] = {
        0: {"timestamp": "2024-03-01T10:00:00Z"},
        1: {"timestamp": "2024-01-15T08:00:00"},
        2: {"timestamp": "2024-05-01T12:00:00+02:00"},
        3: {"timestamp": "not a date", "posting_timestamp": "2024-06-01T00:00:00Z"},
        4: {"timestamp": "garbage"},
    }
    _add_blobs(client, payload=lambda i, _pid: {"reference_metadata": stamps[i]} if i in stamps else {})
    stray = _ids(1, offset=700)[0]
    add_points(
        client,
        COLL,
        [
            (
                stray,
                np.eye(1, DIM, 6)[0],
                {"search_text": "stray passage words", "reference_metadata": {"timestamp": "2020-01-01T00:00:00Z"}},
            )
        ],
    )
    outcome = _run(_service(client, root), "build")
    assert outcome.run_id is not None
    assert _topics(client)[stray] == -1
    run = TopicStore(root).load(COLL, outcome.run_id)
    assert run is not None
    first = run.table.topics[0]
    assert (first.first_seen, first.last_seen) == ("2024-01-15T08:00:00+00:00", "2024-06-01T00:00:00+00:00")
    assert (run.table.topics[1].first_seen, run.table.topics[1].last_seen) == (None, None)


def test_the_service_logs_counts_only(client: QdrantClient, root: Path) -> None:
    """The job line carries counts and ids; no text, label or keyword reaches the log.

    Args:
        client: Qdrant client.
        root: Store root.
    """
    _add_blobs(client, payload=lambda i, pid: {"search_text": f"passage {pid} {SECRET_TEXT} words"})
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level="DEBUG", format="{message}")
    try:
        outcome = _run(_service(client, root, labeller=_labeller(SECRET_LABEL)), "build")
    finally:
        logger.remove(sink)
    expected = f"Topics build | run={outcome.run_id} topics=3 added=3 assigned={outcome.assigned} deferred=false"
    assert any(line.strip() == expected for line in lines)
    assert not [line for line in lines if "secret" in line.lower()]


@pytest.mark.parametrize(
    ("stage", "done", "total", "message"),
    [
        ("starting", 0, 0, "Starting topics run"),
        ("waiting", 3, 0, "Waiting for another topics run on this collection: 3 s"),
        ("scanning", 512, 0, "Scanning chunks: 512 chunks scanned"),
        ("sampling", 256, 900, "Reading the sample: 256/900 chunks read"),
        ("fitting", 15, 0, "Fitting topics: 15 s"),
        ("labelling", 2, 7, "Labelling topics: 2/7 topics labelled"),
        ("assigning", 300, 1000, "Assigning topics: 300/1000 chunks processed"),
        ("aligning", 12, 40, "Aligning topics: 12/40 chunks processed"),
    ],
)
def test_progress_messages_match_the_published_shapes(stage: str, done: int, total: int, message: str) -> None:
    """Each stage renders its published copy, and the payload carries the parts.

    Args:
        stage: Stage name.
        done: Units done.
        total: Units in all.
        message: The expected copy.
    """
    assert stage in service.STAGES
    assert progress_message(stage, done, total) == message
    assert progress_payload(stage, done, total) == {"message": message, "stage": stage, "done": done, "total": total}


def test_the_outcome_artifact_carries_the_published_keys() -> None:
    """The job artifact names the action, run and counts."""
    outcome = TopicsOutcome(action="build", run_id="r", topics=3, added=3, assigned=10, empty=False)
    assert outcome.artifact() == {
        "action": "build",
        "run_id": "r",
        "topics": 3,
        "added": 3,
        "assigned": 10,
        "deferred": False,
    }


def test_the_topics_package_imports_nothing_from_rag_state_or_api() -> None:
    """No topics module loads the engine, the state stores or the API."""
    modules = sorted(f"docint.core.topics.{info.name}" for info in pkgutil.iter_modules(docint.core.topics.__path__))
    script = (
        "import importlib, sys\n"
        "for name in sys.argv[1:]:\n"
        "    importlib.import_module(name)\n"
        "banned = ('docint.core.rag', 'docint.core.state', 'docint.core.api')\n"
        "print(' '.join(sorted(m for m in sys.modules if any(m == b or m.startswith(b + '.') for b in banned))))\n"
    )

    loaded = subprocess.run([sys.executable, "-c", script, *modules], capture_output=True, text=True, check=True)

    assert "docint.core.topics.service" in modules
    assert loaded.stdout.split() == []
