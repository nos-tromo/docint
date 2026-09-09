"""Per-file preprocessing pool: the heavy ingest stages, in parallel, on arrival.

Three stages of an ingest cost minutes per file and are idempotent by content
hash — a PDF's layout/OCR pass (artifacts under ``PIPELINE_ARTIFACTS_DIR``),
an image's caption/OCR/CLIP point (looked up in ``{collection}_images``), and a
clip's Nextext transcript (cached in the ingest manifest). Because each of them
checks its own cache before calling a model, the work can start the moment a
file lands on disk and the job later finds it done; and because each is keyed
by the file's hash, running them across files at once needs no new state.

This module holds the one bounded pool both callers share: ``/ingest/upload``
submits a file as soon as it is saved, and the job submits every file up
front, then joins them in its existing order. ``submit`` never runs a key
twice while it is in flight; ``run`` joins an in-flight task or runs it
inline; a finished key is evicted, so the pool holds no results — the caches
are the memory, and a failed task is simply tried again.

Concurrency is threads, not asyncio, for the reasons ``streaming_executor``
gives: the sync ingest path is canonical and every stage here is HTTP-bound.
The class itself has no docint domain imports; the task builders below import
theirs lazily so the readers and pre-passes that call back into this pool do
not close an import cycle.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger

from docint.core.ingest.images_service import ImageAsset, ImageIngestionService, IngestContext
from docint.utils.env_cfg import load_ingestion_env, load_nextext_env
from docint.utils.hashing import compute_file_hash
from docint.utils.mimetype import get_mimetype

if TYPE_CHECKING:
    from docint.core.ingest.media_transcribe import MediaTranscribeResult
    from docint.core.readers.documents.models import DocumentManifest

T = TypeVar("T")

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".gif"})

__all__ = [
    "IMAGE_EXTENSIONS",
    "PreprocessPool",
    "get_preprocess_pool",
    "prefetch_batch",
    "preprocess_image",
    "preprocess_key",
    "preprocess_media",
    "preprocess_pdf",
    "run_all",
    "shutdown_preprocess_pool",
    "standalone_image_asset",
    "submit_file",
]


class PreprocessPool:
    """Bounded thread pools that run at most one task per key at a time.

    Two executors behind one registry, because the stages wait on two
    different services: a clip holds a slot for its whole Nextext round trip
    (minutes), while images and PDF pages queue on the vision/OCR endpoint.
    A single FIFO pool let a few hundred images uploaded first starve Nextext
    entirely, so ``media`` keys get their own executor and the rest share one.

    Args:
        max_workers (int): Worker threads for the vision/OCR stages; floored at one.
        media_workers (int | None): Worker threads for Nextext clips; defaults
            to ``max_workers``.
    """

    def __init__(self, max_workers: int, *, media_workers: int | None = None) -> None:
        """Create the executors and the in-flight registry."""
        self._executors: dict[str, ThreadPoolExecutor] = {
            "": ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="docint-preprocess"),
            "media": ThreadPoolExecutor(
                max_workers=max(1, media_workers or max_workers), thread_name_prefix="docint-preprocess-media"
            ),
        }
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)

    def submit(self, key: str, fn: Callable[[], T]) -> Future[T]:
        """Start *fn* under *key*, or return the future already running it.

        Args:
            key (str): Dedupe key, from :func:`preprocess_key`.
            fn (Callable[[], T]): Zero-argument task.

        Returns:
            Future[T]: The in-flight future for *key*.
        """
        with self._lock:
            existing = self._futures.get(key)
            if existing is not None:
                return existing
            future: Future[T] = self._executor_for(key).submit(fn)
            self._futures[key] = future
        future.add_done_callback(lambda done, key=key: self._evict(key, done))
        return future

    def run(self, key: str, fn: Callable[[], T]) -> T:
        """Join the task running under *key*, or run *fn* now, and return its result.

        Args:
            key (str): Dedupe key, from :func:`preprocess_key`.
            fn (Callable[[], T]): Zero-argument task to run when nothing is in flight.

        Returns:
            T: The task's result.

        Raises:
            Exception: Whatever the task raised; the key is evicted first so a
                retry runs it again.
        """
        return self.submit(key, fn).result()

    def join(self, key: str) -> None:
        """Wait for the task running under *key*, if any; a failure is theirs to report.

        For a caller about to consult a cache another task is filling: an
        in-flight task means the answer is seconds away, so waiting beats
        doing the work twice.

        Args:
            key (str): Dedupe key, from :func:`preprocess_key`.
        """
        with self._lock:
            future = self._futures.get(key)
        if future is None:
            return
        try:
            future.result()
        except Exception:
            return

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until no task is in flight.

        Args:
            timeout (float | None): Seconds to wait, or forever when ``None``.

        Returns:
            bool: ``True`` when the pool went idle within the timeout.
        """
        with self._idle:
            return self._idle.wait_for(lambda: not self._futures, timeout=timeout)

    def shutdown(self) -> None:
        """Stop accepting work and drop queued tasks; running ones finish on their own."""
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)

    def _executor_for(self, key: str) -> ThreadPoolExecutor:
        """Pick the executor by the key's kind prefix (``media`` or everything else)."""
        kind = key.partition(":")[0]
        return self._executors.get(kind, self._executors[""])

    def _evict(self, key: str, future: Future[Any]) -> None:
        """Forget a finished key and log a failure once, on the thread that saw it."""
        with self._idle:
            if self._futures.get(key) is future:
                del self._futures[key]
            self._idle.notify_all()
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.warning("Preprocess task '{}' failed: {}", key, exc)
        else:
            logger.debug("Preprocess task '{}' done.", key)


def run_all(pool: PreprocessPool | None, jobs: list[tuple[str, Callable[[], T]]]) -> list[T]:
    """Run keyed jobs through *pool* — all submitted first, results in order — or inline.

    Args:
        pool (PreprocessPool | None): The pool, or ``None`` to run the jobs
            one after another on the calling thread (a task already on a pool
            worker must not wait on the pool it occupies).
        jobs (list[tuple[str, Callable[[], T]]]): ``(key, task)`` pairs.

    Returns:
        list[T]: One result per job, in the order given.
    """
    if pool is None:
        return [task() for _, task in jobs]
    # Keep the futures: a task that finishes before its turn is evicted, and
    # re-keying it through ``run`` would run it a second time.
    futures = [pool.submit(key, task) for key, task in jobs]
    return [future.result() for future in futures]


_pool: PreprocessPool | None = None
_pool_lock = threading.Lock()
_service: ImageIngestionService | None = None


def get_preprocess_pool() -> PreprocessPool:
    """Return the process-wide pool, sized by ``INGEST_PREPROCESS_WORKERS``.

    Returns:
        PreprocessPool: The shared pool.
    """
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = PreprocessPool(
                load_ingestion_env().ingest_preprocess_workers,
                media_workers=load_nextext_env().nextext_max_concurrency,
            )
        return _pool


def shutdown_preprocess_pool() -> None:
    """Stop the shared pool; the next ``get_preprocess_pool`` builds a fresh one.

    Called from the API lifespan so a queued upload-time task cannot hold the
    process open. Resetting the singleton rather than keeping a stopped one
    is what lets a test client's lifespan run more than once per process.
    """
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown()


def prefetch_batch(
    batch_dir: Path,
    collection: str,
    *,
    skip_hashes: set[str],
    image_service: ImageIngestionService | None = None,
    pool: PreprocessPool | None = None,
) -> int:
    """Submit every file in a batch that has a heavy stage, before the lanes read them.

    Called once at the top of an ingest run, so the PDF lane, the social and
    standalone media passes and the image sweep — which still consume files
    in their own order — find that order's work already running or done. A
    file the collection already holds is skipped; a clip is not, since its
    identity is its transcript's hash and the transcript cache makes a
    repeat cheap.

    Args:
        batch_dir (Path): The staged batch tree.
        collection (str): Physical collection name.
        skip_hashes (set[str]): File hashes the collection already holds.
        image_service (ImageIngestionService | None): Service for the image
            stages; the shared one when ``None``.
        pool (PreprocessPool | None): Pool to submit to; the shared one when ``None``.

    Returns:
        int: How many files were submitted.
    """
    heavy = {".pdf", *IMAGE_EXTENSIONS, *(ext.lower() for ext in load_ingestion_env().media_filetypes)}
    submitted = 0
    for path in sorted(candidate for candidate in batch_dir.rglob("*") if candidate.is_file()):
        if path.suffix.lower() not in heavy:
            continue
        file_hash = compute_file_hash(path)
        if file_hash in skip_hashes:
            continue
        if submit_file(path, collection, pool=pool, file_hash=file_hash, image_service=image_service) is not None:
            submitted += 1
    if submitted:
        logger.info("Preprocess prefetch | collection={!r} files={}", collection, submitted)
    return submitted


def _image_service(image_service: ImageIngestionService | None) -> ImageIngestionService:
    """Return the caller's service, else the one shared by every task thread."""
    global _service
    if image_service is not None:
        return image_service
    with _pool_lock:
        if _service is None:
            _service = ImageIngestionService()
        return _service


def preprocess_key(kind: str, collection: str, file_hash: str) -> str:
    """Build the pool key for one file's heavy stage.

    Args:
        kind (str): ``pdf``, ``image`` or ``media`` — the same bytes mean
            different work under different extensions.
        collection (str): Physical collection the file is staged for.
        file_hash (str): The file's content hash.

    Returns:
        str: ``"{kind}:{collection}:{file_hash}"``.
    """
    return f"{kind}:{collection}:{file_hash}"


def standalone_image_asset(path: Path) -> ImageAsset:
    """The asset an image file gets when nothing links it to a posting.

    One construction shared by the reader and the pool, so a pre-processed
    image and one read cold by the job are byte-identical points.

    Args:
        path (Path): The image file.

    Returns:
        ImageAsset: The standalone asset.
    """
    return ImageAsset(
        source_type="standalone",
        image_path=path,
        image_bytes=path.read_bytes(),
        source_path=str(path),
        mime_type=get_mimetype(path),
    )


def preprocess_image(path: Path, collection: str, *, image_service: ImageIngestionService | None = None) -> Any:
    """Caption, read and embed one image into the collection's ``_images`` companion.

    Args:
        path (Path): The image file.
        collection (str): Physical collection name.
        image_service (ImageIngestionService | None): Service to use; the
            shared one when ``None``.

    Returns:
        StoredImageRecord: What the service stored or found cached.
    """
    return _image_service(image_service).ingest_image(
        standalone_image_asset(path), context=IngestContext(source_collection=collection)
    )


def preprocess_pdf(
    path: Path, collection: str, *, image_service: ImageIngestionService | None = None
) -> DocumentManifest:
    """Run the page pipeline on one PDF and ingest its figures.

    Args:
        path (Path): The PDF.
        collection (str): Physical collection name (for the figures' companion).
        image_service (ImageIngestionService | None): Service to use; the
            shared one when ``None``.

    Returns:
        DocumentManifest: The pipeline's manifest for the document.
    """
    from docint.core.readers.documents.orchestrator import DocumentPipelineOrchestrator
    from docint.core.readers.documents.reader import ingest_pipeline_images

    orchestrator = DocumentPipelineOrchestrator()
    manifest = orchestrator.process(path)
    if manifest.status == "completed":
        ingest_pipeline_images(
            _image_service(image_service),
            collection,
            file_path=path,
            doc_id=manifest.doc_id,
            artifacts_dir=Path(orchestrator.config.artifacts_dir),
        )
    return manifest


def preprocess_media(
    path: Path, collection: str, *, image_service: ImageIngestionService | None = None
) -> MediaTranscribeResult:
    """Transcribe one clip through Nextext and store its keyframes with file identity.

    The transcript lands in the collection's ingest manifest and the keyframes
    in its ``_images`` companion, both keyed by the clip's own hash. A social
    export that later claims the clip relinks those keyframes to the posting;
    the transcript Documents returned here are discarded, since the job
    re-reads them from the cache with the identity it knows.

    Args:
        path (Path): The audio/video file.
        collection (str): Physical collection name.
        image_service (ImageIngestionService | None): Service to use; the
            shared one when ``None``.

    Returns:
        MediaTranscribeResult: The transcriber's result.
    """
    from docint.core.ingest.media_transcribe import MediaTranscriber
    from docint.core.ingest.standalone_media import standalone_clip
    from docint.core.storage.ingest_manifest import open_ingest_manifest
    from docint.utils.nextext_client import NextextClient

    nextext_cfg = load_nextext_env()
    manifest = open_ingest_manifest(collection)
    try:
        return MediaTranscriber(
            image_service=_image_service(image_service),
            nextext_client=NextextClient(nextext_cfg),
            target_collection=collection,
            manifest=manifest,
            keyframe_dedup_cosine=nextext_cfg.keyframe_dedup_cosine,
            nextext_max_concurrency=1,
        ).run([standalone_clip(path)])
    finally:
        manifest.close()


def submit_file(
    path: Path,
    collection: str,
    *,
    pool: PreprocessPool | None = None,
    file_hash: str | None = None,
    image_service: ImageIngestionService | None = None,
) -> Future[Any] | None:
    """Queue the heavy stage a staged file needs, if it needs one.

    Args:
        path (Path): The file, inside the collection's batch directory.
        collection (str): Physical collection name.
        pool (PreprocessPool | None): Pool to submit to; the shared one when ``None``.
        file_hash (str | None): The file's content hash when the caller already
            has it (the upload handler does); read from disk otherwise.
        image_service (ImageIngestionService | None): Service for the image
            stages; the shared one when ``None``.

    Returns:
        Future[Any] | None: The in-flight future, or ``None`` for a file with
        no heavy stage — an unsupported extension, or a clip with no Nextext
        configured.
    """
    ext = path.suffix.lower()
    task: Callable[..., Any]
    if ext == ".pdf":
        kind, task = "pdf", preprocess_pdf
    elif ext in IMAGE_EXTENSIONS:
        kind, task = "image", preprocess_image
    elif ext in {e.lower() for e in load_ingestion_env().media_filetypes}:
        if not load_nextext_env().enabled:
            return None
        kind, task = "media", preprocess_media
    else:
        return None
    key = preprocess_key(kind, collection, file_hash or compute_file_hash(path))
    return (pool or get_preprocess_pool()).submit(key, lambda: task(path, collection, image_service=image_service))
