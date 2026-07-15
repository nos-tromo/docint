"""Tests for the shared MediaTranscriber engine (media_transcribe.py)."""

from pathlib import Path
from typing import Any

from loguru import logger

from docint.core.ingest.media_transcribe import MediaClip, MediaTranscriber
from docint.utils.nextext_client import NextextResult


class _FakeNextext:
    """Nextext stub that returns a fixed result for every call."""

    def __init__(self, result: NextextResult) -> None:
        """Initialise with the fixed result to return on every call.

        Args:
            result: The NextextResult returned by every `process_media` call.
        """
        self.result = result
        self.calls: list[Path] = []

    def process_media(self, path: Path) -> NextextResult:
        """Record the call and return the fixed result.

        Args:
            path: Path to the media file (recorded but otherwise ignored).

        Returns:
            The fixed NextextResult supplied at construction time.
        """
        self.calls.append(path)
        return self.result


class _FakeImages:
    """In-memory image-service stub that records keyframe calls without touching Qdrant."""

    def __init__(self) -> None:
        """Initialise with an empty keyframe-call log."""
        self.keyframe_calls: list[dict[str, Any]] = []

    def ingest_keyframe_set(self, frames: list[bytes], **kwargs: Any) -> list[Any]:
        """Record the keyframe call and return an empty list.

        Args:
            frames: Keyframe bytes (recorded but not stored).
            **kwargs: All other keyword arguments, recorded alongside ``frames``.

        Returns:
            An empty list (no records stored in the stub).
        """
        self.keyframe_calls.append({"frames": frames, **kwargs})
        return []


def _clip(path: Path) -> MediaClip:
    """Return a video-keyframe-flavored MediaClip (standalone-style identity) for *path*.

    Args:
        path: The media file this clip wraps.

    Returns:
        A MediaClip stamped with a fixed hash-based identity and standalone-style
        keyframe/transcript metadata (no posting link).
    """
    return MediaClip(
        path=path,
        source_doc_id="hash-1",
        media_hash="hash-1",
        keyframe_source_type="video_keyframe",
        keyframe_link_field=None,
        keyframe_extra_metadata={"media_file_hash": "hash-1"},
        transcript_extra_info={"source_file": path.name, "file_hash": "hash-1"},
    )


def test_video_yields_transcript_documents_and_keyframes(tmp_path: Path) -> None:
    """A completed Nextext run yields one transcript Document and one keyframe call."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    nextext = _FakeNextext(
        NextextResult(
            status="completed",
            transcript_jsonl=b'{"text":"hello","start_seconds":0,"end_seconds":1}\n',
            keyframes=[b"f0", b"f1"],
        )
    )
    images = _FakeImages()
    result = MediaTranscriber(images, nextext, target_collection="c", manifest=None).run([_clip(clip)])

    assert len(result.transcript_documents) == 1
    assert result.transcript_documents[0].metadata["source_file"] == "clip.mp4"
    assert clip in result.consumed_paths
    assert (tmp_path / "clip.mp4.nextext.jsonl") in result.consumed_paths
    assert len(images.keyframe_calls) == 1
    call = images.keyframe_calls[0]
    assert call["keyframe_source_type"] == "video_keyframe"
    assert call["link_field"] is None
    assert call["source_doc_id"] == "hash-1"


def test_cache_hit_skips_nextext(tmp_path: Path) -> None:
    """A manifest cache hit must prevent the Nextext job from being submitted."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    class _Manifest:
        """Manifest stub with a pre-seeded cache hit; a cache write would be a bug."""

        def get_nextext_transcript(self, collection: str, file_hash: str) -> str | None:
            """Return a fixed cached transcript regardless of collection/hash.

            Args:
                collection: Collection name (ignored in stub).
                file_hash: Media file hash (ignored in stub).

            Returns:
                A fixed cached transcript JSONL string.
            """
            return '{"text":"cached","start_seconds":0,"end_seconds":1}\n'

        def cache_nextext_transcript(self, *a: Any) -> None:  # pragma: no cover
            """Fail the test if invoked; a cache hit must never write back.

            Args:
                *a: Ignored positional arguments.

            Raises:
                AssertionError: Always — this must not be called on a cache hit.
            """
            raise AssertionError("must not write on a cache hit")

    nextext = _FakeNextext(NextextResult(status="error"))
    result = MediaTranscriber(_FakeImages(), nextext, target_collection="c", manifest=_Manifest()).run([_clip(clip)])
    assert nextext.calls == []  # cache hit → Nextext never called
    assert len(result.transcript_documents) == 1


def test_transient_transcript_removed_from_disk(tmp_path: Path) -> None:
    """The ``.nextext.jsonl`` transient must not survive the run on disk.

    A leftover transient in a shared source directory (``DATA_PATH`` /
    ``ingest`` CLI) gets swept into other collections by the generic JSON
    reader on a later ingest, where ``consumed_paths`` no longer shields it.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    nextext = _FakeNextext(
        NextextResult(
            status="completed",
            transcript_jsonl=b'{"text":"hello","start_seconds":0,"end_seconds":1}\n',
            keyframes=[],
        )
    )
    result = MediaTranscriber(_FakeImages(), nextext, target_collection="c", manifest=None).run([_clip(clip)])

    transient = tmp_path / "clip.mp4.nextext.jsonl"
    assert not transient.exists()
    # Still marked consumed so this run's generic sweep skips it even if a
    # failed unlink left it behind.
    assert transient in result.consumed_paths
    assert len(result.transcript_documents) == 1


def test_transient_transcript_removed_on_cache_hit(tmp_path: Path) -> None:
    """The cache-hit ingest path also removes its ``.nextext.jsonl`` transient."""
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    class _Manifest:
        """Manifest stub with a pre-seeded cache hit."""

        def get_nextext_transcript(self, collection: str, file_hash: str) -> str | None:
            """Return a fixed cached transcript regardless of collection/hash.

            Args:
                collection: Collection name (ignored in stub).
                file_hash: Media file hash (ignored in stub).

            Returns:
                A fixed cached transcript JSONL string.
            """
            return '{"text":"cached","start_seconds":0,"end_seconds":1}\n'

        def cache_nextext_transcript(self, *a: Any) -> None:  # pragma: no cover
            """Fail the test if invoked; a cache hit must never write back.

            Args:
                *a: Ignored positional arguments.

            Raises:
                AssertionError: Always — this must not be called on a cache hit.
            """
            raise AssertionError("must not write on a cache hit")

    MediaTranscriber(
        _FakeImages(), _FakeNextext(NextextResult(status="error")), target_collection="c", manifest=_Manifest()
    ).run([_clip(clip)])
    assert not (tmp_path / "clip.mp4.nextext.jsonl").exists()


def test_empty_clip_list_is_noop(tmp_path: Path) -> None:
    """Running with no clips returns an empty result and touches nothing."""
    result = MediaTranscriber(_FakeImages(), _FakeNextext(NextextResult(status="error")), target_collection="c").run([])
    assert result.consumed_paths == set()
    assert result.transcript_documents == []


# ---------------------------------------------------------------------------
# The three tests below were relocated from tests/test_social_linker_routing.py
# (test_route_media_clip_warns_when_nextext_not_completed,
# test_route_media_clips_processes_all_clips,
# test_route_media_clips_isolates_a_failed_clip), which previously exercised
# this exact engine through SocialLinker._route_media_clips before Task 4
# extracted it into MediaTranscriber. They are ported here verbatim in spirit
# (same scenarios/assertions, MediaLink -> MediaClip, SocialLinker -> MediaTranscriber)
# since SocialLinker no longer has a method to call them against.
# ---------------------------------------------------------------------------


def test_warns_when_nextext_does_not_complete(tmp_path: Path) -> None:
    """A clip Nextext can't process (e.g. disabled) logs a warning, not silence."""

    class _DisabledNextext:
        """Nextext stub whose every call reports the job as disabled."""

        def process_media(self, path: Path) -> NextextResult:
            """Return a fixed disabled result regardless of the input path.

            Args:
                path: Path to the media file (ignored).

            Returns:
                A NextextResult with status "disabled".
            """
            return NextextResult(status="disabled")

    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"\x00")
    transcriber = MediaTranscriber(_FakeImages(), _DisabledNextext(), target_collection="c")

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="WARNING", format="{message}")
    try:
        transcriber.run([_clip(clip_path)])
    finally:
        logger.remove(sink_id)

    assert any("did not process" in line and "clip.mp4" in line and "disabled" in line for line in lines)


def test_processes_every_clip_in_a_batch(tmp_path: Path) -> None:
    """A batch of clips is processed concurrently but each is fully ingested.

    Every clip must end up with its keyframes ingested and exactly one
    transcript Document, and every clip's path must be recorded as consumed —
    proving the bounded thread pool used for the Nextext round-trips does not
    drop or duplicate work for any clip in the batch.
    """
    images = _FakeImages()
    nextext = _FakeNextext(
        NextextResult(
            status="completed",
            transcript_jsonl=b'{"text":"spoken","start_seconds":0,"end_seconds":1}\n',
            keyframes=[b"\xff\xd8\xff0"],
        )
    )
    clips = []
    for i in range(3):
        path = tmp_path / f"clip{i}.mp4"
        path.write_bytes(f"video-{i}".encode())
        clips.append(MediaClip(path=path, source_doc_id=f"u{i}", media_hash=f"hash-{i}"))

    result = MediaTranscriber(images, nextext, target_collection="c").run(clips)

    assert len(images.keyframe_calls) == 3
    assert len(result.transcript_documents) == 3
    clip_paths = {clip.path for clip in clips}
    assert clip_paths.issubset(result.consumed_paths)


def test_isolates_a_failed_clip_in_a_batch(tmp_path: Path) -> None:
    """A clip that errors out at Nextext must not block the other clips in the batch.

    One of three clips returns a Nextext ``error`` status (no transcript, no
    keyframes); the other two are ``completed``. The two good clips must still
    be fully ingested, and a warning naming the failed file must be logged —
    proving a single bad clip degrades gracefully instead of aborting the batch
    or silently losing sibling results.
    """
    clips = []
    for i in range(3):
        path = tmp_path / f"clip{i}.mp4"
        path.write_bytes(f"video-{i}".encode())
        clips.append(MediaClip(path=path, source_doc_id=f"u{i}", media_hash=f"hash-{i}"))
    failing_path = clips[1].path

    class _SelectiveNextext:
        """Nextext stub that fails one specific path and completes the rest."""

        def process_media(self, path: Path) -> NextextResult:
            """Return an error result for the failing path, completed otherwise.

            Args:
                path: Path to the media file.

            Returns:
                A ``status="error"`` NextextResult for ``failing_path``, else a
                completed NextextResult with one transcript segment and one keyframe.
            """
            if path == failing_path:
                return NextextResult(status="error")
            return NextextResult(
                status="completed",
                transcript_jsonl=b'{"text":"spoken","start_seconds":0,"end_seconds":1}\n',
                keyframes=[b"\xff\xd8\xff0"],
            )

    images = _FakeImages()
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="WARNING", format="{message}")
    try:
        result = MediaTranscriber(images, _SelectiveNextext(), target_collection="c").run(clips)
    finally:
        logger.remove(sink_id)

    assert len(images.keyframe_calls) == 2
    assert len(result.transcript_documents) == 2
    assert any("did not process" in line and failing_path.name in line for line in lines)
