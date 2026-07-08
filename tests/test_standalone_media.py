"""Tests for StandaloneMediaIngestor (standalone_media.py)."""

from pathlib import Path
from typing import Any

from docint.core.ingest.media_transcribe import MediaClip, MediaTranscribeResult
from docint.core.ingest.standalone_media import StandaloneMediaIngestor


class _RecordingTranscriber:
    """Transcriber stub that records the clips it receives instead of calling Nextext."""

    def __init__(self) -> None:
        """Initialise with an empty clip log."""
        self.clips: list[MediaClip] = []

    def run(self, clips: list[MediaClip]) -> MediaTranscribeResult:
        """Record the clips and return a result that consumes each clip's path.

        Args:
            clips: The media clips the ingestor built for this run.

        Returns:
            A MediaTranscribeResult whose ``consumed_paths`` mirrors the clip
            paths and whose ``transcript_documents`` is empty (no real
            transcription happens in this stub).
        """
        self.clips = clips
        return MediaTranscribeResult(consumed_paths={c.path for c in clips}, transcript_documents=[])


def _ingestor(transcriber: Any, *, enabled: bool = True) -> StandaloneMediaIngestor:
    """Return a StandaloneMediaIngestor wired to *transcriber* for mp4/mp3 discovery.

    Args:
        transcriber: The (fake) transcriber to delegate discovered clips to.
        enabled: Value to pass as ``nextext_enabled``. Defaults to True.

    Returns:
        A StandaloneMediaIngestor configured with ``{".mp4", ".mp3"}`` as its
        discoverable media file types.
    """
    return StandaloneMediaIngestor(transcriber, media_filetypes={".mp4", ".mp3"}, nextext_enabled=enabled)


def test_discovers_unclaimed_av_and_builds_file_identity_clips(tmp_path: Path) -> None:
    """Unclaimed audio/video is discovered and stamped with content-hash identity.

    Non-A/V files are ignored, files already in ``already_consumed`` are
    skipped, and each resulting clip is anchored to its own file hash rather
    than any posting/manifest identity.
    """
    (tmp_path / "a.mp4").write_bytes(b"v")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mp3").write_bytes(b"a")
    (tmp_path / "notes.txt").write_text("x")  # non-A/V: ignored
    claimed = tmp_path / "claimed.mp4"
    claimed.write_bytes(b"c")

    transcriber = _RecordingTranscriber()
    result = _ingestor(transcriber).run(tmp_path, already_consumed={claimed})

    paths = sorted(c.path.name for c in transcriber.clips)
    assert paths == ["a.mp4", "b.mp3"]  # claimed.mp4 skipped; notes.txt ignored
    clip = next(c for c in transcriber.clips if c.path.name == "a.mp4")
    assert clip.keyframe_source_type == "video_keyframe"
    assert clip.keyframe_link_field is None
    assert clip.source_doc_id == clip.media_hash  # anchored to content hash
    assert clip.transcript_extra_info["source_file"] == "a.mp4"
    assert clip.transcript_extra_info["file_hash"] == clip.media_hash
    assert claimed.name not in [c.path.name for c in transcriber.clips]
    assert {p.name for p in result.consumed_paths} == {"a.mp4", "b.mp3"}


def test_noop_when_nextext_disabled(tmp_path: Path) -> None:
    """When Nextext is disabled, discovered media is never handed to the transcriber."""
    (tmp_path / "a.mp4").write_bytes(b"v")
    transcriber = _RecordingTranscriber()
    result = _ingestor(transcriber, enabled=False).run(tmp_path, already_consumed=set())
    assert transcriber.clips == []  # engine never invoked
    assert result.transcript_documents == []
    assert result.consumed_paths == set()
