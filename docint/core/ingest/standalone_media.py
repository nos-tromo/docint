"""Standalone audio/video ingestion — transcribe loose media without social tables.

Runs as a pipeline pre-pass right after the social linker. It walks the batch
tree for audio/video files the linker did not already claim and routes each
through the shared :class:`MediaTranscriber`, anchoring every artifact to the
media file's own content hash (no postings/media manifest required). Enabled
whenever Nextext is configured (``NEXTEXT_API_BASE`` set); a no-op otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from docint.core.ingest.media_transcribe import MediaClip, MediaTranscribeResult
from docint.utils.hashing import compute_file_hash

__all__ = ["StandaloneMediaIngestor"]


class StandaloneMediaIngestor:
    """Discover unclaimed audio/video files and transcribe them with file identity."""

    def __init__(self, transcriber: Any, *, media_filetypes: set[str], nextext_enabled: bool) -> None:
        """Create the ingestor.

        Args:
            transcriber (Any): A :class:`MediaTranscriber` (or compatible) to route clips through.
            media_filetypes (set[str]): Lowercase, dot-prefixed audio/video extensions to discover.
            nextext_enabled (bool): Whether the Nextext client is configured. When
                ``False`` the pass is a no-op (with a one-line warning if media is present).
        """
        self._transcriber = transcriber
        self._media_filetypes = {ext.lower() for ext in media_filetypes}
        self._nextext_enabled = nextext_enabled

    def _discover(self, data_dir: Path, already_consumed: set[Path]) -> list[Path]:
        """Return audio/video files under ``data_dir`` not already consumed.

        Args:
            data_dir (Path): The batch tree root.
            already_consumed (set[Path]): Paths the social linker already claimed.

        Returns:
            list[Path]: Discovered, unclaimed media files (sorted, deterministic).
        """
        found: list[Path] = []
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self._media_filetypes:
                continue
            if path in already_consumed:
                continue
            found.append(path)
        return found

    def run(self, data_dir: Path, already_consumed: set[Path]) -> MediaTranscribeResult:
        """Transcribe every unclaimed audio/video file under ``data_dir``.

        Args:
            data_dir (Path): The batch tree root.
            already_consumed (set[Path]): Paths the social linker already claimed
                (excluded here so manifest-linked media is never double-ingested).

        Returns:
            MediaTranscribeResult: Consumed paths + transcript Documents (empty when
                no unclaimed media is present or Nextext is disabled).
        """
        media_files = self._discover(data_dir, already_consumed)
        if not media_files:
            return MediaTranscribeResult()
        if not self._nextext_enabled:
            logger.warning(
                "{} audio/video file(s) found but NOT ingested: Nextext is not configured "
                "(set NEXTEXT_API_BASE to enable transcription).",
                len(media_files),
            )
            return MediaTranscribeResult()

        clips: list[MediaClip] = []
        for path in media_files:
            media_hash = compute_file_hash(path)
            clips.append(
                MediaClip(
                    path=path,
                    source_doc_id=media_hash,
                    media_hash=media_hash,
                    keyframe_source_type="video_keyframe",
                    keyframe_link_field=None,
                    keyframe_extra_metadata={
                        "media_file_hash": media_hash,
                        "source_file": path.name,
                        "source_path": str(path),
                    },
                    transcript_extra_info={
                        "filename": path.name,
                        "file_name": path.name,
                        "file_path": str(path),
                        "source_file": path.name,
                        "file_hash": media_hash,
                        "media_file_hash": media_hash,
                    },
                )
            )
        logger.info("Standalone media: transcribing {} audio/video file(s).", len(clips))
        return self._transcriber.run(clips)
