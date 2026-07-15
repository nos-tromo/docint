"""Shared per-file media→(transcript, keyframes) engine.

Extracted from the social linker so both the social path and the standalone
media path route audio/video through Nextext identically. Pure orchestration:
hash → transcript-cache lookup → (miss) Nextext round-trip (bounded concurrency)
→ keyframes to CLIP → transcript to segment Documents → write-through cache.
No media decoding and no model inference here — those live in Nextext and the
image service. Everything path-specific (posting identity vs. file identity) is
carried on each :class:`MediaClip`, not baked into the engine.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import Document
from loguru import logger

from docint.core.ingest.images_service import IngestContext
from docint.core.readers.json import CustomJSONReader
from docint.utils.hashing import compute_file_hash
from docint.utils.nextext_client import NextextResult

__all__ = ["MediaClip", "MediaTranscribeResult", "MediaTranscriber"]


@dataclass(frozen=True)
class MediaClip:
    """One audio/video file to route through Nextext, with the identity to stamp.

    Attributes:
        path (Path): The media file on disk.
        source_doc_id (str | None): The anchor id stamped on keyframe points
            (posting UUID for social; media content hash for standalone).
        media_hash (str | None): Precomputed content hash used as the transcript
            cache key. When ``None`` the engine computes it (single hash per file).
        keyframe_source_type (str): ``source_type`` payload value for keyframes.
        keyframe_link_field (str | None): Payload key aliasing ``source_doc_id``
            on keyframes (``"posting_uuid"`` for social; ``None`` for standalone).
        keyframe_extra_metadata (dict[str, Any]): Extra keyframe payload fields.
        transcript_extra_info (dict[str, Any]): Extra info merged into each
            transcript segment (e.g. ``source_file``/``file_hash`` or posting ids).
    """

    path: Path
    source_doc_id: str | None
    media_hash: str | None = None
    keyframe_source_type: str = "social_media_keyframe"
    keyframe_link_field: str | None = "posting_uuid"
    keyframe_extra_metadata: dict[str, Any] = field(default_factory=dict)
    transcript_extra_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaTranscribeResult:
    """Consumed paths + transcript Documents produced by a transcriber run."""

    consumed_paths: set[Path] = field(default_factory=set)
    transcript_documents: list[Document] = field(default_factory=list)


@dataclass
class MediaTranscriber:
    """Route a batch of media clips through Nextext and ingest their artifacts."""

    image_service: Any
    nextext_client: Any
    target_collection: str | None
    manifest: Any = None
    keyframe_dedup_cosine: float = 0.95
    nextext_max_concurrency: int = 4

    def run(self, clips: list[MediaClip]) -> MediaTranscribeResult:
        """Transcribe + keyframe every clip, returning consumed paths + Documents.

        The Nextext round-trips (submit + poll — the slow part) run in a bounded
        thread pool so a batch of clips overlaps instead of serializing. Cache
        lookups and the ingestion of results (keyframes → CLIP, transcript →
        Documents, transcript-cache writes) run on the calling thread, keeping
        Qdrant / image-service / manifest writes single-threaded. On a cache hit
        Nextext is not called at all.

        Args:
            clips (list[MediaClip]): The media clips to process.

        Returns:
            MediaTranscribeResult: Consumed paths + transcript Documents.
        """
        result = MediaTranscribeResult()
        if not clips:
            return result
        context = IngestContext(source_collection=self.target_collection)
        collection = self.target_collection or ""
        for clip in clips:
            result.consumed_paths.add(clip.path)
        # Phase 1 (serial): hash + transcript-cache lookup.
        hashes: dict[Path, str] = {}
        cached: dict[Path, bytes] = {}
        to_fetch: list[MediaClip] = []
        for clip in clips:
            media_hash = clip.media_hash or compute_file_hash(clip.path)
            hashes[clip.path] = media_hash
            hit = self.manifest.get_nextext_transcript(collection, media_hash) if self.manifest else None
            if hit is not None:
                cached[clip.path] = hit.encode("utf-8")
            else:
                to_fetch.append(clip)
        # Phase 2 (concurrent): Nextext round-trips only (HTTP is concurrency-safe).
        outcomes: dict[Path, NextextResult] = {}
        if to_fetch:
            workers = max(1, min(self.nextext_max_concurrency, len(to_fetch)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self.nextext_client.process_media, clip.path): clip for clip in to_fetch}
                for future in as_completed(futures):
                    clip = futures[future]
                    try:
                        outcomes[clip.path] = future.result()
                    except Exception as exc:  # defensive: a raised call must not abort the batch
                        logger.warning("Nextext call raised for {!r}: {}", clip.path.name, exc)
                        outcomes[clip.path] = NextextResult(status="error", error=str(exc))
        # Phase 3 (serial): ingest each clip's transcript + keyframes.
        for clip in clips:
            if clip.path in cached:
                self._ingest_transcript(clip, cached[clip.path], result)
                continue
            outcome = outcomes[clip.path]
            if outcome.transcript_jsonl is not None and self.manifest is not None:
                self.manifest.cache_nextext_transcript(
                    collection, hashes[clip.path], outcome.transcript_jsonl.decode("utf-8")
                )
            if outcome.status != "completed":
                logger.warning(
                    "Nextext did not process media {!r} (status={!r}); no transcript/keyframes ingested. "
                    "If unexpected, set NEXTEXT_API_BASE (include the /api/v1 suffix) and ensure Nextext is reachable.",
                    clip.path.name,
                    outcome.status,
                )
            if outcome.keyframes:
                self.image_service.ingest_keyframe_set(
                    outcome.keyframes,
                    context=context,
                    source_doc_id=clip.source_doc_id,
                    extra_metadata=clip.keyframe_extra_metadata,
                    dedup_cosine=self.keyframe_dedup_cosine,
                    keyframe_source_type=clip.keyframe_source_type,
                    link_field=clip.keyframe_link_field,
                )
            if outcome.transcript_jsonl:
                self._ingest_transcript(clip, outcome.transcript_jsonl, result)
        return result

    def _ingest_transcript(self, clip: MediaClip, transcript: bytes, result: MediaTranscribeResult) -> None:
        """Parse transcript JSONL into segment Documents stamped with the clip identity.

        Writes a transient ``.nextext.jsonl`` next to the media file purely so
        ``CustomJSONReader`` (which reads from a path) can parse it; the file is
        unlinked as soon as parsing finishes (and marked consumed as a fallback
        for this run's sweep should the unlink fail). Leaving it on disk let a
        later ingest of the same source directory — where ``consumed_paths`` no
        longer covers it, e.g. a shared ``DATA_PATH`` or CLI ingest into another
        collection — sweep it up via the generic JSON reader as an orphaned,
        posting-less transcript. The durable cache of record is the ingest
        manifest, not this file.

        Args:
            clip (MediaClip): The clip whose ``transcript_extra_info`` is stamped.
            transcript (bytes): The transcript NDJSON bytes.
            result (MediaTranscribeResult): Accumulator for consumed paths + Documents.
        """
        transient = clip.path.parent / (clip.path.name + ".nextext.jsonl")
        transient.write_bytes(transcript)
        result.consumed_paths.add(transient)
        try:
            docs = CustomJSONReader(is_jsonl=True).iter_documents(
                transient, extra_info=dict(clip.transcript_extra_info)
            )
            result.transcript_documents.extend(docs)
        finally:
            try:
                transient.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove transient transcript {!r}: {}", str(transient), exc)
