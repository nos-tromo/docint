# Standalone Media (Audio/Video) Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest loose audio/video files via the remote Nextext service (transcript + video keyframes) with no `postings.csv`/`media.csv` scaffolding, linking every artifact to the media file itself.

**Architecture:** Extract the social linker's per-file Nextext→(transcript, keyframes) engine into a shared `MediaTranscriber` (Approach A). Add a `StandaloneMediaIngestor` pipeline pre-pass that runs right after the social linker, walks the batch for audio/video files the linker did not claim, and routes each through the shared engine with file-derived identity. Retrieval/citation/reporting are unchanged — standalone transcript segments and keyframes flow as independent, normally-ranked sources.

**Tech Stack:** Python ≥3.11,<3.12 (uv), llama-index, Qdrant, httpx, pandas; pytest.

**Companion spec:** `docs/specs/2026-07-08-standalone-media-ingestion-design.md`.

## Global Constraints

- **Python ≥3.11,<3.12**; manage deps with `uv` (`uv add`/`uv remove`). Tests: `uv run pytest`. Lint/format/type: `uv run pre-commit run --all-files` (ruff check, ruff format, pyrefly) before **every** commit.
- **No new heavy media dependency.** No `ffmpeg`/`av`/`PyAV`/`moviepy`. All decode/ASR/frame-sampling stays in Nextext over HTTP (`httpx`, already a dep).
- **All `os.getenv` and config dataclasses live in `docint/utils/env_cfg.py`.** Other modules import loaders from there.
- **Google-style docstrings** on every new/modified function and class.
- **Tests with every functional change** (pytest under `tests/`; `conftest.py` provides mock stubs for external deps like `magic`).
- **Fail-soft posture** (matches rerank/image-tagging/social): a dead/timed-out Nextext job, a missing/corrupt media file, or a disabled client skips *that item*, logs, and never fails the batch.
- **Trigger = automatic when `NEXTEXT_API_BASE` is set** (reuse `NextextConfig.enabled`; no new enable knob).
- **Identity = media file content hash** (`compute_file_hash`, `docint/utils/hashing.py`) — the same key the transcript cache and image dedup already use.
- **The social path's observable behavior must not change.** Its keyframe payloads and transcript metadata stay byte-for-byte; the existing `tests/test_social_linker_*`, `tests/test_pipeline_social_linker.py`, `tests/test_keyframe_dedup.py`, `tests/test_json_posting_link.py`, `tests/test_nextext_client.py` suites must stay green.
- **Commit trailer** (every commit): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Work on branch `feat/standalone-media-ingestion` (already created; the design spec is committed there).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `docint/core/ingest/images_service.py` (modify) | Parameterize `ingest_keyframe_set` (`keyframe_source_type`, `link_field`) | 1 |
| `docint/core/readers/json.py` (modify) | Transcript `source_file` falls back to `extra_info`/`base` | 2 |
| `docint/utils/env_cfg.py` (modify) | `IngestionConfig.media_filetypes` + `MEDIA_FILETYPES` override | 3 |
| `docint/core/ingest/media_transcribe.py` (create) | Shared per-file engine: `MediaClip`, `MediaTranscribeResult`, `MediaTranscriber` | 4 |
| `docint/core/ingest/social_linker.py` (modify) | Build `MediaClip`s + delegate to `MediaTranscriber`; drop moved code | 4 |
| `docint/core/ingest/standalone_media.py` (create) | Discover unclaimed A/V; build file-identity `MediaClip`s; delegate | 5 |
| `docint/core/ingest/ingestion_pipeline.py` (modify) | `_open_ingest_manifest()` helper + `_run_standalone_media()`; merge accumulators; drop obsolete unreferenced-media warning | 6 |
| `README.md`, `docs/ingestion.md`, `docs/architecture.md`, `CLAUDE.md` (modify) | Document the standalone path | 7 |

---

## Task 1: Parameterize `ingest_keyframe_set` (source_type + link field)

**Files:**
- Modify: `docint/core/ingest/images_service.py:953-1063` (`ingest_keyframe_set`)
- Test: `tests/test_keyframe_source_type.py` (create)

**Interfaces:**
- Produces: `ImageIngestionService.ingest_keyframe_set(frames, *, context, source_doc_id, extra_metadata=None, dedup_cosine=0.95, keyframe_source_type="social_media_keyframe", link_field="posting_uuid") -> list[StoredImageRecord]`.
- The two new keyword args **default to today's hard-coded values**, so existing callers (the social path) are unaffected.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_keyframe_source_type.py
from typing import Any

from docint.core.ingest.images_service import ImageIngestionService, IngestContext


class _FakeEmbed:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, image_bytes: bytes) -> list[float]:
        self.calls += 1
        return [1.0, 0.0]

    def embed_text(self, text: str) -> list[float]:  # pragma: no cover - unused
        return [1.0, 0.0]


def _service(monkeypatch) -> tuple[ImageIngestionService, list[Any]]:
    svc = ImageIngestionService(qdrant_client=None)
    monkeypatch.setattr(svc, "_get_embedding_backend", lambda: _FakeEmbed())
    monkeypatch.setattr(svc, "_get_tagging_backend", lambda: None)
    monkeypatch.setattr(svc, "_resolve_collection_name", lambda name: f"{name}_images")
    monkeypatch.setattr(svc, "_ensure_collection", lambda **kw: None)
    stored: list[Any] = []

    class _Store:
        def add(self, nodes: list[Any]) -> None:
            stored.extend(nodes)

    monkeypatch.setattr(svc, "_get_vector_store", lambda name: _Store())
    return svc, stored


def test_defaults_preserve_social_keyframe_payload(monkeypatch) -> None:
    svc, stored = _service(monkeypatch)
    svc.ingest_keyframe_set(
        [b"f0"],
        context=IngestContext(source_collection="c"),
        source_doc_id="uuid-1",
        extra_metadata={"posting_id": "P", "media_id": "P_0", "source_type": "social_media", "posting_uuid": "uuid-1"},
    )
    payload = stored[0].metadata
    assert payload["source_type"] == "social_media"  # extra_metadata override, exactly as today
    assert payload["posting_uuid"] == "uuid-1"
    assert payload["source_doc_id"] == "uuid-1"


def test_standalone_video_keyframe_has_no_posting_uuid(monkeypatch) -> None:
    svc, stored = _service(monkeypatch)
    svc.ingest_keyframe_set(
        [b"f0"],
        context=IngestContext(source_collection="c"),
        source_doc_id="hash-1",
        keyframe_source_type="video_keyframe",
        link_field=None,
        extra_metadata={"media_file_hash": "hash-1", "source_file": "clip.mp4"},
    )
    payload = stored[0].metadata
    assert payload["source_type"] == "video_keyframe"
    assert "posting_uuid" not in payload
    assert payload["source_doc_id"] == "hash-1"
    assert payload["source_file"] == "clip.mp4"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_keyframe_source_type.py -v`
Expected: FAIL — `test_standalone_video_keyframe_has_no_posting_uuid` fails because `posting_uuid` is currently hard-coded onto every keyframe payload (and `ingest_keyframe_set` has no `keyframe_source_type`/`link_field` kwargs → `TypeError`).

- [ ] **Step 3: Add the two keyword args to the signature**

In `docint/core/ingest/images_service.py`, change the `ingest_keyframe_set` signature (currently ending at line 960) to add the two keyword-only args:

```python
    def ingest_keyframe_set(
        self,
        frames: list[bytes],
        *,
        context: IngestContext,
        source_doc_id: str | None,
        extra_metadata: dict[str, Any] | None = None,
        dedup_cosine: float = 0.95,
        keyframe_source_type: str = "social_media_keyframe",
        link_field: str | None = "posting_uuid",
    ) -> list[StoredImageRecord]:
```

- [ ] **Step 4: Use the args in the payload**

In the same method, replace the payload literal + `extra_metadata` merge (currently lines 1024-1039) with:

```python
            payload: dict[str, Any] = {
                "image_id": image_id,
                "source_type": keyframe_source_type,
                "source_collection": context.source_collection,
                "source_doc_id": source_doc_id,
                "mime_type": "image/jpeg",
                "mimetype": "image/jpeg",
                "file_type": "image/jpeg",
                "llm_description": description,
                "llm_tags": tags,
                "vector_name": self.img_ingestion_config.vector_name,
                "image_collection": target_collection,
            }
            if link_field:
                payload[link_field] = source_doc_id
            if extra_metadata:
                payload.update(extra_metadata)
```

(The hard-coded `"source_type": "social_media_keyframe"` and `"posting_uuid": source_doc_id` lines are gone; the defaults reproduce them.) Also update the Args section of the docstring to document `keyframe_source_type` and `link_field`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_keyframe_source_type.py tests/test_keyframe_dedup.py -v`
Expected: PASS (new file 2 passed; the existing `test_keyframe_dedup.py` still green — social defaults unchanged).

- [ ] **Step 6: Commit**

```bash
uv run pre-commit run --all-files
git add docint/core/ingest/images_service.py tests/test_keyframe_source_type.py
git commit -m "feat(images): parameterize keyframe source_type + link field" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Transcript `source_file` falls back to `extra_info`

**Files:**
- Modify: `docint/core/readers/json.py:515` (inside `_build_segment_metadata`)
- Test: `tests/test_json_transcript_source_file.py` (create)

**Interfaces:**
- Consumes: `CustomJSONReader(is_jsonl=True).iter_documents(path, extra_info={...})` — `extra_info` is merged into `base_extra_info` (`json.py:723-724`) and passed as `base` to `_build_segment_metadata`.
- Produces: when a transcript segment has no `source_file` field, the segment's `source_file` (top-level **and** `reference_metadata["source_file"]`) falls back to `base["source_file"]` (i.e. the caller's `extra_info`). Backward compatible: no caller sets `base["source_file"]` today, so nothing changes for existing paths.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_json_transcript_source_file.py
from pathlib import Path

from docint.core.readers.json import CustomJSONReader


def _write(tmp_path: Path) -> Path:
    jsonl = tmp_path / "clip.mp4.nextext.jsonl"
    jsonl.write_text(
        '{"text":"hello","start_seconds":0,"end_seconds":1}\n'
        '{"text":"world","start_seconds":1,"end_seconds":2}\n',
        encoding="utf-8",
    )
    return jsonl


def test_source_file_falls_back_to_extra_info(tmp_path: Path) -> None:
    docs = list(
        CustomJSONReader(is_jsonl=True).iter_documents(
            _write(tmp_path),
            extra_info={"source_file": "clip.mp4", "file_hash": "hash-1"},
        )
    )
    assert len(docs) == 2
    for doc in docs:
        assert doc.metadata["source_file"] == "clip.mp4"
        assert doc.metadata["reference_metadata"]["source_file"] == "clip.mp4"
        assert doc.metadata["file_hash"] == "hash-1"


def test_no_source_file_anywhere_omits_it(tmp_path: Path) -> None:
    # Regression: the social path passes no source_file; reference_metadata must not gain one.
    docs = list(CustomJSONReader(is_jsonl=True).iter_documents(_write(tmp_path)))
    assert "source_file" not in docs[0].metadata["reference_metadata"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_json_transcript_source_file.py -v`
Expected: FAIL — `test_source_file_falls_back_to_extra_info` fails: `reference_metadata["source_file"]` is absent because it is currently sourced only from the segment (`json.py:515`), which has no `source_file` here.

- [ ] **Step 3: Add the fallback**

In `docint/core/readers/json.py`, in `_build_segment_metadata`, change line 515 from:

```python
        source_file = segment.get("source_file")
```

to:

```python
        # Fall back to the caller-provided base (extra_info) so a standalone
        # media ingest can cite the original clip (e.g. clip.mp4) even when the
        # Nextext segment carries no source_file of its own. Social/other callers
        # set no base source_file, so their behavior is unchanged.
        source_file = segment.get("source_file") or base.get("source_file")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_json_transcript_source_file.py tests/test_json_posting_link.py tests/test_json_reader_nextext.py -v`
Expected: PASS (new file 2 passed; existing transcript tests still green).

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run --all-files
git add docint/core/readers/json.py tests/test_json_transcript_source_file.py
git commit -m "feat(readers): transcript source_file falls back to extra_info" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `IngestionConfig.media_filetypes` in env_cfg

**Files:**
- Modify: `docint/utils/env_cfg.py` (`IngestionConfig` at ~line 712; `load_ingestion_env` at ~line 715)
- Test: `tests/test_env_cfg_media_filetypes.py` (create)

**Interfaces:**
- Produces: `IngestionConfig.media_filetypes: list[str]` — the audio/video extensions (lowercase, dot-prefixed) the standalone pass discovers. Default is the canonical A/V set; overridable via `MEDIA_FILETYPES` (comma-separated).
- Also exports module constant `DEFAULT_MEDIA_FILETYPES: list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_cfg_media_filetypes.py
from docint.utils.env_cfg import DEFAULT_MEDIA_FILETYPES, load_ingestion_env


def test_default_media_filetypes(monkeypatch) -> None:
    monkeypatch.delenv("MEDIA_FILETYPES", raising=False)
    cfg = load_ingestion_env()
    assert ".mp4" in cfg.media_filetypes
    assert ".mp3" in cfg.media_filetypes
    assert ".wav" in cfg.media_filetypes
    assert cfg.media_filetypes == DEFAULT_MEDIA_FILETYPES
    # A/V must NOT be in the generic reader whitelist (they route via the pre-pass).
    assert ".mp4" not in cfg.supported_filetypes


def test_media_filetypes_override(monkeypatch) -> None:
    monkeypatch.setenv("MEDIA_FILETYPES", ".mp4, .mov")
    cfg = load_ingestion_env()
    assert cfg.media_filetypes == [".mp4", ".mov"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_env_cfg_media_filetypes.py -v`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_MEDIA_FILETYPES'` (and `IngestionConfig` has no `media_filetypes`).

- [ ] **Step 3: Add the constant + config field + loader logic**

In `docint/utils/env_cfg.py`, add the module constant near the other ingestion constants (above `load_ingestion_env`):

```python
DEFAULT_MEDIA_FILETYPES: list[str] = [
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
]
```

Add the field to the `IngestionConfig` dataclass (after `supported_filetypes: list[str]` at line 712):

```python
    media_filetypes: list[str]
```

In `load_ingestion_env`, build the value from the env override and pass it into the `IngestionConfig(...)` constructor. Add this just before the `return IngestionConfig(` (after the `default_supported_filetypes` block ends at line 802):

```python
    media_override = os.getenv("MEDIA_FILETYPES")
    if media_override:
        media_filetypes = [ext.strip().lower() for ext in media_override.split(",") if ext.strip()]
    else:
        media_filetypes = list(DEFAULT_MEDIA_FILETYPES)
```

And add `media_filetypes=media_filetypes,` to the `IngestionConfig(...)` call. Also add a one-line description of `media_filetypes` to the `load_ingestion_env` docstring's returns list.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_env_cfg_media_filetypes.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run --all-files
git add docint/utils/env_cfg.py tests/test_env_cfg_media_filetypes.py
git commit -m "feat(config): add media_filetypes (audio/video extensions)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Extract the shared `MediaTranscriber` engine; adopt it in the social linker

This is a behavior-preserving "extract class" refactor: the per-file engine
(`SocialLinker._route_media_clips` + `_ingest_transcript`, `social_linker.py:328-432`)
moves into a new module both paths call. The social linker's observable output
is unchanged.

**Files:**
- Create: `docint/core/ingest/media_transcribe.py`
- Modify: `docint/core/ingest/social_linker.py` (remove `_route_media_clips`/`_ingest_transcript`; build `MediaClip`s + delegate in `run()`)
- Test: `tests/test_media_transcriber.py` (create)

**Interfaces:**
- Consumes: `ImageIngestionService.ingest_keyframe_set(..., keyframe_source_type=, link_field=)` (Task 1); `CustomJSONReader.iter_documents(path, extra_info=)`; `NextextClient.process_media(path) -> NextextResult`; `IngestManifest.get/cache_nextext_transcript`.
- Produces:
  - `MediaClip(path: Path, source_doc_id: str | None, media_hash: str | None = None, keyframe_source_type: str = "social_media_keyframe", keyframe_link_field: str | None = "posting_uuid", keyframe_extra_metadata: dict = {}, transcript_extra_info: dict = {})`
  - `MediaTranscribeResult(consumed_paths: set[Path], transcript_documents: list[Document])`
  - `MediaTranscriber(image_service, nextext_client, target_collection, manifest=None, keyframe_dedup_cosine=0.95, nextext_max_concurrency=4)` with `run(clips: list[MediaClip]) -> MediaTranscribeResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_media_transcriber.py
from pathlib import Path
from typing import Any

from docint.core.ingest.media_transcribe import MediaClip, MediaTranscriber
from docint.utils.nextext_client import NextextResult


class _FakeNextext:
    def __init__(self, result: NextextResult) -> None:
        self.result = result
        self.calls: list[Path] = []

    def process_media(self, path: Path) -> NextextResult:
        self.calls.append(path)
        return self.result


class _FakeImages:
    def __init__(self) -> None:
        self.keyframe_calls: list[dict[str, Any]] = []

    def ingest_keyframe_set(self, frames: list[bytes], **kwargs: Any) -> list[Any]:
        self.keyframe_calls.append({"frames": frames, **kwargs})
        return []


def _clip(path: Path) -> MediaClip:
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
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    class _Manifest:
        def get_nextext_transcript(self, collection: str, file_hash: str) -> str | None:
            return '{"text":"cached","start_seconds":0,"end_seconds":1}\n'

        def cache_nextext_transcript(self, *a: Any) -> None:  # pragma: no cover
            raise AssertionError("must not write on a cache hit")

    nextext = _FakeNextext(NextextResult(status="error"))
    result = MediaTranscriber(_FakeImages(), nextext, target_collection="c", manifest=_Manifest()).run([_clip(clip)])
    assert nextext.calls == []  # cache hit → Nextext never called
    assert len(result.transcript_documents) == 1


def test_empty_clip_list_is_noop(tmp_path: Path) -> None:
    result = MediaTranscriber(_FakeImages(), _FakeNextext(NextextResult(status="error")), target_collection="c").run([])
    assert result.consumed_paths == set()
    assert result.transcript_documents == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_media_transcriber.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docint.core.ingest.media_transcribe'`.

- [ ] **Step 3: Create the shared engine module**

```python
# docint/core/ingest/media_transcribe.py
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
                self.manifest.cache_nextext_transcript(collection, hashes[clip.path], outcome.transcript_jsonl.decode("utf-8"))
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
        ``CustomJSONReader`` (which reads from a path) can parse it; that
        transient is marked consumed so the generic sweep ignores it. The durable
        cache of record is the ingest manifest, not this file.

        Args:
            clip (MediaClip): The clip whose ``transcript_extra_info`` is stamped.
            transcript (bytes): The transcript NDJSON bytes.
            result (MediaTranscribeResult): Accumulator for consumed paths + Documents.
        """
        transient = clip.path.parent / (clip.path.name + ".nextext.jsonl")
        transient.write_bytes(transcript)
        result.consumed_paths.add(transient)
        docs = CustomJSONReader(is_jsonl=True).iter_documents(transient, extra_info=dict(clip.transcript_extra_info))
        result.transcript_documents.extend(docs)
```

- [ ] **Step 4: Run the engine tests to verify they pass**

Run: `uv run pytest tests/test_media_transcriber.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Rewrite `SocialLinker.run` to build `MediaClip`s and delegate**

In `docint/core/ingest/social_linker.py`:

(a) Add imports near the other Task-10 imports (after `from docint.utils.nextext_client import NextextResult` at line 234):

```python
from docint.core.ingest.media_transcribe import MediaClip, MediaTranscriber  # noqa: E402
```

(b) Replace the clip-routing tail of `run()` (currently lines 309-326, from `clips: list[MediaLink] = []` through `return result`) with:

```python
        clips: list[MediaClip] = []
        for link in links:
            if is_image(link.path):
                result.consumed_paths.add(link.path)
                extra = {"posting_id": link.posting_id, "media_id": link.media_id, "source_type": "social_media"}
                self.image_service.ingest_image(
                    ImageAsset.from_path(
                        path=link.path,
                        source_type="social_media",
                        source_doc_id=link.posting_uuid,
                        extra_metadata={**extra, "posting_uuid": link.posting_uuid},
                    ),
                    context=context,
                )
            else:
                clips.append(
                    MediaClip(
                        path=link.path,
                        source_doc_id=link.posting_uuid,
                        keyframe_extra_metadata={
                            "posting_id": link.posting_id,
                            "media_id": link.media_id,
                            "source_type": "social_media",
                            "posting_uuid": link.posting_uuid,
                        },
                        transcript_extra_info={
                            "posting_uuid": link.posting_uuid,
                            "posting_id": link.posting_id,
                            "media_id": link.media_id,
                        },
                    )
                )
        sub = MediaTranscriber(
            image_service=self.image_service,
            nextext_client=self.nextext_client,
            target_collection=self.target_collection,
            manifest=self.manifest,
            keyframe_dedup_cosine=self.keyframe_dedup_cosine,
            nextext_max_concurrency=self.nextext_max_concurrency,
        ).run(clips)
        result.consumed_paths |= sub.consumed_paths
        result.transcript_documents.extend(sub.transcript_documents)
        return result
```

(`MediaClip` uses the default `keyframe_source_type="social_media_keyframe"` / `keyframe_link_field="posting_uuid"`; combined with the `source_type`/`posting_uuid` in `keyframe_extra_metadata`, the stored keyframe payload is byte-for-byte identical to today.)

(c) Delete the now-unused methods `_route_media_clips` (lines 328-406) and `_ingest_transcript` (lines 408-432). Remove the now-unused imports at the top of the file that only they used: `from concurrent.futures import ThreadPoolExecutor, as_completed` (line 12), `from docint.utils.hashing import compute_file_hash` (line 233), and the `CustomJSONReader` import (line 231) **only if** no other code references it (grep first: `rg 'CustomJSONReader|ThreadPoolExecutor|compute_file_hash' docint/core/ingest/social_linker.py`). `NextextResult` (line 234) is also now unused — remove it if grep confirms.

- [ ] **Step 6: Run the social suite to verify no behavior change**

Run: `uv run pytest tests/test_social_linker_join.py tests/test_social_linker_routing.py tests/test_pipeline_social_linker.py tests/test_media_transcriber.py -v`
Expected: PASS (all green — the social path is unchanged; the engine is now shared).

- [ ] **Step 7: Commit**

```bash
uv run pre-commit run --all-files
git add docint/core/ingest/media_transcribe.py docint/core/ingest/social_linker.py tests/test_media_transcriber.py
git commit -m "refactor(ingest): extract shared MediaTranscriber; social linker delegates" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `StandaloneMediaIngestor` — discover unclaimed A/V, delegate with file identity

**Files:**
- Create: `docint/core/ingest/standalone_media.py`
- Test: `tests/test_standalone_media.py` (create)

**Interfaces:**
- Consumes: `MediaTranscriber`, `MediaClip`, `MediaTranscribeResult` (Task 4); `load_ingestion_env().media_filetypes` (Task 3); `compute_file_hash`.
- Produces: `StandaloneMediaIngestor(transcriber: MediaTranscriber, *, media_filetypes: set[str], nextext_enabled: bool)` with `run(data_dir: Path, already_consumed: set[Path]) -> MediaTranscribeResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_standalone_media.py
from pathlib import Path
from typing import Any

from docint.core.ingest.media_transcribe import MediaClip, MediaTranscribeResult
from docint.core.ingest.standalone_media import StandaloneMediaIngestor


class _RecordingTranscriber:
    def __init__(self) -> None:
        self.clips: list[MediaClip] = []

    def run(self, clips: list[MediaClip]) -> MediaTranscribeResult:
        self.clips = clips
        return MediaTranscribeResult(consumed_paths={c.path for c in clips}, transcript_documents=[])


def _ingestor(transcriber: Any, *, enabled: bool = True) -> StandaloneMediaIngestor:
    return StandaloneMediaIngestor(transcriber, media_filetypes={".mp4", ".mp3"}, nextext_enabled=enabled)


def test_discovers_unclaimed_av_and_builds_file_identity_clips(tmp_path: Path) -> None:
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
    (tmp_path / "a.mp4").write_bytes(b"v")
    transcriber = _RecordingTranscriber()
    result = _ingestor(transcriber, enabled=False).run(tmp_path, already_consumed=set())
    assert transcriber.clips == []  # engine never invoked
    assert result.transcript_documents == []
    assert result.consumed_paths == set()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_standalone_media.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docint.core.ingest.standalone_media'`.

- [ ] **Step 3: Create the standalone pass**

```python
# docint/core/ingest/standalone_media.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_standalone_media.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run --all-files
git add docint/core/ingest/standalone_media.py tests/test_standalone_media.py
git commit -m "feat(ingest): standalone audio/video ingestor (file identity)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Wire the standalone pass into the ingestion pipeline

**Files:**
- Modify: `docint/core/ingest/ingestion_pipeline.py` (add `_open_ingest_manifest()`; add `_run_standalone_media()`; call it after `_run_social_linker()` at line 851; merge accumulators)
- Modify: `docint/core/ingest/social_linker.py` (drop the now-obsolete `_warn_unreferenced_media` call + function)
- Test: `tests/test_pipeline_standalone_media.py` (create)

**Interfaces:**
- Consumes: `StandaloneMediaIngestor` (Task 5); `MediaTranscriber` (Task 4); `load_nextext_env`, `load_ingestion_env` (env_cfg); `NextextClient`; `IngestManifest`/`NullIngestManifest`.
- Reuses the existing pipeline hooks: `self.social_link_consumed` (skipped by the generic sweep at `ingestion_pipeline.py:394-396`) and `self.social_link_documents` (yielded at `:415-416`). Both passes feed them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_standalone_media.py
from pathlib import Path
from typing import Any

from docint.core.ingest import ingestion_pipeline as pipe_mod
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.ingest.media_transcribe import MediaTranscribeResult
from llama_index.core import Document


class _StubManifest:
    def close(self) -> None:
        pass


def _pipeline(tmp_path: Path, monkeypatch) -> DocumentIngestionPipeline:
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path, ner_model=None, progress_callback=None, target_collection="c"
    )
    # Isolate _run_standalone_media from real services/disk: it constructs an
    # ImageIngestionService and opens a SQLite manifest before delegating.
    monkeypatch.setattr(pipe_mod, "ImageIngestionService", lambda *a, **k: object())
    monkeypatch.setattr(pipeline, "_open_ingest_manifest", lambda: _StubManifest())
    return pipeline


def test_standalone_pass_merges_into_prepass_accumulators(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.mp4").write_bytes(b"v")
    pipeline = _pipeline(tmp_path, monkeypatch)

    doc = Document(text="hello", metadata={"docint_doc_kind": "transcript_segment"})
    captured: dict[str, Any] = {}

    class _FakeIngestor:
        def run(self, data_dir: Path, already_consumed: set[Path]) -> MediaTranscribeResult:
            captured["already_consumed"] = set(already_consumed)
            return MediaTranscribeResult(consumed_paths={tmp_path / "a.mp4"}, transcript_documents=[doc])

    monkeypatch.setattr(pipe_mod, "StandaloneMediaIngestor", lambda *a, **k: _FakeIngestor())

    pipeline._run_standalone_media()

    assert (tmp_path / "a.mp4") in pipeline.social_link_consumed
    assert doc in pipeline.social_link_documents
    assert captured["already_consumed"] == set()  # social consumed nothing


def test_standalone_pass_excludes_socially_consumed_paths(tmp_path: Path, monkeypatch) -> None:
    claimed = tmp_path / "linked.mp4"
    claimed.write_bytes(b"v")
    pipeline = _pipeline(tmp_path, monkeypatch)
    pipeline.social_link_consumed = {claimed}

    seen: dict[str, Any] = {}

    class _FakeIngestor:
        def run(self, data_dir: Path, already_consumed: set[Path]) -> MediaTranscribeResult:
            seen["already_consumed"] = set(already_consumed)
            return MediaTranscribeResult()

    monkeypatch.setattr(pipe_mod, "StandaloneMediaIngestor", lambda *a, **k: _FakeIngestor())
    pipeline._run_standalone_media()
    assert claimed in seen["already_consumed"]  # the linker's claim is passed through so it is skipped
```

(The fixture patches `pipe_mod.ImageIngestionService` and `pipeline._open_ingest_manifest` so `_run_standalone_media` builds no real image service and touches no SQLite. `NextextClient` and `MediaTranscriber` construct trivially — an httpx client with an empty base URL and a plain dataclass — so they need no patch.)

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_pipeline_standalone_media.py -v`
Expected: FAIL — `AttributeError: 'DocumentIngestionPipeline' object has no attribute '_run_standalone_media'`.

- [ ] **Step 3: Add a manifest helper and refactor `_run_social_linker` to use it**

In `docint/core/ingest/ingestion_pipeline.py`, add this method just above `_run_social_linker` (line 853). It factors out the manifest-open logic so both passes share it:

```python
    def _open_ingest_manifest(self) -> Any:
        """Open the ingest manifest for this collection, or a no-op stub.

        Returns:
            IngestManifest | NullIngestManifest: A manifest keyed by
            ``(collection, file_hash)`` when enabled and a sources root + target
            collection are configured; otherwise a no-op stub. Callers must
            ``close()`` the returned object.
        """
        from docint.core.storage.ingest_manifest import IngestManifest, NullIngestManifest
        from docint.utils.env_cfg import load_ingestion_env, load_path_env

        try:
            sources_root = load_path_env().qdrant_sources
            if load_ingestion_env().ingest_manifest_enabled and sources_root and self.target_collection:
                target = self.target_collection
                return IngestManifest(sources_root / target / f"{target}_ingest_manifest.db")
        except Exception as exc:  # pragma: no cover - fail-soft guard
            logger.debug("Manifest unavailable: {}", exc)
        return NullIngestManifest()
```

Then replace the manifest-building block inside `_run_social_linker` (lines 864-871) with a single call:

```python
        manifest = self._open_ingest_manifest()
```

(The `from docint.core.storage.ingest_manifest import IngestManifest, NullIngestManifest` and `from docint.utils.env_cfg import load_ingestion_env, load_path_env` imports inside `_run_social_linker` are now unused there — the helper owns them; remove them from `_run_social_linker`. Keep `from docint.utils.env_cfg import load_nextext_env` and `from docint.utils.nextext_client import NextextClient`, and keep `from docint.core.ingest.social_linker import SocialLinker`.)

- [ ] **Step 4: Add `_run_standalone_media` and call it**

In `docint/core/ingest/ingestion_pipeline.py`, add the top-level import near the other ingest imports at the top of the file:

```python
from docint.core.ingest.standalone_media import StandaloneMediaIngestor
```

Add the method immediately after `_run_social_linker` (after line 889):

```python
    def _run_standalone_media(self) -> None:
        """Transcribe loose audio/video files the social linker did not claim.

        Runs after :meth:`_run_social_linker`; merges its consumed paths +
        transcript Documents into the shared pre-pass accumulators
        (``social_link_consumed`` / ``social_link_documents``) that the generic
        sweep skips / yields. Fail-soft: any error logs a warning and is swallowed
        so ingestion of the rest of the batch proceeds.
        """
        from docint.core.ingest.media_transcribe import MediaTranscriber
        from docint.utils.env_cfg import load_ingestion_env, load_nextext_env
        from docint.utils.nextext_client import NextextClient

        manifest = self._open_ingest_manifest()
        try:
            nextext_cfg = load_nextext_env()
            transcriber = MediaTranscriber(
                image_service=self.image_ingestion_service or ImageIngestionService(),
                nextext_client=NextextClient(nextext_cfg),
                target_collection=self.target_collection,
                manifest=manifest,
                keyframe_dedup_cosine=nextext_cfg.keyframe_dedup_cosine,
                nextext_max_concurrency=nextext_cfg.nextext_max_concurrency,
            )
            result = StandaloneMediaIngestor(
                transcriber,
                media_filetypes=set(load_ingestion_env().media_filetypes),
                nextext_enabled=nextext_cfg.enabled,
            ).run(self.data_dir, self.social_link_consumed)
        except Exception as exc:  # pragma: no cover - fail-soft guard
            logger.warning("Standalone media ingestion skipped due to error: {}", exc)
            return
        finally:
            manifest.close()
        self.social_link_consumed = self.social_link_consumed | result.consumed_paths
        self.social_link_documents = [*self.social_link_documents, *result.transcript_documents]
```

Then call it right after the social linker in `_load_doc_readers` — change line 851 from:

```python
        self._run_social_linker()
```

to:

```python
        self._run_social_linker()
        self._run_standalone_media()
```

- [ ] **Step 5: Run the pipeline test to verify it passes**

Run: `uv run pytest tests/test_pipeline_standalone_media.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Drop the now-obsolete unreferenced-media warning**

Because the standalone pass now ingests loose audio/video that the manifest did
not link, `SocialLinker`'s "present but not linked → NOT ingested" warning is no
longer accurate. In `docint/core/ingest/social_linker.py`:

- Remove the `_warn_unreferenced_media(media_csv.parent, {link.path for link in links})` call (line 305).
- Delete the `_warn_unreferenced_media` function (lines 197-220).
- Remove the now-unused `_AV_EXTS` constant (lines 21-38) — grep first (`rg '_AV_EXTS' docint/`) to confirm nothing else references it; the canonical set now lives in `env_cfg.DEFAULT_MEDIA_FILETYPES`.
- If a test asserts the warning (`rg -l "not linked|unreferenced|_warn_unreferenced" tests/`), delete or update that specific test.

- [ ] **Step 7: Run the social + standalone + pipeline suites**

Run: `uv run pytest tests/test_social_linker_routing.py tests/test_social_linker_join.py tests/test_pipeline_social_linker.py tests/test_standalone_media.py tests/test_pipeline_standalone_media.py -v`
Expected: PASS (all green).

- [ ] **Step 8: Full suite + lint**

Run: `uv run pytest` then `uv run pre-commit run --all-files`
Expected: full suite green; lint/format/type clean.

- [ ] **Step 9: Commit**

```bash
git add docint/core/ingest/ingestion_pipeline.py docint/core/ingest/social_linker.py tests/test_pipeline_standalone_media.py
git commit -m "feat(ingest): run standalone media pass in the pipeline" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Documentation

**Files:**
- Modify: `docs/ingestion.md` (correct the "silently skipped" statement; document the standalone path)
- Modify: `README.md` (add a "Standalone media (audio/video)" subsection beside "Social Multimodal Media")
- Modify: `docs/architecture.md` (mention `standalone_media.py` + `media_transcribe.py`)
- Modify: `CLAUDE.md` (add module bullets)

- [ ] **Step 1: Update `docs/ingestion.md`**

Replace the paragraph at `docs/ingestion.md:30-40` (the "Only the file types listed above are ingested when uploaded standalone… Audio and video are instead handled through the social-multimodal path…" block) with text stating that audio/video dropped anywhere in a batch are transcribed by Nextext (transcript + video keyframes) with **no** `postings.csv`/`media.csv` required, linked to the media file; the social-manifest path still additionally links media to postings; both require `NEXTEXT_API_BASE`; when it is unset, audio/video are skipped with a warning. Add the A/V extensions (`MEDIA_FILETYPES`, default from `DEFAULT_MEDIA_FILETYPES`) to the "Supported file types" section as a **Media** category (noting they are routed via the pre-pass, not the generic reader).

- [ ] **Step 2: Update `README.md`**

Under (or beside) the "Social Multimodal Media" section (`README.md:301-354`), add a short "Standalone media (audio/video)" subsection: drop audio/video files into any ingest batch (SPA folder upload or `DATA_PATH`); with `NEXTEXT_API_BASE` set they are transcribed and (for video) keyframed automatically; artifacts cite the source clip; no manifest needed. Cross-reference the existing Nextext env vars (`NEXTEXT_API_BASE`, `KEYFRAMES_PER_MINUTE`, `KEYFRAMES_MAX`, `KEYFRAME_DEDUP_COSINE`, `NEXTEXT_MAX_CONCURRENCY`) and note the new `MEDIA_FILETYPES` override.

- [ ] **Step 3: Update `docs/architecture.md`**

Add `docint/core/ingest/media_transcribe.py` (shared per-file Nextext engine) and `docint/core/ingest/standalone_media.py` (standalone A/V pre-pass) to the module list, noting the social linker now delegates its per-file routing to the shared engine.

- [ ] **Step 4: Update `CLAUDE.md`**

Add two bullets to the "Key modules" list mirroring the existing `social_linker.py` bullet: one for `media_transcribe.py` (the shared engine both passes call) and one for `standalone_media.py` (loose A/V → Nextext, file identity, automatic on `NEXTEXT_API_BASE`, runs after the social linker). Update the `social_linker.py` bullet to note it delegates per-file routing to `media_transcribe.py`.

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run --all-files
git add docs/ingestion.md README.md docs/architecture.md CLAUDE.md
git commit -m "docs: document standalone audio/video ingestion" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the full suite: `uv run pytest` — all green.
- [ ] Run lint/format/type: `uv run pre-commit run --all-files` — clean.
- [ ] Manual smoke (optional, needs a reachable Nextext + Qdrant): set `NEXTEXT_API_BASE`, ingest a folder containing a single `.mp4` and no CSVs; confirm the collection gains transcript-segment nodes citing the clip and keyframe image points in `{collection}_images`, and that a query returns them as independent sources.

## Spec coverage check

- Independent-source retrieval, no new postprocessor → Tasks 1-6 add only ingestion; no `rag.py` change. ✅
- Automatic trigger on `NEXTEXT_API_BASE` → Task 5 `nextext_enabled` gate; Task 6 passes `nextext_cfg.enabled`. ✅
- Anchor to content hash → Task 5 `source_doc_id=media_hash`, `file_hash=media_hash`. ✅
- Approach A shared engine, social byte-for-byte → Task 4. ✅
- Social linker runs first, standalone excludes claimed paths → Task 6. ✅
- `ingest_keyframe_set` parameterization → Task 1. ✅
- Citation shows the clip (verify item a) → Task 2 + Task 5 `transcript_extra_info`. ✅
- Keyframe preview via `source_doc_id or file_hash` (verify item b) → Task 5 `source_doc_id=media_hash`, no `posting_uuid`. ✅
- Config in env_cfg → Task 3. ✅
- Fail-soft everywhere → Task 4 (per-clip), Task 5 (disabled no-op), Task 6 (pass-level guard). ✅
- Docs → Task 7. ✅
- Frontend/API/CLI unchanged → no task needed (agents confirmed A/V already accepted/persisted/streamed). ✅
