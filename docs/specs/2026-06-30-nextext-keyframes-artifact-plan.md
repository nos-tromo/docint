# Nextext `keyframes.zip` Artifact — Implementation Plan

> **Repo:** this plan operates on the **Nextext** repo (`../Nextext`, relative to this workspace), not docint. It is the prerequisite for docint's social-multimodal video *visuals* (docint Phase 2 consumes `keyframes.zip`). docint degrades gracefully without it — video transcripts still flow; keyframes stay empty until this ships.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** During a Nextext job, extract a rate-sampled, capped set of video keyframes and expose them as a downloadable `keyframes.zip` artifact.

**Architecture:** Reuse the PyAV decode already in `nextext/core/audio.py` to pull *video* frames (decoding only keyframes via `skip_frame="NONKEY"` — fast and naturally scene-diverse). The pipeline (`_run_pipeline_blocking`) stashes JPEG bytes in `result["keyframes"]`; `render_artifact` serves them as a ZIP through the existing `/jobs/{id}/artifacts/{name}` route. docint's `NextextClient` then downloads `keyframes.zip` and prunes near-duplicates on its side.

**Tech Stack:** Python (uv), PyAV (`av`, already used for audio), Pillow (`frame.to_image()`), FastAPI, pytest (+ pytest-asyncio per-test).

## Global Constraints

- Manage deps with `uv`; run tests with `uv run pytest` (full) / `uv run pytest tests/test_keyframes.py` (one file); `make test` runs `uv run pytest -q`. Lint/format/type per the repo's pre-commit before committing.
- `JobOptions` uses `model_config = ConfigDict(extra="forbid")` — new option keys MUST be added to the model or validation rejects them. docint sends `keyframes_per_minute` + `keyframes_max`.
- Keyframe extraction is **fail-soft**: no video stream, an audio-only file, or a decode error yields `[]` (and whatever decoded so far), never an exception that fails the job.
- Frame defaults mirror docint: `keyframes_per_minute=4`, `keyframes_max=20`.
- Commit trailer (every commit): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Branch `feat/keyframes-artifact`.

---

## Task 1: Add keyframe options to `JobOptions`

**Files:**
- Modify: `nextext/api/schemas.py` (`JobOptions`, lines 23-34)
- Test: `tests/test_api/test_job_options_keyframes.py` (create)

**Interfaces:**
- Produces: `JobOptions.keyframes_per_minute: int` (default 4, ge=0) and `JobOptions.keyframes_max: int` (default 20, ge=0, le=200).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api/test_job_options_keyframes.py
import pytest
from pydantic import ValidationError

from nextext.api.schemas import JobOptions


def test_job_options_accepts_keyframe_fields() -> None:
    opts = JobOptions.model_validate({"keyframes_per_minute": 6, "keyframes_max": 30})
    assert opts.keyframes_per_minute == 6
    assert opts.keyframes_max == 30


def test_job_options_keyframe_defaults() -> None:
    opts = JobOptions.model_validate({})
    assert opts.keyframes_per_minute == 4
    assert opts.keyframes_max == 20


def test_job_options_rejects_negative_rate() -> None:
    with pytest.raises(ValidationError):
        JobOptions.model_validate({"keyframes_per_minute": -1})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_api/test_job_options_keyframes.py -v`
Expected: FAIL — `extra="forbid"` rejects `keyframes_per_minute`.

- [ ] **Step 3: Add the fields**

In `nextext/api/schemas.py`, append to `JobOptions` (after `hate_speech: bool = False`):

```python
    keyframes_per_minute: int = Field(default=4, ge=0)
    keyframes_max: int = Field(default=20, ge=0, le=200)
```

(`Field` is already imported — `speakers` uses it.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_api/test_job_options_keyframes.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add nextext/api/schemas.py tests/test_api/test_job_options_keyframes.py
git commit -m "feat(jobs): add keyframe sampling options" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Keyframe extraction module

**Files:**
- Create: `nextext/core/keyframes.py`
- Test: `tests/test_keyframes.py` (create)

**Interfaces:**
- Produces: `extract_keyframes(file_path: Path, *, per_minute: int = 4, max_frames: int = 20) -> list[bytes]` (JPEG bytes) and the pure helper `subsample(items: list[T], target: int) -> list[T]`.

- [ ] **Step 1: Confirm Pillow is available**

Run: `cd ../Nextext && uv run python -c "import PIL; print(PIL.__version__)"`
Expected: prints a version. If it errors, run `uv add pillow` and commit the lock change as part of Step 5.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_keyframes.py
from pathlib import Path

from nextext.core.keyframes import extract_keyframes, subsample


def test_subsample_evenly_picks_target() -> None:
    assert subsample([1, 2, 3, 4, 5, 6, 7, 8], 4) == [1, 3, 5, 7]


def test_subsample_returns_all_when_fewer_than_target() -> None:
    assert subsample([1, 2], 5) == [1, 2]


def test_subsample_zero_target_is_empty() -> None:
    assert subsample([1, 2, 3], 0) == []


def test_extract_keyframes_failsoft_on_non_video(tmp_path: Path) -> None:
    bogus = tmp_path / "not_a_video.bin"
    bogus.write_bytes(b"\x00\x01\x02not media")
    # No video stream / undecodable -> empty list, never raises.
    assert extract_keyframes(bogus, per_minute=4, max_frames=5) == []
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/test_keyframes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nextext.core.keyframes'`

- [ ] **Step 4: Implement the module**

```python
# nextext/core/keyframes.py
"""Extract a small, scene-diverse set of video keyframes as JPEG bytes.

Reuses the PyAV decode already relied on for audio (see
``nextext.core.audio``). To stay cheap and naturally diverse we decode only
keyframes (``skip_frame="NONKEY"``), gather candidates up to a work bound, then
evenly subsample to the rate/cap requested by the job. Audio-only files, files
with no video stream, and decode errors yield ``[]`` (fail-soft) — a clip that
can't be framed must never fail the job.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TypeVar

import av
from av.error import FFmpegError
from loguru import logger

__all__ = ["extract_keyframes", "subsample"]

T = TypeVar("T")

# Decode at most this multiple of the cap before subsampling, so a long video
# does not decode end-to-end just to throw most frames away.
_CANDIDATE_BUDGET_FACTOR = 4


def subsample(items: list[T], target: int) -> list[T]:
    """Evenly pick ``target`` items across ``items`` (order preserved).

    Args:
        items (list[T]): Source items.
        target (int): Desired count. ``<= 0`` yields an empty list; a target
            at or above ``len(items)`` returns all items.

    Returns:
        list[T]: ``target`` evenly-spaced items (or all, or none).
    """
    if target <= 0:
        return []
    if len(items) <= target:
        return list(items)
    step = len(items) / target
    return [items[int(i * step)] for i in range(target)]


def extract_keyframes(file_path: Path, *, per_minute: int = 4, max_frames: int = 20) -> list[bytes]:
    """Return up to a capped, rate-scaled set of video keyframes as JPEG bytes.

    Args:
        file_path (Path): Path to the media file.
        per_minute (int): Target frames per minute of video.
        max_frames (int): Hard ceiling on returned frames.

    Returns:
        list[bytes]: JPEG-encoded frames in time order; ``[]`` when there is no
            decodable video stream.
    """
    if per_minute <= 0 or max_frames <= 0:
        return []
    candidates: list[bytes] = []
    target = 1
    try:
        with av.open(str(file_path)) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            duration_sec = float(container.duration / 1_000_000) if container.duration else 0.0
            target = min(max_frames, max(1, round(duration_sec / 60.0 * per_minute)))
            work_budget = max_frames * _CANDIDATE_BUDGET_FACTOR
            for frame in container.decode(stream):
                buffer = io.BytesIO()
                frame.to_image().save(buffer, format="JPEG")
                candidates.append(buffer.getvalue())
                if len(candidates) >= work_budget:
                    break
    except (FFmpegError, ValueError, OSError) as exc:
        logger.warning("Keyframe extraction failed for {}: {}", file_path.name, exc)
    return subsample(candidates, target) if candidates else []
```

`target` defaults to 1 and is set inside the `with` block; on a mid-decode error we still return whatever subsampled (fail-soft), and an empty `candidates` yields `[]`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_keyframes.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add nextext/core/keyframes.py tests/test_keyframes.py pyproject.toml uv.lock
git commit -m "feat(core): video keyframe extraction (PyAV, scene-diverse)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Produce keyframes during the pipeline run

**Files:**
- Modify: `nextext/api/jobs.py` (`_run_pipeline_blocking`, ~lines 340-466)
- Test: `tests/test_api/test_pipeline_keyframes.py` (create)

**Interfaces:**
- Consumes: `extract_keyframes` (Task 2), `state.file_path`, `state.options.keyframes_per_minute` / `keyframes_max`.
- Produces: `result["keyframes"]: list[bytes]` populated on both the normal and empty-transcript (silent-video) paths.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api/test_pipeline_keyframes.py
from pathlib import Path

import pandas as pd

from nextext.api import jobs as jobs_module
from nextext.api.jobs import JobState, _run_pipeline_blocking
from nextext.api.schemas import JobOptions, JobStatus


def test_pipeline_populates_keyframes(monkeypatch, tmp_path: Path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")

    # Stub the heavy stages so only keyframe wiring is exercised.
    df = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": [""], "text": ["hi"]})
    monkeypatch.setattr(
        "nextext.pipeline.transcription_pipeline", lambda **kwargs: (df, "en")
    )
    monkeypatch.setattr(jobs_module, "extract_keyframes", lambda path, **kw: [b"\xff\xd8\xff0", b"\xff\xd8\xff1"])

    state = JobState(
        job_id="j1",
        owner_id="o",
        file_name="clip.mp4",
        file_path=media,
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({"task": "transcribe"}),
        status=JobStatus.QUEUED,
    )
    result = _run_pipeline_blocking(state, lambda *a, **k: None)
    assert result["keyframes"] == [b"\xff\xd8\xff0", b"\xff\xd8\xff1"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_api/test_pipeline_keyframes.py -v`
Expected: FAIL — `result` has no `"keyframes"` key (and `extract_keyframes` is not imported in `jobs.py`).

- [ ] **Step 3: Wire keyframe extraction into the pipeline**

In `nextext/api/jobs.py`, add the import near the other `nextext.core` imports inside `_run_pipeline_blocking` (or at module top):

```python
    from nextext.core.keyframes import extract_keyframes
```

Immediately after the transcription stage produces `df` (after the `df, updated_src_lang = transcription_pipeline(...)` call and `file_opts["src_lang"] = updated_src_lang`), compute keyframes once:

```python
    keyframes = extract_keyframes(
        state.file_path,
        per_minute=opts.keyframes_per_minute,
        max_frames=opts.keyframes_max,
    )
```

Add `"keyframes": keyframes,` to **both** result payloads: the empty-transcript early-return `payload` dict and the main `result` dict (both already set `"transcript"`, `"summary"`, etc.).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_api/test_pipeline_keyframes.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add nextext/api/jobs.py tests/test_api/test_pipeline_keyframes.py
git commit -m "feat(pipeline): extract keyframes into job result" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Serve `keyframes.zip` as an artifact

**Files:**
- Modify: `nextext/api/artifacts.py` (`SUPPORTED_ARTIFACTS` line 340-355; `render_artifact` line 263-337)
- Test: `tests/test_api/test_keyframes_artifact.py` (create)

**Interfaces:**
- Consumes: `result["keyframes"]` (Task 3), `_zip_members` (artifacts.py:143-157), `_APP_ZIP` (artifacts.py:260).
- Produces: `render_artifact(state, "keyframes.zip") -> (zip_bytes, "application/zip")`; `"keyframes.zip"` in `SUPPORTED_ARTIFACTS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api/test_keyframes_artifact.py
import io
import zipfile
from pathlib import Path

from nextext.api import artifacts
from nextext.api.jobs import JobState
from nextext.api.schemas import JobOptions, JobStatus


def _job_with_keyframes(frames: list[bytes]) -> JobState:
    return JobState(
        job_id="j1",
        owner_id="o",
        file_name="clip.mp4",
        file_path=Path("clip.mp4"),
        source_file_hash="sha256:x",
        options=JobOptions.model_validate({}),
        status=JobStatus.COMPLETED,
        result={"keyframes": frames},
    )


def test_keyframes_zip_contains_each_frame() -> None:
    rendered = artifacts.render_artifact(_job_with_keyframes([b"\xff\xd8\xff0", b"\xff\xd8\xff1"]), "keyframes.zip")
    assert rendered is not None
    payload, content_type = rendered
    assert content_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = sorted(zf.namelist())
    assert len(names) == 2
    assert names[0].endswith(".jpg")


def test_keyframes_zip_absent_returns_none() -> None:
    assert artifacts.render_artifact(_job_with_keyframes([]), "keyframes.zip") is None


def test_keyframes_zip_is_supported() -> None:
    assert "keyframes.zip" in artifacts.SUPPORTED_ARTIFACTS
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_api/test_keyframes_artifact.py -v`
Expected: FAIL — `"keyframes.zip"` not supported; `render_artifact` returns `None`.

- [ ] **Step 3: Add the artifact**

In `nextext/api/artifacts.py`, add `"keyframes.zip"` to the `SUPPORTED_ARTIFACTS` frozenset. Then add a branch in `render_artifact` immediately before the final `return None` (mirrors the `archive.zip` branch, using the existing `_zip_members`):

```python
    if name == "keyframes.zip":
        frames = result.get("keyframes")
        if not frames:
            return None
        members = [(f"frame_{index:03d}.jpg", payload) for index, payload in enumerate(frames)]
        return _zip_members(members), _APP_ZIP
```

- [ ] **Step 4: Run the tests + full suite**

Run: `uv run pytest tests/test_api/test_keyframes_artifact.py -v && uv run pytest -q`
Expected: PASS (3 passed; full suite green).

- [ ] **Step 5: Commit**

```bash
git add nextext/api/artifacts.py tests/test_api/test_keyframes_artifact.py
git commit -m "feat(artifacts): serve keyframes.zip" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Coverage:** keyframe options (Task 1) → extraction (Task 2) → pipeline population, incl. silent-video path (Task 3) → `keyframes.zip` served via the existing artifact route (Task 4, which is gated on `SUPPORTED_ARTIFACTS` so no route change is needed). docint's `NextextClient` fetches `/jobs/{id}/artifacts/keyframes.zip`, matching the wire contract in the docint plan.

**Placeholder scan:** No TBD/TODO; every code step shows complete code. Task 2 Step 1 conditionally runs `uv add pillow` only if the import probe fails — a guarded action, not a blank.

**Type consistency:** `extract_keyframes(file_path, *, per_minute, max_frames) -> list[bytes]` is identical between Task 2 (def) and Task 3 (call). `result["keyframes"]: list[bytes]` is set in Task 3 and read in Task 4. `keyframes.zip` is the exact artifact name in Task 4 and in the docint plan's Nextext wire contract.

**Optional follow-ups (out of scope):** include keyframes in the per-job `archive.zip` via `_render_archive_members`; surface a `keyframes_url` in `JobResult` (mirroring `_wordcloud_url`).
