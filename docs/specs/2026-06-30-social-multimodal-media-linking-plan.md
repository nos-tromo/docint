# Social Multimodal Media Linking — Implementation Plan (docint)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest a social export's media (images + video/audio), link every derived artifact to its posting's UUID, and surface a post and its media as one entity at query time.

**Architecture:** Three phases in docint. (1) The React SPA folder-upload preserves the export's subdir tree end-to-end. (2) A new *social linker* enrichment pass joins the postings table to the media manifest (`Posting ID` ⊂ `Media ID`), routes each media file — images through the existing CLIP path, video/audio to the remote **Nextext** service for a transcript (`docint.jsonl`) plus rate-sampled, near-duplicate-pruned keyframes — and stamps `posting_uuid` on each artifact. (3) A `LinkFollowingPostprocessor` groups a post with its media at retrieval. docint stays CPU-only: all media decoding (transcode, ASR, frame extraction) is delegated to Nextext over HTTP.

**Tech Stack:** Python ≥3.11,<3.12 (uv), FastAPI, llama-index, Qdrant, httpx; React 19 + TypeScript + Vitest (happy-dom + React Testing Library).

**Companion spec:** `docs/specs/2026-06-30-social-multimodal-media-linking-design.md`.

## Global Constraints

- **Python ≥3.11,<3.12**; manage deps with `uv` (`uv add`/`uv remove`). Run tests with `uv run pytest`; lint/format/type with `uv run pre-commit run --all-files` (ruff check, ruff format, pyrefly) before every commit.
- **No new heavy media dependency in docint.** No `ffmpeg`/`av`/`PyAV`/`moviepy`. Media decoding, ASR, and frame extraction are delegated to Nextext over HTTP. The only transport is `httpx` (already a dependency).
- **All `os.getenv` and config dataclasses live in `docint/utils/env_cfg.py`.** Other modules import loaders from there.
- **Google-style docstrings** on every new/modified function and class.
- **Tests with every functional change** (pytest under `tests/`; `conftest.py` provides mock stubs for external deps; frontend Vitest under `frontend/src`).
- **Fail-soft posture** (matches rerank/image-tagging): a dead/timed-out Nextext job, a missing media file, or an orphan media row skips *that item*, logs one warning, and never fails the batch.
- **Link key is `posting_uuid`** (the post's `UUID`). Postings text nodes already carry it as `reference_metadata.uuid`; media artifacts get `posting_uuid` stamped.
- **Keyframe defaults:** `KEYFRAMES_PER_MINUTE=4`, `KEYFRAMES_MAX=20`, `KEYFRAME_DEDUP_COSINE=0.95`. CLIP embeddings are L2-normalized, so cosine similarity == dot product.
- **Commit message trailer** (every commit): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Work on branch `feat/social-multimodal-linking`.

### Nextext wire contract (consumed by Phase 2; produced by the separate Nextext plan)

- `POST {NEXTEXT_API_BASE}/jobs` — multipart `file=<media>` + `options=<JSON JobOptions>` (form field, required) → `201 {"job_id": str, "status": str, "created_at": ...}`.
- `GET {NEXTEXT_API_BASE}/jobs/{job_id}` → `{"status": "queued"|"running"|"completed"|"failed", ...}`.
- `GET {NEXTEXT_API_BASE}/jobs/{job_id}/artifacts/docint.jsonl` → NDJSON bytes (already exists).
- `GET {NEXTEXT_API_BASE}/jobs/{job_id}/artifacts/keyframes.zip` → ZIP of JPEG frames (**Nextext plan adds** this to `SUPPORTED_ARTIFACTS`; `JobOptions` gains optional `keyframes_per_minute` / `keyframes_max` with defaults so docint's `options` payload validates).

---

## Phase 1 — Structure-preserving SPA folder upload

### Task 1: Backend preserves uploaded subdir tree

**Files:**
- Modify: `docint/core/api.py` (add `_safe_relative_dest` near `_resolve_qdrant_src_dir` at 191–204; replace the flatten at `api.py:2576-2577`)
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `_safe_relative_dest(batch_dir: Path, raw_name: str) -> Path` — a path strictly inside `batch_dir`, traversal-neutralized.

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_api.py — append
from pathlib import Path
from docint.core import api as api_module


def test_safe_relative_dest_preserves_subdirs(tmp_path: Path) -> None:
    assert api_module._safe_relative_dest(tmp_path, "media/sub/a.jpg") == tmp_path / "media" / "sub" / "a.jpg"


def test_safe_relative_dest_strips_traversal(tmp_path: Path) -> None:
    dest = api_module._safe_relative_dest(tmp_path, "../../etc/passwd")
    assert dest == tmp_path / "etc" / "passwd"


def test_safe_relative_dest_drops_absolute_leading_slash(tmp_path: Path) -> None:
    assert api_module._safe_relative_dest(tmp_path, "/abs/x.jpg") == tmp_path / "abs" / "x.jpg"


def test_safe_relative_dest_normalizes_backslashes(tmp_path: Path) -> None:
    assert api_module._safe_relative_dest(tmp_path, "media\\sub\\b.png") == tmp_path / "media" / "sub" / "b.png"


def test_safe_relative_dest_empty_falls_back(tmp_path: Path) -> None:
    assert api_module._safe_relative_dest(tmp_path, "") == tmp_path / "upload"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_api.py -k safe_relative_dest -v`
Expected: FAIL — `AttributeError: module 'docint.core.api' has no attribute '_safe_relative_dest'`

- [ ] **Step 3: Implement `_safe_relative_dest`**

Add directly after `_resolve_qdrant_src_dir` (ends at `api.py:204`):

```python
def _safe_relative_dest(batch_dir: Path, raw_name: str) -> Path:
    """Resolve an uploaded file's relative path safely under ``batch_dir``.

    Preserves subdirectories from a browser folder upload (the
    ``webkitRelativePath`` sent as the multipart filename) while neutralizing
    path traversal: backslashes are normalized to ``/`` and empty, ``.`` and
    ``..`` segments are dropped, so the result can never escape ``batch_dir``.

    Args:
        batch_dir (Path): The collection's upload directory (containment root).
        raw_name (str): Client-supplied name, possibly a relative path.

    Returns:
        Path: A path strictly inside ``batch_dir``.
    """
    raw = (raw_name or "upload").replace("\\", "/")
    parts = [segment for segment in raw.split("/") if segment not in ("", ".", "..")]
    if not parts:
        parts = ["upload"]
    return batch_dir.joinpath(*parts)
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_api.py -k safe_relative_dest -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Write the failing endpoint test**

```python
# tests/test_api.py — append (uses the existing `client` fixture + autouse _patch_rag)
def test_ingest_upload_preserves_subdir_structure(client, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)

    def fake_ingest(collection, path, hybrid=True, progress_callback=None) -> None:
        _ = (collection, path, hybrid, progress_callback)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    response = client.post(
        "/ingest/upload",
        data={"collection": "tree", "hybrid": "false"},
        files=[("files", ("media/sub/a.jpg", b"\xff\xd8\xff", "image/jpeg"))],
    )

    assert response.status_code == 200
    assert (tmp_path / "tree" / "media" / "sub" / "a.jpg").is_file()
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_api.py::test_ingest_upload_preserves_subdir_structure -v`
Expected: FAIL — file written flat at `tree/a.jpg`, so the subdir assertion fails.

- [ ] **Step 7: Replace the flatten in `ingest_upload`**

At `api.py:2576-2577`, replace:

```python
            filename = Path(upload.filename or "upload").name
            dest = batch_dir / filename
```

with:

```python
            dest = _safe_relative_dest(batch_dir, upload.filename or "upload")
            dest.parent.mkdir(parents=True, exist_ok=True)
            filename = str(dest.relative_to(batch_dir))
```

(`filename` stays the relative path string used in the `upload_progress` / `file_saved` SSE events.)

- [ ] **Step 8: Run the endpoint test + the existing upload tests**

Run: `uv run pytest tests/test_api.py -k "ingest_upload or safe_relative_dest" -v`
Expected: PASS (existing four `ingest_upload` tests still green + the two new ones)

- [ ] **Step 9: Commit**

```bash
git add docint/core/api.py tests/test_api.py
git commit -m "feat(ingest): preserve uploaded subdir tree (traversal-safe)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 2: Frontend sends `webkitRelativePath` as the upload filename

**Files:**
- Modify: `frontend/src/api/ingest.ts` (extract `buildIngestFormData`; use `webkitRelativePath`)
- Test: `frontend/src/api/ingest.test.ts` (create)

**Interfaces:**
- Produces: `buildIngestFormData(collection: string, files: File[]): FormData`

- [ ] **Step 1: Write the failing test**

```ts
// frontend/src/api/ingest.test.ts
import { describe, it, expect } from 'vitest'
import { buildIngestFormData } from './ingest'

describe('buildIngestFormData', () => {
  it('uses webkitRelativePath as the upload filename when present', () => {
    const f = new File([new Uint8Array([1])], 'a.jpg', { type: 'image/jpeg' })
    Object.defineProperty(f, 'webkitRelativePath', { value: 'export/media/sub/a.jpg' })
    const fd = buildIngestFormData('c1', [f])
    const entries = fd.getAll('files') as File[]
    expect(entries[0].name).toBe('export/media/sub/a.jpg')
    expect(fd.get('collection')).toBe('c1')
  })

  it('falls back to the file name when webkitRelativePath is empty', () => {
    const f = new File([new Uint8Array([1])], 'b.png', { type: 'image/png' })
    const fd = buildIngestFormData('c1', [f])
    expect((fd.getAll('files') as File[])[0].name).toBe('b.png')
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd frontend && pnpm test -- ingest.test.ts`
Expected: FAIL — `buildIngestFormData` is not exported.

- [ ] **Step 3: Refactor `ingest.ts`**

Add the exported helper above the `streamIngestUpload` generator, and replace the inline FormData build (`const fd = new FormData(); fd.append('collection', collection); for (const f of files) fd.append('files', f, f.name)`) with a call to it:

```ts
export function buildIngestFormData(collection: string, files: File[]): FormData {
  const fd = new FormData()
  fd.append('collection', collection)
  for (const f of files) fd.append('files', f, f.webkitRelativePath || f.name)
  return fd
}
```

In the generator body, replace the three FormData lines with:

```ts
  const fd = buildIngestFormData(collection, files)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd frontend && pnpm test -- ingest.test.ts`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/ingest.ts frontend/src/api/ingest.test.ts
git commit -m "feat(frontend): send webkitRelativePath as upload filename" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 3: Frontend folder picker

**Files:**
- Modify: `frontend/src/components/ingest/Dropzone.tsx` (add a `webkitdirectory` folder input + button)
- Test: `frontend/src/components/ingest/Dropzone.test.tsx` (create)

**Interfaces:**
- Consumes: `Dropzone` named export with props `{ onFiles: (files: File[]) => void; disabled?: boolean }`.

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/ingest/Dropzone.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { Dropzone } from './Dropzone'

describe('Dropzone folder picker', () => {
  it('exposes a folder input with webkitdirectory and forwards picked files', () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)

    expect(screen.getByRole('button', { name: /choose a folder/i })).toBeInTheDocument()

    const folderInput = Array.from(
      document.querySelectorAll('input[type="file"]')
    ).find((el) => el.hasAttribute('webkitdirectory')) as HTMLInputElement
    expect(folderInput).toBeTruthy()

    const f = new File([new Uint8Array([1])], 'a.jpg', { type: 'image/jpeg' })
    Object.defineProperty(folderInput, 'files', { value: [f] })
    fireEvent.change(folderInput)
    expect(onFiles).toHaveBeenCalledWith([f])
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd frontend && pnpm test -- Dropzone.test.tsx`
Expected: FAIL — no folder button / `webkitdirectory` input rendered.

- [ ] **Step 3: Add the folder input to `Dropzone.tsx`**

Add a second ref beside `inputRef`:

```tsx
  const folderInputRef = useRef<HTMLInputElement>(null)
```

Inside the dropzone `<div>`, after the existing `<p>...</p>`, add the button and folder input (the `stopPropagation` keeps the surrounding div's click-to-pick-files from also firing):

```tsx
      <button
        type="button"
        className="mt-3 underline"
        onClick={(e) => {
          e.stopPropagation()
          folderInputRef.current?.click()
        }}
      >
        Or choose a folder
      </button>
      <input
        ref={folderInputRef}
        type="file"
        multiple
        className="hidden"
        {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
        onChange={(e) => {
          const list = Array.from(e.target.files ?? [])
          if (list.length) onFiles(list)
          e.target.value = ''
        }}
      />
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd frontend && pnpm test -- Dropzone.test.tsx`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ingest/Dropzone.tsx frontend/src/components/ingest/Dropzone.test.tsx
git commit -m "feat(frontend): add folder picker to ingest dropzone" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — Social linker

### Task 4: Media-manifest detection predicate

**Files:**
- Modify: `docint/core/readers/tables.py` (add `MEDIA_MANIFEST_REQUIRED_COLUMNS` + `is_media_manifest`)
- Test: `tests/test_tables_media_manifest.py` (create)

**Interfaces:**
- Produces: `is_media_manifest(columns: Iterable[Any]) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tables_media_manifest.py
from docint.core.readers.tables import is_media_manifest


def test_is_media_manifest_detects_join_columns() -> None:
    assert is_media_manifest(["Media ID", "Exported media filename", "Extra"])
    assert is_media_manifest(["media id", "exported media filename"])


def test_is_media_manifest_rejects_postings() -> None:
    assert not is_media_manifest(["UUID", "Posting ID", "Text Content"])
    assert not is_media_manifest(["Media ID"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_tables_media_manifest.py -v`
Expected: FAIL — `ImportError: cannot import name 'is_media_manifest'`

- [ ] **Step 3: Implement the predicate**

Add to `docint/core/readers/tables.py` (after `_normalize_column_name`; `Iterable` is already imported from `collections.abc`):

```python
MEDIA_MANIFEST_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        _normalize_column_name("Media ID"),
        _normalize_column_name("Exported media filename"),
    }
)


def is_media_manifest(columns: Iterable[Any]) -> bool:
    """Return whether a table's columns identify it as a social media manifest.

    Detection is fuzzy by design: only the two join columns must be present
    (``Media ID`` + ``Exported media filename``), not an exact header set, so
    the manifest is recognized regardless of platform-specific extra columns.

    Args:
        columns (Iterable[Any]): The table's column names.

    Returns:
        bool: True when both join columns are present (case-insensitively).
    """
    normalized = {_normalize_column_name(column) for column in columns}
    return MEDIA_MANIFEST_REQUIRED_COLUMNS.issubset(normalized)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_tables_media_manifest.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add docint/core/readers/tables.py tests/test_tables_media_manifest.py
git commit -m "feat(readers): add social media-manifest detection predicate" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 5: `NextextConfig` + keyframe knobs in env_cfg

**Files:**
- Modify: `docint/utils/env_cfg.py` (add `NextextConfig` + `load_nextext_env`)
- Test: `tests/test_env_cfg_nextext.py` (create)

**Interfaces:**
- Produces: `NextextConfig(api_base, api_key, timeout, poll_interval, poll_max_seconds, enabled, keyframes_per_minute, keyframes_max, keyframe_dedup_cosine)` and `load_nextext_env(...) -> NextextConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_cfg_nextext.py
import pytest
from docint.utils.env_cfg import load_nextext_env


def test_nextext_disabled_when_base_unset(monkeypatch) -> None:
    monkeypatch.delenv("NEXTEXT_API_BASE", raising=False)
    cfg = load_nextext_env()
    assert cfg.enabled is False
    assert cfg.keyframes_per_minute == 4
    assert cfg.keyframes_max == 20
    assert cfg.keyframe_dedup_cosine == 0.95


def test_nextext_enabled_and_overrides(monkeypatch) -> None:
    monkeypatch.setenv("NEXTEXT_API_BASE", "http://nextext:8000/")
    monkeypatch.setenv("KEYFRAMES_PER_MINUTE", "6")
    cfg = load_nextext_env()
    assert cfg.enabled is True
    assert cfg.api_base == "http://nextext:8000"
    assert cfg.keyframes_per_minute == 6


def test_nextext_rejects_out_of_range_cosine(monkeypatch) -> None:
    monkeypatch.setenv("KEYFRAME_DEDUP_COSINE", "1.5")
    with pytest.raises(ValueError):
        load_nextext_env()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_env_cfg_nextext.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_nextext_env'`

- [ ] **Step 3: Implement the config + loader**

Add to `docint/utils/env_cfg.py` (mirrors `NERClientConfig`/`load_ner_client_env` at 891–939; `os`, `dataclass` already imported):

```python
@dataclass(frozen=True)
class NextextConfig:
    """Remote Nextext media-processing client + keyframe-sampling configuration."""

    api_base: str
    api_key: str | None
    timeout: float
    poll_interval: float
    poll_max_seconds: float
    enabled: bool
    keyframes_per_minute: int
    keyframes_max: int
    keyframe_dedup_cosine: float


def load_nextext_env(
    default_api_base: str = "",
    default_timeout: float = 30.0,
    default_poll_interval: float = 2.0,
    default_poll_max_seconds: float = 1800.0,
    default_keyframes_per_minute: int = 4,
    default_keyframes_max: int = 20,
    default_keyframe_dedup_cosine: float = 0.95,
) -> NextextConfig:
    """Load Nextext client + keyframe settings from the environment.

    docint forwards video/audio to Nextext, which returns a ``docint.jsonl``
    transcript and a ``keyframes.zip`` artifact. The two sampling knobs ride
    along as Nextext job options; the cosine threshold is applied docint-side.
    The client is disabled unless ``NEXTEXT_API_BASE`` is set, so non-social
    dev hosts skip video/audio rows rather than erroring.

    Args:
        default_api_base (str): Fallback base URL (empty ⇒ disabled).
        default_timeout (float): Per-request HTTP timeout (seconds).
        default_poll_interval (float): Delay between job status polls (seconds).
        default_poll_max_seconds (float): Max wall-clock to await a job (seconds).
        default_keyframes_per_minute (int): Frame sampling rate forwarded to Nextext.
        default_keyframes_max (int): Hard candidate-frame ceiling forwarded to Nextext.
        default_keyframe_dedup_cosine (float): Drop a frame whose cosine similarity
            to a kept frame is >= this value. Must be within [0, 1].

    Returns:
        NextextConfig: Resolved configuration.

    Raises:
        ValueError: If ``KEYFRAME_DEDUP_COSINE`` is not within [0, 1].
    """
    raw_base = os.getenv("NEXTEXT_API_BASE", default_api_base).strip()
    raw_key = os.getenv("NEXTEXT_API_KEY")
    api_key = raw_key.strip() if raw_key and raw_key.strip() else None
    cosine = float(os.getenv("KEYFRAME_DEDUP_COSINE", default_keyframe_dedup_cosine))
    if not (0.0 <= cosine <= 1.0):
        raise ValueError(f"KEYFRAME_DEDUP_COSINE={cosine!r} is out of range — must be within [0, 1].")
    return NextextConfig(
        api_base=raw_base.rstrip("/"),
        api_key=api_key,
        timeout=float(os.getenv("NEXTEXT_TIMEOUT", default_timeout)),
        poll_interval=float(os.getenv("NEXTEXT_POLL_INTERVAL", default_poll_interval)),
        poll_max_seconds=float(os.getenv("NEXTEXT_POLL_MAX_SECONDS", default_poll_max_seconds)),
        enabled=bool(raw_base),
        keyframes_per_minute=int(os.getenv("KEYFRAMES_PER_MINUTE", default_keyframes_per_minute)),
        keyframes_max=int(os.getenv("KEYFRAMES_MAX", default_keyframes_max)),
        keyframe_dedup_cosine=cosine,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_env_cfg_nextext.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add docint/utils/env_cfg.py tests/test_env_cfg_nextext.py
git commit -m "feat(config): add NextextConfig + keyframe knobs" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 6: `NextextClient` (submit → poll → fetch artifacts)

**Files:**
- Create: `docint/utils/nextext_client.py`
- Test: `tests/test_nextext_client.py`

**Interfaces:**
- Consumes: `load_nextext_env`, `NextextConfig` (Task 5).
- Produces: `NextextResult(transcript_jsonl: bytes | None, keyframes: list[bytes], status: str, error: str | None)` and `NextextClient(cfg=None, *, client=None)` with `process_media(file_path: Path) -> NextextResult`.

- [ ] **Step 1: Write the failing test (httpx MockTransport, no real network)**

```python
# tests/test_nextext_client.py
import io
import zipfile

import httpx
import pytest

from docint.utils.env_cfg import NextextConfig
from docint.utils.nextext_client import NextextClient, NextextResult


def _cfg() -> NextextConfig:
    return NextextConfig(
        api_base="http://nextext.test",
        api_key=None,
        timeout=5.0,
        poll_interval=0.0,
        poll_max_seconds=5.0,
        enabled=True,
        keyframes_per_minute=4,
        keyframes_max=20,
        keyframe_dedup_cosine=0.95,
    )


def _keyframes_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("frame_0.jpg", b"\xff\xd8\xff0")
        zf.writestr("frame_1.jpg", b"\xff\xd8\xff1")
    return buf.getvalue()


def _handler(request: httpx.Request) -> httpx.Response:
    if request.method == "POST" and request.url.path == "/jobs":
        return httpx.Response(201, json={"job_id": "J1", "status": "queued"})
    if request.url.path == "/jobs/J1" and request.method == "GET":
        return httpx.Response(200, json={"status": "completed"})
    if request.url.path == "/jobs/J1/artifacts/docint.jsonl":
        return httpx.Response(200, content=b'{"text":"hi","start_seconds":0,"end_seconds":1}\n')
    if request.url.path == "/jobs/J1/artifacts/keyframes.zip":
        return httpx.Response(200, content=_keyframes_zip())
    return httpx.Response(404)


def test_process_media_returns_transcript_and_keyframes(tmp_path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fakevideo")
    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(_handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert result.status == "completed"
    assert result.transcript_jsonl is not None and b"hi" in result.transcript_jsonl
    assert len(result.keyframes) == 2


def test_process_media_failsoft_on_job_failure(tmp_path) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"job_id": "J2", "status": "queued"})
        if request.url.path == "/jobs/J2":
            return httpx.Response(200, json={"status": "failed"})
        return httpx.Response(404)

    client = httpx.Client(base_url="http://nextext.test", transport=httpx.MockTransport(handler))
    result = NextextClient(_cfg(), client=client).process_media(media)
    assert result.status == "failed"
    assert result.transcript_jsonl is None
    assert result.keyframes == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_nextext_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docint.utils.nextext_client'`

- [ ] **Step 3: Implement the client**

```python
# docint/utils/nextext_client.py
"""Thin HTTP client for the remote Nextext media-processing service.

Mirrors the posture of ``ner_client``/``clip_client``: docint forwards a
media file to Nextext, which decodes it (PyAV/ffmpeg), runs VAD → diarize →
Whisper, and exposes a ``docint.jsonl`` transcript plus a ``keyframes.zip``
artifact. docint stays CPU-only and media-dependency-free.

All failures are fail-soft: ``process_media`` returns a ``NextextResult`` with
``status='error'`` and empty payloads rather than raising, so one bad clip
never aborts a batch.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from loguru import logger

from docint.utils.env_cfg import NextextConfig, load_nextext_env

__all__ = ["NextextClient", "NextextResult"]

_TERMINAL_OK = "completed"
_TERMINAL_FAIL = {"failed", "error", "cancelled"}


@dataclass(frozen=True)
class NextextResult:
    """Outcome of a single media file processed by Nextext."""

    status: str
    transcript_jsonl: bytes | None = None
    keyframes: list[bytes] = field(default_factory=list)
    error: str | None = None


class NextextClient:
    """Submit a media file to Nextext, await the job, and fetch its artifacts."""

    def __init__(self, cfg: NextextConfig | None = None, *, client: httpx.Client | None = None) -> None:
        """Create the client.

        Args:
            cfg (NextextConfig | None): Configuration; defaults to ``load_nextext_env()``.
            client (httpx.Client | None): Injected HTTP client (tests pass a
                ``MockTransport``-backed client). When ``None``, one is built
                from ``cfg`` with the Bearer header when an API key is set.
        """
        self._cfg = cfg if cfg is not None else load_nextext_env()
        if client is not None:
            self._client = client
        else:
            headers = {"Authorization": f"Bearer {self._cfg.api_key}"} if self._cfg.api_key else {}
            self._client = httpx.Client(base_url=self._cfg.api_base, timeout=self._cfg.timeout, headers=headers)

    def _options_payload(self) -> str:
        """Return the JSON ``options`` form field forwarded to Nextext."""
        return json.dumps(
            {
                "keyframes_per_minute": self._cfg.keyframes_per_minute,
                "keyframes_max": self._cfg.keyframes_max,
            }
        )

    def _await_job(self, job_id: str) -> str:
        """Poll a job until it reaches a terminal status or the budget expires.

        Args:
            job_id (str): The job identifier returned by submission.

        Returns:
            str: The terminal status string (``'completed'`` on success).
        """
        deadline = time.monotonic() + self._cfg.poll_max_seconds
        while True:
            resp = self._client.get(f"/jobs/{job_id}")
            resp.raise_for_status()
            status = str(resp.json().get("status") or "").lower()
            if status == _TERMINAL_OK or status in _TERMINAL_FAIL:
                return status
            if time.monotonic() >= deadline:
                return "timeout"
            time.sleep(self._cfg.poll_interval)

    def _fetch_artifact(self, job_id: str, name: str) -> bytes | None:
        """Fetch one job artifact, returning ``None`` when absent (404)."""
        resp = self._client.get(f"/jobs/{job_id}/artifacts/{name}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def _unzip_jpegs(blob: bytes | None) -> list[bytes]:
        """Return the JPEG members of a keyframes zip in name order."""
        if not blob:
            return []
        frames: list[bytes] = []
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    frames.append(zf.read(name))
        return frames

    def process_media(self, file_path: Path) -> NextextResult:
        """Run one media file through Nextext and return its artifacts.

        Args:
            file_path (Path): Path to the audio/video file to process.

        Returns:
            NextextResult: Transcript JSONL bytes (or ``None`` when no speech)
                and keyframe JPEG bytes. Fail-soft: transport/HTTP/job errors
                yield ``status='error'`` with empty payloads.
        """
        if not self._cfg.enabled:
            return NextextResult(status="disabled")
        try:
            with file_path.open("rb") as handle:
                resp = self._client.post(
                    "/jobs",
                    files={"file": (file_path.name, handle, "application/octet-stream")},
                    data={"options": self._options_payload()},
                )
            resp.raise_for_status()
            job_id = str(resp.json()["job_id"])
            status = self._await_job(job_id)
            if status != _TERMINAL_OK:
                logger.warning("Nextext job {} for {} ended status={}", job_id, file_path.name, status)
                return NextextResult(status=status)
            transcript = self._fetch_artifact(job_id, "docint.jsonl")
            keyframes = self._unzip_jpegs(self._fetch_artifact(job_id, "keyframes.zip"))
            return NextextResult(status="completed", transcript_jsonl=transcript, keyframes=keyframes)
        except Exception as exc:
            logger.warning("Nextext processing failed for {}: {}", file_path.name, exc)
            return NextextResult(status="error", error=str(exc))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_nextext_client.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add docint/utils/nextext_client.py tests/test_nextext_client.py
git commit -m "feat(nextext): add remote media-processing client" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 7: `ImageIngestionService.ingest_keyframe_set` (embed → prune → caption survivors)

**Files:**
- Modify: `docint/core/ingest/images_service.py` (add `ingest_keyframe_set`)
- Test: `tests/test_keyframe_dedup.py` (create)

**Interfaces:**
- Consumes: existing `_get_embedding_backend`, `_get_tagging_backend`, `_resolve_collection_name`, `_ensure_collection`, `_get_vector_store`, `_hash_image_bytes`, `_point_id_from_image_id`, `IngestContext`.
- Produces: `ImageIngestionService.ingest_keyframe_set(frames: list[bytes], *, context: IngestContext, source_doc_id: str | None, extra_metadata: dict[str, Any] | None = None, dedup_cosine: float = 0.95) -> list[StoredImageRecord]`.

- [ ] **Step 1: Write the failing test (inject fake embed + tag backends)**

```python
# tests/test_keyframe_dedup.py
from typing import Any

from docint.core.ingest.images_service import ImageIngestionService, IngestContext


class _FakeEmbed:
    """Returns a preset unit vector per frame; records calls."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    @property
    def dimension(self) -> int:
        return len(self.vectors[0])

    def embed(self, image_bytes: bytes) -> list[float]:
        v = self.vectors[self.calls]
        self.calls += 1
        return v

    def embed_text(self, text: str) -> list[float]:  # pragma: no cover - unused
        return self.vectors[0]


class _FakeTagger:
    def __init__(self) -> None:
        self.calls = 0

    def describe_and_tag(self, image_bytes: bytes, mime_type: str) -> tuple[str, list[str]]:
        self.calls += 1
        return (f"caption {self.calls}", ["tag"])


def _service(monkeypatch, embed: _FakeEmbed, tagger: _FakeTagger) -> ImageIngestionService:
    svc = ImageIngestionService(qdrant_client=None)
    monkeypatch.setattr(svc, "_get_embedding_backend", lambda: embed)
    monkeypatch.setattr(svc, "_get_tagging_backend", lambda: tagger)
    stored: list[Any] = []
    monkeypatch.setattr(svc, "_ensure_collection", lambda **kw: None)

    class _Store:
        def add(self, nodes: list[Any]) -> None:
            stored.extend(nodes)

    monkeypatch.setattr(svc, "_get_vector_store", lambda name: _Store())
    svc._stored_nodes = stored  # type: ignore[attr-defined]
    return svc


def test_prunes_near_duplicate_frames(monkeypatch) -> None:
    # frame0 and frame1 identical (cosine 1.0 -> drop frame1); frame2 orthogonal (keep).
    embed = _FakeEmbed([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tagger = _FakeTagger()
    svc = _service(monkeypatch, embed, tagger)
    records = svc.ingest_keyframe_set(
        [b"f0", b"f1", b"f2"],
        context=IngestContext(source_collection="c"),
        source_doc_id="uuid-1",
        dedup_cosine=0.95,
    )
    stored = [r for r in records if r.status == "stored"]
    assert len(stored) == 2  # frame1 pruned
    assert tagger.calls == 2  # caption only the 2 survivors, not the pruned dup
    assert all(r.payload.get("posting_uuid") == "uuid-1" for r in stored)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_keyframe_dedup.py -v`
Expected: FAIL — `AttributeError: 'ImageIngestionService' object has no attribute 'ingest_keyframe_set'`

- [ ] **Step 3: Implement `ingest_keyframe_set`**

Add to `ImageIngestionService` (CLIP vectors are L2-normalized, so dot product == cosine):

```python
    def ingest_keyframe_set(
        self,
        frames: list[bytes],
        *,
        context: IngestContext,
        source_doc_id: str | None,
        extra_metadata: dict[str, Any] | None = None,
        dedup_cosine: float = 0.95,
    ) -> list[StoredImageRecord]:
        """Embed candidate keyframes, prune near-duplicates, caption survivors.

        Each frame is CLIP-embedded once. A greedy pass keeps a frame only when
        its maximum cosine similarity (== dot product, since CLIP vectors are
        L2-normalized) to an already-kept frame is below ``dedup_cosine``. Only
        survivors are sent to the (expensive) vision tagger and stored, each
        stamped with ``source_doc_id``/``extra_metadata`` so it links to its
        posting. Fail-soft: a frame that fails to embed is skipped, not raised.

        Args:
            frames (list[bytes]): Candidate keyframe image bytes, in time order.
            context (IngestContext): Collection-resolution context.
            source_doc_id (str | None): The posting UUID stamped on each point.
            extra_metadata (dict[str, Any] | None): Extra payload fields
                (e.g. ``posting_id``/``media_id``) merged onto each point.
            dedup_cosine (float): Drop a frame whose cosine similarity to a kept
                frame is >= this value.

        Returns:
            list[StoredImageRecord]: One record per survivor (status ``stored``).
        """
        if not self.img_ingestion_config.enabled or not frames:
            return []
        embedding_backend = self._get_embedding_backend()
        if embedding_backend is None:
            logger.warning("Keyframe ingestion skipped: no embedding backend.")
            return []
        try:
            target_collection = self._resolve_collection_name(context.source_collection)
        except Exception as exc:
            logger.warning("Keyframe ingestion skipped: {}", exc)
            return []

        kept_embeddings: list[list[float]] = []
        records: list[StoredImageRecord] = []
        tagger = self._get_tagging_backend()
        for frame_bytes in frames:
            try:
                embedding = self._run_with_retries(lambda: embedding_backend.embed(frame_bytes))
            except Exception as exc:
                logger.warning("Keyframe embed failed (skipping frame): {}", exc)
                continue
            if any(_cosine(embedding, kept) >= dedup_cosine for kept in kept_embeddings):
                continue
            kept_embeddings.append(embedding)

            description, tags = "", []
            if self.img_ingestion_config.tagging_enabled and tagger is not None:
                try:
                    description, tags = self._run_with_retries(
                        lambda: tagger.describe_and_tag(frame_bytes, "image/jpeg")
                    )
                except Exception as exc:
                    logger.warning("Keyframe tagging failed: {}", exc)

            image_id = self._hash_image_bytes(frame_bytes)
            point_id = self._point_id_from_image_id(image_id)
            text_parts = [description.strip()]
            if tags:
                text_parts.append("Tags: " + ", ".join(tags))
            node_text = "\n\n".join(part for part in text_parts if part).strip()
            payload: dict[str, Any] = {
                "image_id": image_id,
                "source_type": "social_media_keyframe",
                "source_collection": context.source_collection,
                "source_doc_id": source_doc_id,
                "posting_uuid": source_doc_id,
                "mime_type": "image/jpeg",
                "mimetype": "image/jpeg",
                "file_type": "image/jpeg",
                "llm_description": description,
                "llm_tags": tags,
                "vector_name": self.img_ingestion_config.vector_name,
                "image_collection": target_collection,
            }
            if extra_metadata:
                payload.update(extra_metadata)
            node = ImageNode(
                id_=point_id,
                text=node_text,
                metadata=payload,
                image_mimetype="image/jpeg",
                embedding=embedding,
            )
            try:
                self._ensure_collection(collection_name=target_collection, vector_dim=len(embedding))
                self._get_vector_store(target_collection).add([node])
            except Exception as exc:
                logger.warning("Keyframe store failed: {}", exc)
                continue
            records.append(
                StoredImageRecord(
                    point_id=point_id,
                    image_id=image_id,
                    status="stored",
                    payload=payload,
                    llm_description=description,
                    llm_tags=tags,
                )
            )
        return records
```

Add a module-level cosine helper near the top of `images_service.py` (after imports):

```python
def _cosine(a: list[float], b: list[float]) -> float:
    """Return the dot product of two equal-length L2-normalized vectors.

    CLIP embeddings are already L2-normalized, so the dot product equals the
    cosine similarity without re-normalizing.

    Args:
        a (list[float]): First vector.
        b (list[float]): Second vector.

    Returns:
        float: Cosine similarity in ``[-1, 1]``.
    """
    return sum(x * y for x, y in zip(a, b, strict=False))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_keyframe_dedup.py -v`
Expected: PASS (1 passed) — frame1 pruned, tagger called exactly twice.

- [ ] **Step 5: Commit**

```bash
git add docint/core/ingest/images_service.py tests/test_keyframe_dedup.py
git commit -m "feat(images): add keyframe-set ingestion with cosine dedup" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 8: Thread `posting_uuid` onto transcript segments

**Files:**
- Modify: `docint/utils/reference_metadata.py` (add `posting_uuid`)
- Modify: `docint/core/readers/json.py` (`_build_segment_metadata` copies posting link fields from `base` into `reference_metadata`)
- Test: `tests/test_json_posting_link.py` (create)

**Interfaces:**
- Consumes: `CustomJSONReader.iter_documents(file, extra_info={...})` already merges `extra_info` into `base_extra_info` (json.py:718-719), and `_build_segment_metadata` copies `base` via `metadata = dict(base)` (json.py:446).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_json_posting_link.py
from pathlib import Path

from docint.core.readers.json import CustomJSONReader


def test_transcript_segments_carry_posting_uuid(tmp_path: Path) -> None:
    jsonl = tmp_path / "clip.nextext.jsonl"
    jsonl.write_text(
        '{"text":"hello","start_seconds":0,"end_seconds":1}\n'
        '{"text":"world","start_seconds":1,"end_seconds":2}\n',
        encoding="utf-8",
    )
    docs = list(
        CustomJSONReader(is_jsonl=True).iter_documents(
            jsonl,
            extra_info={"posting_uuid": "uuid-9", "posting_id": "P_1", "media_id": "P_1_0"},
        )
    )
    assert len(docs) == 2
    for doc in docs:
        assert doc.metadata["posting_uuid"] == "uuid-9"
        assert doc.metadata["reference_metadata"]["posting_uuid"] == "uuid-9"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_json_posting_link.py -v`
Expected: FAIL — `reference_metadata` has no `posting_uuid` key (it is rebuilt fresh at json.py:532-550, dropping the top-level copy).

- [ ] **Step 3: Add `posting_uuid` to the reference-metadata schema**

In `docint/utils/reference_metadata.py`, add to `REFERENCE_METADATA_FIELDS` (after `"uuid": "UUID",`):

```python
    "posting_uuid": "Posting UUID",
    "posting_id": "Posting ID",
    "media_id": "Media ID",
```

- [ ] **Step 4: Copy the link fields into the segment's `reference_metadata`**

In `docint/core/readers/json.py`, inside `_build_segment_metadata`, immediately before `metadata["reference_metadata"] = reference_metadata` (json.py:550), add:

```python
        for link_field in ("posting_uuid", "posting_id", "media_id"):
            link_value = base.get(link_field)
            if link_value:
                reference_metadata[link_field] = link_value
```

(The top-level `metadata["posting_uuid"]` is already present via `metadata = dict(base)` at json.py:446, because the linker passes it in `extra_info`.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_json_posting_link.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Commit**

```bash
git add docint/utils/reference_metadata.py docint/core/readers/json.py tests/test_json_posting_link.py
git commit -m "feat(readers): carry posting_uuid onto transcript segments" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 9: Social linker — join core (maps, counter-strip match, recursive resolution)

**Files:**
- Create: `docint/core/ingest/social_linker.py` (pure join logic only in this task)
- Test: `tests/test_social_linker_join.py` (create)

**Interfaces:**
- Produces:
  - `strip_counter(media_id: str) -> str` — drops a trailing `_<digits>` segment.
  - `build_posting_index(postings_df) -> dict[str, str]` — `Posting ID → UUID`.
  - `resolve_media_rows(media_df, posting_uuids: dict[str, str], file_index: dict[str, list[Path]], tables_dir: Path) -> list[MediaLink]` where `MediaLink(posting_uuid, posting_id, media_id, path)`.
  - `build_file_index(root: Path) -> dict[str, list[Path]]` — basename → paths (recursive).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_social_linker_join.py
from pathlib import Path

import pandas as pd

from docint.core.ingest.social_linker import (
    build_file_index,
    build_posting_index,
    resolve_media_rows,
    strip_counter,
)


def test_strip_counter_drops_trailing_numeric_segment() -> None:
    assert strip_counter("2603434334845655437_44657421320_0") == "2603434334845655437_44657421320"
    assert strip_counter("2603434334845655437_44657421320_12") == "2603434334845655437_44657421320"


def test_resolve_matches_by_known_posting_id_and_finds_file_recursively(tmp_path: Path) -> None:
    (tmp_path / "media" / "sub").mkdir(parents=True)
    img = tmp_path / "media" / "sub" / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")

    postings = pd.DataFrame({"Posting ID": ["P_1"], "UUID": ["uuid-1"]})
    media = pd.DataFrame({"Media ID": ["P_1_0", "ORPHAN_0"], "Exported media filename": ["a.jpg", "missing.jpg"]})

    links = resolve_media_rows(
        media,
        build_posting_index(postings),
        build_file_index(tmp_path),
        tables_dir=tmp_path,
    )
    assert len(links) == 1
    assert links[0].posting_uuid == "uuid-1"
    assert links[0].media_id == "P_1_0"
    assert links[0].path == img
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_social_linker_join.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'docint.core.ingest.social_linker'`

- [ ] **Step 3: Implement the join core**

```python
# docint/core/ingest/social_linker.py
"""Join a social export's postings table to its media manifest + files.

Pure join logic lives here (counter stripping, set-membership matching,
recursive file resolution). Routing of resolved media into the modality
pipelines (CLIP / Nextext) lives in :class:`SocialLinker` (Task 10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

_COUNTER_SUFFIX = re.compile(r"_\d+$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@dataclass(frozen=True)
class MediaLink:
    """A media file resolved to its owning posting."""

    posting_uuid: str
    posting_id: str
    media_id: str
    path: Path


def strip_counter(media_id: str) -> str:
    """Return the ``Media ID`` with a single trailing ``_<digits>`` counter removed.

    Args:
        media_id (str): The media identifier, e.g. ``"<posting_id>_0"``.

    Returns:
        str: The candidate posting id (``media_id`` itself if no counter).
    """
    return _COUNTER_SUFFIX.sub("", str(media_id), count=1)


def build_posting_index(postings_df: pd.DataFrame) -> dict[str, str]:
    """Return ``{Posting ID: UUID}`` from a postings DataFrame.

    Args:
        postings_df (pd.DataFrame): Table carrying ``Posting ID`` + ``UUID``.

    Returns:
        dict[str, str]: Mapping from posting id to posting UUID.
    """
    index: dict[str, str] = {}
    for _, row in postings_df.iterrows():
        posting_id = str(row.get("Posting ID") or "").strip()
        uuid = str(row.get("UUID") or "").strip()
        if posting_id and uuid:
            index[posting_id] = uuid
    return index


def build_file_index(root: Path) -> dict[str, list[Path]]:
    """Index every file under ``root`` (recursively) by lowercase basename.

    Args:
        root (Path): The batch tree root.

    Returns:
        dict[str, list[Path]]: ``{basename_lower: [paths]}`` (sorted per key).
    """
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            index.setdefault(path.name.lower(), []).append(path)
    return index


def _resolve_path(
    exported_filename: str,
    file_index: dict[str, list[Path]],
    tables_dir: Path,
) -> Path | None:
    """Resolve an ``Exported media filename`` to a file in the batch tree.

    Prefers a path relative to the tables folder when the value carries one;
    otherwise matches by basename across the tree, logging on collision.

    Args:
        exported_filename (str): The manifest's filename (basename or rel path).
        file_index (dict[str, list[Path]]): Recursive basename index.
        tables_dir (Path): Directory holding the manifest (relative-path anchor).

    Returns:
        Path | None: The resolved file, or ``None`` when missing.
    """
    name = str(exported_filename or "").strip().replace("\\", "/")
    if not name:
        return None
    if "/" in name:
        candidate = (tables_dir / name).resolve()
        if candidate.is_file():
            return candidate
    matches = file_index.get(Path(name).name.lower(), [])
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning("Media filename {!r} matched {} files; using {}", name, len(matches), matches[0])
    return matches[0]


def resolve_media_rows(
    media_df: pd.DataFrame,
    posting_uuids: dict[str, str],
    file_index: dict[str, list[Path]],
    *,
    tables_dir: Path,
) -> list[MediaLink]:
    """Resolve manifest rows to ``MediaLink``s, skipping orphans and missing files.

    Args:
        media_df (pd.DataFrame): Manifest with ``Media ID`` + ``Exported media filename``.
        posting_uuids (dict[str, str]): ``Posting ID → UUID`` from the postings table.
        file_index (dict[str, list[Path]]): Recursive basename index.
        tables_dir (Path): Manifest directory (relative-path anchor).

    Returns:
        list[MediaLink]: One per row whose posting is known and file exists.
    """
    links: list[MediaLink] = []
    for _, row in media_df.iterrows():
        media_id = str(row.get("Media ID") or "").strip()
        if not media_id:
            continue
        posting_id = strip_counter(media_id)
        uuid = posting_uuids.get(posting_id)
        if uuid is None:
            logger.debug("Orphan media row {!r} (no posting {!r})", media_id, posting_id)
            continue
        path = _resolve_path(str(row.get("Exported media filename") or ""), file_index, tables_dir)
        if path is None:
            logger.warning("Media file for {!r} not found; skipping.", media_id)
            continue
        links.append(MediaLink(posting_uuid=uuid, posting_id=posting_id, media_id=media_id, path=path))
    return links


def is_image(path: Path) -> bool:
    """Return whether ``path`` has a still-image extension (vs. video/audio)."""
    return path.suffix.lower() in _IMAGE_EXTS
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_social_linker_join.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add docint/core/ingest/social_linker.py tests/test_social_linker_join.py
git commit -m "feat(linker): social join core (counter strip, recursive resolve)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 10: Social linker — routing + transcript ingestion + caching

**Files:**
- Modify: `docint/core/ingest/social_linker.py` (add `SocialLinker` class)
- Test: `tests/test_social_linker_routing.py` (create)

**Interfaces:**
- Consumes: `NextextClient` (Task 6), `ImageIngestionService` (`ingest_image` + `ingest_keyframe_set`, Task 7), `CustomJSONReader` (Task 8), join helpers (Task 9), `is_media_manifest` + `TableReader.schema_profiles` (Task 4).
- Produces: `SocialLinker(image_service, nextext_client, target_collection).run(data_dir: Path) -> SocialLinkResult(consumed_paths: set[Path], transcript_documents: list[Document])`.

- [ ] **Step 1: Write the failing test (fakes for both services)**

```python
# tests/test_social_linker_routing.py
from pathlib import Path
from typing import Any

import pandas as pd

from docint.core.ingest.images_service import IngestContext
from docint.core.ingest.social_linker import SocialLinker
from docint.utils.nextext_client import NextextResult


class _FakeImageService:
    def __init__(self) -> None:
        self.images: list[Any] = []
        self.keyframe_calls: list[dict[str, Any]] = []

    def ingest_image(self, asset: Any, *, context: IngestContext) -> Any:
        self.images.append(asset)
        return None

    def ingest_keyframe_set(self, frames, *, context, source_doc_id, extra_metadata=None, dedup_cosine=0.95):
        self.keyframe_calls.append({"frames": frames, "source_doc_id": source_doc_id})
        return []


class _FakeNextext:
    def process_media(self, file_path: Path) -> NextextResult:
        return NextextResult(
            status="completed",
            transcript_jsonl=b'{"text":"spoken","start_seconds":0,"end_seconds":1}\n',
            keyframes=[b"\xff\xd8\xff0"],
        )


def _write_export(root: Path) -> None:
    (root / "tables").mkdir(parents=True)
    (root / "media").mkdir()
    pd.DataFrame({"Posting ID": ["P_1", "P_2"], "UUID": ["u1", "u2"], "Text Content": ["a", "b"]}).to_csv(
        root / "tables" / "postings.csv", index=False
    )
    pd.DataFrame(
        {"Media ID": ["P_1_0", "P_2_0"], "Exported media filename": ["pic.jpg", "clip.mp4"]}
    ).to_csv(root / "tables" / "media.csv", index=False)
    (root / "media" / "pic.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "media" / "clip.mp4").write_bytes(b"video")


def test_run_routes_image_and_video_and_links(tmp_path: Path) -> None:
    _write_export(tmp_path)
    img = _FakeImageService()
    linker = SocialLinker(image_service=img, nextext_client=_FakeNextext(), target_collection="c")
    result = linker.run(tmp_path)

    # The image went through the CLIP path with the posting UUID.
    assert len(img.images) == 1
    assert img.images[0].source_doc_id == "u1"
    # The video produced keyframes (linked to u2) and a transcript Document.
    assert img.keyframe_calls and img.keyframe_calls[0]["source_doc_id"] == "u2"
    assert len(result.transcript_documents) == 1
    assert result.transcript_documents[0].metadata["posting_uuid"] == "u2"
    # media.csv + both media files are consumed (excluded from the generic sweep).
    consumed_names = {p.name for p in result.consumed_paths}
    assert {"media.csv", "pic.jpg", "clip.mp4"}.issubset(consumed_names)
    # postings.csv is NOT consumed (the sweep ingests it as text nodes).
    assert "postings.csv" not in consumed_names
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_social_linker_routing.py -v`
Expected: FAIL — `ImportError: cannot import name 'SocialLinker'`

- [ ] **Step 3: Implement `SocialLinker`**

Append to `docint/core/ingest/social_linker.py`:

```python
from dataclasses import field

import pandas as pd
from llama_index.core import Document

from docint.core.ingest.images_service import ImageAsset, IngestContext
from docint.core.readers.json import CustomJSONReader
from docint.core.readers.tables import TableReader, is_media_manifest

_POSTINGS_HEADERS = next(
    (p.normalized_headers for p in TableReader.schema_profiles if p.style == "postings"),
    frozenset(),
)


@dataclass
class SocialLinkResult:
    """Outcome of a social-linker pass over one batch tree."""

    consumed_paths: set[Path] = field(default_factory=set)
    transcript_documents: list[Document] = field(default_factory=list)


@dataclass
class SocialLinker:
    """Join + route a social export's media, linking each artifact to its posting."""

    image_service: Any
    nextext_client: Any
    target_collection: str | None

    def _find_tables(self, data_dir: Path) -> tuple[Path | None, Path | None]:
        """Locate the postings table and media manifest anywhere in the tree.

        Args:
            data_dir (Path): The batch tree root.

        Returns:
            tuple[Path | None, Path | None]: ``(postings_csv, media_csv)``.
        """
        postings: Path | None = None
        media: Path | None = None
        for path in sorted(data_dir.rglob("*.csv")):
            try:
                columns = pd.read_csv(path, nrows=0).columns
            except Exception:
                continue
            normalized = {str(c).strip().casefold() for c in columns}
            if media is None and is_media_manifest(columns):
                media = path
            elif postings is None and normalized == _POSTINGS_HEADERS:
                postings = path
        return postings, media

    def run(self, data_dir: Path) -> SocialLinkResult:
        """Run the linker over ``data_dir``; no-op when it is not a social export.

        Args:
            data_dir (Path): The batch tree root.

        Returns:
            SocialLinkResult: Consumed paths + transcript Documents for the pipeline.
        """
        result = SocialLinkResult()
        postings_csv, media_csv = self._find_tables(data_dir)
        if postings_csv is None or media_csv is None:
            return result

        posting_uuids = build_posting_index(pd.read_csv(postings_csv, dtype=str))
        media_df = pd.read_csv(media_csv, dtype=str)
        file_index = build_file_index(data_dir)
        links = resolve_media_rows(media_df, posting_uuids, file_index, tables_dir=media_csv.parent)

        result.consumed_paths.add(media_csv)
        context = IngestContext(source_collection=self.target_collection)
        for link in links:
            result.consumed_paths.add(link.path)
            extra = {"posting_id": link.posting_id, "media_id": link.media_id, "source_type": "social_media"}
            if is_image(link.path):
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
                self._route_media_clip(link, context, result)
        return result

    def _route_media_clip(self, link: MediaLink, context: IngestContext, result: SocialLinkResult) -> None:
        """Send one video/audio file to Nextext; ingest transcript + keyframes.

        Args:
            link (MediaLink): The resolved media file and its posting linkage.
            context (IngestContext): Collection-resolution context.
            result (SocialLinkResult): Accumulator for transcript Documents.
        """
        sidecar = link.path.with_suffix(link.path.suffix + ".nextext.jsonl")
        if sidecar.is_file():
            transcript = sidecar.read_bytes()  # cached: skip the (expensive) re-run
            keyframes: list[bytes] = []
        else:
            outcome = self.nextext_client.process_media(link.path)
            transcript = outcome.transcript_jsonl
            keyframes = outcome.keyframes
            if transcript:
                sidecar.write_bytes(transcript)
                result.consumed_paths.add(sidecar)
        extra = {"posting_id": link.posting_id, "media_id": link.media_id, "source_type": "social_media"}
        if keyframes:
            self.image_service.ingest_keyframe_set(
                keyframes,
                context=context,
                source_doc_id=link.posting_uuid,
                extra_metadata={**extra, "posting_uuid": link.posting_uuid},
            )
        if transcript:
            result.consumed_paths.add(sidecar)
            self._ingest_transcript(sidecar if sidecar.is_file() else link.path, transcript, link, result)

    def _ingest_transcript(
        self, sidecar: Path, transcript: bytes, link: MediaLink, result: SocialLinkResult
    ) -> None:
        """Parse transcript JSONL into segment Documents stamped with the posting link.

        Args:
            sidecar (Path): Path of the persisted transcript JSONL.
            transcript (bytes): The transcript NDJSON bytes.
            link (MediaLink): The posting linkage to stamp.
            result (SocialLinkResult): Accumulator for the produced Documents.
        """
        if not sidecar.is_file():
            sidecar.write_bytes(transcript)
            result.consumed_paths.add(sidecar)
        docs = CustomJSONReader(is_jsonl=True).iter_documents(
            sidecar,
            extra_info={
                "posting_uuid": link.posting_uuid,
                "posting_id": link.posting_id,
                "media_id": link.media_id,
            },
        )
        result.transcript_documents.extend(docs)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_social_linker_routing.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add docint/core/ingest/social_linker.py tests/test_social_linker_routing.py
git commit -m "feat(linker): route media + ingest transcripts/keyframes with caching" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 11: Wire the linker into the ingestion pipeline

**Files:**
- Modify: `docint/core/ingest/ingestion_pipeline.py` (run linker; skip consumed paths; yield transcript docs)
- Test: `tests/test_pipeline_social_linker.py` (create)

**Interfaces:**
- Consumes: `SocialLinker.run` (Task 10); `DocumentIngestionPipeline` fields `image_ingestion_service`, `target_collection`, `data_dir`, `dir_reader` (pipeline 143-199, 388-410).
- Produces: pipeline excludes `result.consumed_paths` from the sweep and appends `result.transcript_documents`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_social_linker.py
from pathlib import Path
from typing import Any

from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.ingest.social_linker import SocialLinkResult


def test_pipeline_skips_consumed_and_yields_transcripts(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "media.csv").write_text("Media ID,Exported media filename\nP_1_0,a.jpg\n", encoding="utf-8")
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "postings.csv").write_text("Posting ID,UUID,Text Content\nP_1,u1,hello\n", encoding="utf-8")

    from llama_index.core import Document

    fake_doc = Document(text="spoken", metadata={"posting_uuid": "u1", "docint_doc_kind": "transcript_segment"})

    def fake_run(self: Any, data_dir: Path) -> SocialLinkResult:
        return SocialLinkResult(
            consumed_paths={tmp_path / "media.csv", tmp_path / "a.jpg"},
            transcript_documents=[fake_doc],
        )

    monkeypatch.setattr("docint.core.ingest.social_linker.SocialLinker.run", fake_run)

    pipeline = DocumentIngestionPipeline(data_dir=tmp_path, ner_model=None, progress_callback=None, target_collection="c")
    pipeline._load_doc_readers()
    batches = list(pipeline._iter_loaded_documents())
    loaded = [doc for batch in batches for doc in batch]

    texts = {doc.text for doc in loaded}
    assert "spoken" in texts  # transcript doc injected
    # The consumed media.csv + a.jpg are not re-ingested by the generic sweep.
    filenames = {doc.metadata.get("filename") for doc in loaded}
    assert "a.jpg" not in filenames
    assert "media.csv" not in filenames
    # postings.csv still flows through the sweep.
    assert any("hello" in (doc.text or "") for doc in loaded)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_pipeline_social_linker.py -v`
Expected: FAIL — the sweep ingests `media.csv`/`a.jpg` and no `"spoken"` transcript doc is yielded.

- [ ] **Step 3: Run the linker in the pipeline**

In `docint/core/ingest/ingestion_pipeline.py`, add two `init=False` fields beside `dir_reader` (pipeline:184):

```python
    social_link_consumed: set[Path] = field(default_factory=set, init=False)
    social_link_documents: list[Document] = field(default_factory=list, init=False)
```

Add a method to run the linker (call it from `_load_doc_readers` end, after `self.dir_reader = SimpleDirectoryReader(...)` at pipeline:843):

```python
    def _run_social_linker(self) -> None:
        """Run the social linker; record consumed paths + transcript Documents.

        No-op (and fail-soft) unless the batch is a social export with both a
        postings table and a media manifest.
        """
        from docint.core.ingest.social_linker import SocialLinker
        from docint.utils.nextext_client import NextextClient

        try:
            result = SocialLinker(
                image_service=self.image_ingestion_service or ImageIngestionService(),
                nextext_client=NextextClient(),
                target_collection=self.target_collection,
            ).run(self.data_dir)
        except Exception as exc:  # pragma: no cover - fail-soft guard
            logger.warning("Social linker skipped due to error: {}", exc)
            return
        self.social_link_consumed = result.consumed_paths
        self.social_link_documents = result.transcript_documents
```

At the end of `_load_doc_readers`, after the `SimpleDirectoryReader(...)` assignment, add:

```python
        self._run_social_linker()
```

- [ ] **Step 4: Skip consumed files and yield transcript docs**

In `_iter_loaded_documents` (pipeline:392), filter the sweep and append the linker's docs. Change the loop header and add a trailing yield:

```python
        for input_file in dir_reader.input_files:
            if Path(input_file) in self.social_link_consumed:
                continue
            ext = input_file.suffix.lower()
```

…and immediately after the `for` loop (before the method returns), add:

```python
        if self.social_link_documents:
            yield self.social_link_documents
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_pipeline_social_linker.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Add the `posting_uuid` payload index on the image companion**

In `ImageIngestionService._ensure_collection` (images_service:678-685), after the `image_id` index creation, add a second index so link-following can filter `_images` by posting:

```python
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="posting_uuid",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as idx_exc:
                logger.debug("Payload index creation on posting_uuid skipped: {}", idx_exc)
```

- [ ] **Step 7: Run the full ingestion test module + pre-commit**

Run: `uv run pytest tests/test_pipeline_social_linker.py tests/test_social_linker_routing.py -v && uv run pre-commit run --all-files`
Expected: PASS; pre-commit clean.

- [ ] **Step 8: Commit**

```bash
git add docint/core/ingest/ingestion_pipeline.py docint/core/ingest/images_service.py tests/test_pipeline_social_linker.py
git commit -m "feat(pipeline): integrate social linker (skip consumed, inject transcripts)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Retrieval: post + media as one entity

### Task 12: Fetch a posting's sibling artifacts by `posting_uuid`

**Files:**
- Modify: `docint/core/rag.py` (add `_fetch_posting_entity_nodes`)
- Test: `tests/test_link_following_fetch.py` (create)

**Interfaces:**
- Produces: `RAG._fetch_posting_entity_nodes(posting_uuid: str, *, exclude_node_ids: set[str]) -> list[NodeWithScore]` — sibling transcript/text nodes (main collection) + image/keyframe caption nodes (`_images`), as `NodeWithScore` with score `None`, excluding ids already present.

- [ ] **Step 1: Write the failing test (inject a fake qdrant client)**

```python
# tests/test_link_following_fetch.py
from types import SimpleNamespace
from typing import Any

from docint.core.rag import RAG


class _FakeQdrant:
    def __init__(self, by_collection: dict[str, list[dict[str, Any]]]) -> None:
        self.by_collection = by_collection

    def scroll(self, collection_name, scroll_filter=None, limit=64, with_payload=True, with_vectors=False):
        points = [
            SimpleNamespace(id=p["id"], payload=p["payload"]) for p in self.by_collection.get(collection_name, [])
        ]
        return points, None


def test_fetch_posting_entity_collects_siblings(monkeypatch) -> None:
    rag = RAG.__new__(RAG)  # bypass heavy __init__
    rag.qdrant_collection = "c"
    rag.qdrant_client = _FakeQdrant(
        {
            "c": [{"id": "t1", "payload": {"posting_uuid": "u1", "text": "spoken words"}}],
            "c_images": [{"id": "i1", "payload": {"posting_uuid": "u1", "llm_description": "a red banner"}}],
        }
    )
    monkeypatch.setattr(rag, "_image_collection_name", lambda: "c_images")

    nodes = rag._fetch_posting_entity_nodes("u1", exclude_node_ids={"already"})
    texts = {n.node.text for n in nodes}
    assert "spoken words" in texts
    assert any("red banner" in t for t in texts)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_link_following_fetch.py -v`
Expected: FAIL — `AttributeError: 'RAG' object has no attribute '_fetch_posting_entity_nodes'`

- [ ] **Step 3: Implement the fetch helper**

Add to `RAG` in `docint/core/rag.py` (uses `qdrant_client.scroll` like `_existing_by_image_id`; builds `TextNode`/`NodeWithScore`):

```python
    def _image_collection_name(self) -> str:
        """Return the ``_images`` companion collection name for the active collection."""
        if self._image_ingestion_service is None:
            self._image_ingestion_service = ImageIngestionService()
        return self._image_ingestion_service._resolve_collection_name(self.qdrant_collection)

    def _fetch_posting_entity_nodes(self, posting_uuid: str, *, exclude_node_ids: set[str]) -> list[NodeWithScore]:
        """Fetch a posting's sibling artifacts across both collections.

        Gathers transcript/text nodes (main collection) and image/keyframe
        caption nodes (``_images``) whose payload carries ``posting_uuid``, so a
        post and its media surface together. Fail-soft: scroll errors yield
        whatever was collected so far.

        Args:
            posting_uuid (str): The posting UUID link key.
            exclude_node_ids (set[str]): Node ids already present in the result
                set (skip to avoid duplicates).

        Returns:
            list[NodeWithScore]: Sibling nodes with score ``None``.
        """
        if not posting_uuid:
            return []
        flt = models.Filter(
            must=[models.FieldCondition(key="posting_uuid", match=models.MatchValue(value=posting_uuid))]
        )
        collected: list[NodeWithScore] = []
        targets = [(self.qdrant_collection, "text"), (self._image_collection_name(), "image")]
        for collection_name, kind in targets:
            if not collection_name or not qdrant_collection_exists(self.qdrant_client, collection_name):
                continue
            try:
                points, _ = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=flt,
                    limit=64,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("Link-following scroll failed for {}: {}", collection_name, exc)
                continue
            for point in points:
                point_id = str(getattr(point, "id", ""))
                if not point_id or point_id in exclude_node_ids:
                    continue
                payload = dict(getattr(point, "payload", {}) or {})
                if kind == "image":
                    text = str(payload.get("llm_description") or "").strip()
                    tags = payload.get("llm_tags")
                    if isinstance(tags, list) and tags:
                        text = f"{text}\n\nTags: {', '.join(str(t) for t in tags)}".strip()
                else:
                    text = str(payload.get("text") or payload.get("_node_content") or "").strip()
                if not text:
                    continue
                exclude_node_ids.add(point_id)
                collected.append(NodeWithScore(node=TextNode(id_=point_id, text=text, metadata=payload), score=None))
        return collected
```

(If `TextNode`/`NodeWithScore` are not already imported in rag.py, add `from llama_index.core.schema import NodeWithScore, TextNode`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_link_following_fetch.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add docint/core/rag.py tests/test_link_following_fetch.py
git commit -m "feat(retrieval): fetch sibling posting artifacts by posting_uuid" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 13: `LinkFollowingPostprocessor`

**Files:**
- Modify: `docint/core/rag.py` (add the postprocessor class)
- Test: `tests/test_link_following_postprocessor.py` (create)

**Interfaces:**
- Consumes: `RAG._fetch_posting_entity_nodes` (Task 12).
- Produces: `LinkFollowingPostprocessor(rag: RAG, max_per_post: int = 12)` (a `BaseNodePostprocessor`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_link_following_postprocessor.py
from types import SimpleNamespace
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode

from docint.core.rag import LinkFollowingPostprocessor


def test_postprocessor_appends_siblings_once() -> None:
    sibling = NodeWithScore(node=TextNode(id_="s1", text="spoken", metadata={"posting_uuid": "u1"}), score=None)

    rag = SimpleNamespace(
        _fetch_posting_entity_nodes=lambda uuid, *, exclude_node_ids: [sibling] if uuid == "u1" else []
    )
    pp = LinkFollowingPostprocessor(rag=rag)  # type: ignore[arg-type]

    hit = NodeWithScore(
        node=TextNode(id_="p1", text="post text", metadata={"reference_metadata": {"uuid": "u1"}}), score=0.9
    )
    out = pp._postprocess_nodes([hit], None)
    ids = [n.node.node_id for n in out]
    assert "p1" in ids and "s1" in ids
    # Idempotent: a second sibling for an already-included id is not duplicated.
    out2 = pp._postprocess_nodes(out, None)
    assert out2.count(sibling) <= 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_link_following_postprocessor.py -v`
Expected: FAIL — `ImportError: cannot import name 'LinkFollowingPostprocessor'`

- [ ] **Step 3: Implement the postprocessor**

Add to `docint/core/rag.py` near `SocialSourceDiversityPostprocessor` (589):

```python
class LinkFollowingPostprocessor(BaseNodePostprocessor):
    """Expand each retrieved post to include its linked media (and vice versa).

    For every hit, resolves the posting UUID (``reference_metadata.uuid`` or a
    top-level ``posting_uuid``) and appends the post's sibling artifacts —
    transcript segments and image/keyframe captions — so the generator sees a
    post and its media as one evidence block. Bounded by ``max_per_post`` and
    deduplicated by node id; triggering is bidirectional (a media hit pulls in
    its post's siblings too).
    """

    rag: Any
    max_per_post: int = 12

    class Config:
        arbitrary_types_allowed = True

    @override
    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier."""
        return "LinkFollowingPostprocessor"

    @staticmethod
    def _posting_uuid(node: NodeWithScore) -> str:
        """Extract the posting UUID link key from a node's metadata."""
        metadata = getattr(node, "metadata", {}) or {}
        direct = str(metadata.get("posting_uuid") or "").strip()
        if direct:
            return direct
        reference_metadata = metadata.get("reference_metadata")
        if isinstance(reference_metadata, dict):
            return str(reference_metadata.get("posting_uuid") or reference_metadata.get("uuid") or "").strip()
        return ""

    @override
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Append linked sibling artifacts for each retrieved posting.

        Args:
            nodes (list[NodeWithScore]): Retrieved nodes.
            query_bundle (QueryBundle | None): Unused.

        Returns:
            list[NodeWithScore]: Original nodes plus bounded, deduped siblings.
        """
        _ = query_bundle
        present: set[str] = {n.node.node_id for n in nodes}
        additions: list[NodeWithScore] = []
        seen_posts: set[str] = set()
        for node in nodes:
            posting_uuid = self._posting_uuid(node)
            if not posting_uuid or posting_uuid in seen_posts:
                continue
            seen_posts.add(posting_uuid)
            try:
                siblings = self.rag._fetch_posting_entity_nodes(posting_uuid, exclude_node_ids=present)
            except Exception as exc:
                logger.warning("Link-following expansion failed for {}: {}", posting_uuid, exc)
                continue
            additions.extend(siblings[: self.max_per_post])
        return nodes + additions
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_link_following_postprocessor.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add docint/core/rag.py tests/test_link_following_postprocessor.py
git commit -m "feat(retrieval): add LinkFollowingPostprocessor" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 14: Wire `LinkFollowingPostprocessor` into the query engine + index the main collection

**Files:**
- Modify: `docint/core/rag.py` (`build_query_engine` social-table branch at 3960-3963; add `posting_uuid` index on the main collection at index-creation)
- Test: `tests/test_build_query_engine_linkfollow.py` (create)

**Interfaces:**
- Consumes: `LinkFollowingPostprocessor` (Task 13); `build_query_engine` (3912-3977).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_query_engine_linkfollow.py
from typing import Any

from docint.core.rag import LinkFollowingPostprocessor


def test_link_following_added_for_social_collections(monkeypatch) -> None:
    from docint.core import rag as rag_module

    captured: dict[str, Any] = {}

    class _FakeQE:
        @classmethod
        def from_args(cls, **kwargs: Any) -> "_FakeQE":
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(rag_module, "RetrieverQueryEngine", _FakeQE)

    rag = rag_module.RAG.__new__(rag_module.RAG)
    # Minimal seams build_query_engine relies on:
    monkeypatch.setattr(rag, "create_index", lambda: None)
    rag.index = object()
    monkeypatch.setattr(rag, "_infer_collection_profile", lambda: {"is_social_table": True})
    monkeypatch.setattr(rag, "_resolve_runtime_retrieval_settings", lambda **k: {"parent_context_enabled": False})
    monkeypatch.setattr(rag, "_resolve_chat_response_mode", lambda: "compact")
    monkeypatch.setattr(rag, "_build_retriever", lambda **k: object())
    monkeypatch.setattr(rag, "_build_grounded_text_qa_template", lambda **k: None)
    monkeypatch.setattr(rag, "_build_grounded_refine_template", lambda **k: None)
    rag.post_retrieval_text_model = None
    rag.social_summary_diversity_limit = 2

    rag.build_query_engine()
    kinds = [type(p).__name__ for p in captured["node_postprocessors"]]
    assert "LinkFollowingPostprocessor" in kinds
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_build_query_engine_linkfollow.py -v`
Expected: FAIL — `LinkFollowingPostprocessor` not in the postprocessor list.

- [ ] **Step 3: Append the postprocessor in the social-table branch**

In `build_query_engine`, inside the existing `if bool(profile.get("is_social_table")):` block (rag.py:3960-3963), after the `SocialSourceDiversityPostprocessor` append, add:

```python
            node_postprocessors.append(LinkFollowingPostprocessor(rag=self))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_build_query_engine_linkfollow.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Add the `posting_uuid` payload index to the main collection**

Find where the main collection's payload indexes are created (search: `grep -n "create_payload_index" docint/core/rag.py`). Alongside the existing index creation for the primary collection, add (idempotent, fail-soft):

```python
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.qdrant_collection,
                field_name="posting_uuid",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_exc:
            logger.debug("posting_uuid index on {} skipped: {}", self.qdrant_collection, idx_exc)
```

If no such index-creation site exists, create the index lazily inside `_fetch_posting_entity_nodes` on first use (wrap the existing scroll: on a "needs index" error, call `create_payload_index` then retry once). Choose the approach that matches the surrounding code; document which in the commit message.

- [ ] **Step 6: Run tests + pre-commit**

Run: `uv run pytest tests/test_build_query_engine_linkfollow.py tests/test_link_following_postprocessor.py tests/test_link_following_fetch.py -v && uv run pre-commit run --all-files`
Expected: PASS; pre-commit clean.

- [ ] **Step 7: Commit**

```bash
git add docint/core/rag.py tests/test_build_query_engine_linkfollow.py
git commit -m "feat(retrieval): wire link-following into social query engine" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 15: Group sources by posting in the response payload

**Files:**
- Modify: `docint/core/rag.py` (`_normalize_response_data` at 4122-4140 — attach a `posting_uuid` group key to each source)
- Test: `tests/test_source_grouping.py` (create)

**Interfaces:**
- Consumes: `_source_from_payload` adds `reference_metadata` (rag.py:3259-3261); `_retrieve_image_sources` (2448).
- Produces: each source dict carries a `posting_group` key (the posting UUID) when linkable, so the UI can render a post + its media as one entity.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_source_grouping.py
from docint.core.rag import _attach_posting_group


def test_attach_posting_group_from_reference_metadata() -> None:
    sources = [
        {"text": "post", "reference_metadata": {"uuid": "u1"}},
        {"text": "caption", "posting_uuid": "u1"},
        {"text": "unrelated"},
    ]
    grouped = _attach_posting_group(sources)
    assert grouped[0]["posting_group"] == "u1"
    assert grouped[1]["posting_group"] == "u1"
    assert "posting_group" not in grouped[2]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_source_grouping.py -v`
Expected: FAIL — `ImportError: cannot import name '_attach_posting_group'`

- [ ] **Step 3: Implement the grouping helper and call it**

Add a module-level helper in `docint/core/rag.py`:

```python
def _attach_posting_group(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag each source with its posting UUID group key when linkable.

    Reads the posting UUID from a top-level ``posting_uuid`` or from
    ``reference_metadata.uuid``/``reference_metadata.posting_uuid`` and writes
    it as ``posting_group`` so the UI can render a post and its media as one
    entity. Sources without a link are left untouched.

    Args:
        sources (list[dict[str, Any]]): Normalized source dicts.

    Returns:
        list[dict[str, Any]]: The same list, mutated with ``posting_group``.
    """
    for source in sources:
        group = str(source.get("posting_uuid") or "").strip()
        if not group:
            reference_metadata = source.get("reference_metadata")
            if isinstance(reference_metadata, dict):
                group = str(reference_metadata.get("posting_uuid") or reference_metadata.get("uuid") or "").strip()
        if group:
            source["posting_group"] = group
    return sources
```

In `_normalize_response_data`, immediately after the `sources.extend(self._retrieve_image_sources(...))` block (rag.py:4134-4140), add:

```python
        sources = _attach_posting_group(sources)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_source_grouping.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Full suite + pre-commit + README**

Update `README.md` with a short "Social multimodal media" subsection (the linker, the `NEXTEXT_*` / `KEYFRAMES_*` env vars, and the folder-upload requirement). Then:

Run: `uv run pytest && uv run pre-commit run --all-files && (cd frontend && pnpm test && pnpm build)`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add docint/core/rag.py tests/test_source_grouping.py README.md
git commit -m "feat(retrieval): group sources by posting; document social multimodal flow" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (each design section → task):
- Structure-preserving SPA folder upload → Tasks 1–3.
- Media-manifest detection (not ingested as content) → Task 4 + Task 10 (`consumed_paths` excludes `media.csv`).
- `Posting ID`↔`Media ID` join (counter strip + set membership), recursive resolution → Task 9.
- Routing: image→CLIP; video/audio→Nextext transcript + keyframes → Task 10 (+ Task 6 client, Task 7 keyframe dedup).
- Keyframe rate-sampling + cosine prune before captioning → Task 5 (knobs), Task 7 (prune), Nextext plan (sampling).
- `posting_uuid` linkage on every artifact → Task 7 (images/keyframes), Task 8 (transcripts), postings already carry `reference_metadata.uuid`.
- Caching (don't re-transcribe) → Task 10 (`.nextext.jsonl` sidecar). *Note:* this realizes the spec's "don't re-run Nextext" intent via a persisted-artifact sidecar rather than an `IngestManifest` column — simpler and reuses the filesystem; update the spec's Idempotency section to match.
- Fail-soft degradation → Tasks 6, 10, 11, 12, 13 (all media/scroll/job paths log-and-skip).
- Retrieval link-following (bidirectional, grouped) → Tasks 12–15.
- Payload indexes on `posting_uuid` (both collections) → Task 11 (images), Task 14 (main).

**Placeholder scan:** Task 14 Step 5 leaves a *documented* either/or (index at creation vs. lazy) keyed to "match the surrounding code" — resolve it during implementation by grepping `create_payload_index`; not a blank placeholder. No other TBD/TODO.

**Type consistency:** `MediaLink(posting_uuid, posting_id, media_id, path)` used identically in Tasks 9–10. `NextextResult(status, transcript_jsonl, keyframes, error)` consistent across Tasks 6 and 10. `_fetch_posting_entity_nodes(posting_uuid, *, exclude_node_ids)` signature matches between Task 12 (def) and Task 13 (call). `ingest_keyframe_set(frames, *, context, source_doc_id, extra_metadata, dedup_cosine)` matches between Task 7 (def) and Task 10 (call).

**Out of scope (separate plan):** the Nextext `keyframes.zip` artifact + `JobOptions` keyframe fields. Phase 2 degrades gracefully without it (the transcript still flows; `keyframes` is simply empty until Nextext ships the artifact).
