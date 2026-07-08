# Standalone media (audio/video) ingestion — design

- **Date:** 2026-07-08
- **Status:** Draft for review
- **Scope:** docint ingestion only. No Nextext-repo change (reuses the existing
  `/jobs` wire contract). No retrieval-layer change. No frontend/API/CLI change
  to *accept* files.

## Problem

Audio/video files can only be transcribed and ingested today when they are
referenced by a social-media **`media.csv`** manifest, which itself requires a
**`postings.csv`** table. The `SocialLinker` pass no-ops unless *both* tables are
found (`social_linker.py:297-298`), and the generic ingestion sweep drops A/V
files at the `required_exts`/`supported_filetypes` whitelist
(`env_cfg.py:783-802`, enforced inside `SimpleDirectoryReader`). So a user who
drops a folder of `.mp4`/`.mp3`/`.wav` files (no CSVs) gets a **silent
"no files found" soft-empty ingest** — the media is dropped without a trace, and
the user-facing docs even instruct people to build a social manifest to ingest
video (`docs/ingestion.md:30-40`, `README.md:301-354`).

The transcription/keyframe machinery is not social-specific — it is a generic
Nextext delegation. This feature exposes it as a **first-class standalone
ingestion path**: drop audio/video files anywhere in a batch and have docint
transcribe them (and, for video, sample keyframes) with no CSV scaffolding,
linking every artifact to the **media file itself**.

## Goals

- Ingest loose audio/video files (no `postings.csv`/`media.csv`) by delegating
  to the remote Nextext service, exactly as the social path does:
  **video → transcript + keyframes; audio → transcript.**
- Link each derived artifact to the **media file's own identity** (content hash +
  filename/path), so transcript segments and keyframes retrieve and cite as
  ordinary, independently-ranked sources that name the source clip.
- **Automatic when Nextext is configured** (`NEXTEXT_API_BASE` set) — the same
  gate the social path already uses. No new enable knob, no per-ingest toggle.
- **Coexist cleanly** with the social path: a social export is unchanged; a
  mixed batch (manifest + loose clips) links the manifest media *and* ingests the
  loose clips; a pure-A/V batch is fully handled by the new pass.
- Keep docint **CPU-only and media-dependency-free**: all decode/ASR/frame
  sampling stays in Nextext (no `ffmpeg`/`av`/`PyAV`).
- **Share one implementation** with the social path (Approach A): extract the
  per-file Nextext→keyframes+transcript engine so the two callers cannot drift.

## Non-goals (YAGNI / parked)

- **Media-entity grouping / co-retrieval.** Transcript segments and keyframes of
  the same clip flow as *independent* sources (the decided retrieval model). No
  `LinkFollowingPostprocessor` analog, no per-file grouping key at query time.
- **Per-frame dense video analysis** — reuse the existing rate-sampled,
  near-duplicate-pruned keyframe set (`KEYFRAMES_PER_MINUTE`, `KEYFRAMES_MAX`,
  `KEYFRAME_DEDUP_COSINE`).
- **Background-job / async ingest redesign.** The pass inherits the existing
  synchronous, request-thread posture and the file-hash manifest resume. No new
  job queue.
- **New SPA controls.** The frontend, `/ingest/upload`, and the CLI already
  accept and persist A/V unfiltered; only docs/help-text change.
- **In-process transcode/ASR/frame extraction inside docint** — delegated to
  Nextext.
- **A separate "video document" node** beyond the derived artifacts (transcript
  segments + keyframes carry the media identity; no empty placeholder node).

## Key decisions (locked during brainstorming)

1. **Retrieval = independent sources.** No new retrieval code; the downstream
   audit confirms transcript nodes + keyframe image points without `posting_uuid`
   retrieve, rerank, cite, and report correctly. The social postprocessors are
   gated behind a collection-level `is_social_table` flag that a standalone
   collection never trips (`rag.py:4288-4292`; transcripts are `source="transcript"`,
   not `"table"`), and keyframes surface via the **unconditional** CLIP image
   co-retrieval (`_retrieve_image_sources`, `rag.py:2757-2858`).
2. **Automatic trigger, gated on `NEXTEXT_API_BASE`.** Runs as a pipeline
   pre-pass beside the social linker. Nextext unset/unreachable → A/V skipped
   fail-soft with a one-line warning; batch still succeeds.
3. **Anchor artifacts to the media file's content hash** (`compute_file_hash`,
   `utils/hashing.py`) — the same key the transcript cache and image dedup
   already use, so re-ingestion is idempotent.
4. **Approach A (shared engine).** Lift the per-file routine out of
   `SocialLinker` into a reusable helper both passes call; the social path keeps
   byte-for-byte behavior by passing posting identity, the standalone path passes
   file identity.
5. **Social linker runs first**, standalone second, so manifest-claimed media is
   never double-ingested.

## Architecture overview

```
Batch tree under qdrant_src/{collection}/  (recursive)
 │
 ├─ _run_social_linker()      ── if postings.csv + media.csv present ──▶ claims manifest media,
 │  (ingestion_pipeline.py)      routes it (CLIP / Nextext), stamps posting_uuid,
 │                               fills pre-pass consumed-paths + injected-documents
 │
 └─ _run_standalone_media()   ── walks tree for A/V exts NOT already consumed ──▶
    (NEW, runs right after)       per clip → shared MediaTranscriber:
                                    ├─ hash → manifest transcript-cache lookup
                                    ├─ (miss) Nextext job  [bounded concurrency]
                                    ├─ keyframes (video) ─▶ ingest_keyframe_set(source_type="video_keyframe",
                                    │                        source_doc_id=media_hash) → {collection}_images
                                    └─ transcript ─▶ CustomJSONReader → segment Documents
                                                     (source_file=clip.mp4, media_file_hash=…, NO posting fields)
                                    marks each clip + transient .jsonl consumed;
                                    appends segment Documents to injected-documents

Generic sweep: skips every consumed path (ingestion_pipeline.py:394-396),
               yields injected transcript Documents as a final batch (:415-416)
               ─▶ normal chunk→embed→Qdrant path (main {collection})

Query ─▶ normal retrieval; transcript segments cite as text sources naming the clip,
         keyframes surface via CLIP image co-retrieval. No grouping. (unchanged rag.py)
```

## Components

### New

- **`docint/core/ingest/standalone_media.py`** — the standalone pass. One job:
  given a batch dir and the set of paths the social linker already consumed, walk
  the tree for A/V extensions, build a `MediaClip` per unclaimed file (identity =
  content hash + filename/path), and route them through the shared
  `MediaTranscriber`. Pure orchestration; no model inference, no media decoding.
  Returns the same `(consumed_paths, transcript_documents)` shape the pipeline
  already consumes. Testable in isolation with a stubbed transcriber.

### Refactor (Approach A — the shared engine)

- **Extract `MediaTranscriber`** (new small module, e.g.
  `docint/core/ingest/media_transcribe.py`) from `SocialLinker._route_media_clips`
  + `_ingest_transcript` (`social_linker.py:328-432`). Its responsibility: take a
  list of media jobs and, per job, hash → transcript-cache lookup → (miss) Nextext
  round-trip (bounded `ThreadPoolExecutor`) → ingest keyframes via
  `ingest_keyframe_set` → parse transcript via `CustomJSONReader` into segment
  Documents → write-through the transcript cache. Everything social-specific is
  lifted into the **job's identity payload**, not the engine:

  ```python
  @dataclass(frozen=True)
  class MediaClip:
      path: Path
      source_doc_id: str | None            # anchor id stamped on keyframes + transcript link
      keyframe_source_type: str            # "social_media" (social) | "video_keyframe" (standalone)
      keyframe_extra_metadata: dict[str, Any]   # merged onto each keyframe point
      transcript_extra_info: dict[str, Any]     # merged into each transcript segment
      display_source_file: str             # the media filename to cite (never the .nextext.jsonl transient)

  class MediaTranscriber:
      def __init__(self, image_service, nextext_client, target_collection,
                   manifest=None, keyframe_dedup_cosine=0.95, nextext_max_concurrency=4): ...
      def run(self, clips: list[MediaClip]) -> MediaTranscribeResult: ...
      # MediaTranscribeResult(consumed_paths: set[Path], transcript_documents: list[Document])
  ```

  `SocialLinker.run()` then builds `MediaClip`s from its `MediaLink`s (posting
  identity) and calls the transcriber; `StandaloneMediaIngestor` builds them from
  file identity. `SocialLinker`'s join logic (`_find_tables`, `resolve_media_rows`,
  `build_posting_index`, `_derive_posting_id`, `_warn_unreferenced_media`, the
  image-routing branch) stays put. **Net effect: the social path's observable
  behavior is unchanged; its per-file routing simply moves behind a shared call.**

### Modified

- **`docint/core/ingest/images_service.py` (`ingest_keyframe_set`,
  `:953-1063`)** — parameterize the two hard-coded payload fields
  (`source_type="social_media_keyframe"` at `:1026`, `posting_uuid=source_doc_id`
  at `:1029`) into arguments that **default to today's values**, so the social
  path is unchanged and the standalone path passes `source_type="video_keyframe"`
  (and no `posting_uuid` alias). Keep the greedy CLIP near-duplicate pruning
  intact — it is the reason to reuse this method rather than per-frame
  `ingest_image`.
- **`docint/core/ingest/ingestion_pipeline.py`** — after `_run_social_linker()`
  (`:851`), call a new `_run_standalone_media()` that constructs
  `StandaloneMediaIngestor`, passes the social linker's already-consumed set, and
  **merges** its results into the pipeline's pre-pass accumulators. Generalize the
  two existing hook fields (`social_link_consumed`, `social_link_documents`,
  consumed at `:394-396` / `:415-416`) into neutral names
  (`_prepass_consumed_paths`, `_prepass_documents`) that both passes feed — a
  light rename, no behavior change. Fail-soft: any exception in the pass →
  `logger.warning` + continue (mirrors `_run_social_linker`, `:883-885`).
- **`docint/utils/env_cfg.py`** — lift the A/V extension set (`_AV_EXTS`,
  currently `social_linker.py:21-38`) to a shared home so both passes and any
  future config reads share one source of truth (per the "all config lives in
  env_cfg" rule). No new enable env var — reuse `NextextConfig.enabled`
  (`NEXTEXT_API_BASE`). `KEYFRAMES_*` / `KEYFRAME_DEDUP_COSINE` /
  `NEXTEXT_MAX_CONCURRENCY` are reused as-is.
- **`docint/core/ingest/social_linker.py`** — import `_AV_EXTS` from its new home
  (drop the local copy); build `MediaClip`s and delegate to `MediaTranscriber`
  (its `_route_media_clips`/`_ingest_transcript` bodies move to the transciber).
  `_warn_unreferenced_media` (`:197-220`) is **downgraded**: loose A/V that the
  manifest didn't link is now ingested standalone rather than dropped, so the
  "were NOT ingested" warning is removed or converted to a debug note (the
  standalone pass owns those clips now).
- **Docs** — `docs/ingestion.md` (correct the "silently skipped" statement + list
  A/V as a supported standalone input), `README.md` (add a "Standalone media"
  subsection beside "Social Multimodal Media"), `docs/architecture.md`, and
  `CLAUDE.md` (new module bullet).

### Unchanged (verified, no work)

- Frontend upload (`Dropzone.tsx`, `api/ingest.ts`), `/ingest/upload`
  (`api.py:3148`), and the CLI (`cli/ingest.py`) already accept, byte-batch,
  persist, and stream progress for A/V. No functional change.
- `NextextClient.process_media` (`nextext_client.py:119`),
  `CustomJSONReader.iter_documents` (`json.py:658`), and
  `IngestManifest.cache/get_nextext_transcript` (`ingest_manifest.py:290/321`) are
  already generic. Reused as-is.
- All retrieval/citation/reporting (`rag.py`, `report_render.py`) — the 14-point
  coupling audit found only `ingest_keyframe_set` needed a change; everything else
  no-ops gracefully without `posting_uuid`.

## Discovery & coexistence ordering

1. `_run_social_linker()` runs first (unchanged). If a social export is present it
   claims its manifest media into `_prepass_consumed_paths` and stamps
   `posting_uuid`.
2. `_run_standalone_media()` runs second. It walks `data_dir` recursively for
   files whose suffix ∈ A/V-exts and that are **not** already in
   `_prepass_consumed_paths` (this is what prevents double-ingest of
   manifest-linked clips) and are not transient `.nextext.jsonl` files (excluded
   by extension anyway). Each surviving file becomes a `MediaClip` with file
   identity.
3. Both passes' consumed paths and transcript Documents are merged; the generic
   sweep skips all consumed paths and yields all injected Documents.

Because the standalone pass now ingests loose A/V, the social linker's
"present but not linked → dropped" warning is no longer accurate and is removed.

## Linking / identity model

Each standalone clip is anchored by `media_hash = compute_file_hash(path)`:

- **Transcript segment nodes** — parsed by `CustomJSONReader` with
  `extra_info = {source_file: <clip filename>, source_path: <clip path>,
  media_file_hash: <hash>}` and **no** posting fields. They must cite the
  **original media file** (e.g. `clip.mp4`), never the transient
  `clip.mp4.nextext.jsonl` the reader parses from — `display_source_file` on the
  `MediaClip` carries the media filename so the segment's `source_file` /
  citation is the clip, not the sidecar. `docint_doc_kind="transcript_segment"`,
  `source="transcript"`, timing/speaker metadata all flow as today.
- **Keyframe image points** — stored in `{collection}_images` via
  `ingest_keyframe_set(source_type="video_keyframe", source_doc_id=media_hash,
  extra_metadata={media_file_hash, source_file, source_path})`. Previewable and
  CLIP-co-retrievable exactly like social keyframes; frame-bytes hash dedup still
  applies underneath.

There is no cross-modal grouping key: a transcript hit and a keyframe hit for the
same clip are two independent sources that both name the clip.

## Retrieval — why no change is needed

Confirmed by the coupling audit: `is_social_table` stays off for a standalone
collection, so `SocialSourceDiversityPostprocessor` and `LinkFollowingPostprocessor`
are never instantiated (`rag.py:4288-4292`); even if they were, both no-op per
node without `posting_uuid`. `_source_from_payload` already special-cases
`docint_doc_kind == "transcript_segment"` for the citation row locator
(`rag.py:3531-3536`), and image co-retrieval is unconditional. So transcript
segments cite as normal text sources and keyframes as normal image sources.

## Error handling & degradation (inherited fail-soft posture)

- Nextext unreachable / job failed / timed out → skip **that clip**, log one
  warning, keep the rest of the batch (`NextextClient` already returns a
  non-`completed` status with empty payloads; the engine logs and moves on).
- `NEXTEXT_API_BASE` unset → the standalone pass is a no-op with a single
  info/warning line (mirrors the social path's "video/audio skipped gracefully").
- A corrupt/unreadable media file → skip + log; never abort the batch.
- Empty transcript (silent video) → keyframes only; not an error.
- Any unexpected exception in the pass → caught at the pipeline call site
  (`logger.warning`), batch continues.

## Idempotency / caching

- Transcript reuse: `IngestManifest.get/cache_nextext_transcript(collection,
  media_hash)` — re-ingesting an unchanged clip skips the Nextext job entirely
  (keyframe points already persist in `{collection}_images` from the first run).
- Keyframe points dedupe by frame-bytes hash (existing `image_id` dedup) beneath
  the CLIP near-duplicate prune.
- The pass is re-runnable; the file-hash ledger lets a re-run skip
  already-completed clips (important given the synchronous, disconnect-fragile
  ingest — a re-upload resumes rather than re-transcribes).

## Testing strategy

- **Standalone discovery:** A/V files found recursively; non-A/V ignored; files
  already in the social-consumed set are skipped (no double-ingest);
  `.nextext.jsonl` transients excluded.
- **Coexistence ordering:** a batch with a social manifest + extra loose clips
  links the manifest media (posting_uuid) *and* standalone-ingests the loose clips
  (media_hash); a pure-A/V batch is fully handled; assert no path is consumed
  twice.
- **`MediaTranscriber` (shared engine), mocked Nextext client:** video →
  transcript + keyframes; audio → transcript only; cache hit skips the Nextext
  call; write-through populates the cache; concurrency bound respected; a raised
  `process_media` degrades that clip, not the batch.
- **`ingest_keyframe_set` parameterization:** default args reproduce the social
  payload byte-for-byte (regression guard); `source_type="video_keyframe"` +
  no `posting_uuid` for standalone; frame-dedup unchanged (caption/store only
  survivors).
- **Standalone linkage:** transcript segments carry `media_file_hash` +
  `source_file == <clip filename>` (NOT the `.nextext.jsonl` transient) and **no**
  `posting_uuid`; keyframes carry `source_type="video_keyframe"` + `media_hash`.
- **Downstream smoke:** a standalone transcript node and a standalone keyframe
  normalize to citable sources via `_source_from_payload` / `_retrieve_image_sources`
  (no social linkage required).
- **Fail-soft:** Nextext disabled → no-op; unreachable/timeout/failed job →
  clip skipped, batch green; corrupt file skipped.
- **Social regression:** the existing `test_social_linker_*` /
  `test_pipeline_social_linker` suites stay green after the refactor.
- `conftest.py` mock-stub conventions; every functional change ships with tests
  (project rule); frontend untouched (no Vitest change expected beyond docs).

## Open assumptions / risks

1. **Synchronous ingest + slow transcription.** Nextext jobs run on the ingest
   request thread; a large video dump is a long, disconnect-fragile call. Accepted
   for v1 (same as the social path); the file-hash manifest makes re-runs resume.
   Documented, not re-architected.
2. **Large single-file uploads** are bounded by `DOCINT_CLIENT_MAX_BODY_SIZE`; the
   SPA already byte-batches uploads (commit `cebf7a7`). No change; note in docs.
3. **Cost/volume.** Automatic transcription of every loose clip is intended
   (decided), but a large accidental drop-in transcribes everything. Mitigations:
   fail-soft, manifest resume, and `NEXTEXT_MAX_CONCURRENCY`. A future opt-in
   guard is out of scope.
4. **Citation & preview parity (verify in implementation).** Two details are
   asserted as "parity with the social path" and must be confirmed in the plan,
   not assumed: (a) whether `CustomJSONReader` lets `extra_info` override the
   path-derived `source_file` so transcript segments cite `clip.mp4` and not the
   transient `clip.mp4.nextext.jsonl` (if it does not, the sidecar is named/handled
   so citations still show the clip); and (b) the keyframe `source_doc_id` value
   (`media_hash` vs. `None`) that makes a standalone keyframe's `preview_url`
   resolve correctly in `_retrieve_image_sources` (`rag.py:2827`), where the URL
   key is `source_doc_id or file_hash`. Both are concrete requirements resolved in
   the plan.
5. Reusing the Nextext `/jobs` contract as-is; no Nextext-repo change needed.
