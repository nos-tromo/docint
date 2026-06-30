# Social multimodal media linking — design

- **Date:** 2026-06-30
- **Status:** Draft for review
- **Scope:** docint ingestion + retrieval; one prerequisite change in the sibling Nextext repo.

## Problem

Social ingestion today extracts only *text*. The `TableReader` "postings" schema
profile (`docint/core/readers/tables.py`) turns each posting row into one text
`Document` (`doc_id = Posting ID`, `reference_metadata.uuid = UUID`), but the
multimedia attached to a post — images, memes, stories, voice/video clips — is
ignored. For modern social platforms that is most of the meaning, and for
investigative work "just the visuals is a dead end" and just the text doubly so.

We have the media. It lives as files named by an **`Exported media filename`**
column in a **separate media table** whose **`Media ID`** is the posting's
`Posting ID` plus a counter suffix:

```
Posting ID :  2603434334845655437_44657421320
Media ID   :  2603434334845655437_44657421320_0
             2603434334845655437_44657421320_1   ...
```

We need to (1) join media rows to their posting, (2) ingest each media item
through the right modality path, (3) link every derived artifact to the
posting's **UUID**, and (4) at query time treat a post and its media as **one
entity**.

## Goals

- Join postings ↔ media-table ↔ media-files and ingest the media.
- **Audio and video visuals are both a required baseline** for video. Audio
  (transcript) is the deep channel; video also yields a capped visual sample.
- Link every artifact to the posting UUID; surface post + media as one entity
  in retrieval and citations.
- **Works through the React SPA upload as the primary path.** The server is
  deployed where network users cannot copy files to it (no CLI/mount access), so
  the browser upload must ingest the export's subdir tree directly — no manual
  zipping, no local tooling.
- Keep docint **CPU-only and dependency-free** for media: all heavy media work
  (transcode, ASR, frame extraction) stays in the remote services, consistent
  with the existing CLIP/GLiNER/rerank/Whisper delegation.

## Non-goals (parked / future)

- **Media-digest booster** (appending a short media summary to the post node's
  text for extra recall) — deliberately parked; full fidelity comes from
  link-following.
- **Per-frame dense video analysis** — only a rate-sampled, near-duplicate-pruned
  keyframe set.
- **Voiceless-clip handling beyond keyframes** — a silent video contributes its
  keyframes only (no transcript; that is correct, not a gap).
- **In-process ASR/transcode/frame extraction inside docint** — explicitly out;
  delegated to Nextext.
- **Manual client-side zipping before upload** — rejected: slow for large exports
  and archiving software is not guaranteed on locked-down client machines. The
  SPA uploads the folder directly, preserving relative paths.

## Key decisions (locked during brainstorming)

1. **docint builds glue, not inference.** Nextext already owns media→transcript
   (VAD → diarize → Whisper → PyAV transcode → `docint.jsonl`, see
   `Nextext/nextext/core/docint_transcript.py`), and docint's `json.py` already
   ingests that JSONL shape. We add orchestration + linking, not transcription.
2. **Live Nextext integration.** Per matched video/audio file, docint calls
   Nextext's job API and ingests the returned artifacts. (Not pre-computed
   sidecars.)
3. **Video = transcript + keyframes.** Both are baseline. Audio has depth
   priority in v1, but visuals are never dropped.
4. **Keyframes extracted by Nextext** (not docint) — a new `keyframes` artifact,
   **rate-sampled at `KEYFRAMES_PER_MINUTE`, capped at `KEYFRAMES_MAX`** candidates
   per video. Reuses the video decode Nextext already does for audio; keeps docint
   dependency-free.
5. **Near-duplicate keyframes are pruned in docint**, reusing the CLIP embedding
   docint already computes for storage (no CLIP in Nextext), via a greedy cosine
   filter (`KEYFRAME_DEDUP_COSINE`) that runs **before** the vision-LLM caption so
   only survivors are captioned. Nextext samples pixels; docint makes the
   embedding/semantic decisions.
6. **Retrieval = link-following co-retrieval**, keyed on `posting_uuid`. One
   vector per artifact (modality-specific recall); group into one posting entity
   at assembly time.

## Architecture overview

```
Batch tree under qdrant_src/{collection}/ (React SPA folder upload preserves subdirs → searched RECURSIVELY)
 ├─ tables/postings.csv ──TableReader "postings" profile──▶ text nodes (main collection)
 │                                                    doc_id=Posting ID, reference_metadata.uuid=UUID
 ├─ tables/media.csv    ──fuzzy-detected MEDIA MANIFEST──▶  (NOT ingested as content; join metadata)
 └─ media/**/ (packed in subdirs) ──linker resolves recursively, routes each matched row──▶
        image/meme/photo-story ─▶ ImageIngestionService (CLIP vector + vision caption)
        │                          ─▶ {collection}_images point   [posting_uuid stamped]
        video/audio ─▶ NextextClient job (per file):
                         ├─ docint.jsonl  ─▶ json.py transcript segment nodes (main)  [posting_uuid stamped]
                         └─ keyframes(N/min, capped) ─▶ embed+prune dups ─▶ caption survivors ─▶ {collection}_images pts [posting_uuid stamped]

Query ─▶ normal retrieval (per-modality vectors) ─▶ LinkFollowingPostprocessor (group by posting_uuid)
      ─▶ one "posting entity" context block + one grouped source (media nested)
```

## Components

### New

- **`docint/core/ingest/social_linker.py`** — the orchestration unit. One job:
  given a batch dir, detect the postings + media tables, build the join maps,
  match each media row to its posting, resolve its file **recursively** across the
  batch tree, route it to the modality path, and stamp `posting_uuid`. Pure orchestration; no model
  inference, no media decoding. Testable in isolation with a stubbed
  `NextextClient` and image service.
- **`docint/utils/nextext_client.py`** — thin httpx client mirroring
  `ner_client.py` / `clip_client.py`. Methods: submit a media file
  (`POST {NEXTEXT_API_BASE}/jobs`, multipart `file`), poll
  (`GET /jobs/{id}` until `COMPLETED`/failed), and fetch the per-job
  `docint.jsonl` and `keyframes` artifacts. Bounded polling with timeout;
  fail-soft (returns an empty result + logs on error).

### Modified (docint)

- **`ingestion_pipeline.py`** — run the social-linker enrichment pass when a
  batch contains a postings table; exclude the media manifest from the generic
  directory sweep; route video/audio files (currently filtered by
  `SimpleDirectoryReader`) through the linker instead of dropping them.
- **`readers/tables.py`** — add a small **media-manifest predicate** (a column
  set containing `Media ID` + `Exported media filename`) so the pipeline can
  recognize and skip it. (Detection helper only; the linker owns join logic.)
- **`ingest/images_service.py`** — already supports `source_doc_id` +
  `extra_metadata` + `occurrences`. Ensure the linker passes
  `source_doc_id = posting_uuid` and `extra_metadata = {posting_id, media_id,
  counter, source_type: "social_media"}`. Add a **payload index on
  `posting_uuid`** when creating `{collection}_images`. New
  **`ingest_keyframe_set(frames, …)`**: CLIP-embed each candidate once → greedy
  cosine prune (`KEYFRAME_DEDUP_COSINE`) → caption + store survivors only (so the
  vision LLM never runs on dropped near-dups).
- **`readers/json.py`** — transcript ingestion exists; thread an `extra_info`
  passthrough so each segment node carries `posting_uuid` (+ `posting_id`,
  `media_id`).
- **`core/rag.py`** — new **`LinkFollowingPostprocessor`** keyed on
  `posting_uuid`; payload index on `posting_uuid` in the main collection;
  integrate grouping into query assembly and `_source_from_node_with_score`
  source shaping; add `posting_uuid`/`posting_id`/`media_id` to the
  reference-metadata allowlists.
- **`utils/env_cfg.py`** — new `NextextConfig` dataclass (`NEXTEXT_API_BASE`,
  `NEXTEXT_API_KEY`, `NEXTEXT_TIMEOUT`, poll interval/budget, enable flag) plus
  keyframe knobs: `KEYFRAMES_PER_MINUTE` (default 4), `KEYFRAMES_MAX` (default
  20), `KEYFRAME_DEDUP_COSINE` (default 0.95). The two sampling knobs are
  forwarded to Nextext as job options; the cosine threshold is docint-side. All
  `os.getenv` access stays here per project rule.
- **`storage/ingest_manifest.py`** — extend to cache Nextext job results keyed
  by media-file hash (see Idempotency).
- **`core/api.py` (`/ingest/upload`)** — preserve uploaded subdir structure:
  write each file to its **sanitized relative path** (taken from
  `UploadFile.filename`) under the batch dir instead of flattening to `.name`;
  create parent subdirs; guard against path traversal (`..`, absolute paths,
  any escape outside the batch dir). This is what makes recursive resolution
  meaningful via the SPA.
- **`frontend/` upload control** — allow selecting or drag-dropping a **folder**
  (`webkitdirectory` + drag-drop entry walking), sending each file with its
  `webkitRelativePath` as the multipart filename; per-file progress (the endpoint
  already streams). Large folders upload as chunked multipart batches. No
  client-side zipping.
- **`README.md`** — document the social-media multimodal flow and new env vars.

### Prerequisite (Nextext repo — separate, independently useful)

- **Keyframe extraction** rate-sampled at `KEYFRAMES_PER_MINUTE` and capped at
  `KEYFRAMES_MAX` (both honored as per-job options forwarded by docint), reusing
  the `av.open` decode already in `Nextext/nextext/core/audio.py` (or a new
  `core/keyframes.py`). No embedding or dedup in Nextext — docint prunes.
- **`keyframes` artifact** wired into `nextext/api/artifacts.py` +
  `routes/jobs.py` (alongside `docint.jsonl`), as a zip or per-frame images.
- Optional `JobOptions` toggle / cap. Tracked as its own small PR; the docint
  work depends on it.

## The join algorithm (precise)

The linker first locates the postings table and media manifest **anywhere in the
batch tree** (recursive; identified by schema/columns, not by path or filename),
and indexes every file in the tree once for resolution.

1. Build `posting_ids: set[str]` and `posting_id → uuid` from the postings
   table (`Posting ID`, `UUID` columns).
2. Build `media_id → exported_filename` from the media manifest.
3. For each `media_id`, strip a **trailing `_<digits>`** counter and check the
   remainder against `posting_ids` (match against the *known set*, not a blind
   prefix — Posting IDs themselves contain `_`, so set-membership disambiguates;
   a media id whose stripped remainder is not a known Posting ID is an orphan →
   log + skip).
4. Resolve `exported_filename` against the **recursive** file index (the default
   export packs media into subdirs beneath the tables folder). Prefer resolving
   it as a path **relative to the tables folder** when it carries one; otherwise
   match by basename across the tree. On a basename collision across subdirs, log
   and pick deterministically (first by sorted path) rather than guess silently.
   Then route by type:
   - image extensions (`.jpg/.jpeg/.png/.webp/.gif`) → CLIP path.
   - video/audio extensions → Nextext job.
   - missing file → log + skip.

## Linking model

Every derived artifact carries `posting_uuid` (the join key) plus `posting_id`,
`media_id`, `counter`, and `source_type: "social_media"`:

- **Image / keyframe points** → in the `{collection}_images` payload
  (`source_doc_id = posting_uuid` + `extra_metadata`). Hash dedup means a media
  image also swept standalone just gets a posting `occurrence` appended.
- **Transcript segment nodes** → `posting_uuid` in node metadata via the
  `json.py` `extra_info` passthrough.
- **Posting text node** → already carries `reference_metadata.uuid`.

## Keyframe sampling & near-duplicate pruning

Per video, Nextext emits up to `KEYFRAMES_MAX` candidate frames sampled at
`KEYFRAMES_PER_MINUTE` (e.g. 4/min → one every 15s), the rate forwarded as a job
option so docint stays in control. docint then, per video:

1. CLIP-embeds each candidate once (the embedding it would store anyway).
2. **Greedy prune:** keep the first frame; drop any later frame whose max cosine
   similarity to an already-kept frame ≥ `KEYFRAME_DEDUP_COSINE`.
3. Caption (vision LLM) + store **only survivors** in `{collection}_images`,
   each linked to `posting_uuid`.

Captioning — the expensive call — runs only after the prune, so a duplicate frame
costs one cheap CLIP embed and nothing more. The existing exact-byte `image_id`
hash dedup still applies underneath this semantic prune.

## Retrieval: link-following co-retrieval

Keep one vector per artifact so each modality retrieves on its own strength
(spoken words → transcript, visual content → keyframe, written caption → post).
A new `LinkFollowingPostprocessor`:

1. For every retrieved hit (text node, transcript segment, **or** image source
   from `_retrieve_image_sources`), resolve `posting_uuid`.
2. Gather the whole posting entity across both collections — post text + all
   keyframe/image captions + all transcript segments — within a context budget.
3. Dedupe to one grouped unit per `posting_uuid`; emit one coherent evidence
   block to the LLM and one grouped source (media nested) to the UI.

Triggering is **bidirectional**: a hit in any modality pulls in the rest.
Reuses `SocialSourceDiversityPostprocessor` (grouping/dedup of social rows),
parent-context expansion, and the image-source shaping in
`_retrieve_image_sources`.

## Error handling & degradation

Consistent with docint's existing fail-soft posture (rerank, image tagging):

- Nextext unreachable / job failed / timed out → skip *that media item*, log,
  keep the posting and other media. Never fail the batch.
- Media file referenced but missing → log + skip.
- Media row with no matching posting → log + skip (orphan).
- Empty transcript (voiceless clip) → keyframes only; not an error.
- Nextext disabled (`NEXTEXT_API_BASE` unset) → video/audio rows skipped with a
  one-line warning; images still flow through CLIP.

## Idempotency / caching

- Image points already dedupe by content hash.
- Nextext jobs are expensive; cache results by **media-file hash** in
  `IngestManifest` (already keyed by `collection, file_hash`) so re-ingestion of
  an unchanged batch does not re-submit transcription/keyframe jobs.
- The linker is re-runnable; re-stamping `posting_uuid` is a no-op set_payload.

## Testing strategy

- **Join unit tests:** counter stripping, set-membership disambiguation,
  multi-underscore Posting IDs, orphan media, posting-with-no-media.
- **Recursive resolution:** media found in nested subdirs; relative-path vs
  bare-basename resolution; basename collision across subdirs logged + resolved
  deterministically.
- **Structure-preserving upload:** relative paths recreate subdirs under the
  batch dir; path-traversal inputs (`..`, absolute) rejected; frontend folder
  selection sends `webkitRelativePath` (Vitest).
- **Media-manifest detection:** positive/negative column sets; ensure it is not
  ingested as content.
- **Routing:** image → CLIP path; video/audio → Nextext (mocked client);
  missing file; unknown extension.
- **Keyframe sampling & prune:** rate → candidate count; `KEYFRAMES_MAX` ceiling;
  greedy cosine prune keeps/drops at the threshold boundary; caption backend
  invoked for survivors only (assert not called on pruned frames).
- **Nextext client (mocked httpx):** submit → poll → fetch; failure/timeout
  degradation; empty-transcript path.
- **Linking:** `posting_uuid` stamped on image points and transcript nodes.
- **LinkFollowingPostprocessor:** grouping by `posting_uuid`, dedupe,
  bidirectional triggering, budget enforcement.
- **End-to-end fixture:** `postings.csv` + `media.csv` + a small image +
  a stubbed Nextext response → one grouped posting entity at query time.
- `conftest.py` mock-stub conventions for external deps; every functional change
  ships with tests (project rule).

## Open assumptions to confirm in review

1. The React SPA is the **primary ingestion path** (the server forbids file
   copy / CLI access). The upload endpoint is extended to **preserve the export's
   subdir tree** (relative paths, traversal-sanitized) so the recursive linker and
   the generic sweep (`reader_recursive=True`) see the real structure. Manual
   client-side zipping is explicitly rejected (UX + no guaranteed local tooling).
   The CLI / `data_dir` path remains available and also preserves structure.
2. The media manifest is reliably identified by the presence of
   `Media ID` + `Exported media filename` (fuzzy), not an exact header set.
3. The counter is always a **trailing `_<digits>`** segment of `Media ID`.
4. Nextext per-job artifacts (`docint.jsonl`, `keyframes`) are retrievable by the
   docint caller (owner scoping satisfied by the deployment).
