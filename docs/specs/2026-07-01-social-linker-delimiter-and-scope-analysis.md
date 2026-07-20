# Social multimodal ingestion — "images from the larger dataset appear" — analysis & fix

**Date:** 2026-07-01
**Reporter symptom:** ingesting a reduced multimodal test set (`<test-batch-dir>`, a local test-batch directory: `..._postings.csv`, `<prefix>_media.csv`, 5 images + 3 videos), but images belonging to the *full* dataset (`<full-dataset-dir>`, an unrelated filesystem location outside the ingestion source) appear in Qdrant. Question raised: *"Does the current algo search my filesystem?"*

## TL;DR

1. **Your current test collection is actually clean.** `videoingest-test-1_images` holds exactly the **5 physical test images**, ingested as plain `standalone` images (no posting linkage). No full-dataset images are in it, and **no collection anywhere** contains social-media images or any path under `<full-dataset-dir>`.
2. **The social multimodal linker never ran on your data.** Your CSVs are **semicolon-delimited with a UTF-8 BOM** (typical German export). `social_linker` reads them with the default *comma* delimiter, which collapses the whole header into one column, so it fails to recognise the postings/media tables and **silently no-ops**. Your 5 images were picked up only by the generic image sweep; the videos, keyframes, transcripts, and posting links were all skipped.
3. **The real danger is latent, and your instinct is right.** `<prefix>_media.csv` is the **full manifest — 73,969 rows** (you copied in only 8 files). The linker resolves each row's `Exported media filename` and, in one branch, **does not confine the result to the batch** — an absolute or `../` path would resolve to a file *anywhere on disk*. Combined with a full manifest, any scan root that reaches the full dataset would pull in the whole corpus. That is almost certainly what you saw in an earlier run.

## Evidence

| Check | Finding |
|---|---|
| CSV format | `media.csv` & `postings.csv` are `;`-delimited **with a BOM** (`﻿`). |
| `is_media_manifest` (comma read, what the linker does) | **False** — header seen as 1 column. |
| `is_media_manifest` (`;` read) | **True** — 18 columns. |
| `TableReader._detect_delimiter` | Auto-sniffs `;` (so the generic reader parses these fine — the inconsistency). |
| `media.csv` row count | **73,969** (full dataset manifest, not reduced). |
| Physical media in test set | 8 files in `.../media/` (5 `.jpg`, 3 `.mp4`); all 8 basenames present in `media.csv`'s `Exported media filename`. |
| `videoingest-test-1_images` (Qdrant) | **5 points**, `source_type=standalone`, `source_path=/var/lib/docint/sources/videoingest-test-1/.../media/*.jpg`, `source_doc_id=null` (no linkage). |
| Any social-media / full-dataset images in Qdrant? | **None.** `<dataset>` `showcase`/`test-long-table` collections have no `_images` companion; `voc-37*_images` are PDF-page images. |

## Root cause 1 — semicolon exports silently no-op the linker

`docint/core/ingest/social_linker.py` reads CSVs with the default comma delimiter in three places:

- `_find_tables`: `pd.read_csv(path, nrows=0).columns` → header becomes one giant column → `is_media_manifest(...)` and the postings-header match both fail.
- `run()`: `pd.read_csv(postings_csv, ...)` and `pd.read_csv(media_csv, ...)` — same problem.

Because `_find_tables` returns `(None, None)`, `run()` returns early and the entire media join is skipped **with no error** (it's a fail-soft feature). `TableReader` already sniffs the delimiter (`_detect_delimiter`), so the generic pipeline parses the same files correctly — the two disagree.

**Net effect for you:** the multimodal linking you're testing isn't happening at all on this data. The 5 images you see are the generic image reader picking up the physical `.jpg`s; videos/keyframes/transcripts/posting-links are absent.

## Root cause 2 — media resolution isn't confined to the batch (latent, security-relevant)

`_resolve_path` resolves `Exported media filename` two ways, and the path branch has **no containment check**:

```python
if "/" in name:
    candidate = (tables_dir / name).resolve()   # tables_dir = folder holding media.csv
    if candidate.is_file():
        return candidate                         # ← no "is this inside the batch?" check
matches = file_index.get(Path(name).name.lower(), [])   # basename index, scoped to the batch root
```

`(tables_dir / name)` follows an **absolute** path straight out of the batch (`Path("/batch") / "/full/x.jpg" == "/full/x.jpg"`), and `.resolve()` follows `../`. So a manifest whose filenames are absolute/parent paths into `<full-dataset-dir>` would ingest those files directly — an arbitrary-file-read / cross-dataset leak. (Your current `media.csv` uses bare basenames, so this specific branch isn't firing today — but with the full 73,969-row manifest it is exactly the mechanism that pulls in the larger dataset once resolution reaches it.)

The basename branch is scoped to `build_file_index(data_dir)`, so it is safe **as long as `data_dir` is only the batch**. If ingestion is pointed at (or the batch lives inside) a directory that also contains the full dataset, even bare basenames match the full corpus.

## Hypothesis for the images you saw

Given the current collection is clean and the linker no-ops on `;`, the full-dataset images you observed were **not** produced by the current, upload-scoped run. Most likely one of:

- An **earlier run** with the ingestion source (`DATA_PATH` / the ingested folder) pointing at — or containing — `<full-dataset-dir>`, so either the generic image sweep or a then-working linker matched the 73,969-row manifest against the full corpus; or
- A **non-fresh collection** carried over from prior full-dataset work.

Either way it reduces to the same code weakness: **a full manifest + basename/path resolution that isn't hard-confined to the batch.** The fix closes that; the operational note below closes the rest.

## The fix (`docint/core/ingest/social_linker.py`)

- **A. Sniff the delimiter + read BOM-safe.** Add `_sniff_delimiter()` (mirrors `TableReader`: `csv.Sniffer` over a sample, frequency-count fallback, default `,`), and read all three sites with `sep=_sniff_delimiter(path), encoding="utf-8-sig"`. The linker now recognises `;`/tab/pipe exports.
- **B. Confine resolution to the batch tree.** Thread the batch root into `_resolve_path`/`resolve_media_rows` and **refuse** any path-branch candidate that is not `is_relative_to(root)` (logs a warning, falls through to the batch-scoped basename index). The linker can now only ingest files that physically live inside the batch — regardless of what the manifest claims.

New tests: `;`+BOM export is detected and linked; a full manifest links only the files actually present (skips the rest); and an absolute/`../` manifest reference to a file *outside* the batch is refused.

## What to check / do on your side

1. **Use a fresh collection** for the test (or delete `videoingest-test-1` + `videoingest-test-1_images` first) so nothing carries over.
2. **Scope the ingestion source to the test batch only.** Prefer the SPA folder-upload of `<test-batch-dir>`, or ensure `DATA_PATH` points *only* at it — never at a parent of `<full-dataset-dir>`.
3. Re-ingest after the fix and confirm: the linker now links the 8 present media to their postings (images via CLIP, videos via Nextext keyframes+transcript), and the other 73,961 manifest rows are skipped as "file not found" rather than pulled from disk.
4. (Optional) If you want the reduced test to be self-contained, trim `media.csv` to just the rows for the copied files — but with fix B this is no longer required for safety.

## Fix status

Implemented in commit **`d68c2ff`** — `fix(social): sniff CSV delimiter (;/BOM exports) and confine media resolution to the batch tree` (on `feat/social-multimodal-linking`, **not pushed** — awaiting review).

- `_sniff_delimiter()` added; the three `pd.read_csv` sites in `_find_tables`/`run()` now use `sep=_sniff_delimiter(path), encoding="utf-8-sig"`.
- `_resolve_path` takes a `root`; the path branch returns a candidate only when `candidate.is_relative_to(root.resolve())`, else logs `"…resolves outside the batch tree; refusing."` and falls through to the batch-scoped basename index. `run()` passes `root=data_dir`.

## Verification

- 3 new tests in `tests/test_social_linker_routing.py` (staged TDD): `;`+BOM export is detected & linked; a full manifest links only the media physically present; an absolute **and** a `../` reference to a file *outside* the batch are both refused.
- `tests/test_social_linker_routing.py` + `tests/test_social_linker_join.py`: 9 passed.
- Full suite: **961 passed** (958 baseline + 3), one pre-existing unrelated deprecation warning.
- `ruff check .`, `ruff format --check .`, `pyrefly check` (project-wide), and `pre-commit run --all-files`: all clean.

---

## Addendum — "robust to large `media.csv` drop-ins with only a few referenced files"

Ran the fixed linker against the real batch (73,969-row manifest, 8 files present). Three findings.

### 1. Log-volume robustness — FIXED (commit `41b7b27`)
`resolve_media_rows` logged one line per skipped row → **73,969 log lines** on the real data (~4.2s, all "orphan" here). Now aggregated into a single `INFO` summary: *"Social linker: N media linked, M skipped (… no matching posting, … no local file) across R rows."* New test `test_resolve_media_rows_aggregates_skips_for_large_manifest` asserts one summary, not per-row. Combined with the batch-containment guard (fix B) and basename index (scoped to the batch), the large-drop-in shape is now handled: only files physically in the batch can ever be ingested, and skips don't flood the logs.

### 2. The join key is wrong for this export → **0 links** on your data
The linker derives the posting id as `strip_counter(Media ID)` (strip a trailing `_<digits>`) and matches it against `Posting ID`. In this export:
- `Posting ID` = `<postingId>_<accountId>` (e.g. `3847537156143369215_77503905789`).
- `Media ID` = `[<albumId>_]<postingId>_<accountId>` — so `strip_counter` strips the **account** (the wrong end) and never matches.
- Measured: raw `Media ID` equals a `Posting ID` in **138** rows, and **`Network ID` equals a `Posting ID` in 138 rows** — i.e. the true media→posting key here is `Network ID` (or the raw `Media ID`), **not** `strip_counter(Media ID)`. Over the full manifest, `strip_counter` yields **0** links.
- **Recommendation (needs confirmation — this is a join-semantics/export-format decision):** key media→posting on `Network ID` (fallback: raw `Media ID`), or make the key column part of a per-export profile. Deliberately **not** changed unilaterally: it alters the documented join contract and could affect other export formats.

### 3. The reduced subsets are inconsistent → the 8 copied files can't link regardless
The 8 files in `media/` belong to postings whose ids (`3748824911910182974_77503905789`, …) are **not** among the 138 rows in `postings.csv`. Even with the correct join key, they have no parent posting in the batch. **Data-side fix:** reduce `postings.csv` and `media.csv` to the *same* postings — keep the postings that own the copied media (or copy media belonging to the kept postings).

### Net
- **Robustness to large drop-ins: now covered** — aggregated logging, batch-confined resolution, only-present-files linked.
- **Useful linking on your data still needs:** (2) the correct join key confirmed, and (3) consistent postings/media subsets.

---

## Update — resolution simplified to one flat directory (commit `268fe50`)

Per request, media resolution is now **flat and single-directory**: `postings.csv`, `media.csv`, and every media file live in one directory, and each media row is resolved by *basename within the manifest's own directory*.

- **Removed** the recursive `build_file_index` (`rglob`) and the relative/absolute-path branch of `_resolve_path` — and with them the batch-containment guard (fix B), now unnecessary: only a basename is ever used, looked up only in that one directory, so resolution provably cannot leave it. A new test asserts a basename that exists only in a *different* directory is not ingested.
- **Kept** the delimiter/BOM sniffing (fix A) and the aggregated skip summary (large drop-ins stay quiet — a full manifest with a handful of present files links only those and logs one line).
- Net −116 / +70 lines; full suite **963 passing**.

This does not change the two open items above: the media→posting **join key** (2) and **consistent subsets** (3) still determine whether anything actually links on your data.

---

## Update — `Network ID` join key implemented (commit `45c7868`)

Open item (2) is resolved. `_derive_posting_id` now keys media→posting on **`Network ID`** first, then the raw **`Media ID`**, then `strip_counter(Media ID)`, using the first that names a known posting (backward compatible with the `<Posting ID>_<counter>` format). On the real 73,969-row manifest this derives a posting for **138 rows** (was **0** with `strip_counter` alone).

**Only (3) — data consistency — remains for your test to link.** Measured: **0 of your 8 present files** have a parent posting in the 138-row `postings.csv`. To get links, make the subsets consistent — copy the media files that belong to the 138 kept postings, or keep the postings that own the 8 files you copied.
