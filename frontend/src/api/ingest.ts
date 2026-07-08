import { streamUpload, UploadHttpError } from './upload'
import { streamSse } from './sse'
import { planUploadBatches } from '@/lib/uploadBatches'
import { formatBytes } from '@/lib/ingestStatus'
import type { IngestEvent } from './types'

/**
 * Fraction of the server's per-request upload ceiling the client packs each
 * batch up to. The headroom absorbs multipart framing overhead (part boundaries
 * and per-file headers) so a batch sized right at the byte budget never tips
 * over the real nginx `client_max_body_size` and gets 413'd.
 */
export const UPLOAD_SAFETY_MARGIN = 0.9

export function buildIngestFormData(
  collection: string,
  files: File[],
  deferIngest = false
): FormData {
  const fd = new FormData()
  fd.append('collection', collection)
  // Staged-only upload: save the files but defer ingestion to a single
  // /ingest/finalize pass (see streamIngestUploadBatched). Omitted when false so
  // the single-request default keeps its existing save+ingest behaviour.
  if (deferIngest) fd.append('defer_ingest', 'true')
  for (const f of files) fd.append('files', f, f.webkitRelativePath || f.name)
  return fd
}

/** A batch that failed to upload, retained to summarise partial failures. */
export interface BatchFailure {
  batch: number
  total: number
  files: string[]
  /** HTTP status if the server responded (413 = too large); undefined = transport error. */
  status?: number
}

/**
 * Build a human-readable, actionable message for a single failed batch.
 *
 * A 413 after client-side batching means one individual file is larger than the
 * hard server limit (batches are packed under it), so the guidance is to raise
 * the limit or drop that file — not to retry blindly.
 *
 * @param f - The failed batch descriptor.
 * @param limitBytes - The server's per-request ceiling, for the 413 message.
 * @returns A sentence describing the failure and how to recover.
 */
export function describeBatchFailure(f: BatchFailure, limitBytes: number): string {
  const label = f.files.length === 1 ? `"${f.files[0]}"` : `${f.files.length} files`
  if (f.status === 413) {
    return (
      `Batch ${f.batch}/${f.total} (${label}) is larger than the ` +
      `${formatBytes(limitBytes)} per-upload limit and was skipped. Raise ` +
      `DOCINT_CLIENT_MAX_BODY_SIZE and restart, or remove the oversized file, ` +
      `then re-ingest — already-ingested files are skipped automatically.`
    )
  }
  const suffix = f.status ? ` (HTTP ${f.status})` : ' (network error)'
  return `Batch ${f.batch}/${f.total} (${label}) failed${suffix} and was skipped; other batches continued.`
}

const fileLabel = (f: File): string => f.webkitRelativePath || f.name

/**
 * Upload a file selection in size-bounded batches, then run a single ingestion
 * pass — yielding one normalised ingest event stream.
 *
 * Why batch: nginx caps each request body at `client_max_body_size`, so one
 * multipart POST carrying every file is rejected with 413 once the selection
 * exceeds the ceiling (the original crash). Splitting into batches that each
 * stay under the ceiling makes the total upload size unbounded by that cap.
 *
 * Why decouple upload from ingestion: the batches are uploaded **staged-only**
 * (`defer_ingest`), then one `/ingest/finalize` call ingests the whole staged
 * directory at once. Ingesting per batch instead would (a) re-init the models
 * for every batch, and (b) hard-fail any batch that happened to hold only
 * reader-unsupported files — e.g. a batch of just audio/video, which the backend
 * `required_exts` whitelist filters out, yields "No files found". A single
 * finalize pass sees the whole selection, so the images/docs are always found.
 * Being fileless, finalize can't 413, so ingestion still runs even if some
 * upload batch failed.
 *
 * The streams are normalised so downstream consumers (`deriveIngestStatus`, the
 * Ingest view) still see one logical ingest:
 * - one synthetic `start` up front listing every file;
 * - each batch's `upload_progress` / `file_saved` forwarded (progress
 *   accumulates); per-batch `start` and `upload_complete` swallowed;
 * - finalize's `ingestion_started` / `ingestion_progress` / `warning` forwarded;
 * - one synthetic terminal: `ingestion_complete`, or `error` if every upload
 *   batch failed or finalize itself failed.
 *
 * Upload failures are non-fatal: a batch that errors (a lone oversize file →
 * 413, or a transient drop) is reported as a `warning` and the rest still
 * upload; the terminal `ingestion_complete` carries `failed_files` /
 * `failed_message` so the UI can flag partial failures.
 *
 * @param collection - Target logical collection name.
 * @param files - The full selection to upload, in user order.
 * @param limitBytes - The server's per-request upload ceiling in bytes (from
 *   `/config` `max_upload_bytes`); the packing budget is this times
 *   `UPLOAD_SAFETY_MARGIN`.
 * @param signal - Optional abort signal cancelling the in-flight request.
 * @yields Normalised `IngestEvent`s, each stamped with `receivedAt`.
 */
export async function* streamIngestUploadBatched(
  collection: string,
  files: File[],
  limitBytes: number,
  signal?: AbortSignal
): AsyncGenerator<IngestEvent, void, unknown> {
  const budgetBytes = Math.max(1, Math.floor(limitBytes * UPLOAD_SAFETY_MARGIN))
  const batches = planUploadBatches(files, budgetBytes)

  // Stamp each event with its arrival time so `deriveIngestStatus` can compute
  // the elapsed timer purely from `receivedAt` (see IngestEvent.receivedAt).
  const stamp = (event: IngestEvent['event'], data: Record<string, unknown>): IngestEvent => ({
    event,
    data,
    receivedAt: Date.now()
  })

  // One synthetic `start` for the whole run so the reducer's `totalFiles`
  // covers every batch instead of resetting to the last batch's file count.
  yield stamp('start', { collection, files: files.map(fileLabel) })

  // Stage 1 — upload every batch staged-only (no ingestion yet).
  const failures: BatchFailure[] = []
  let anySaved = false

  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i]
    const batchNames = batch.map(fileLabel)
    try {
      for await (const ev of streamUpload(
        '/ingest/upload',
        buildIngestFormData(collection, batch, true),
        signal
      )) {
        const data = (ev.data ?? {}) as Record<string, unknown>
        // Swallow the per-batch `start` (one synthetic start already emitted)
        // and `upload_complete` (staged-only terminal); forward save progress.
        if (ev.event === 'start' || ev.event === 'upload_complete') continue
        yield stamp(ev.event as IngestEvent['event'], data)
      }
      anySaved = true
    } catch (err) {
      const status = err instanceof UploadHttpError ? err.status : undefined
      const failure: BatchFailure = { batch: i + 1, total: batches.length, files: batchNames, status }
      failures.push(failure)
      // Surface inline (shows in the event log) but keep going — one bad batch
      // must not sink the rest of a large upload.
      yield stamp('warning', { message: describeBatchFailure(failure, limitBytes) })
    }
  }

  if (!anySaved) {
    const anyTooLarge = failures.some((f) => f.status === 413)
    const message = anyTooLarge
      ? `Upload failed: every batch exceeded the ${formatBytes(limitBytes)} per-upload limit. ` +
        `Raise DOCINT_CLIENT_MAX_BODY_SIZE and restart, or upload smaller files.`
      : `Upload failed: none of the ${failures.length} batch(es) were accepted. ` +
        `Check that the backend is running and reachable, then retry.`
    yield stamp('error', { message })
    return
  }

  // Stage 2 — one ingestion pass over everything staged above.
  const failedFiles = failures.flatMap((f) => f.files)
  let empty = false
  let finalizeError: string | null = null
  try {
    for await (const ev of streamSse('/ingest/finalize', { collection, hybrid: true }, signal)) {
      const data = (ev.data ?? {}) as Record<string, unknown>
      if (ev.event === 'ingestion_complete') {
        if (data.empty === true) empty = true
        continue // fold into the single synthetic terminal below
      }
      if (ev.event === 'error') {
        finalizeError = typeof data.message === 'string' ? data.message : 'Ingestion failed'
        continue
      }
      yield stamp(ev.event as IngestEvent['event'], data)
    }
  } catch (err) {
    finalizeError = err instanceof Error ? err.message : String(err)
  }

  if (finalizeError) {
    yield stamp('error', { message: finalizeError })
    return
  }

  yield stamp('ingestion_complete', {
    collection,
    empty,
    ...(failedFiles.length > 0
      ? {
          failed_files: failedFiles,
          failed_message:
            `Completed, but ${failedFiles.length} file(s) were skipped: ` +
            `${failedFiles.join(', ')}. See the warnings above for details.`
        }
      : {})
  })
}

export const sourcePreviewUrl = (collection: string, file_hash: string) =>
  `/sources/preview?collection=${encodeURIComponent(collection)}&file_hash=${encodeURIComponent(file_hash)}`
