import { streamUpload, UploadHttpError } from './upload'
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

export function buildIngestFormData(collection: string, files: File[]): FormData {
  const fd = new FormData()
  fd.append('collection', collection)
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
 * Upload a file selection to `/ingest/upload` as one or more size-bounded
 * batches and yield a *single* normalised ingest event stream.
 *
 * Why batch: nginx caps each request body at `client_max_body_size`, so one
 * multipart POST carrying every file is rejected with 413 once the selection
 * exceeds the ceiling (the original crash). Splitting into batches that each
 * stay under the ceiling makes the total upload size unbounded by that
 * per-request cap. The backend ingestion is idempotent by file hash, so
 * uploading batches sequentially to the same collection never double-ingests.
 *
 * The per-batch SSE streams are normalised so downstream consumers
 * (`deriveIngestStatus`, the Ingest view) still see one logical ingest:
 * - exactly one synthetic `start` up front listing every file;
 * - every batch's `upload_progress` / `file_saved` / `ingestion_*progress` /
 *   `warning` events, forwarded verbatim (progress accumulates across batches);
 * - per-batch `start` and `ingestion_complete` are swallowed;
 * - exactly one synthetic terminal: `ingestion_complete` if any batch
 *   committed, or `error` if every batch failed.
 *
 * Failures are non-fatal: a batch that errors (e.g. a lone oversize file → 413,
 * or a transient network drop) is recorded and reported as a `warning`, and the
 * remaining batches still upload. The terminal `ingestion_complete` carries
 * `failed_files` / `failed_message` so the UI can flag partial failures without
 * discarding the batches that did succeed.
 *
 * @param collection - Target logical collection name.
 * @param files - The full selection to upload, in user order.
 * @param limitBytes - The server's per-request upload ceiling in bytes (from
 *   `/config` `max_upload_bytes`); the packing budget is this times
 *   `UPLOAD_SAFETY_MARGIN`.
 * @param signal - Optional abort signal cancelling the in-flight batch.
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

  const failures: BatchFailure[] = []
  let anySucceeded = false
  let anyContent = false

  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i]
    const batchNames = batch.map(fileLabel)
    try {
      let batchEmpty = false
      for await (const ev of streamUpload('/ingest/upload', buildIngestFormData(collection, batch), signal)) {
        const data = (ev.data ?? {}) as Record<string, unknown>
        if (ev.event === 'start') continue // already emitted one synthetic start
        if (ev.event === 'ingestion_complete') {
          if (data.empty === true) batchEmpty = true
          continue // swallow; one synthetic terminal is emitted after all batches
        }
        yield stamp(ev.event as IngestEvent['event'], data)
      }
      anySucceeded = true
      if (!batchEmpty) anyContent = true
    } catch (err) {
      const status = err instanceof UploadHttpError ? err.status : undefined
      const failure: BatchFailure = { batch: i + 1, total: batches.length, files: batchNames, status }
      failures.push(failure)
      // Surface inline (shows in the event log) but keep going — one bad batch
      // must not sink the rest of a large upload.
      yield stamp('warning', { message: describeBatchFailure(failure, limitBytes) })
    }
  }

  if (!anySucceeded) {
    const anyTooLarge = failures.some((f) => f.status === 413)
    const message = anyTooLarge
      ? `Upload failed: every batch exceeded the ${formatBytes(limitBytes)} per-upload limit. ` +
        `Raise DOCINT_CLIENT_MAX_BODY_SIZE and restart, or upload smaller files.`
      : `Upload failed: none of the ${failures.length} batch(es) were accepted. ` +
        `Check that the backend is running and reachable, then retry.`
    yield stamp('error', { message })
    return
  }

  const failedFiles = failures.flatMap((f) => f.files)
  yield stamp('ingestion_complete', {
    collection,
    empty: !anyContent,
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
