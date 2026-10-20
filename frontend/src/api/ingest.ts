import { streamUpload, UploadHttpError, UploadStalledError } from './upload'
import { streamErrorText } from './errorMessage'
import { url, withOwner } from './client'
import { MAX_BATCH_BYTES, planUploadBatches } from '@/lib/uploadBatches'
import { formatBytes } from '@/lib/ingestStatus'
import type { IngestEvent } from './types'
import type { Strings } from '@/i18n'
import { defaultT } from '@/i18n/defaultT'

type Translate = (key: keyof Strings, vars?: Record<string, string | number>) => string

/**
 * Fraction of the server's per-request upload ceiling the client packs each
 * batch up to. The headroom absorbs multipart framing overhead (part boundaries
 * and per-file headers) so a batch sized right at the byte budget never tips
 * over the real nginx `client_max_body_size` and gets 413'd.
 */
export const UPLOAD_SAFETY_MARGIN = 0.9

/**
 * Pause before each extra attempt at a batch whose request stalled, dropped
 * or met a gateway error — two retries, spread far enough apart to outlast a
 * backend restart. Only the batch's files not yet saved are sent again.
 */
export const UPLOAD_RETRY_DELAYS_MS = [5_000, 15_000]

/** Gateway statuses a retry can cure; any other HTTP status is the request's own fault. */
const RETRYABLE_STATUSES = new Set([502, 503, 504])

/**
 * Build the multipart body for a staged upload batch.
 *
 * `/ingest/upload` stages files only — it no longer runs the pipeline, so it
 * has no use for enrichment flags. Those travel solely on the
 * `createIngestJob` (`/ingest/finalize`) call the run controller makes once
 * every batch is staged (see `stores/ingestRun.ts`).
 */
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
  /** The request stopped making progress and was aborted. */
  stalled?: boolean
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
 * @param t - Translate function; defaults to the English catalog for pure/test callers.
 * @returns A sentence describing the failure and how to recover.
 */
export function describeBatchFailure(
  f: BatchFailure,
  limitBytes: number,
  t: Translate = defaultT
): string {
  const label =
    f.files.length === 1 ? `"${f.files[0]}"` : t('ingest.batch_files_count', { count: f.files.length })
  if (f.status === 413) {
    return t('ingest.batch_too_large', {
      batch: f.batch,
      total: f.total,
      label,
      limit: formatBytes(limitBytes)
    })
  }
  if (f.status) {
    return t('ingest.batch_failed_http', { batch: f.batch, total: f.total, label, status: f.status })
  }
  if (f.stalled) {
    return t('ingest.batch_stalled', { batch: f.batch, total: f.total, label })
  }
  return t('ingest.batch_failed_network', { batch: f.batch, total: f.total, label })
}

const fileLabel = (f: File): string => f.webkitRelativePath || f.name

/** Drop one pending file per save ack, so same-named files are not all cleared by one. */
function withoutSaved(pending: File[], saved: string[]): File[] {
  const left = [...pending]
  for (const name of saved) {
    const i = left.findIndex((f) => fileLabel(f) === name)
    if (i !== -1) left.splice(i, 1)
  }
  return left
}

/**
 * Upload a file selection in size-bounded batches, staging every batch on the
 * server — yielding one normalised upload event stream.
 *
 * Why batch: nginx caps each request body at `client_max_body_size`, so one
 * multipart POST carrying every file is rejected with 413 once the selection
 * exceeds the ceiling (the original crash). Splitting into batches that each
 * stay under the ceiling makes the total upload size unbounded by that cap.
 *
 * This function owns only the upload leg. Ingestion itself is queued
 * separately as a server-owned job (`createIngestJob` in `api/jobs.ts`,
 * called by `stores/ingestRun.ts` once this generator returns) rather than
 * run inline here — a browser reload or navigation no longer severs the
 * run's only view, because the job survives independently of this call.
 * This generator therefore no longer manufactures a terminal
 * `ingestion_complete`/`error` event for ingestion; its return value reports
 * only the upload outcome.
 *
 * The stream is normalised so downstream consumers (`deriveIngestStatus`, the
 * ingest run store) see one logical upload:
 * - one synthetic `start` up front listing every file;
 * - each batch's `upload_progress` / `file_saved` forwarded (progress
 *   accumulates); per-batch `start` and `upload_complete` swallowed;
 * - a terminal `error` only when every batch failed to upload.
 *
 * A batch is done only when the server says so (`upload_complete`). One that
 * stalls, drops, meets a gateway error or ends early is sent again — only its
 * files not yet saved — after each pause in `UPLOAD_RETRY_DELAYS_MS`, with a
 * `warning` per retry, retracted by a `warning_resolved` once the batch goes
 * through. Upload failures are non-fatal: a batch still failing
 * after that, or failing in a way a retry cannot cure (a lone oversize file
 * → 413, a server-side save error), is reported and the rest still upload;
 * the returned `failures` list names exactly the files that never made it to
 * the server.
 *
 * @param collection - Target logical collection name.
 * @param files - The full selection to upload, in user order.
 * @param limitBytes - The server's per-request upload ceiling in bytes (from
 *   `/config` `max_upload_bytes`); the packing budget is this times
 *   `UPLOAD_SAFETY_MARGIN`.
 * @param signal - Optional abort signal cancelling the in-flight request.
 * @param t - Translate function; defaults to the English catalog for pure/test callers.
 * @param owner - Owner every attempt is sent under, fixed when the run starts
 *   (see `withOwner`); omitted, each request uses the one selected at the time.
 * @yields Normalised `IngestEvent`s, each stamped with `receivedAt`.
 * @returns Whether any batch saved, and the list of batches that failed.
 */
export async function* streamIngestUploadBatched(
  collection: string,
  files: File[],
  limitBytes: number,
  signal?: AbortSignal,
  t: Translate = defaultT,
  owner?: string | null
): AsyncGenerator<IngestEvent, { anySaved: boolean; failures: BatchFailure[] }, unknown> {
  const budgetBytes = Math.max(1, Math.min(MAX_BATCH_BYTES, Math.floor(limitBytes * UPLOAD_SAFETY_MARGIN)))
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
    let pending = batches[i]
    // This batch's retry notices: retracted if it gets through, kept if it fails.
    const notices: string[] = []
    for (let attempt = 0; ; attempt++) {
      const saved: string[] = []
      let complete = false
      let serverError = false
      let caught: unknown
      try {
        for await (const ev of streamUpload(
          '/ingest/upload',
          buildIngestFormData(collection, pending),
          signal,
          owner
        )) {
          const data = (ev.data ?? {}) as Record<string, unknown>
          // Swallow the per-batch `start` (one synthetic start already emitted)
          // and `upload_complete` (staged-only terminal); forward save progress.
          // Once `upload_complete` is in, the batch is done whatever then
          // happens to the rest of the stream.
          if (ev.event === 'upload_complete') complete = true
          if (ev.event === 'start' || ev.event === 'upload_complete') continue
          if (ev.event === 'file_saved' && typeof data.filename === 'string') saved.push(data.filename)
          if (ev.event === 'error') {
            serverError = true
            // Backend error events are protocol flags — never forward their
            // message. A save_failed event names the failing file in a
            // structured field; render it only when it matches a file the
            // client itself uploaded (echo-of-client-data, provably not prose).
            const echoed =
              typeof data.filename === 'string' && files.map(fileLabel).includes(data.filename)
                ? (data.filename as string)
                : null
            const message = echoed
              ? streamErrorText(t, data.code, 'ingest.save_failed_file', { filename: echoed })
              : streamErrorText(t, data.code, 'ingest.failed_default')
            yield stamp('error', { message })
            continue
          }
          yield stamp(ev.event as IngestEvent['event'], data)
        }
      } catch (err) {
        caught = err
      }
      if (saved.length > 0) anySaved = true
      if (!complete) pending = withoutSaved(pending, saved)
      if (complete || pending.length === 0) {
        if (notices.length > 0) yield stamp('warning_resolved', { messages: notices })
        break
      }

      const status = caught instanceof UploadHttpError ? caught.status : undefined
      const aborted = caught instanceof DOMException && caught.name === 'AbortError'
      const retryable = !serverError && !aborted && (status === undefined || RETRYABLE_STATUSES.has(status))
      if (retryable && attempt < UPLOAD_RETRY_DELAYS_MS.length) {
        const message = t('ingest.batch_retrying', {
          batch: i + 1,
          total: batches.length,
          attempt: attempt + 1,
          max: UPLOAD_RETRY_DELAYS_MS.length
        })
        notices.push(message)
        yield stamp('warning', { message })
        await new Promise((resolve) => setTimeout(resolve, UPLOAD_RETRY_DELAYS_MS[attempt]))
        continue
      }

      const failure: BatchFailure = {
        batch: i + 1,
        total: batches.length,
        files: pending.map(fileLabel),
        status,
        ...(caught instanceof UploadStalledError ? { stalled: true } : {})
      }
      failures.push(failure)
      // Surface inline (shows in the event log) but keep going — one bad batch
      // must not sink the rest of a large upload. A server-side save error has
      // already been surfaced as its own `error` event.
      if (!serverError) yield stamp('warning', { message: describeBatchFailure(failure, limitBytes, t) })
      break
    }
  }

  if (!anySaved) {
    const anyTooLarge = failures.some((f) => f.status === 413)
    const message = anyTooLarge
      ? t('ingest.upload_failed_too_large', { limit: formatBytes(limitBytes) })
      : t('ingest.upload_failed_rejected', { count: failures.length })
    yield stamp('error', { message })
    return { anySaved, failures }
  }

  // Stage 2 is no longer this function's job: ingestion is queued separately
  // as a server-owned job (see `stores/ingestRun.ts` -> `createIngestJob`), so
  // a browser reload no longer severs the run's only view.
  return { anySaved, failures }
}

// Used as an <a href>, not fetched — must carry the sub-path base itself.
export const sourcePreviewUrl = (collection: string, file_hash: string) =>
  url(
    withOwner(
      `/sources/preview?collection=${encodeURIComponent(collection)}&file_hash=${encodeURIComponent(file_hash)}`
    )
  )

// The picture behind an image finding, addressed by the image rather than by
// the file it was cut from (a keyframe's file is its clip, a figure's its PDF).
// Used as an <img src>, so it carries the sub-path base too.
export const sourceThumbnailUrl = (collection: string, imageId: string) =>
  url(
    withOwner(
      `/sources/thumbnail?collection=${encodeURIComponent(collection)}&image_id=${encodeURIComponent(imageId)}`
    )
  )
