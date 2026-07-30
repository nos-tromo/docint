import { streamUpload, UploadHttpError } from './upload'
import { streamErrorText } from './errorMessage'
import { url, withOwner } from './client'
import { planUploadBatches } from '@/lib/uploadBatches'
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

export interface IngestEnrichmentOptions {
  /** Per-request NER override; omitted -> deployment env default. */
  ner?: boolean
  /** Per-request hate-speech override; omitted -> deployment env default. */
  hateSpeech?: boolean
}

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
  return t('ingest.batch_failed_network', { batch: f.batch, total: f.total, label })
}

const fileLabel = (f: File): string => f.webkitRelativePath || f.name

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
 * Upload failures are non-fatal: a batch that errors (a lone oversize file →
 * 413, or a transient drop) is reported as a `warning` and the rest still
 * upload; the returned `failures` list is how the caller learns which files
 * never made it to the server.
 *
 * @param collection - Target logical collection name.
 * @param files - The full selection to upload, in user order.
 * @param limitBytes - The server's per-request upload ceiling in bytes (from
 *   `/config` `max_upload_bytes`); the packing budget is this times
 *   `UPLOAD_SAFETY_MARGIN`.
 * @param signal - Optional abort signal cancelling the in-flight request.
 * @param t - Translate function; defaults to the English catalog for pure/test callers.
 * @param _options - Accepted only so `Ingest.tsx`'s existing call site keeps
 *   compiling until it is rebuilt; enrichment now travels solely on the
 *   `createIngestJob` call the run controller makes after this generator
 *   returns, so it is not applied here.
 * @yields Normalised `IngestEvent`s, each stamped with `receivedAt`.
 * @returns Whether any batch saved, and the list of batches that failed.
 */
export async function* streamIngestUploadBatched(
  collection: string,
  files: File[],
  limitBytes: number,
  signal?: AbortSignal,
  t: Translate = defaultT,
  _options: IngestEnrichmentOptions = {}
): AsyncGenerator<IngestEvent, { anySaved: boolean; failures: BatchFailure[] }, unknown> {
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
      for await (const ev of streamUpload('/ingest/upload', buildIngestFormData(collection, batch), signal)) {
        const data = (ev.data ?? {}) as Record<string, unknown>
        // Swallow the per-batch `start` (one synthetic start already emitted)
        // and `upload_complete` (staged-only terminal); forward save progress.
        if (ev.event === 'start' || ev.event === 'upload_complete') continue
        if (ev.event === 'error') {
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
      anySaved = true
    } catch (err) {
      const status = err instanceof UploadHttpError ? err.status : undefined
      const failure: BatchFailure = { batch: i + 1, total: batches.length, files: batchNames, status }
      failures.push(failure)
      // Surface inline (shows in the event log) but keep going — one bad batch
      // must not sink the rest of a large upload.
      yield stamp('warning', { message: describeBatchFailure(failure, limitBytes, t) })
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
