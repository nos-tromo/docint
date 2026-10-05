import { apiGet, apiPost, apiDelete, ApiError } from './client'
import type { IngestJobSnapshot } from './types'

export const INGEST_JOB_EVENTS_PATH = '/ingest/jobs/events'

export interface CreateIngestJobPayload {
  collection: string
  hybrid?: boolean
  ner?: boolean
  hate_speech?: boolean
  /**
   * Whether the run rebuilds the collection summary when it finishes.
   * Omitted means the deployment's own `SUMMARY_ON_INGEST` decides.
   */
  summary?: boolean
  /**
   * How long this run spent uploading, in ms. The run starts when the user
   * hits ingest, but the job only exists from here on, so the client reports
   * the leg the server never saw and the backend folds it into the one
   * duration it logs and reports back. A duration, never a timestamp — the
   * server trusts no client clock, and clamps this. Omitted by the re-run
   * path, which re-finalizes already-staged batches without uploading.
   */
  upload_elapsed_ms?: number
}

/**
 * POST to a route that queues a background job, adopting the run already in
 * flight when the route refuses a second one.
 *
 * Every queueing route answers that refusal with 409 and names the in-flight
 * `job_id` — under `detail` when FastAPI wraps an `HTTPException`, at the top
 * level when the route writes the body itself — so both shapes are read. The
 * user has usually just re-submitted after a reload, so it is not an error for
 * them; the caller re-attaches to the existing run.
 *
 * @param path - The queueing route.
 * @param body - Its JSON body.
 * @returns The job id, and whether it was adopted from an in-flight run.
 */
export async function queueJob(
  path: string,
  body?: unknown
): Promise<{ job_id: string; adopted: boolean }> {
  try {
    const res = await apiPost<{ job_id: string }>(path, body)
    return { job_id: res.job_id, adopted: false }
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) {
      const detail = e.detail as { detail?: { job_id?: unknown }; job_id?: unknown } | null
      const jobId = detail?.detail?.job_id ?? detail?.job_id
      if (typeof jobId === 'string' && jobId) return { job_id: jobId, adopted: true }
    }
    throw e
  }
}

/**
 * Queue an ingest job over a collection's staged upload batches.
 *
 * A 409 means that collection already has a run in flight, whose `job_id` is
 * adopted (see {@link queueJob}).
 *
 * @param payload - Collection and per-run enrichment overrides.
 * @returns The job id, and whether it was adopted from an in-flight run.
 */
export const createIngestJob = (payload: CreateIngestJobPayload) =>
  queueJob('/ingest/finalize', payload)

/** One file already on the server, as a re-picking client recognises it. */
export interface StagedFile {
  /** Path relative to the batch directory — the upload's own `webkitRelativePath`. */
  name: string
  bytes: number
}

/** One preprocessing stage's tally for a collection.
 *
 * `stage` is protocol, English in every locale: `pdf`, `image` and `media`
 * count files, `ocr_pages` the scanned pages inside PDFs and `keyframes` the
 * frames inside clips. `failed` counts files whose task raised and are waiting
 * for whichever lane next needs them.
 */
export interface PreprocessStage {
  stage: string
  done: number
  total: number
  failed: number
}

/** What the server has staged for a collection, and what it is still reading. */
export interface StagedBatch {
  collection: string
  files: number
  bytes: number
  /** Transfers cut off mid-file. Counted, never staged, never ingested. */
  partial: number
  entries: StagedFile[]
  /** True when more files are staged than `entries` lists. */
  entries_truncated: boolean
  preprocess: { running: number; queued: number; stages: PreprocessStage[] }
}

/**
 * Ask what is staged for a collection.
 *
 * Uploading stages bytes and finalizing queues the job, so a browser that
 * dies between the two leaves files no job accounts for — and, before this,
 * nothing on screen to say so. The named entries are what let a re-picked
 * folder finish an interrupted upload instead of re-sending it whole.
 *
 * @param collection - The caller's logical collection name.
 * @param opts - `includeEntries: false` leaves the names out, which is what a
 *   progress poll wants: a folder-sized batch lists tens of thousands of them.
 * @returns The staged file count, total size, names, and preprocessing counts.
 */
export const getStagedBatch = (collection: string, opts?: { includeEntries?: boolean }) =>
  apiGet<StagedBatch>('/ingest/staged', {
    collection,
    include_entries: opts?.includeEntries === false ? false : undefined
  })

/** List the caller's jobs, newest first. Powers reload re-discovery. */
export const listIngestJobs = () => apiGet<{ jobs: IngestJobSnapshot[] }>('/ingest/jobs')

/** Fetch one job's snapshot. Rejects with a 404 `ApiError` when it is gone. */
export const getIngestJob = (id: string) => apiGet<IngestJobSnapshot>(`/ingest/jobs/${id}`)

/** Dismiss a finished job. Rejects with 409 while it is still running. */
export const dismissIngestJob = (id: string) => apiDelete<{ ok: boolean }>(`/ingest/jobs/${id}`)

/**
 * Ask a running job to stop.
 *
 * Resolving means the request was accepted, not that the job has stopped: a
 * worker thread cannot be killed, so the run ends at its next progress
 * checkpoint and one in-flight model call finishes first. Watch for the
 * terminal `ingestion_cancelled` frame. Rejects with 404 (unknown) or 409
 * (already finished).
 */
export const cancelIngestJob = (id: string) =>
  apiPost<{ ok: boolean }>(`/ingest/jobs/${id}/cancel`, {})
