import { apiGet, apiPost, apiDelete, ApiError } from './client'
import type { IngestJobSnapshot } from './types'

export const INGEST_JOB_EVENTS_PATH = '/ingest/jobs/events'

export interface CreateIngestJobPayload {
  collection: string
  hybrid?: boolean
  ner?: boolean
  hate_speech?: boolean
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
 * Queue an ingest job over a collection's staged upload batches.
 *
 * A 409 means that collection already has a run in flight. That is not an
 * error condition for the user — it happens when they re-submit after a
 * reload — so the in-flight `job_id` is adopted and the caller simply
 * re-attaches to the existing run.
 *
 * @param payload - Collection and per-run enrichment overrides.
 * @returns The job id, and whether it was adopted from an in-flight run.
 */
export async function createIngestJob(
  payload: CreateIngestJobPayload
): Promise<{ job_id: string; adopted: boolean }> {
  try {
    const res = await apiPost<{ job_id: string }>('/ingest/finalize', payload)
    return { job_id: res.job_id, adopted: false }
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) {
      const detail = e.detail as { detail?: { job_id?: string } } | { job_id?: string }
      const nested = (detail as { detail?: { job_id?: string } }).detail
      const jobId = nested?.job_id ?? (detail as { job_id?: string }).job_id
      if (jobId) return { job_id: jobId, adopted: true }
    }
    throw e
  }
}

/** List the caller's jobs, newest first. Powers reload re-discovery. */
export const listIngestJobs = () => apiGet<{ jobs: IngestJobSnapshot[] }>('/ingest/jobs')

/** Fetch one job's snapshot. Rejects with a 404 `ApiError` when it is gone. */
export const getIngestJob = (id: string) => apiGet<IngestJobSnapshot>(`/ingest/jobs/${id}`)

/** Dismiss a finished job. Rejects with 409 while it is still running. */
export const dismissIngestJob = (id: string) => apiDelete<{ ok: boolean }>(`/ingest/jobs/${id}`)
