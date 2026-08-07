import { apiPost } from './client'
import type { SummarizeResult } from './types'

/**
 * Build a query string for the summary endpoint. `refresh` bypasses the
 * cached collection summary; `collection` is the caller's logical name, which
 * WS2's stateless backend owner-gates and scopes per request (no server-side
 * active collection). Both are optional and omitted when falsy.
 */
function summaryQuery(refresh?: boolean, collection?: string): string {
  const qs = [
    refresh ? 'refresh=true' : '',
    collection ? `collection=${encodeURIComponent(collection)}` : ''
  ]
    .filter(Boolean)
    .join('&')
  return qs ? `?${qs}` : ''
}

/**
 * Fetch the collection summary, or queue a rebuild.
 *
 * A cache hit (or a fresh build finishing synchronously) answers 200 with the
 * full `SummaryResponse`. A miss — or an explicit `refresh` — queues a
 * background job and answers 202 with just `{job_id}`; the caller then
 * follows progress on the owner-multiplexed `GET /ingest/jobs/events` stream
 * and re-calls `summarize(false, collection)` once the job completes. A 409
 * (a build already in flight for this collection) surfaces as a thrown
 * `ApiError` carrying the in-flight `job_id` under `detail.job_id` — callers
 * adopt it the same way `createIngestJob` does.
 */
export const summarize = (refresh?: boolean, collection?: string) =>
  apiPost<SummarizeResult>('/summarize' + summaryQuery(refresh, collection))
