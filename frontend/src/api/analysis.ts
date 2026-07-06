import { apiPost } from './client'
import { streamSse } from './sse'
import type { SummaryResponse } from './types'

/**
 * Build a query string for the summary endpoints. `refresh` bypasses the
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

export const summarize = (refresh?: boolean, collection?: string) =>
  apiPost<SummaryResponse>('/summarize' + summaryQuery(refresh, collection))

export const streamSummary = (refresh?: boolean, collection?: string) =>
  streamSse('/summarize/stream' + summaryQuery(refresh, collection))
