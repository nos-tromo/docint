import { apiGet, apiPost, getOwnerParam, url } from './client'
import type {
  AggregateRequest,
  AggregateResult,
  ChunkText,
  GroupByField,
  MetadataFilter,
  SearchRequest,
  SearchResult,
} from './types'

/**
 * Run a full-text keyword search over the caller's collection.
 *
 * Keywords are ANDed server-side and matched case-insensitively on word
 * prefixes. The response's `status` distinguishes "never indexed" from
 * "backfill incomplete" from "no matches" — callers must keep the three apart.
 *
 * @param body - Query, collection, metadata filters and paging.
 * @returns The hits plus the collection's search-index state.
 */
export const searchCollection = (body: SearchRequest) => apiPost<SearchResult>('/search', body)

/**
 * Fetch one chunk's full text, for expanding a single search hit.
 *
 * Search responses cap every preview at 600 characters, so the rest is fetched
 * only for the hit an investigator actually opens rather than inflating every
 * response with text most hits never show. A **404 is a real answer** — a
 * re-ingest mints new point ids, so a hit from an earlier search can outlive
 * its chunk — and callers must render it as "gone", not as an empty chunk.
 *
 * @param id - The hit's Qdrant point id.
 * @param collection - Caller's logical collection, owner-gated server-side.
 * @returns The chunk id and its full text.
 */
export const fetchChunkText = (id: string, collection?: string) =>
  apiGet<ChunkText>('/search/chunk', { id, collection })

/**
 * Group every matching chunk by one payload field (`POST /search/aggregate`).
 *
 * The exhaustive counterpart to `searchCollection`: no top-k, no ranking. A
 * blank `question` groups the whole (filtered) collection.
 *
 * @param body - Query, group-by field, collection, filters and sizing.
 * @returns The groups plus the collection's search-index state.
 */
export const aggregateCollection = (body: AggregateRequest) =>
  apiPost<AggregateResult>('/search/aggregate', body)

/**
 * Build an absolute URL for the search CSV export (`GET /search/export.csv`).
 *
 * One endpoint, two lanes, selected the same way as the JSON endpoints:
 * `groupBy` omitted streams the keyword hits lane (the backend requires
 * `question` there), `groupBy` set streams the grouped/social lane (a blank
 * `question` groups the whole collection). `sessionId` and `markedIds` both
 * mark chunks for the export server-side — a stored chat scope and an
 * unsaved local selection respectively — and are unioned there, so passing
 * both is safe. Callers should still omit `markedIds` once `sessionId` is
 * set, though: the scope it would name is already persisted server-side by
 * then (see the SearchPanel.tsx call site), and a scope has no count cap —
 * only a token budget — so a legitimate selection can run to hundreds of
 * ids, long enough to overflow the gateway's header limits if serialized
 * here too. Use this as the `href` of a download anchor so the browser
 * handles the streaming response natively.
 *
 * @param collection - Caller's logical collection.
 * @param opts.question - Whitespace-separated keywords; required by the
 *   backend for the hits lane, optional for the grouped lane.
 * @param opts.groupBy - The payload field to group by, or omitted for the
 *   keyword hits lane.
 * @param opts.filters - Metadata filters currently applied.
 * @param opts.sessionId - The open session, if any, whose stored chat scope
 *   should count as marked too.
 * @param opts.markedIds - Qdrant point ids the caller has selected locally.
 * @returns An absolute URL to the CSV export endpoint.
 */
export function searchExportHref(
  collection: string,
  opts: {
    question: string
    groupBy?: GroupByField
    filters: MetadataFilter[]
    sessionId?: string | null
    markedIds?: string[]
  }
): string {
  const owner = getOwnerParam()
  const params = new URLSearchParams()
  params.set('collection', collection)
  if (opts.question.trim()) params.set('question', opts.question.trim())
  if (opts.groupBy) params.set('group_by', opts.groupBy)
  if (opts.filters.length) params.set('metadata_filters', JSON.stringify(opts.filters))
  if (opts.sessionId) params.set('session_id', opts.sessionId)
  if (opts.markedIds?.length) params.set('marked_ids', opts.markedIds.join(','))
  if (owner) params.set('owner', owner)
  return url(`/search/export.csv?${params}`)
}
