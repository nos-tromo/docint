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
 * Build an absolute URL for the grouped-search CSV download
 * (`GET /search/aggregate/export.csv`).
 *
 * Mirrors `aggregateCollection`'s inputs so the export always reflects
 * exactly what the panel currently shows. Use this as the `href` of a
 * download anchor so the browser handles the streaming response natively.
 *
 * @param collection - Caller's logical collection.
 * @param groupBy - The payload field to group by.
 * @param question - Whitespace-separated keywords; may be blank.
 * @param filters - Metadata filters currently applied.
 * @returns An absolute URL to the CSV export endpoint.
 */
export function aggregateExportHref(
  collection: string,
  groupBy: GroupByField,
  question: string,
  filters: MetadataFilter[]
): string {
  const owner = getOwnerParam()
  const params = new URLSearchParams()
  params.set('collection', collection)
  params.set('group_by', groupBy)
  if (question.trim()) params.set('question', question.trim())
  if (filters.length) params.set('metadata_filters', JSON.stringify(filters))
  if (owner) params.set('owner', owner)
  return url(`/search/aggregate/export.csv?${params}`)
}
