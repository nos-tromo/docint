import { apiGet, apiPost } from './client'
import type { ChunkText, SearchRequest, SearchResult } from './types'

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
