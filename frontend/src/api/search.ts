import { apiPost } from './client'
import type { SearchRequest, SearchResult } from './types'

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
