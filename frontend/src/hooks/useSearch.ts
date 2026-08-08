import { useMutation, useQuery } from '@tanstack/react-query'
import { searchCollection } from '@/api/search'
import { clearScope, setScope } from '@/api/scope'
import type { MetadataFilter, SearchResult } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useUiStore } from '@/stores/ui'

export const searchQueryKey = (
  collection: string | null,
  query: string,
  filters: MetadataFilter[]
) => ['search', collection, query, filters] as const

/**
 * Run a keyword search against the active collection.
 *
 * The panel's metadata filters are part of the key, so tightening a filter
 * re-runs the search rather than showing hits the filter excludes. Disabled
 * until both a collection and a non-empty query exist — a keyword-less search
 * would be an unfiltered scan of the whole collection.
 *
 * @param query - The submitted keywords.
 * @returns The TanStack Query result carrying hits and index status.
 */
export function useSearch(query: string) {
  const collection = useUiStore((s) => s.selectedCollection)
  // Whole-store subscription (the idiom Chat.tsx already uses) so any filter
  // edit re-renders and rebuilds the payload. The payload is a fresh array
  // every render, which is harmless — React Query hashes keys structurally.
  const filters = useChatFiltersStore().buildPayload()
  const trimmed = query.trim()

  return useQuery<SearchResult>({
    queryKey: searchQueryKey(collection, trimmed, filters),
    queryFn: () =>
      searchCollection({
        question: trimmed,
        collection: collection ?? undefined,
        metadata_filters: filters
      }),
    enabled: !!collection && trimmed.length > 0,
    // A blank/too-short query is a 422 the user must fix; retrying it just
    // delays the message.
    retry: false
  })
}

/**
 * Write and clear a session's pinned scope.
 *
 * @param sessionId - The session to scope, or null before the backend has
 *     minted one (the mutations are then inert).
 * @returns `set` and `clear` mutations over the scope endpoints.
 */
export function useScope(sessionId: string | null) {
  const collection = useUiStore((s) => s.selectedCollection)
  const set = useMutation({
    mutationFn: (chunkIds: string[]) =>
      setScope(sessionId as string, chunkIds, collection ?? undefined),
    // 422 means the selection cannot fit the context window. Retrying sends
    // the identical body and gets the identical refusal.
    retry: false
  })
  const clear = useMutation({
    mutationFn: () => clearScope(sessionId as string),
    retry: false
  })
  return { set, clear }
}
