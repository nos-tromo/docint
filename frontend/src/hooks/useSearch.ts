import { useMutation, useQuery } from '@tanstack/react-query'
import { fetchChunkText, searchCollection } from '@/api/search'
import { clearScope, setScope } from '@/api/scope'
import type { ChunkText, MetadataFilter, SearchField, SearchResult } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useUiStore } from '@/stores/ui'

export const searchQueryKey = (
  collection: string | null,
  query: string,
  field: SearchField,
  filters: MetadataFilter[]
) => ['search', collection, query, field, filters] as const

/**
 * Run a keyword search against the active collection.
 *
 * The panel's metadata filters are part of the key, so tightening a filter
 * re-runs the search rather than showing hits the filter excludes. Disabled
 * until both a collection and a non-empty query exist — a keyword-less search
 * would be an unfiltered scan of the whole collection. `field` chooses which
 * payload field the keywords match; it is part of the key so switching the
 * picker re-runs the search.
 *
 * @param query - The submitted keywords.
 * @param field - The payload field to match; defaults to the chunk text.
 * @returns The TanStack Query result carrying hits and index status.
 */
export function useSearch(query: string, field: SearchField = 'text') {
  const collection = useUiStore((s) => s.selectedCollection)
  // Whole-store subscription (the idiom Chat.tsx already uses) so any filter
  // edit re-renders and rebuilds the payload. The payload is a fresh array
  // every render, which is harmless — React Query hashes keys structurally.
  const filters = useChatFiltersStore().buildPayload()
  const trimmed = query.trim()

  return useQuery<SearchResult>({
    queryKey: searchQueryKey(collection, trimmed, field, filters),
    queryFn: () =>
      searchCollection({
        question: trimmed,
        collection: collection ?? undefined,
        metadata_filters: filters,
        field
      }),
    enabled: !!collection && trimmed.length > 0,
    // A blank/too-short query is a 422 the user must fix; retrying it just
    // delays the message.
    retry: false
  })
}

export const chunkTextQueryKey = (collection: string | null, id: string) =>
  ['search-chunk', collection, id] as const

/**
 * Fetch one hit's full chunk text, on demand.
 *
 * Gated on `enabled` so nothing is fetched until a hit is actually expanded —
 * a search page of 20 hits would otherwise cost 20 extra requests for text
 * nobody asked to read. Once fetched the text is immutable for that point id
 * (a re-ingest mints new ids rather than rewriting a chunk), so it never goes
 * stale and collapsing then re-expanding is free.
 *
 * A 404 is terminal and meaningful — the chunk is gone — so it is neither
 * retried nor smoothed into an empty result; the caller renders it as its own
 * state.
 *
 * @param id - The hit's Qdrant point id.
 * @param enabled - Whether the hit is expanded.
 * @returns The TanStack Query result carrying the chunk text.
 */
export function useChunkText(id: string, enabled: boolean) {
  const collection = useUiStore((s) => s.selectedCollection)

  return useQuery<ChunkText>({
    queryKey: chunkTextQueryKey(collection, id),
    queryFn: () => fetchChunkText(id, collection ?? undefined),
    enabled: enabled && !!collection && id.length > 0,
    staleTime: Infinity,
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
