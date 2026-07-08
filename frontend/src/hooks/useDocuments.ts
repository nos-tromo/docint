import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { getDocumentsCount, getDocumentsPage, getDocumentsSummary } from '@/api/collections'
import { useUiStore } from '@/stores/ui'

/**
 * Paginated document records for the active collection. The inspector uses
 * this to render rows without materializing the whole list.
 */
export function useDocumentsPages() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useInfiniteQuery({
    queryKey: ['documents-pages', collection],
    queryFn: ({ pageParam }) =>
      getDocumentsPage({
        cursor: pageParam as string | null,
        limit: 50,
        collection: collection ?? undefined
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: !!collection
  })
}

/**
 * Unique-document count for the active collection. Backed by a cheap
 * server endpoint that shares the per-collection cache with the paginated
 * inspector — the first warm pays one full scroll, subsequent calls are O(1).
 */
export function useDocumentsCount() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['documents-count', collection],
    queryFn: () => getDocumentsCount(collection ?? undefined),
    enabled: !!collection
  })
}

/**
 * Collection-wide document aggregates (document/node totals + file-type and
 * entity-type breakdown) for the inspector's KPI strip. Computed server-side
 * over the whole collection, so the summary cards stay accurate regardless of
 * how many pages {@link useDocumentsPages} has loaded.
 */
export function useDocumentsSummary() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['documents-summary', collection],
    queryFn: () => getDocumentsSummary(collection ?? undefined),
    enabled: !!collection
  })
}
