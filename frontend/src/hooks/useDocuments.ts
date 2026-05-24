import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { getDocumentsCount, getDocumentsPage } from '@/api/collections'
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
      getDocumentsPage({ cursor: pageParam as string | null, limit: 50 }),
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
    queryFn: getDocumentsCount,
    enabled: !!collection
  })
}
