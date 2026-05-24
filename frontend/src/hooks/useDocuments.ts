import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { getDocumentsPage, listDocuments } from '@/api/collections'
import { useUiStore } from '@/stores/ui'

/**
 * Full-collection document list. Backed by the legacy ``/collections/documents``
 * endpoint, which still returns every record in one shot. Currently used by
 * the dashboard for a total-documents KPI; large collections (100k+ nodes)
 * pay the full scroll on entry to that page. Long-term follow-up: add a
 * dedicated count endpoint backed by a payload index so the KPI no longer
 * materializes the whole list.
 */
export function useDocuments() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['documents', collection],
    queryFn: listDocuments,
    enabled: !!collection
  })
}

/**
 * Paginated document records for the active collection. Replaces the
 * single-shot ``useDocuments`` for the inspector so it no longer
 * materializes the whole document list on every collection switch.
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
