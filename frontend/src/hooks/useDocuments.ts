import { useQuery } from '@tanstack/react-query'
import { listDocuments } from '@/api/collections'
import { useUiStore } from '@/stores/ui'

export function useDocuments() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['documents', collection],
    queryFn: listDocuments,
    enabled: !!collection
  })
}
