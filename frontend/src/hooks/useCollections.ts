import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteCollection, listCollections, selectCollection } from '@/api/collections'
import { reportsKey } from '@/hooks/useReports'

export const collectionsKey = ['collections'] as const

export function useCollections() {
  return useQuery({ queryKey: collectionsKey, queryFn: listCollections })
}

export function useSelectCollection() {
  return useMutation({ mutationFn: (name: string) => selectCollection(name) })
}

/** Deleting a collection deletes its reports too, so both listings are refreshed. */
export function useDeleteCollection() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => deleteCollection(name),
    onSuccess: () =>
      Promise.all([
        qc.invalidateQueries({ queryKey: collectionsKey }),
        qc.invalidateQueries({ queryKey: reportsKey })
      ])
  })
}
