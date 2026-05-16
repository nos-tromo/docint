import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteCollection, listCollections, selectCollection } from '@/api/collections'

export const collectionsKey = ['collections'] as const

export function useCollections() {
  return useQuery({ queryKey: collectionsKey, queryFn: listCollections })
}

export function useSelectCollection() {
  return useMutation({ mutationFn: (name: string) => selectCollection(name) })
}

export function useDeleteCollection() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => deleteCollection(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: collectionsKey })
  })
}
