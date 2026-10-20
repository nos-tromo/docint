import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteSession, getSessionHistory, listSessions } from '@/api/sessions'
import { useUiStore } from '@/stores/ui'

export const sessionsKey = ['sessions'] as const
export const sessionHistoryKey = (id: string) => ['sessions', id, 'history'] as const

export function useSessions() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: [...sessionsKey, collection],
    queryFn: () => listSessions(collection ?? undefined),
    enabled: !!collection
  })
}

/**
 * A session's history, read through the active collection.
 *
 * The id comes from the URL, and Back reopens an old one after a collection
 * switch: the server answers 404 outside the session's own collection, and
 * the collection is part of the key so a cached copy from the old collection
 * is not served in the new one meanwhile.
 */
export function useSessionHistory(id: string | null) {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: id ? [...sessionHistoryKey(id), collection] : ['sessions', 'none'],
    queryFn: () => getSessionHistory(id!, collection!),
    enabled: !!id && !!collection
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => deleteSession(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionsKey })
  })
}
