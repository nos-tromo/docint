import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteSession, getSessionHistory, listSessions } from '@/api/sessions'
import { useActiveCollection } from '@/hooks/useActiveCollection'

export const sessionsKey = ['sessions'] as const
export const sessionHistoryKey = (id: string) => ['sessions', id, 'history'] as const

export function useSessions() {
  const scope = useActiveCollection()
  return useQuery({
    queryKey: [...sessionsKey, scope],
    queryFn: () => listSessions(scope.collection ?? undefined),
    enabled: !!scope.collection
  })
}

/**
 * A session's history, read through the active collection.
 *
 * The id comes from the URL, and Back reopens an old one after a collection
 * switch: the server answers 404 outside the session's own collection, and
 * the collection and its owner are part of the key so a cached copy from the
 * old one is not served in the new one meanwhile.
 */
export function useSessionHistory(id: string | null) {
  const scope = useActiveCollection()
  return useQuery({
    queryKey: id ? [...sessionHistoryKey(id), scope] : ['sessions', 'none'],
    queryFn: () => getSessionHistory(id!, scope.collection!),
    enabled: !!id && !!scope.collection
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => deleteSession(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionsKey })
  })
}
