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

export function useSessionHistory(id: string | null) {
  return useQuery({
    queryKey: id ? sessionHistoryKey(id) : ['sessions', 'none'],
    queryFn: () => getSessionHistory(id!),
    enabled: !!id
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => deleteSession(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionsKey })
  })
}
