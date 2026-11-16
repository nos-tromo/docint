import { useEffect } from 'react'
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { deleteSession, getSessionAnalysis, getSessionHistory, listSessions } from '@/api/sessions'
import type { Analysis } from '@/api/types'
import { useActiveCollection, type CollectionScope } from '@/hooks/useActiveCollection'

export const sessionsKey = ['sessions'] as const
export const sessionHistoryKey = (id: string) => ['sessions', id, 'history'] as const
// Outside the `sessions` prefix on purpose: a finished turn invalidates that prefix, and a stored analysis
// (up to a MiB of charts) never changes, so it must not be read again with every message.
export const sessionAnalysesKey = ['session-analyses'] as const
export const sessionAnalysisKey = (id: string, analysisId: string) => [...sessionAnalysesKey, id, analysisId] as const

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

/**
 * One analysis of a session, read through the active collection like its history.
 *
 * A stored analysis never changes, so it is read once and kept. The collection and its owner are part of
 * the key, as the history's are: the server answers 404 outside the session's own collection.
 *
 * @param sessionId - The session it belongs to; nothing is read without one.
 * @param analysisId - The analysis's id.
 * @param enabled - Whether to read it: a summary from history does, one the final frame carried does not.
 */
export function useSessionAnalysis(sessionId: string | null, analysisId: string, enabled: boolean) {
  const scope = useActiveCollection()
  return useQuery({
    queryKey: [...sessionAnalysisKey(sessionId ?? '', analysisId), scope] as const,
    queryFn: () => getSessionAnalysis(sessionId!, analysisId, scope.collection!),
    enabled: enabled && !!sessionId && !!scope.collection,
    staleTime: Infinity
  })
}

/**
 * Keep an analysis the final frame carried where `useSessionAnalysis` looks.
 *
 * A finished turn refreshes the session's history, which describes the analysis by its shape only; its
 * card then reads the analysis through its route. Seeded, that read is already answered, and the card
 * does not blink back to loading.
 *
 * @param sessionId - The session it belongs to.
 * @param analysis - The analysis in full, or null when the card holds only a summary.
 */
export function useSeedSessionAnalysis(sessionId: string | null, analysis: Analysis | null) {
  const qc = useQueryClient()
  const scope = useActiveCollection()
  useEffect(() => {
    if (analysis && sessionId && scope.collection) {
      qc.setQueryData([...sessionAnalysisKey(sessionId, analysis.id), scope], analysis)
    }
  }, [qc, scope, sessionId, analysis])
}

/** Forget the analyses cached for a collection's chats, whichever owner's. */
export function dropCollectionAnalyses(qc: QueryClient, collection: string) {
  qc.removeQueries({
    queryKey: sessionAnalysesKey,
    // The scope is the fourth part of the key, after the session and the analysis (see `useSessionAnalysis`).
    predicate: (query) => (query.queryKey[3] as CollectionScope | undefined)?.collection === collection
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => deleteSession(id),
    onSuccess: (_data, id) => {
      // Its analyses are up to a MiB each, and nothing reads them once it is gone.
      qc.removeQueries({ queryKey: [...sessionAnalysesKey, id] })
      return qc.invalidateQueries({ queryKey: sessionsKey })
    }
  })
}
