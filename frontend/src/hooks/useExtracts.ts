import { useEffect, useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { createExtract, deleteExtract, listExtracts } from '@/api/extracts'
import { useAppendixFields } from '@/hooks/useAppendixFields'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useUiStore } from '@/stores/ui'
import type { IngestEvent } from '@/api/types'

/** Progress of the extract build currently in flight, if any. */
export interface ExtractProgress {
  jobId: string
  rendered: number | null
  totalUnits: number | null
  error: string | null
}

/** Stored extracts for the active collection, newest first. */
export function useExtracts() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['extracts', collection],
    queryFn: () => listExtracts(collection ?? ''),
    enabled: !!collection
  })
}

/** Delete one stored extract and refresh the listing. */
export function useDeleteExtract() {
  const collection = useUiStore((s) => s.selectedCollection)
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (extractId: string) => deleteExtract(collection ?? '', extractId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['extracts', collection] })
  })
}

/** Queue an extract build. Used by the panel and by a per-row 413 fallback. */
export function useStartExtract() {
  const collection = useUiStore((s) => s.selectedCollection)
  const appendix = useAppendixFields()
  return async (target?: string) => {
    if (!collection) return null
    return createExtract(collection, target, appendix)
  }
}

/** Fold one job's frames into the panel's progress view. */
function readProgress(jobId: string, events: IngestEvent[]): ExtractProgress | null {
  let progress: ExtractProgress | null = null
  for (const event of events) {
    const data = (event.data ?? {}) as Record<string, unknown>
    if (event.event === 'extract_started') {
      progress = { jobId, rendered: null, totalUnits: null, error: null }
    } else if (event.event === 'extract_progress') {
      progress = {
        jobId,
        rendered: typeof data.rendered === 'number' ? data.rendered : null,
        totalUnits: typeof data.total_units === 'number' ? data.total_units : null,
        error: null
      }
    } else if (event.event === 'extract_completed') {
      return null
    } else if (event.event === 'error' && progress) {
      return { ...progress, error: String(data.code ?? 'extract_failed') }
    }
  }
  return progress
}

/**
 * The extract build in flight for the active collection, if any.
 *
 * Frames come from the shared job store, which the single owner-multiplexed
 * stream in `Shell` already feeds — so a build survives navigating away from
 * the Inspector and a reload re-attaches to it like any other job. When a
 * build ends, the stored listing is refreshed.
 */
export function useExtractJob() {
  const collection = useUiStore((s) => s.selectedCollection)
  const queryClient = useQueryClient()
  const events = useIngestJobsStore((s) => s.events)
  const start = useStartExtract()

  const progress = useMemo(() => {
    for (const [jobId, frames] of Object.entries(events)) {
      if (!frames.some((frame) => frame.event === 'extract_started')) continue
      const current = readProgress(jobId, frames)
      if (current) return current
    }
    return null
  }, [events])

  const finished = useMemo(
    () =>
      Object.values(events).some((frames) =>
        frames.some((frame) => frame.event === 'extract_completed')
      ),
    [events]
  )

  useEffect(() => {
    if (finished) void queryClient.invalidateQueries({ queryKey: ['extracts', collection] })
  }, [finished, collection, queryClient])

  return { progress, start, running: progress !== null && !progress.error }
}
