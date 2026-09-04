import { useMemo, useState } from 'react'
import { Button } from '@infra/ui'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { dismissIngestJob } from '@/api/jobs'
import { useIngestJobs, ingestJobsKey } from '@/hooks/useIngestJobs'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useIngestRunStore } from '@/stores/ingestRun'
import { IngestJobCard } from '@/components/ingest/IngestJobCard'
import { isFinishedCard, mergeJobCards } from '@/lib/ingestJobCards'
import { useT } from '@/i18n/LanguageContext'

interface ClearResult {
  cleared: number
  failed: number
}

/**
 * Dismiss several finished jobs, reporting how many actually went.
 *
 * A job the server has already forgotten (404) counts as cleared: it is gone
 * either way, and leaving its card behind is the one outcome a "clear"
 * control must not have.
 *
 * @param jobIds - The jobs to dismiss.
 * @returns How many were cleared and how many refused.
 */
async function dismissAll(jobIds: string[]): Promise<ClearResult> {
  const results = await Promise.allSettled(
    jobIds.map(async (jobId) => {
      try {
        await dismissIngestJob(jobId)
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 404)) throw e
      }
      return jobId
    })
  )
  const { dropJob } = useIngestJobsStore.getState()
  const { untrackJob } = useIngestRunStore.getState()
  let cleared = 0
  for (const result of results) {
    if (result.status !== 'fulfilled') continue
    cleared += 1
    dropJob(result.value)
    untrackJob(result.value)
  }
  return { cleared, failed: results.length - cleared }
}

/**
 * Every ingest job the caller owns, newest first — queued, running, finished
 * and interrupted alike.
 *
 * Merges what this browser queued with what the server lists, so a run stays
 * visible when the next one starts, after a reload, and across tabs.
 */
export function IngestJobList() {
  const t = useT()
  const qc = useQueryClient()
  const tracked = useIngestRunStore((s) => s.trackedJobs)
  const { data: listed } = useIngestJobs()
  // A map that changes once per run, not once per progress frame — see the
  // store. Subscribing to the event log here would re-render every card in
  // the list on every frame of every job.
  const terminal = useIngestJobsStore((s) => s.terminal)
  const [clearError, setClearError] = useState<string | null>(null)

  const cards = useMemo(() => mergeJobCards(tracked, listed ?? []), [tracked, listed])
  const finishedIds = useMemo(
    () => cards.filter((card) => isFinishedCard(card, terminal)).map((card) => card.jobId),
    [cards, terminal]
  )

  const clearMutation = useMutation({
    mutationFn: () => dismissAll(finishedIds),
    onSuccess: ({ cleared, failed }) => {
      setClearError(
        failed > 0
          ? t('ingest.clear_partial_failure', { cleared, total: cleared + failed, failed })
          : null
      )
      void qc.invalidateQueries({ queryKey: ingestJobsKey })
    },
    onError: () => setClearError(t('ingest.failed_default'))
  })

  if (cards.length === 0) return null

  return (
    <div className="space-y-4">
      {finishedIds.length > 0 && (
        <div className="flex items-center justify-end gap-2">
          {clearError && <span className="text-sm text-[var(--status-amber-fg)]">{clearError}</span>}
          {/* No confirmation: this removes job records, not indexed data, and
              the per-card control it batches has none either. */}
          <Button
            variant="secondary"
            disabled={clearMutation.isPending}
            onClick={() => {
              setClearError(null)
              clearMutation.mutate()
            }}
          >
            {t('ingest.clear_finished', { count: finishedIds.length })}
          </Button>
        </div>
      )}
      {cards.map((card) => (
        <IngestJobCard
          key={card.jobId}
          jobId={card.jobId}
          collection={card.collection}
          listItem={card.listItem}
        />
      ))}
    </div>
  )
}
