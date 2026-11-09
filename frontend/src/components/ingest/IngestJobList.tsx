import { useMemo, useState } from 'react'
import { Button, SectionHeader } from '@infra/ui'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { dismissIngestJob } from '@/api/jobs'
import { useIngestJobs, ingestJobsKey } from '@/hooks/useIngestJobs'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useIngestRunStore } from '@/stores/ingestRun'
import { useUiStore } from '@/stores/ui'
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
 * @param owner - The owner they were queued under.
 * @returns How many were cleared and how many refused.
 */
async function dismissAll(jobIds: string[], owner: string | null): Promise<ClearResult> {
  const results = await Promise.allSettled(
    jobIds.map(async (jobId) => {
      try {
        await dismissIngestJob(jobId, owner)
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
 * visible when the next one starts, after a reload, and across tabs. Both
 * halves are the selected owner's: a job queued in another namespace is
 * unknown to this one, so its card could only ever read as interrupted.
 */
export function IngestJobList() {
  const t = useT()
  const qc = useQueryClient()
  const owner = useUiStore((s) => s.selectedOwner)
  const tracked = useIngestRunStore((s) => s.trackedJobs)
  const { data: listed } = useIngestJobs()
  // A map that changes once per run, not once per progress frame — see the
  // store. Subscribing to the event log here would re-render every card in
  // the list on every frame of every job.
  const terminal = useIngestJobsStore((s) => s.terminal)
  // An error is about the jobs a clear was asked to remove, and goes with them.
  const [clearError, setClearError] = useState<{ message: string; jobIds: string[] } | null>(null)

  const cards = useMemo(
    () => mergeJobCards(tracked.filter((job) => job.owner === owner), listed ?? []),
    [tracked, listed, owner]
  )
  const finishedIds = useMemo(
    () => cards.filter((card) => isFinishedCard(card, terminal)).map((card) => card.jobId),
    [cards, terminal]
  )

  const clearMutation = useMutation({
    mutationFn: (jobIds: string[]) => dismissAll(jobIds, owner),
    onSuccess: ({ cleared, failed }, jobIds) => {
      setClearError(
        failed > 0
          ? { message: t('ingest.clear_partial_failure', { cleared, total: cleared + failed, failed }), jobIds }
          : null
      )
      void qc.invalidateQueries({ queryKey: ingestJobsKey })
    },
    onError: (_error, jobIds) => setClearError({ message: t('ingest.failed_default'), jobIds })
  })
  const showClearError = clearError !== null && clearError.jobIds.some((id) => finishedIds.includes(id))

  if (cards.length === 0) return null

  return (
    <div className="space-y-4">
      <SectionHeader
        title={t('ingest.jobs_heading')}
        actions={
          // No confirmation: this removes job records, not indexed data, and
          // the per-card control it batches has none either.
          finishedIds.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              disabled={clearMutation.isPending}
              onClick={() => {
                setClearError(null)
                clearMutation.mutate(finishedIds)
              }}
            >
              {t('ingest.clear_finished', { count: finishedIds.length })}
            </Button>
          )
        }
      />
      {showClearError && (
        <p role="alert" className="text-sm text-[var(--status-amber-fg)]">
          {clearError.message}
        </p>
      )}
      {cards.map((card) => (
        <IngestJobCard
          key={card.jobId}
          jobId={card.jobId}
          collection={card.collection}
          owner={owner}
          listItem={card.listItem}
        />
      ))}
    </div>
  )
}
