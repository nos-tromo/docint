import { useMemo } from 'react'
import { Button, Card } from '@infra/ui'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { createIngestJob, dismissIngestJob, getIngestJob } from '@/api/jobs'
import { ingestJobsKey } from '@/hooks/useIngestJobs'
import { useIngestCompletion } from '@/hooks/useIngestCompletion'
import { useIngestJobsStore, selectJobEvents } from '@/stores/ingestJobs'
import { useIngestRunStore } from '@/stores/ingestRun'
import { IngestionStatus } from '@/components/ingest/IngestionStatus'
import { deriveIngestStatus, withServerTimes, type IngestStatus } from '@/lib/ingestStatus'
import { useT } from '@/i18n/LanguageContext'
import type { IngestJobSnapshot } from '@/api/types'

/** How often to re-check a job that is still waiting on a worker slot. */
const QUEUED_POLL_INTERVAL_MS = 2_000

/**
 * What the server's own status says the card should show when no frame has
 * arrived — which for a queued job is every frame, since one waiting on a
 * worker slot emits none at all.
 */
const PHASE_BY_STATUS: Record<string, IngestStatus['phase'] | undefined> = {
  queued: 'queued',
  running: 'processing',
  completed: 'complete',
  failed: 'error'
}

interface IngestJobCardProps {
  jobId: string
  /** The collection this job was queued against, shown on the card. */
  collection: string
  /** The server's snapshot, when this card came from the jobs list. */
  listItem?: IngestJobSnapshot
}

/**
 * One ingest job: its warnings, its live progress and the controls that end
 * it. Reads progress from the owner-multiplexed SSE stream's store and the
 * run's own timeline from the server snapshot.
 *
 * @param jobId - The job to render.
 * @param collection - The collection it was queued against.
 * @param listItem - The server's snapshot, if the jobs list carried one.
 */
export function IngestJobCard({ jobId, collection, listItem }: IngestJobCardProps) {
  const t = useT()
  const qc = useQueryClient()
  const jobEvents = useIngestJobsStore(selectJobEvents(jobId))
  const uploadEvents = useIngestRunStore((s) => s.uploadEventsByJob[jobId])
  const handled = useIngestRunStore((s) => s.handledJobIds.includes(jobId))

  // Queried directly by job id rather than inferred from the list snapshot —
  // see the `interrupted` derivation below for why. A card the list already
  // vouched for starts from that snapshot and does not refetch; a card known
  // only locally has nothing to start from and asks.
  const jobQuery = useQuery({
    queryKey: ['ingest-job', jobId],
    queryFn: () => getIngestJob(jobId),
    initialData: listItem,
    retry: false,
    staleTime: 30_000,
    // A queued job emits no SSE frames until a worker slot frees up, so
    // nothing else would refresh this snapshot — the "waiting for a slot"
    // notice would outlive the queue by up to `staleTime`. Poll only while
    // queued; every other status is driven by the event stream.
    refetchInterval: (query) =>
      query.state.data?.status === 'queued' ? QUEUED_POLL_INTERVAL_MS : false
  })

  const status: IngestStatus = useMemo(() => {
    // The upload leg belongs to the job it produced (stores/ingestRun.ts), so
    // the card's timeline spans both legs — the same log the single-card view
    // merged, now scoped to one job instead of "whichever job is active".
    const merged = uploadEvents ? [...uploadEvents, ...jobEvents] : jobEvents
    const derived = deriveIngestStatus(merged)
    // A reattached log has no synthetic upload `start` frame, so the elapsed
    // timer has no client anchor — fall back to the server snapshot's
    // `run_started_at`/`finished_at` (already fetched by `jobQuery`).
    const timed = withServerTimes(derived, jobQuery.data)
    // The card must name its collection even before a frame carries one —
    // with several runs listed, an unnamed card belongs to no run.
    const named = timed.collection ? timed : { ...timed, collection }
    // A merged log can start mid-stream — reattaching to a job whose
    // `ingestion_started` frame arrived before this tab did — and still carry
    // real progress. Treat "has events" as never idle so progress stays
    // visible even without an explicit phase-setting frame.
    if (named.phase !== 'idle') return named
    if (jobEvents.length > 0) return { ...named, phase: 'processing' }
    // No frames at all: a job discovered from the server's list, or one whose
    // frames aged out of the replay. The snapshot is the only account of it,
    // and rendering nothing is how a run vanishes from view.
    const phase = PHASE_BY_STATUS[jobQuery.data?.status ?? '']
    return phase ? { ...named, phase } : named
  }, [uploadEvents, jobEvents, jobQuery.data, collection])

  useIngestCompletion(jobId, jobEvents, collection)

  // A job the server 404s on is an interrupted run: the backend restarted
  // while it was in flight (jobs are in-memory by design). Answered by
  // querying the job directly — the authoritative source — rather than
  // inferred from a `/ingest/jobs` *list* snapshot, which went through three
  // failed attempts at this exact spot (a stale-list false positive; a
  // permanently-stuck false negative from gating on historical SSE events; a
  // transient false-positive flash from invalidating a shared list query but
  // still rendering against its pre-refetch data for one commit). Every one
  // of those was really an attempt to reason about a snapshot's freshness
  // relative to job creation. This query's key includes the job id, so a job
  // this browser has never asked about is a brand-new cache entry with no
  // stale data to race against. A completed-but-undismissed job also 404s
  // after a backend restart, but `handledJobIds` already records "this job
  // reached `ingestion_complete`" — so excluding that case keeps a run that
  // actually finished from reading as interrupted.
  const interrupted =
    jobQuery.isError &&
    jobQuery.error instanceof ApiError &&
    jobQuery.error.status === 404 &&
    !handled

  function forget() {
    useIngestJobsStore.getState().dropJob(jobId)
    useIngestRunStore.getState().untrackJob(jobId)
    void qc.invalidateQueries({ queryKey: ingestJobsKey })
  }

  const dismissMutation = useMutation({
    mutationFn: () => dismissIngestJob(jobId),
    onSuccess: () => {
      forget()
      // The card unmounts in the same tick, which leaves this query inactive
      // — invalidate it anyway so a dismissed-then-somehow-revisited job id
      // never serves a stale cached snapshot.
      void qc.invalidateQueries({ queryKey: ['ingest-job', jobId] })
    }
  })

  // Re-queuing an interrupted run does NOT re-upload: the batches this run
  // already staged are still in the collection's server-side upload directory
  // (only the in-memory job registry was lost), so re-finalizing directly is
  // what "re-running skips whatever was already indexed" (the banner copy)
  // actually promises. Runs against the collection the interrupted job was
  // queued for, never the live form field the user can edit meanwhile.
  const rerunMutation = useMutation({
    mutationFn: () => {
      const run = useIngestRunStore.getState()
      return createIngestJob({ collection, ner: run.ner, hate_speech: run.hate })
    },
    onSuccess: ({ job_id }) => {
      useIngestRunStore.getState().trackJob(job_id, collection)
      if (job_id !== jobId) forget()
    }
  })

  const dismissable = status.phase === 'complete' || status.phase === 'error'

  return (
    <div className="space-y-2">
      {/* Derived from the merged upload+job log (`status.warnings`), not the
          run's own upload warnings — a job-stream warning (a soft-empty
          ingest, a failed post-ingest entity resolution) is just as
          actionable and must not go unreported. Kept with its own job: with
          several runs listed, a page-level warning names no run. */}
      {status.warnings.length > 0 && (
        <ul className="text-sm text-[var(--status-amber-fg)] space-y-1" role="alert">
          {status.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}

      {interrupted ? (
        <Card className="text-sm space-y-2" role="status">
          <CollectionLabel collection={collection} />
          <p className="text-muted-foreground">{t('ingest.job_interrupted')}</p>
          <div className="flex gap-2">
            <Button
              variant="primary"
              disabled={rerunMutation.isPending}
              onClick={() => rerunMutation.mutate()}
            >
              {t('ingest.job_rerun')}
            </Button>
            {/* An interrupted, never-going-to-report-progress-again job would
                otherwise sit here forever — nothing else clears it. Purely
                local: the server has already forgotten it. */}
            <Button variant="secondary" onClick={forget}>
              {t('ingest.dismiss')}
            </Button>
          </div>
        </Card>
      ) : (
        status.phase !== 'idle' && (
          <>
            <IngestionStatus status={status} />
            {dismissable && (
              <Button
                variant="secondary"
                disabled={dismissMutation.isPending}
                onClick={() => dismissMutation.mutate()}
              >
                {t('ingest.dismiss')}
              </Button>
            )}
          </>
        )
      )}
    </div>
  )
}

/** The collection the interrupted banner is about — the only thing telling
 *  two dead runs apart. `IngestionStatus` carries its own. */
function CollectionLabel({ collection }: { collection: string }) {
  if (!collection) return null
  return <p className="truncate text-sm text-foreground">{collection}</p>
}
