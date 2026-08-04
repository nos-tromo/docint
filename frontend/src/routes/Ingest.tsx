import { useEffect, useMemo, useState } from 'react'
import { Banner, Button, Card, FileList, Input, PageHeader } from '@infra/ui'
import { useQueryClient, useMutation, useQuery } from '@tanstack/react-query'
import { useIngestRunStore } from '@/stores/ingestRun'
import { useIngestJobsStore, selectJobEvents } from '@/stores/ingestJobs'
import { useIngestDefaults } from '@/hooks/useIngestDefaults'
import { dismissIngestJob, createIngestJob, getIngestJob } from '@/api/jobs'
import { ApiError } from '@/api/client'
import { selectCollection } from '@/api/collections'
import { useCollections, collectionsKey } from '@/hooks/useCollections'
import { useConfig } from '@/hooks/useConfig'
import { useUiStore } from '@/stores/ui'
import { Dropzone } from '@/components/ingest/Dropzone'
import { IngestionStatus } from '@/components/ingest/IngestionStatus'
import { deriveIngestStatus, type IngestStatus } from '@/lib/ingestStatus'
import { useT } from '@/i18n/LanguageContext'

/**
 * Per-request upload ceiling assumed only until `/config` loads (or if that
 * fetch fails). Deliberately well under the 1 GiB nginx default so batches
 * never 413 even before the real `max_upload_bytes` is known.
 */
const FALLBACK_UPLOAD_LIMIT_BYTES = 512 * 1024 * 1024
/** How often to re-check a job that is still waiting on a worker slot. */
const QUEUED_POLL_INTERVAL_MS = 2_000

export function Ingest() {
  const t = useT()
  const run = useIngestRunStore()
  const jobEvents = useIngestJobsStore(selectJobEvents(run.activeJobId))
  const streamLost = useIngestJobsStore((s) => s.streamLost)
  // Queried directly by job id rather than inferred from a list snapshot —
  // see the `interrupted` derivation below for why. `enabled: false` when
  // there's no active job keeps this a no-op the rest of the time.
  const jobQuery = useQuery({
    queryKey: ['ingest-job', run.activeJobId],
    queryFn: () => getIngestJob(run.activeJobId!),
    enabled: !!run.activeJobId,
    retry: false,
    staleTime: 30_000,
    // A queued job emits no SSE frames until a worker slot frees up, so
    // nothing else would refresh this snapshot — the "waiting for a slot"
    // notice would outlive the queue by up to `staleTime`. Poll only while
    // queued; every other status is driven by the event stream.
    refetchInterval: (query) =>
      query.state.data?.status === 'queued' ? QUEUED_POLL_INTERVAL_MS : false
  })
  const { data: ingestDefaults } = useIngestDefaults()
  const { data: collections } = useCollections()
  const { data: config } = useConfig()
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const qc = useQueryClient()

  const limitBytes = config?.max_upload_bytes ?? FALLBACK_UPLOAD_LIMIT_BYTES

  // A drop that resolves to no files at all (e.g. a folder of only
  // unreadable entries) never reaches the store — it is a transient,
  // view-only notice, not part of the run.
  const [dropError, setDropError] = useState<string | null>(null)

  // Seed the enrichment toggles once from the deployment defaults; the
  // user's own picks win afterwards for the rest of this mount. Reads the
  // setters non-reactively (store actions are stable) so they need not be
  // tracked as effect dependencies.
  const [seeded, setSeeded] = useState(false)
  useEffect(() => {
    if (seeded || !ingestDefaults) return
    const { setNer, setHate } = useIngestRunStore.getState()
    setNer(ingestDefaults.ner)
    setHate(ingestDefaults.hate_speech)
    setSeeded(true)
  }, [seeded, ingestDefaults])

  // Upload events and job events fold into one timeline, so the existing
  // (already tested) status reducer keeps working unchanged across the seam
  // between "uploading to the server" and "the server is ingesting".
  const fileSizes = useMemo(() => {
    const sizes: Record<string, number> = {}
    for (const f of run.files) sizes[f.webkitRelativePath || f.name] = f.size
    return sizes
  }, [run.files])

  const status: IngestStatus = useMemo(() => {
    // While a fresh upload is in flight, `activeJobId` still points at the
    // *previous* run's job — `run.start()` only overwrites it once
    // `createIngestJob` resolves for the new run (stores/ingestRun.ts). So a
    // second run in the same tab must not merge in the old job's log: its
    // trailing `ingestion_complete` would land after this run's own upload
    // events and flip `deriveIngestStatus` to `phase: 'complete'`, wiping the
    // live `uploadingFile`/`uploadingBytes` fields it clears on that event.
    const merged = run.uploading ? run.uploadEvents : [...run.uploadEvents, ...jobEvents]
    const derived = deriveIngestStatus(merged, fileSizes)
    // A merged log can start mid-stream — reattaching to a job whose
    // `ingestion_started` frame arrived before this tab did — and still
    // carry real progress. Treat "an active job has events" as never idle so
    // progress stays visible even without an explicit phase-setting frame.
    if (derived.phase === 'idle' && !run.uploading && run.activeJobId && jobEvents.length > 0) {
      return { ...derived, phase: 'processing' }
    }
    return derived
  }, [run.uploadEvents, jobEvents, fileSizes, run.activeJobId, run.uploading])

  // A job the server 404s on is an interrupted run: the backend restarted
  // while it was in flight (jobs are in-memory by design). Answered by
  // querying the job directly — the authoritative source — rather than
  // inferred from a `/ingest/jobs` *list* snapshot, which went through three
  // failed attempts at this exact spot (a stale-list false positive; a
  // permanently-stuck false negative from gating on historical SSE events;
  // a transient false-positive flash from invalidating a shared list query
  // but still rendering against its pre-refetch data for one commit). Every
  // one of those was really an attempt to reason about a snapshot's
  // freshness relative to job creation. `jobQuery`'s key includes the job
  // id, so a new `activeJobId` is a brand-new cache entry with no stale data
  // to race against: `data`/`error` start undefined and `interrupted` stays
  // false until an actual 404 comes back — no invalidation effect needed.
  // A completed-but-undismissed job also 404s after a backend restart (its
  // in-memory record is gone either way), but `handledJobId` already records
  // "this job reached `ingestion_complete`" — so excluding that case keeps a
  // run that actually finished from reading as interrupted.
  const interrupted =
    jobQuery.isError &&
    jobQuery.error instanceof ApiError &&
    jobQuery.error.status === 404 &&
    run.handledJobId !== run.activeJobId

  // A job waiting on the server's concurrency semaphore emits zero frames
  // until it starts running (docint/core/jobs.py) — without this, `status`
  // never leaves `phase: 'idle'` and the whole status block below is gated
  // out, so the run vanishes from view with no card, no spinner, no error.
  const queued = jobQuery.data?.status === 'queued'

  // Post-ingest side effects: select the collection and refresh the owned
  // list once, the moment the active job's log reaches its terminal
  // `ingestion_complete` frame. Guarded by `handledJobId` — a *persisted*
  // store field, not a component ref — so this fires once per job id no
  // matter how many times it's observed: within a mount (a reconnect replay
  // re-delivers the same terminal frame in a new event-log array) and across
  // mounts (navigating away and back, or a reload, while the job's log lives
  // on in the module-level job store and gets replayed by the SSE stream).
  useEffect(() => {
    const last = jobEvents[jobEvents.length - 1]
    if (!run.activeJobId || !last || last.event !== 'ingestion_complete') return
    if (run.handledJobId === run.activeJobId) return
    const jobId = run.activeJobId
    const data = last.data as { collection?: unknown }
    const name = typeof data.collection === 'string' ? data.collection : run.collection
    // Mark handled synchronously, before the async work below, so a
    // re-render triggered by that very write (or any other update while the
    // work is in flight) can't slip past the guard a second time.
    useIngestRunStore.getState().markJobHandled(jobId)
    if (!name) return
    void (async () => {
      await selectCollection(name)
      // Refresh the owned-collections list BEFORE selecting: the Sidebar's
      // reconcile effect clears any active collection not present in that
      // cached list, so selecting a brand-new collection while the list is
      // stale would immediately snap the selection back to null. Awaiting the
      // refetch first ensures the new name is in the list before we select it.
      await qc.invalidateQueries({ queryKey: collectionsKey })
      setSelected(name)
    })()
  }, [jobEvents, run.activeJobId, run.handledJobId, run.collection, qc, setSelected])

  const dismissMutation = useMutation({
    mutationFn: (jobId: string) => dismissIngestJob(jobId),
    onSuccess: (_data, jobId) => {
      useIngestJobsStore.getState().dropJob(jobId)
      run.dismissActive()
      // `dismissActive()` clears `activeJobId` in the same tick, which
      // disables `jobQuery` on the next render — but invalidate its cache
      // entry too, so a dismissed-then-somehow-revisited job id never serves
      // a stale cached snapshot.
      void qc.invalidateQueries({ queryKey: ['ingest-job', jobId] })
    }
  })

  // Re-queuing an interrupted run does NOT re-upload: the batches this run
  // already staged are still in the collection's server-side upload
  // directory (only the in-memory job registry was lost), so re-finalizing
  // directly is what "re-running skips whatever was already indexed" (the
  // banner copy) actually promises. `run.start()` would be a no-op here
  // anyway — `run.files` is always empty by the time a run can be observed
  // as interrupted (cleared the moment the original job was queued).
  const rerunMutation = useMutation({
    mutationFn: () =>
      // Re-run against the collection the interrupted job was actually
      // queued for, not the live form field — the user can edit `run.collection`
      // between the interruption and clicking "Run again". Falls back to the
      // live field only for state persisted before `activeJobCollection`
      // existed.
      createIngestJob({
        collection: run.activeJobCollection ?? run.collection,
        ner: run.ner,
        hate_speech: run.hate
      }),
    onSuccess: ({ job_id }) => {
      // No cache invalidation needed: `adoptJob` points `activeJobId` at the
      // new job id, which `jobQuery` (keyed by that id) picks up as a
      // brand-new, uncached query on its own.
      useIngestRunStore.getState().adoptJob(job_id)
    }
  })

  const busy = run.uploading

  return (
    <div className="p-8">
      <PageHeader title={t('ingest.title')} caption={t('ingest.caption')} />
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(22rem,1fr)_minmax(0,1fr)]">
        <Card className="space-y-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-xs uppercase text-muted-foreground">{t('common.collection')}</span>
            <Input
              list="existing-collections"
              value={run.collection}
              onChange={(e) => run.setCollection(e.target.value)}
              placeholder="my-collection"
            />
            <datalist id="existing-collections">
              {collections?.mine.map((c) => (
                <option key={c} value={c} />
              ))}
            </datalist>
          </label>

          <Dropzone
            disabled={busy}
            onFiles={(v) => {
              setDropError(null)
              run.addFiles(v)
            }}
            onEmpty={() => setDropError(t('ingest.drop_empty'))}
          />

          <FileList
            files={run.files}
            onRemove={(i) => run.removeFile(i)}
            onClear={() => run.clearFiles()}
            labels={{
              files: (n) => t(n === 1 ? 'upload.files_one' : 'upload.files_other', { count: n }),
              clearAll: t('upload.clear_all'),
              remove: t('common.remove')
            }}
          />

          <fieldset className="space-y-1 text-sm" disabled={busy}>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={run.ner} onChange={(e) => run.setNer(e.target.checked)} />
              {t('ingest.opt_ner')}
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={run.hate}
                onChange={(e) => run.setHate(e.target.checked)}
              />
              {t('ingest.opt_hate')}
            </label>
          </fieldset>

          <Button
            variant="primary"
            onClick={() => void run.start(limitBytes, t)}
            disabled={busy || !run.collection || run.files.length === 0}
          >
            {run.uploading ? t('ingest.busy') : t('ingest.button')}
          </Button>

          {(dropError || run.error) && <Banner variant="danger">{dropError ?? run.error}</Banner>}
        </Card>

        <div className="min-w-0 space-y-4">
          {/* Derived from the merged upload+job log (`status.warnings`), not
              `run.warnings` (upload leg only) — a job-stream warning (a
              soft-empty ingest, a failed post-ingest entity resolution) is just
              as actionable and must not go unreported. */}
          {status.warnings.length > 0 && (
            <ul className="text-sm text-[var(--status-amber-fg)] space-y-1" role="alert">
              {status.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}

          {queued && !interrupted && (
            <Card className="text-sm text-muted-foreground" role="status">
              {t('ingest.job_queued')}
            </Card>
          )}

          {interrupted && (
            <Card className="text-sm space-y-2" role="status">
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
                    otherwise strand this banner forever — `activeJobId` is
                    persisted and nothing else clears it. */}
                <Button variant="secondary" size="sm" onClick={() => run.dismissActive()}>
                  {t('ingest.dismiss')}
                </Button>
              </div>
            </Card>
          )}

          {streamLost && (
            <div className="flex items-center gap-3 text-sm text-[var(--status-amber-fg)]" role="alert">
              <span>{t('ingest.stream_lost')}</span>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => useIngestJobsStore.getState().retryStream()}
              >
                {t('ingest.reconnect')}
              </Button>
            </div>
          )}

          {status.phase !== 'idle' && (
            <div className="space-y-2">
              <IngestionStatus status={status} />
              {(status.phase === 'complete' || status.phase === 'error') && run.activeJobId && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => dismissMutation.mutate(run.activeJobId!)}
                >
                  {t('ingest.dismiss')}
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
