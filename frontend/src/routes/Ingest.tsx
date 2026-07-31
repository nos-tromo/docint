import { useEffect, useMemo, useState } from 'react'
import { Button, FileList } from '@infra/ui'
import { useQueryClient, useMutation } from '@tanstack/react-query'
import { useIngestRunStore } from '@/stores/ingestRun'
import { useIngestJobsStore, selectJobEvents } from '@/stores/ingestJobs'
import { useIngestJobs, ingestJobsKey } from '@/hooks/useIngestJobs'
import { useIngestDefaults } from '@/hooks/useIngestDefaults'
import { dismissIngestJob, createIngestJob } from '@/api/jobs'
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

export function Ingest() {
  const t = useT()
  const run = useIngestRunStore()
  const jobEvents = useIngestJobsStore(selectJobEvents(run.activeJobId))
  const streamLost = useIngestJobsStore((s) => s.streamLost)
  const { data: jobs } = useIngestJobs()
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

  // A persisted job id the server does not list is an interrupted run: the
  // backend restarted while it was in flight (jobs are in-memory by design).
  // Derived from list membership alone — NOT also gated on "has this job ever
  // produced an SSE frame". An earlier version required `jobEvents.length ===
  // 0` too, reasoning that a job with live evidence can't be interrupted; but
  // a job that emitted `ingestion_started`/progress and *then* the backend
  // restarted still has non-empty `jobEvents` forever (nothing ever clears a
  // job's log once appended), which made that job permanently exempt from
  // ever being flagged interrupted again, no matter how many times the jobs
  // list confirmed it was gone — a worse stuck state than the one the
  // feature exists to catch. The real staleness problem (a job the user just
  // created not yet showing up in a 30s-stale `/ingest/jobs` fetch) is
  // instead fixed directly below, by invalidating that query the moment
  // `activeJobId` changes.
  const interrupted =
    !!run.activeJobId && !!jobs && !jobs.jobs.some((j) => j.job_id === run.activeJobId)

  // Refetch the jobs list the moment a job becomes active (including a fresh
  // one), so `interrupted` above is never computed against a pre-creation
  // snapshot. `createIngestJob` (`POST /ingest/finalize`) resolves only after
  // the backend has registered the job, so this refetch is guaranteed to see
  // it — no race that could reintroduce a false-positive interrupted flag.
  useEffect(() => {
    if (run.activeJobId) void qc.invalidateQueries({ queryKey: ingestJobsKey })
  }, [run.activeJobId, qc])

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
      void qc.invalidateQueries({ queryKey: ingestJobsKey })
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
      createIngestJob({ collection: run.collection, hybrid: true, ner: run.ner, hate_speech: run.hate }),
    onSuccess: ({ job_id }) => {
      useIngestRunStore.getState().adoptJob(job_id)
      void qc.invalidateQueries({ queryKey: ingestJobsKey })
    }
  })

  const busy = run.uploading

  return (
    <div className="p-8 max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">{t('ingest.title')}</h1>

      <label className="flex flex-col gap-1 text-sm max-w-sm">
        <span className="text-xs uppercase text-muted-foreground">{t('common.collection')}</span>
        <input
          list="existing-collections"
          value={run.collection}
          onChange={(e) => run.setCollection(e.target.value)}
          placeholder="my-collection"
          className="bg-muted border border-border rounded-md px-2 py-1 text-sm"
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

      {(dropError || run.error) && (
        <div className="text-[var(--status-red-fg)] text-sm">{dropError ?? run.error}</div>
      )}
      {run.warnings.length > 0 && (
        <ul className="text-amber-400 text-sm space-y-1" role="alert">
          {run.warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}

      {interrupted && (
        <div className="rounded-md border border-border p-3 text-sm space-y-2" role="status">
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
        </div>
      )}

      {streamLost && (
        <div className="flex items-center gap-3 text-sm text-amber-400" role="alert">
          <span>{t('ingest.stream_lost')}</span>
          <button
            type="button"
            className="px-2 py-1 rounded-md border border-border"
            onClick={() => useIngestJobsStore.getState().retryStream()}
          >
            {t('ingest.reconnect')}
          </button>
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
  )
}
