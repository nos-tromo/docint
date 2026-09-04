import { useEffect, useMemo, useState } from 'react'
import { Banner, Button, Card, FileList, Input, PageHeader, ToggleButton } from '@infra/ui'
import { useQueryClient } from '@tanstack/react-query'
import { useIngestRunStore } from '@/stores/ingestRun'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useIngestDefaults } from '@/hooks/useIngestDefaults'
import { ingestJobsKey } from '@/hooks/useIngestJobs'
import { useCollections } from '@/hooks/useCollections'
import { useConfig } from '@/hooks/useConfig'
import { Dropzone } from '@/components/ingest/Dropzone'
import { IngestionStatus } from '@/components/ingest/IngestionStatus'
import { IngestJobList } from '@/components/ingest/IngestJobList'
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
  const streamLost = useIngestJobsStore((s) => s.streamLost)
  const { data: ingestDefaults } = useIngestDefaults()
  const { data: collections } = useCollections()
  const { data: config } = useConfig()
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

  const fileSizes = useMemo(() => {
    const sizes: Record<string, number> = {}
    for (const f of run.files) sizes[f.webkitRelativePath || f.name] = f.size
    return sizes
  }, [run.files])

  // The upload leg only. Its events move to the job they produced the moment
  // that job is queued (stores/ingestRun.ts), so this card describes the
  // transfer in flight and nothing else — and an upload that failed before
  // reaching a job, which has no job card to live on.
  const uploadStatus: IngestStatus = useMemo(
    () => deriveIngestStatus(run.uploadEvents, fileSizes),
    [run.uploadEvents, fileSizes]
  )

  const busy = run.uploading

  return (
    <div className="p-8">
      <PageHeader title={t('ingest.title')} caption={t('ingest.caption')} />
      <div className="space-y-6">
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
            labels={{
              files: (n) => t(n === 1 ? 'upload.files_one' : 'upload.files_other', { count: n }),
              remove: t('common.remove')
            }}
          />

          {/* One row of toggle panels, mirroring Nextext's upload form: what
              a run includes is read as lit and unlit buttons rather than
              hunted for in checkbox marks. `flex-1` shares the span evenly;
              the minimum width makes them wrap instead of crush on a narrow
              screen or in a long-worded locale. The fieldset carries the one
              `disabled` for both. */}
          <fieldset className="flex flex-wrap gap-2" disabled={busy}>
            <ToggleButton
              className="min-w-32 flex-1"
              pressed={run.ner}
              onClick={() => run.setNer(!run.ner)}
            >
              {t('ingest.opt_ner')}
            </ToggleButton>
            <ToggleButton
              className="min-w-32 flex-1"
              pressed={run.hate}
              onClick={() => run.setHate(!run.hate)}
            >
              {t('ingest.opt_hate')}
            </ToggleButton>
          </fieldset>

          {/* Full width: it is the card's one action, and everything above it —
              the collection field, the dropzone, the file list — already runs
              edge to edge. A small left-aligned button under all of that read
              as an afterthought. */}
          <Button
            variant="primary"
            className="w-full"
            onClick={() => {
              void run.start(limitBytes, t).then(() => {
                // The new job exists server-side now; the list is what makes
                // it visible to any other tab (and to this one after a reload).
                void qc.invalidateQueries({ queryKey: ingestJobsKey })
              })
            }}
            disabled={busy || !run.collection || run.files.length === 0}
          >
            {run.uploading ? t('ingest.busy') : t('ingest.button')}
          </Button>

          {(dropError || run.error) && <Banner variant="danger">{dropError ?? run.error}</Banner>}
        </Card>

        <div className="min-w-0 space-y-4">
          {uploadStatus.warnings.length > 0 && (
            <ul className="text-sm text-[var(--status-amber-fg)] space-y-1" role="alert">
              {uploadStatus.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}

          {uploadStatus.phase !== 'idle' && <IngestionStatus status={uploadStatus} />}

          {streamLost && (
            <div className="space-y-2 text-sm text-[var(--status-amber-fg)]" role="alert">
              <p>{t('ingest.stream_lost')}</p>
              <Button
                variant="secondary"
                onClick={() => useIngestJobsStore.getState().retryStream()}
              >
                {t('ingest.reconnect')}
              </Button>
            </div>
          )}

          <IngestJobList />
        </div>
      </div>
    </div>
  )
}
