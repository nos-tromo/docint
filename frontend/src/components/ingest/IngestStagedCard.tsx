import { useQueryClient } from '@tanstack/react-query'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Banner, Button, Card } from '@infra/ui'
import { createIngestJob, getStagedBatch } from '@/api/jobs'
import { ingestJobsKey } from '@/hooks/useIngestJobs'
import { useIngestRunStore } from '@/stores/ingestRun'
import { formatBytes } from '@/lib/ingestStatus'
import { useT } from '@/i18n/LanguageContext'

/** How often to re-ask while the server is still preprocessing the batch. */
const PREPROCESS_POLL_INTERVAL_MS = 5_000

interface IngestStagedCardProps {
  /** The logical collection whose staged batch to describe. */
  collection: string
}

/**
 * The files already on the server for a collection, and a way to ingest them.
 *
 * An upload stages bytes; `POST /ingest/finalize` queues the job. A browser
 * that dies in between — a closed tab, a hung page — leaves a staged batch
 * that no job describes, so every screen showed nothing while the server went
 * on reading the files. This card is what makes that state visible and
 * actionable: finalizing over a staged batch is safe at any time, since
 * ingestion is idempotent by file hash.
 *
 * @param collection - The collection to describe.
 */
export function IngestStagedCard({ collection }: IngestStagedCardProps) {
  const t = useT()
  const qc = useQueryClient()

  const staged = useQuery({
    queryKey: ['ingest-staged', collection],
    queryFn: () => getStagedBatch(collection),
    enabled: collection.length > 0,
    retry: false,
    // Only while the pool still holds work: a settled batch changes only when
    // the user uploads again, which refetches this anyway.
    refetchInterval: (query) => {
      const p = query.state.data?.preprocess
      return p && p.running + p.queued > 0 ? PREPROCESS_POLL_INTERVAL_MS : false
    }
  })

  const finalize = useMutation({
    mutationFn: async () => {
      const { ner, hate } = useIngestRunStore.getState()
      return createIngestJob({ collection, ner, hate_speech: hate })
    },
    onSuccess: ({ job_id }) => {
      useIngestRunStore.getState().trackJob(job_id, collection)
      void qc.invalidateQueries({ queryKey: ingestJobsKey })
      void qc.invalidateQueries({ queryKey: ['ingest-staged', collection] })
    }
  })

  const data = staged.data
  if (!data || data.files === 0) return null

  const { running, queued } = data.preprocess
  const working = running + queued > 0

  return (
    <Card className="space-y-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm text-foreground">
          {t('ingest.staged_title', { collection })}
        </span>
        <span className="tabular-nums text-xs text-muted-foreground">
          {t('upload.files_other', { count: data.files })} · {formatBytes(data.bytes)}
        </span>
      </div>

      <p className="text-xs text-muted-foreground">
        {working
          ? t('ingest.staged_preprocessing', { running, queued })
          : t('ingest.staged_idle')}
      </p>

      <Button
        variant="secondary"
        className="w-full"
        onClick={() => finalize.mutate()}
        disabled={finalize.isPending}
      >
        {finalize.isPending ? t('ingest.busy') : t('ingest.staged_ingest')}
      </Button>

      {finalize.isError && <Banner variant="danger">{t('ingest.failed_default')}</Banner>}
    </Card>
  )
}
