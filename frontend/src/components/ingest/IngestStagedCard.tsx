import { useQuery } from '@tanstack/react-query'
import { Card, WarningIcon } from '@infra/ui'
import { getStagedBatch } from '@/api/jobs'
import { formatBytes } from '@/lib/ingestStatus'
import { useT } from '@/i18n/LanguageContext'

/** How often to re-ask while the server is still preprocessing the batch. */
const PREPROCESS_POLL_INTERVAL_MS = 5_000

interface IngestStagedCardProps {
  /** The logical collection whose staged batch to describe. */
  collection: string
}

/**
 * What an interrupted upload left on the server, and how to finish it.
 *
 * An upload stages bytes; `POST /ingest/finalize` queues the job. A browser
 * that dies in between — a closed tab, a hung page — leaves a staged batch
 * that no job describes, so every screen showed nothing while the server went
 * on reading the files.
 *
 * This card reports that state; it does not offer to ingest it. The only way
 * to reach it is an upload that stopped early, so what is on disk is a
 * fraction of what the user picked, and the client no longer knows what the
 * whole was — an ingest started here would silently index an arbitrary subset.
 * The recovery is to finish the upload: picking the same folder again sends
 * only the files listed nowhere here (see `filesNotYetStaged`).
 *
 * @param collection - The collection to describe.
 */
export function IngestStagedCard({ collection }: IngestStagedCardProps) {
  const t = useT()

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

  const data = staged.data
  if (!data || data.files === 0) return null

  const { running, queued } = data.preprocess
  const working = running + queued > 0

  return (
    <Card className="space-y-3 border-[var(--status-amber-fg)]/40">
      <div className="flex items-baseline justify-between gap-2">
        <span className="flex items-center gap-2 text-sm text-[var(--status-amber-fg)]">
          <WarningIcon className="size-4 shrink-0" />
          {t('ingest.staged_title', { collection })}
        </span>
        <span className="tabular-nums text-xs text-muted-foreground">
          {t('upload.files_other', { count: data.files })} · {formatBytes(data.bytes)}
        </span>
      </div>

      <p className="text-sm text-[var(--status-amber-fg)]">{t('ingest.staged_incomplete')}</p>

      {data.partial > 0 && (
        <p className="text-xs text-muted-foreground">{t('ingest.staged_partial')}</p>
      )}

      <p className="text-xs text-muted-foreground">
        {working
          ? t('ingest.staged_preprocessing', { running, queued })
          : t('ingest.staged_idle')}
      </p>
    </Card>
  )
}
