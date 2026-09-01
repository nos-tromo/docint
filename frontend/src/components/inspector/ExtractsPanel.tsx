import { Button, DeleteButton, DownloadLink } from '@infra/ui'
import { extractDownloadHref } from '@/api/extracts'
import { useDeleteExtract, useExtractJob, useExtracts } from '@/hooks/useExtracts'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'
import type { ExtractRecord } from '@/api/types'

/** Render a byte count the way a downloads list does. */
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`
}

/** Summarize a bundle's contents as a short, translated line. */
function useCountsLabel(): (record: ExtractRecord) => string {
  const t = useT()
  return (record: ExtractRecord) => {
    const counts = record.counts ?? {}
    const parts: string[] = []
    const add = (key: string, label: string) => {
      const value = counts[key]
      if (typeof value === 'number' && value > 0) parts.push(`${value} ${label}`)
    }
    add('documents', t('extract.count_documents'))
    add('media', t('extract.count_media'))
    add('postings', t('extract.count_postings'))
    add('images', t('extract.count_images'))
    add('figures', t('extract.count_figures'))
    return parts.join(' · ')
  }
}

/**
 * The Inspector's extracts panel: build one, then download or delete the
 * bundles already stored for this collection.
 *
 * The listing comes from the server's on-disk store rather than its job
 * registry — jobs are in-memory and finished ones are evicted, so a bundle
 * built before a backend restart must still be downloadable here.
 */
export function ExtractsPanel() {
  const t = useT()
  const collection = useUiStore((s) => s.selectedCollection)
  const { data, isLoading } = useExtracts()
  const { progress, start, running } = useExtractJob()
  const remove = useDeleteExtract()
  const countsLabel = useCountsLabel()
  if (!collection) return null

  const extracts = data?.extracts ?? []
  return (
    <section className="rounded-lg border border-border bg-card p-4 space-y-3">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-medium">{t('extract.title')}</h2>
          <p className="text-xs text-muted-foreground">{t('extract.caption')}</p>
        </div>
        <Button onClick={() => void start()} disabled={running}>
          {t('extract.build')}
        </Button>
      </div>

      {progress && (
        <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs">
          {progress.error ? (
            <span className="text-destructive">{t('extract.failed')}</span>
          ) : progress.totalUnits ? (
            t('extract.running_progress', {
              done: String(progress.rendered ?? 0),
              total: String(progress.totalUnits)
            })
          ) : (
            t('extract.running')
          )}
        </div>
      )}

      {isLoading ? (
        <p className="text-xs text-muted-foreground">{t('common.loading_ellipsis')}</p>
      ) : extracts.length === 0 ? (
        <p className="text-xs text-muted-foreground">{t('extract.none')}</p>
      ) : (
        <ul className="divide-y divide-border">
          {extracts.map((record) => (
            <li key={record.extract_id} className="flex items-center gap-3 py-2 text-xs">
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium">{record.filename}</div>
                <div className="text-muted-foreground">
                  {record.created_at.slice(0, 16).replace('T', ' ')} · {formatSize(record.size)}
                  {countsLabel(record) ? ` · ${countsLabel(record)}` : ''}
                  {record.pdf_skipped ? ` · ${t('extract.pdf_skipped')}` : ''}
                </div>
              </div>
              <DownloadLink
                href={extractDownloadHref(collection, record.extract_id)}
                download={record.filename}
                label={t('extract.download')}
              />
              <DeleteButton
                label={t('extract.delete')}
                onClick={() => {
                  if (confirm(t('extract.delete_confirm'))) remove.mutate(record.extract_id)
                }}
              />
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
