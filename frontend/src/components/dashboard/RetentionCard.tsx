import { useId } from 'react'
import { Card, WarningIcon } from '@infra/ui'
import { useCollectionsRetention, useRetentionWindow } from '@/hooks/useRetention'
import { formatDate } from '@/lib/formatDate'
import { entryKey } from '@/lib/collectionEntries'
import { cn } from '@/lib/cn'
import { useLang, useT } from '@/i18n/LanguageContext'

/** Rows shown before the rest are only counted: the soonest deletions are the ones that matter. */
const SHOWN = 8

/**
 * When each collection is deleted for inactivity (`COLLECTION_RETENTION`).
 *
 * Renders nothing while retention is off. The collection in use is always
 * freshly active, so the rows that matter here are the others — the ones
 * nobody has opened in a while — which is why this lives on the Dashboard
 * rather than beside the active collection.
 */
export function RetentionCard() {
  const t = useT()
  const lang = useLang()
  const headingId = useId()
  const retention = useRetentionWindow()
  const { data } = useCollectionsRetention()
  if (!retention || !data) return null

  const rows = data.collections ?? []
  return (
    <Card role="region" aria-labelledby={headingId}>
      <h2 id={headingId} className="text-lg font-semibold text-primary">
        {t('dashboard.retention_title')}
      </h2>
      <p className="mt-1 text-sm text-muted-foreground">
        {t('dashboard.retention_rule', { months: Number.parseInt(retention, 10) })}
      </p>
      {rows.length === 0 ? (
        <div className="mt-3 text-sm text-muted-foreground">{t('dashboard.retention_empty')}</div>
      ) : (
        <ul className="mt-3 space-y-1 text-sm">
          {rows.slice(0, SHOWN).map((row) => (
            <li key={entryKey(row)} className="flex items-center justify-between gap-3">
              <span className="min-w-0 truncate">
                {row.name}
                {row.owner ? t('common.owned_by_suffix', { owner: row.owner }) : ''}
              </span>
              <span
                className={cn(
                  'flex shrink-0 items-center gap-1.5 tabular-nums',
                  row.warning ? 'text-[var(--status-amber-fg)]' : 'text-muted-foreground'
                )}
              >
                {row.warning && (
                  <span
                    role="img"
                    aria-label={t('dashboard.retention_warning')}
                    title={t('dashboard.retention_warning')}
                    className="inline-flex shrink-0"
                  >
                    <WarningIcon className="size-3.5" />
                  </span>
                )}
                {row.expires_at
                  ? t('dashboard.retention_due', { date: formatDate(row.expires_at, lang) })
                  : t('dashboard.retention_never')}
              </span>
            </li>
          ))}
        </ul>
      )}
      {rows.length > SHOWN && (
        <div className="mt-2 text-xs text-muted-foreground">
          {t('dashboard.retention_more', { count: rows.length - SHOWN })}
        </div>
      )}
    </Card>
  )
}
