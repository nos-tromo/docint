import { useId } from 'react'
import { Card } from '@infra/ui'
import type { RetentionItem } from '@/api/types'
import { useRetentionNotice, useRetentionPeriod } from '@/hooks/useRetention'
import { formatDate } from '@/lib/formatDate'
import { useLang, useT } from '@/i18n/LanguageContext'

/** Rows shown before the rest are only counted: the soonest deletions are the ones that matter. */
const SHOWN = 8

const KIND_KEY = {
  document: 'dashboard.retention_kind_document',
  session: 'dashboard.retention_kind_session',
  report: 'dashboard.retention_kind_report'
} as const

interface Group {
  key: string
  collection: string
  owner: string | null
  items: RetentionItem[]
}

/** Group the shown items by collection, in the order their soonest item comes. */
function groupByCollection(items: RetentionItem[]): Group[] {
  const groups = new Map<string, Group>()
  for (const item of items) {
    const key = JSON.stringify([item.owner, item.collection])
    const group = groups.get(key) ?? { key, collection: item.collection, owner: item.owner, items: [] }
    group.items.push(item)
    groups.set(key, group)
  }
  return [...groups.values()]
}

/**
 * What is deleted, and when (`DOCUMENT_RETENTION_DAYS`).
 *
 * Lists the owner's documents, chats and reports due within the notice
 * period, grouped by collection. Renders nothing while retention is off.
 * Nothing here can keep an item: the wording names the date it goes.
 */
export function RetentionCard() {
  const t = useT()
  const lang = useLang()
  const headingId = useId()
  const period = useRetentionPeriod()
  const { data } = useRetentionNotice()
  if (!period || !data) return null

  const items = (data.items ?? []).filter((item) => item.warning)
  const shown = items.slice(0, SHOWN)
  return (
    <Card role="region" aria-labelledby={headingId}>
      <h2 id={headingId} className="text-lg font-semibold text-primary">
        {t('dashboard.retention_title')}
      </h2>
      <p className="mt-1 text-sm text-muted-foreground">{t('dashboard.retention_rule', { days: period.days })}</p>
      {items.length === 0 ? (
        <div className="mt-3 text-sm text-muted-foreground">
          {t('dashboard.retention_empty', { days: data.notice_days })}
        </div>
      ) : (
        <div className="mt-3 space-y-3">
          {groupByCollection(shown).map((group, index) => (
            <section key={group.key} role="group" aria-labelledby={`${headingId}-g${index}`}>
              <h3 id={`${headingId}-g${index}`} className="truncate text-sm font-medium text-primary">
                {group.collection}
                {group.owner ? t('common.owned_by_suffix', { owner: group.owner }) : ''}
              </h3>
              <ul className="mt-1 space-y-1 text-sm">
                {group.items.map((item) => (
                  <li key={`${item.kind}:${item.id}`} className="flex items-center justify-between gap-3">
                    <span className="flex min-w-0 items-baseline gap-2">
                      <span className="shrink-0 text-xs text-muted-foreground">{t(KIND_KEY[item.kind])}</span>
                      <span className="min-w-0 truncate">{item.name ?? item.id}</span>
                    </span>
                    <span className="shrink-0 tabular-nums text-[var(--status-amber-fg)]">
                      {t('dashboard.retention_due', { date: formatDate(item.expires_at, lang) })}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
      {items.length > SHOWN && (
        <div className="mt-2 text-xs text-muted-foreground">
          {t('dashboard.retention_more', { count: items.length - SHOWN })}
        </div>
      )}
    </Card>
  )
}
