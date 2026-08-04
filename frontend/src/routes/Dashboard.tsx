import { useState } from 'react'
import { PageHeader } from '@infra/ui'
import { useCollections } from '@/hooks/useCollections'
import { useDocumentsCount } from '@/hooks/useDocuments'
import { useNerStats } from '@/hooks/useNer'
import { useSessions } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { ENTITY_MERGE_MODE } from '@/api/types'
import { KpiCard } from '@/components/common/KpiCard'
import { TopEntitiesChart } from '@/components/dashboard/TopEntitiesChart'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

export function Dashboard() {
  const t = useT()
  const collection = useUiStore((s) => s.selectedCollection)
  const { data: collections, isError } = useCollections()
  const { data: documentsCount } = useDocumentsCount()
  const { data: sessionsData } = useSessions()
  const [topK, setTopK] = useState(15)
  const [minMentions, setMinMentions] = useState(2)
  const stats = useNerStats({
    top_k: topK,
    min_mentions: minMentions,
    include_relations: false,
    entity_merge_mode: ENTITY_MERGE_MODE
  })

  return (
    <div className="p-8 space-y-6">
      <PageHeader title={t('dashboard.title')} caption={t('dashboard.caption')} />

      <div className="grid grid-cols-4 gap-4">
        <KpiCard
          label={t('dashboard.kpi_backend')}
          value={
            <span className="flex items-center gap-2">
              <span
                data-testid="backend-status-dot"
                aria-hidden="true"
                className={cn(
                  'h-2 w-2 shrink-0 rounded-full',
                  isError
                    ? 'bg-red-400 shadow-[0_0_6px_rgb(248_113_113_/_0.8)]'
                    : 'bg-primary shadow-[0_0_6px_var(--color-primary)]'
                )}
              />
              {isError ? t('dashboard.status_offline') : t('dashboard.status_online')}
            </span>
          }
        />
        <KpiCard label={t('dashboard.kpi_collections')} value={collections?.mine.length ?? null} />
        <KpiCard
          label={t('table.aria_documents')}
          value={collection ? documentsCount?.count ?? null : '—'}
          hint={collection ? t('dashboard.kpi_hint_in', { collection }) : t('dashboard.kpi_hint_select')}
        />
        <KpiCard
          label={t('common.sessions')}
          value={collection ? sessionsData?.sessions.length ?? null : '—'}
          hint={collection ? t('dashboard.kpi_hint_in', { collection }) : t('dashboard.kpi_hint_select')}
        />
      </div>

      <section className="rounded-lg border border-border bg-muted p-4">
        <header className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium">{t('dashboard.top_entities')}</h2>
          <div className="flex items-center gap-3 text-sm">
            <label className="flex items-center gap-2">
              {t('dashboard.top_k_label')}
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-16 bg-muted border border-border rounded-md px-2 py-1"
              />
            </label>
            <label className="flex items-center gap-2">
              {t('dashboard.min_mentions_label')}
              <input
                type="number"
                min={1}
                value={minMentions}
                onChange={(e) => setMinMentions(Number(e.target.value))}
                className="w-16 bg-muted border border-border rounded-md px-2 py-1"
              />
            </label>
          </div>
        </header>
        {!collection ? (
          <div className="text-sm text-muted-foreground">{t('dashboard.select_collection_entities')}</div>
        ) : (
          <TopEntitiesChart data={stats.data?.top_entities ?? []} />
        )}
      </section>

      <section className="rounded-lg border border-border bg-muted p-4">
        <h2 className="text-lg font-medium mb-3">{t('dashboard.recent_sessions')}</h2>
        {!collection ? (
          <div className="text-sm text-muted-foreground">{t('common.select_collection_to_see_chats')}</div>
        ) : (
          <ul className="space-y-1 text-sm">
            {sessionsData?.sessions.slice(0, 10).map((s) => (
              <li key={s.id}>{s.title?.trim() || s.id.slice(0, 8)}</li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
