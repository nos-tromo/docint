import { useState } from 'react'
import { useCollections } from '@/hooks/useCollections'
import { useDocumentsCount } from '@/hooks/useDocuments'
import { useNerStats } from '@/hooks/useNer'
import { useSessions } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { KpiCard } from '@/components/common/KpiCard'
import { MergeModeToggle } from '@/components/common/MergeModeToggle'
import { TopEntitiesChart } from '@/components/dashboard/TopEntitiesChart'
import { cn } from '@/lib/cn'

export function Dashboard() {
  const collection = useUiStore((s) => s.selectedCollection)
  const mergeMode = useUiStore((s) => s.entityMergeMode)
  const { data: collections, isError } = useCollections()
  const { data: documentsCount } = useDocumentsCount()
  const { data: sessionsData } = useSessions()
  const [topK, setTopK] = useState(15)
  const [minMentions, setMinMentions] = useState(2)
  const stats = useNerStats({
    top_k: topK,
    min_mentions: minMentions,
    include_relations: false,
    entity_merge_mode: mergeMode
  })

  return (
    <div className="p-8 space-y-6">
      <h1 className="text-2xl font-semibold">Dashboard</h1>

      <div className="grid grid-cols-4 gap-4">
        <KpiCard
          label="Backend"
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
              {isError ? 'offline' : 'online'}
            </span>
          }
        />
        <KpiCard label="Collections" value={collections?.length ?? null} />
        <KpiCard
          label="Documents"
          value={collection ? documentsCount?.count ?? null : '—'}
          hint={collection ? `in ${collection}` : 'select a collection'}
        />
        <KpiCard label="Sessions" value={sessionsData?.sessions.length ?? null} />
      </div>

      <section className="rounded-lg border border-border bg-zinc-900 p-4">
        <header className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium">Top entities</h2>
          <div className="flex items-center gap-3 text-sm">
            <MergeModeToggle />
            <label className="flex items-center gap-2">
              top-k
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-16 bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
            <label className="flex items-center gap-2">
              min mentions
              <input
                type="number"
                min={1}
                value={minMentions}
                onChange={(e) => setMinMentions(Number(e.target.value))}
                className="w-16 bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
          </div>
        </header>
        {!collection ? (
          <div className="text-sm text-muted-foreground">Select a collection to see entities.</div>
        ) : (
          <TopEntitiesChart data={stats.data?.top_entities ?? []} />
        )}
      </section>

      <section className="rounded-lg border border-border bg-zinc-900 p-4">
        <h2 className="text-lg font-medium mb-3">Recent sessions</h2>
        <ul className="space-y-1 text-sm">
          {sessionsData?.sessions.slice(0, 10).map((s) => (
            <li key={s.id} className="flex justify-between">
              <span>{s.title?.trim() || s.id.slice(0, 8)}</span>
              {s.collection && (
                <span className="text-muted-foreground">{s.collection}</span>
              )}
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
