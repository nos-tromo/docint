import { useState } from 'react'
import { useHateSpeech, useNer, useNerStats } from '@/hooks/useNer'
import { useUiStore } from '@/stores/ui'
import { EntityInspector } from '@/components/analysis/EntityInspector'
import { HateSpeechTable } from '@/components/analysis/HateSpeechTable'
import { SummaryPanel } from '@/components/analysis/SummaryPanel'
import { cn } from '@/lib/cn'

const TABS = ['NER', 'Hate speech', 'Summary'] as const
type Tab = (typeof TABS)[number]

export function Analysis() {
  const [tab, setTab] = useState<Tab>('NER')
  const collection = useUiStore((s) => s.selectedCollection)
  // Stats give us the entity dropdown (aggregated, ranked); /collections/ner
  // gives us the raw mention rows we filter client-side to show the chunks
  // for the picked entity, mirroring the deleted Streamlit drill-down.
  const stats = useNerStats({ top_k: 500, min_mentions: 1, include_relations: false })
  const sources = useNer()
  const hate = useHateSpeech()

  return (
    <div className="p-8 space-y-4">
      <h1 className="text-2xl font-semibold">Analysis</h1>
      <nav className="flex gap-2 border-b border-border">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={cn(
              'px-3 py-2 text-sm -mb-px border-b-2',
              tab === t ? 'border-foreground' : 'border-transparent text-muted-foreground'
            )}
          >
            {t}
          </button>
        ))}
      </nav>
      {tab === 'NER' && (
        <div className="space-y-3">
          {!collection ? (
            <p className="text-sm text-muted-foreground">
              Select a collection to inspect entities.
            </p>
          ) : sources.isLoading || stats.isLoading ? (
            <p className="text-sm text-muted-foreground">Loading entities…</p>
          ) : (
            <EntityInspector
              entities={stats.data?.top_entities ?? []}
              sources={sources.data?.sources ?? []}
            />
          )}
        </div>
      )}
      {tab === 'Hate speech' && (
        <HateSpeechTable rows={hate.data?.results ?? []} />
      )}
      {tab === 'Summary' && <SummaryPanel />}
    </div>
  )
}
