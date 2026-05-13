import { useState } from 'react'
import { useHateSpeech, useNerStats } from '@/hooks/useNer'
import { NerTable } from '@/components/analysis/NerTable'
import { HateSpeechTable, type HateSpeechRow } from '@/components/analysis/HateSpeechTable'
import { SummaryPanel } from '@/components/analysis/SummaryPanel'
import { cn } from '@/lib/cn'

const TABS = ['NER', 'Hate speech', 'Summary'] as const
type Tab = (typeof TABS)[number]

export function Analysis() {
  const [tab, setTab] = useState<Tab>('NER')
  const stats = useNerStats({ top_k: 100, min_mentions: 1, include_relations: true })
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
      {tab === 'NER' && <NerTable rows={stats.data?.top_entities ?? []} />}
      {tab === 'Hate speech' && (
        <HateSpeechTable rows={(hate.data?.results ?? []) as HateSpeechRow[]} />
      )}
      {tab === 'Summary' && <SummaryPanel />}
    </div>
  )
}
