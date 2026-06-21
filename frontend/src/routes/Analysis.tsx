import { useEffect, useMemo, useState } from 'react'
import { useHateSpeechPages, useNerSources, useNerStats } from '@/hooks/useNer'
import { useReportDedupeKeys } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { EntityInspector } from '@/components/analysis/EntityInspector'
import { HateSpeechTable } from '@/components/analysis/HateSpeechTable'
import { SummaryPanel } from '@/components/analysis/SummaryPanel'
import { warmCollectionNer } from '@/api/collections'
import { MergeModeToggle } from '@/components/common/MergeModeToggle'
import { cn } from '@/lib/cn'

const TABS = ['NER', 'Hate speech', 'Summary'] as const
type Tab = (typeof TABS)[number]

const keyOf = (text: string | null | undefined, type: string | null | undefined) =>
  `${text ?? ''}::${type ?? ''}`

export function Analysis() {
  const [tab, setTab] = useState<Tab>('NER')
  const collection = useUiStore((s) => s.selectedCollection)
  const mergeMode = useUiStore((s) => s.entityMergeMode)
  // Report-builder context, computed once and threaded into both analysis
  // views so each virtualized row only does a Set lookup (no per-row query).
  const activeReportId = useReportStore((s) => s.activeReportId)
  const reportDedupeKeys = useReportDedupeKeys(activeReportId)

  const stats = useNerStats({
    top_k: 500,
    min_mentions: 1,
    include_relations: false,
    entity_merge_mode: mergeMode
  })

  // Background-warm the NER aggregate as soon as a collection is selected;
  // fire-and-forget so the slow scroll happens off the main interaction.
  useEffect(() => {
    if (!collection) return
    warmCollectionNer().catch(() => {
      /* warm is best-effort */
    })
  }, [collection])

  const entities = useMemo(() => stats.data?.top_entities ?? [], [stats.data])
  const [selectedEntityKey, setSelectedEntityKey] = useState<string | null>(null)

  // Reset selection when the collection changes so we don't keep an entity
  // key that doesn't exist in the new collection's aggregate.
  useEffect(() => {
    setSelectedEntityKey(null)
  }, [collection])

  const ner = useNerSources(selectedEntityKey)
  const findings = useMemo(
    () => (ner.data?.pages ?? []).flatMap((p) => p.items),
    [ner.data]
  )

  const hate = useHateSpeechPages()
  const hateRows = useMemo(
    () => (hate.data?.pages ?? []).flatMap((p) => p.items),
    [hate.data]
  )

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
          {collection && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>Merge mode</span>
              <MergeModeToggle />
            </div>
          )}
          {!collection ? (
            <p className="text-sm text-muted-foreground">
              Select a collection to inspect entities.
            </p>
          ) : stats.isLoading ? (
            <p className="text-sm text-muted-foreground">Loading entities…</p>
          ) : (
            <EntityInspector
              entities={entities}
              selectedKey={selectedEntityKey}
              onSelectEntity={setSelectedEntityKey}
              findings={findings}
              isFetchingFindings={ner.isFetching}
              hasNextPage={!!ner.hasNextPage}
              onLoadMore={() => ner.fetchNextPage()}
              collection={collection}
              keyOf={(e) => keyOf(e.text, e.type)}
              entityMergeMode={mergeMode}
              reportDedupeKeys={reportDedupeKeys}
            />
          )}
        </div>
      )}
      {tab === 'Hate speech' && (
        <HateSpeechTable
          rows={hateRows}
          isFetching={hate.isFetching}
          hasNextPage={!!hate.hasNextPage}
          onLoadMore={() => hate.fetchNextPage()}
          collection={collection ?? ''}
          reportDedupeKeys={reportDedupeKeys}
        />
      )}
      {tab === 'Summary' && <SummaryPanel reportDedupeKeys={reportDedupeKeys} />}
    </div>
  )
}
