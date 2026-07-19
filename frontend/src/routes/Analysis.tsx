import { useCallback, useEffect, useMemo, useState } from 'react'
import { useHateSpeechPages, useNerGraph, useNerSources, useNerStats } from '@/hooks/useNer'
import { useReportDedupeKeys } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { EntitySelect } from '@/components/analysis/EntitySelect'
import { EntityGraph } from '@/components/analysis/EntityGraph'
import { EntityFindingsTable } from '@/components/analysis/EntityFindingsTable'
import { HateSpeechTable } from '@/components/analysis/HateSpeechTable'
import { SummaryPanel } from '@/components/analysis/SummaryPanel'
import { warmCollectionNer } from '@/api/collections'
import { MergeModeToggle } from '@/components/common/MergeModeToggle'
import { useConfig } from '@/hooks/useConfig'
import { resolveGraphTopK } from '@/lib/graphTopK'
import type { NerEntityRow } from '@/api/types'
import { cn } from '@/lib/cn'

const TABS = ['NER', 'Hate speech', 'Summary'] as const
type Tab = (typeof TABS)[number]

const NER_VIEWS = [
  { value: 'table', label: 'Table' },
  { value: 'graph', label: 'Graph' }
] as const
type NerView = (typeof NER_VIEWS)[number]['value']

const keyOf = (text: string | null | undefined, type: string | null | undefined) =>
  `${text ?? ''}::${type ?? ''}`

export function Analysis() {
  const [tab, setTab] = useState<Tab>('NER')
  const [nerView, setNerView] = useState<NerView>('table')
  const collection = useUiStore((s) => s.selectedCollection)
  const mergeMode = useUiStore((s) => s.entityMergeMode)
  const cfg = useConfig()
  const graphTopK = useUiStore((s) => s.graphTopK)
  const setGraphTopK = useUiStore((s) => s.setGraphTopK)
  const effectiveTopK = resolveGraphTopK(graphTopK, cfg.data)
  const graphMax = cfg.data?.graph_max_top_k ?? 500
  // Reset the node count to the deploy default by clearing the user override;
  // resolveGraphTopK then falls back to the server's graph_top_k (env
  // `NER_GRAPH_TOP_K`, default 80). Stable so it doesn't rebuild the graph sim.
  const resetGraphTopK = useCallback(() => setGraphTopK(null), [setGraphTopK])
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
    warmCollectionNer(collection).catch(() => {
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

  // Seed a sensible default selection (the top entity) once the list loads, so
  // the findings panel and dropdown aren't empty on arrival. Scoped to the
  // table view: the graph view dims every non-neighbor of the active
  // selection, so auto-selecting there would leave the graph permanently
  // washed-out on first arrival instead of starting bright. (The
  // `EntityGraph` adapter separately avoids adopting a selection that was
  // already set before it mounted, so switching table -> graph after this
  // effect ran in table view doesn't retroactively dim it either.)
  useEffect(() => {
    if (nerView !== 'table' || selectedEntityKey || entities.length === 0) return
    const top = entities.find((e) => (e.text ?? '').trim().length > 0)
    if (top) setSelectedEntityKey(keyOf(top.text, top.type))
  }, [entities, selectedEntityKey, nerView])

  // The selected entity row (for highlight terms / labels / CSV). Falls back to
  // a minimal row parsed from the key so graph clicks on entities outside the
  // currently-loaded stats page still drive the findings table.
  const selectedEntity = useMemo<NerEntityRow | null>(() => {
    if (!selectedEntityKey) return null
    const hit = entities.find((e) => keyOf(e.text, e.type) === selectedEntityKey)
    if (hit) return hit
    const idx = selectedEntityKey.lastIndexOf('::')
    if (idx < 0) return null
    return {
      text: selectedEntityKey.slice(0, idx),
      type: selectedEntityKey.slice(idx + 2),
      mentions: 0
    }
  }, [entities, selectedEntityKey])

  const ner = useNerSources(selectedEntityKey)
  const findings = useMemo(
    () => (ner.data?.pages ?? []).flatMap((p) => p.items),
    [ner.data]
  )

  // Graph payload is only fetched while the graph view is active.
  const graph = useNerGraph({ topKNodes: effectiveTopK, enabled: nerView === 'graph' })

  const hate = useHateSpeechPages()
  const hateRows = useMemo(
    () => (hate.data?.pages ?? []).flatMap((p) => p.items),
    [hate.data]
  )

  // Stable selection-key builders: an inline arrow would rebuild the graph
  // simulation (resetting its layout) on every Analysis re-render.
  const entityKeyOf = useCallback(
    (e: { text: string; type: string }) => keyOf(e.text, e.type),
    []
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
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div
                role="group"
                aria-label="Entity view"
                className="inline-flex overflow-hidden rounded-md border border-border text-sm"
              >
                {NER_VIEWS.map((v) => (
                  <button
                    key={v.value}
                    type="button"
                    aria-pressed={nerView === v.value}
                    onClick={() => setNerView(v.value)}
                    className={cn(
                      'px-3 py-1 transition-colors',
                      nerView === v.value
                        ? 'bg-zinc-800 text-foreground'
                        : 'text-muted-foreground hover:text-foreground'
                    )}
                  >
                    {v.label}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>Merge mode</span>
                <MergeModeToggle />
              </div>
            </div>
          )}
          {!collection ? (
            <p className="text-sm text-muted-foreground">
              Select a collection to inspect entities.
            </p>
          ) : stats.isLoading ? (
            <p className="text-sm text-muted-foreground">Loading entities…</p>
          ) : entities.length === 0 ? (
            <p className="text-sm text-muted-foreground">No entities found in this collection.</p>
          ) : (
            <div className="space-y-4">
              {nerView === 'table' ? (
                <EntitySelect
                  entities={entities}
                  selectedKey={selectedEntityKey}
                  onSelectEntity={setSelectedEntityKey}
                  keyOf={entityKeyOf}
                />
              ) : (
                <EntityGraph
                  nodes={graph.data?.nodes ?? []}
                  edges={graph.data?.edges ?? []}
                  selectedKey={selectedEntityKey}
                  onSelectEntity={setSelectedEntityKey}
                  keyForNode={entityKeyOf}
                  isLoading={graph.isLoading}
                  nodeCount={effectiveTopK}
                  nodeCountMax={graphMax}
                  onNodeCountChange={setGraphTopK}
                  onResetNodeCount={resetGraphTopK}
                  exportName={collection ?? undefined}
                />
              )}
              <EntityFindingsTable
                selected={selectedEntity}
                findings={findings}
                isFetchingFindings={ner.isFetching}
                hasNextPage={!!ner.hasNextPage}
                onLoadMore={() => ner.fetchNextPage()}
                collection={collection}
                entityMergeMode={mergeMode}
                reportDedupeKeys={reportDedupeKeys}
              />
            </div>
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
