import { useCallback, useEffect, useMemo } from 'react'
import { useHateSpeechPages, useNerGraph, useNerSources, useNerStats } from '@/hooks/useNer'
import { useReportDedupeKeys } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { selectEntityKeyFor, useAnalysisUiStore } from '@/stores/analysisUi'
import { EntitySelect } from '@/components/analysis/EntitySelect'
import { EntityGraph } from '@/components/analysis/EntityGraph'
import { EntityFindingsTable } from '@/components/analysis/EntityFindingsTable'
import { HateSpeechTable } from '@/components/analysis/HateSpeechTable'
import { SummaryPanel } from '@/components/analysis/SummaryPanel'
import { warmCollectionNer } from '@/api/collections'
import { useConfig } from '@/hooks/useConfig'
import { resolveGraphTopK } from '@/lib/graphTopK'
import { ENTITY_MERGE_MODE, type NerEntityRow } from '@/api/types'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

const TAB_IDS = ['ner', 'hate', 'summary'] as const
type Tab = (typeof TAB_IDS)[number]

const NER_VIEW_IDS = ['table', 'graph'] as const
type NerView = (typeof NER_VIEW_IDS)[number]

const keyOf = (text: string | null | undefined, type: string | null | undefined) =>
  `${text ?? ''}::${type ?? ''}`

export function Analysis() {
  const t = useT()
  const tab = useAnalysisUiStore((s) => s.tab)
  const setTab = useAnalysisUiStore((s) => s.setTab)
  const nerView = useAnalysisUiStore((s) => s.nerView)
  const setNerView = useAnalysisUiStore((s) => s.setNerView)
  const collection = useUiStore((s) => s.selectedCollection)
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
    entity_merge_mode: ENTITY_MERGE_MODE
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
  const selectedEntityKey = useAnalysisUiStore(selectEntityKeyFor(collection))
  const setEntity = useAnalysisUiStore((s) => s.setEntity)
  const setSelectedEntityKey = useCallback(
    (key: string | null) => setEntity(key, collection),
    [setEntity, collection]
  )

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
  }, [entities, selectedEntityKey, nerView, setSelectedEntityKey])

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

  const tabLabel: Record<Tab, string> = {
    ner: t('analysis.tab_ner'),
    hate: t('analysis.tab_hate'),
    summary: t('analysis.tab_summary')
  }
  const nerViewLabel: Record<NerView, string> = {
    table: t('entities.view_table'),
    graph: t('entities.view_graph')
  }

  return (
    <div className="p-8 space-y-4">
      <h1 className="text-2xl font-semibold">{t('analysis.title')}</h1>
      <nav className="flex gap-2 border-b border-border">
        {TAB_IDS.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={cn(
              'px-3 py-2 text-sm -mb-px border-b-2',
              tab === id ? 'border-foreground' : 'border-transparent text-muted-foreground'
            )}
          >
            {tabLabel[id]}
          </button>
        ))}
      </nav>
      {tab === 'ner' && (
        <div className="space-y-3">
          {collection && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div
                role="group"
                aria-label={t('entities.view_group_aria')}
                className="inline-flex overflow-hidden rounded-md border border-border text-sm"
              >
                {NER_VIEW_IDS.map((v) => (
                  <button
                    key={v}
                    type="button"
                    aria-pressed={nerView === v}
                    onClick={() => setNerView(v)}
                    className={cn(
                      'px-3 py-1 transition-colors',
                      nerView === v
                        ? 'bg-muted text-foreground'
                        : 'text-muted-foreground hover:text-foreground'
                    )}
                  >
                    {nerViewLabel[v]}
                  </button>
                ))}
              </div>
            </div>
          )}
          {!collection ? (
            <p className="text-sm text-muted-foreground">{t('entities.select_collection')}</p>
          ) : stats.isLoading ? (
            <p className="text-sm text-muted-foreground">{t('entities.loading')}</p>
          ) : entities.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t('entities.empty')}</p>
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
                entityMergeMode={ENTITY_MERGE_MODE}
                reportDedupeKeys={reportDedupeKeys}
              />
            </div>
          )}
        </div>
      )}
      {tab === 'hate' && (
        <HateSpeechTable
          rows={hateRows}
          isFetching={hate.isFetching}
          hasNextPage={!!hate.hasNextPage}
          onLoadMore={() => hate.fetchNextPage()}
          collection={collection ?? ''}
          reportDedupeKeys={reportDedupeKeys}
        />
      )}
      {tab === 'summary' && <SummaryPanel reportDedupeKeys={reportDedupeKeys} />}
    </div>
  )
}
