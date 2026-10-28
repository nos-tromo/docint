import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { PageHeader } from '@infra/ui'
import { useHateSpeechPages, useNerGraph, useNerSources, useNerStats } from '@/hooks/useNer'
import { useReportDedupeKeys } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { selectEntityKeyFor, useAnalysisUiStore } from '@/stores/analysisUi'
import { useTopicFocusStore } from '@/stores/topicFocus'
import { EntitySelect } from '@/components/analysis/EntitySelect'
import { EntityGraph } from '@/components/analysis/EntityGraph'
import { EntityFindingsTable } from '@/components/analysis/EntityFindingsTable'
import { HateSpeechTable } from '@/components/analysis/HateSpeechTable'
import { SummaryPanel } from '@/components/analysis/SummaryPanel'
import { TopicsPanel } from '@/components/analysis/TopicsPanel'
import { warmCollectionNer } from '@/api/collections'
import type { TopicSummary } from '@/api/topics'
import { useConfig } from '@/hooks/useConfig'
import { resolveGraphTopK } from '@/lib/graphTopK'
import { entityKey, resolveEntityKey } from '@/lib/entityKey'
import { ENTITY_MERGE_MODE, type NerEntityRow } from '@/api/types'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

const TAB_IDS = ['ner', 'hate', 'summary', 'topics'] as const
type Tab = (typeof TAB_IDS)[number]

const NER_VIEW_IDS = ['table', 'graph'] as const
type NerView = (typeof NER_VIEW_IDS)[number]

const keyOf = entityKey

/**
 * Scroll a tab strip sideways just far enough to show its current tab.
 *
 * Only the strip moves: `scrollIntoView` would also scroll the page to it.
 *
 * @param strip - The scrolling strip holding the tabs.
 */
function revealCurrentTab(strip: HTMLElement): void {
  const tab = strip.querySelector('[aria-current]')
  if (!tab) return
  const box = strip.getBoundingClientRect()
  const rect = tab.getBoundingClientRect()
  if (rect.right > box.right) strip.scrollLeft += rect.right - box.right
  else if (rect.left < box.left) strip.scrollLeft -= box.left - rect.left
}

export function Analysis() {
  const t = useT()
  const navigate = useNavigate()
  const tab = useAnalysisUiStore((s) => s.tab)
  const setTab = useAnalysisUiStore((s) => s.setTab)
  const nerView = useAnalysisUiStore((s) => s.nerView)
  const setNerView = useAnalysisUiStore((s) => s.setNerView)
  const collection = useUiStore((s) => s.selectedCollection)
  const owner = useUiStore((s) => s.selectedOwner)
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
  }, [collection, owner])

  const entities = useMemo(() => stats.data?.top_entities ?? [], [stats.data])
  const selectedEntityKey = useAnalysisUiStore(selectEntityKeyFor(collection))
  const setEntity = useAnalysisUiStore((s) => s.setEntity)
  const setSelectedEntityKey = useCallback(
    (key: string | null) => setEntity(key, collection),
    [setEntity, collection]
  )

  // A key can arrive from outside this screen — a chat entity pill, a graph
  // click — built from a raw chunk surface, while the aggregate here is merged
  // orthographically. Canonicalize it against the loaded aggregate so the
  // findings query and the dropdown agree on which entity is selected; a key
  // the aggregate cannot place is left alone rather than guessed at.
  useEffect(() => {
    if (!selectedEntityKey || entities.length === 0) return
    const canonical = resolveEntityKey(selectedEntityKey, entities)
    if (canonical && canonical !== selectedEntityKey) setSelectedEntityKey(canonical)
  }, [entities, selectedEntityKey, setSelectedEntityKey])

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

  // The tab is remembered across sessions, so one saved while topics were on
  // reads as the entities tab while the deployment has them off.
  const topicsEnabled = cfg.data?.topics_enabled === true
  const tabIds = topicsEnabled ? TAB_IDS : TAB_IDS.filter((id) => id !== 'topics')
  const activeTab: Tab = tab === 'topics' && !topicsEnabled ? 'ner' : tab
  const tabStripRef = useRef<HTMLDivElement>(null)
  const setTopicFocus = useTopicFocusStore((s) => s.setFocus)
  // Stable, so a frame of a running topics job never re-renders the topic rows: the collection is
  // read when asked rather than closed over.
  const askInChat = useCallback(
    (topic: TopicSummary, runId: string) => {
      const { selectedCollection, selectedOwner, setCurrentSessionId } = useUiStore.getState()
      if (!selectedCollection) return
      // A topic gets a chat of its own: dropped here rather than by the chat
      // route, so the chat never shows the open session's topic first.
      setCurrentSessionId(null)
      setTopicFocus({
        collection: selectedCollection,
        owner: selectedOwner,
        runId,
        topicId: topic.id,
        label: topic.label
      })
      navigate('/chat')
    },
    [setTopicFocus, navigate]
  )

  // A remembered tab past a narrow screen's edge is scrolled into view.
  useEffect(() => {
    const strip = tabStripRef.current
    const current = strip?.querySelector('[aria-current]')
    if (!strip || !current) return
    revealCurrentTab(strip)
    if (typeof ResizeObserver === 'undefined') return
    // The web font widening the tabs, or the page's scrollbar arriving with the
    // content and narrowing the strip, can push the current tab back out.
    const observer = new ResizeObserver(() => revealCurrentTab(strip))
    observer.observe(strip)
    observer.observe(current)
    return () => observer.disconnect()
  }, [activeTab])

  const tabLabel: Record<Tab, string> = {
    ner: t('analysis.tab_ner'),
    hate: t('analysis.tab_hate'),
    summary: t('analysis.tab_summary'),
    topics: t('analysis.tab_topics')
  }
  const nerViewLabel: Record<NerView, string> = {
    table: t('entities.view_table'),
    graph: t('entities.view_graph')
  }

  return (
    <div className="p-8 space-y-4">
      <PageHeader title={t('analysis.title')} caption={t('analysis.caption')} />
      <nav className="border-b border-border">
        {/* Scrolls rather than wraps, which would float a row's underline; the strip
            rather than each tab overlaps the border, so the scroll box clips no tab. */}
        <div ref={tabStripRef} className="-mb-px flex gap-1 overflow-x-auto [scrollbar-width:thin] sm:gap-2">
          {tabIds.map((id) => (
            <button
              key={id}
              type="button"
              aria-current={activeTab === id || undefined}
              onClick={() => setTab(id)}
              className={cn(
                // The focus outline is drawn inside the tab, where the scrolling strip cannot clip it.
                'shrink-0 whitespace-nowrap px-2 py-2 text-sm border-b-2 sm:px-3',
                'focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-primary',
                activeTab === id ? 'border-foreground' : 'border-transparent text-muted-foreground'
              )}
            >
              {tabLabel[id]}
            </button>
          ))}
        </div>
      </nav>
      {activeTab === 'ner' && (
        <div className="space-y-3">
          {collection && (
            // Full width, halves: the two views are peers, and the pair now
            // spans exactly the findings table below it — which is simply
            // 100% of this column, so `w-full` lands flush by construction.
            // The wrapper this used to sit in had one child and a
            // `justify-between` left over from whatever once sat beside it.
            <div
              role="group"
              aria-label={t('entities.view_group_aria')}
              className="flex w-full overflow-hidden rounded-md border border-border text-sm"
            >
              {NER_VIEW_IDS.map((v) => (
                <button
                  key={v}
                  type="button"
                  aria-pressed={nerView === v}
                  onClick={() => setNerView(v)}
                  className={cn(
                    'flex-1 px-3 py-1.5 text-center transition-colors',
                    nerView === v
                      ? 'bg-muted text-foreground'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                >
                  {nerViewLabel[v]}
                </button>
              ))}
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
      {activeTab === 'hate' && (
        <HateSpeechTable
          rows={hateRows}
          isFetching={hate.isFetching}
          hasNextPage={!!hate.hasNextPage}
          onLoadMore={() => hate.fetchNextPage()}
          collection={collection ?? ''}
          reportDedupeKeys={reportDedupeKeys}
        />
      )}
      {activeTab === 'summary' && <SummaryPanel reportDedupeKeys={reportDedupeKeys} />}
      {activeTab === 'topics' && <TopicsPanel onAsk={askInChat} reportDedupeKeys={reportDedupeKeys} />}
    </div>
  )
}
