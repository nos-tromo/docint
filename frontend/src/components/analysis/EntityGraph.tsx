import { useCallback, useMemo, useRef, useState } from 'react'
import {
  Button,
  ForceGraph,
  downloadText,
  toGraphHtml,
  toGraphJson,
  toGraphML,
  type ForceGraphHandle
} from '@infra/ui'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'
import { ENTITY_EDGE_STYLES, legendForNodes, nodeStylesForTypes, toEntityForceGraph } from '@/lib/entityGraphElements'
import { GraphTopKControl } from './GraphTopKControl'

interface Props {
  nodes: NerGraphNode[]
  edges: NerGraphEdge[]
  selectedKey: string | null
  /** `key === null` means the selection was cleared (e.g. background click). */
  onSelectEntity: (key: string | null) => void
  /** Map a graph node to the `${text}::${type}` selection key used elsewhere. */
  keyForNode: (n: NerGraphNode) => string
  isLoading?: boolean
  /**
   * Node-count control, rendered inline with the other graph controls. The
   * parent owns this state (it drives the graph fetch + is persisted), so it is
   * threaded in rather than held locally. Omit the handlers to hide the control.
   */
  nodeCount?: number
  nodeCountMax?: number
  onNodeCountChange?: (n: number) => void
  /** Reset the node count to the deploy default (env `NER_GRAPH_TOP_K`). */
  onResetNodeCount?: () => void
}

/**
 * Thin adapter around the shared `@infra/ui` `ForceGraph` primitive. Maps
 * docint's NER graph payload (`NerGraphNode`/`NerGraphEdge`) onto the
 * primitive's `{nodes, edges}` shape (`lib/entityGraphElements.ts`) and
 * bridges docint's single-entity selection model (`${text}::${type}` keys)
 * onto the primitive's node-id-based, multi-select-capable selection API.
 *
 * The force simulation, zoom/pan, drag, min-edges filter, edge-length
 * spread, maximize, and fit-to-view controls all now live in `ForceGraph`
 * itself — this component only owns the docint-specific node-count control
 * and the id↔key selection bridge.
 */
export function EntityGraph({
  nodes,
  edges,
  selectedKey,
  onSelectEntity,
  keyForNode,
  isLoading,
  nodeCount,
  nodeCountMax,
  onNodeCountChange,
  onResetNodeCount
}: Props) {
  // View-only node removal (mirrors chorus's expand-on-click graphs): removed
  // ids are local component state, never sent upstream. A fresh `nodes` array
  // (new fetch, top-K change, merge-mode switch) resets it, so removal never
  // outlives the payload it was applied to.
  const [removedIds, setRemovedIds] = useState<ReadonlySet<string>>(new Set())
  const [prevNodes, setPrevNodes] = useState(nodes)
  if (nodes !== prevNodes) {
    setPrevNodes(nodes)
    setRemovedIds(new Set())
  }

  const visibleNodes = useMemo(
    () => (removedIds.size === 0 ? nodes : nodes.filter((n) => !removedIds.has(n.id))),
    [nodes, removedIds]
  )
  const visibleEdges = useMemo(
    () =>
      removedIds.size === 0
        ? edges
        : edges.filter((e) => !removedIds.has(e.source) && !removedIds.has(e.target)),
    [edges, removedIds]
  )

  const fg = useMemo(() => toEntityForceGraph(visibleNodes, visibleEdges), [visibleNodes, visibleEdges])

  const nodeStyles = useMemo(() => {
    const types = new Set(visibleNodes.map((n) => n.type || 'Unlabeled'))
    return nodeStylesForTypes(types)
  }, [visibleNodes])

  const legend = useMemo(() => legendForNodes(visibleNodes), [visibleNodes])

  // Imperative access to the canvas's live layout, for baking positions into
  // the HTML export.
  const apiRef = useRef<ForceGraphHandle | null>(null)

  // Node id -> docint selection key (`${text}::${type}`), used both to derive
  // `selectedIds` from `selectedKey` and to translate a ForceGraph selection
  // back into a key for `onSelectEntity`.
  const keyById = useMemo(() => {
    const m = new Map<string, string>()
    for (const n of nodes) m.set(n.id, keyForNode(n))
    return m
  }, [nodes, keyForNode])

  // The graph's own multi-select state — this, not `selectedKey`, drives
  // `<ForceGraph selectedIds>`. Docint's findings panel is single-entity, so
  // it can't represent a marquee/shift-click set; the graph canvas can, and
  // this state is what lets it keep all of them highlighted instead of
  // snapping back to one.
  const [graphSelection, setGraphSelection] = useState<string[]>([])

  // The key corresponding to the last id in `graphSelection` (or null when
  // empty) — used to tell an external `selectedKey` change (EntitySelect
  // dropdown, a findings-table pick) apart from one the graph itself just
  // caused via `onSelectEntity`.
  const graphSelectionKey = useMemo(() => {
    if (graphSelection.length === 0) return null
    return keyById.get(graphSelection[graphSelection.length - 1]) ?? null
  }, [graphSelection, keyById])

  // The last `selectedKey` value this adapter has already reconciled
  // `graphSelection` against. Initialized to the mount-time `selectedKey` so
  // a pre-existing selection (e.g. carried over from the table view) is
  // *not* adopted on mount — the graph starts undimmed; only a subsequent
  // change to `selectedKey` triggers a highlight.
  const [syncedKey, setSyncedKey] = useState<string | null>(selectedKey)

  // Render-time state adjustment (same pattern as the `prevNodes` reset
  // above): when `selectedKey` changes to something the graph didn't just
  // report itself, replace `graphSelection` with the matching node (or
  // clear it). When it changed *because* the graph reported it, just record
  // it as synced without touching the selection that's already correct.
  if (selectedKey !== syncedKey) {
    setSyncedKey(selectedKey)
    if (selectedKey !== graphSelectionKey) {
      let matchId: string | null = null
      for (const [id, key] of keyById) {
        if (key === selectedKey) {
          matchId = id
          break
        }
      }
      setGraphSelection(matchId ? [matchId] : [])
    }
  }

  // Docint's findings panel is single-entity: a marquee/shift-click multi
  // selection has no representation there, so the panel follows whichever
  // node was most recently added to the selection. The canvas itself keeps
  // the full set via `graphSelection`. An empty selection (background click)
  // clears both, same as chorus.
  const handleSelectionChange = useCallback(
    (ids: string[]) => {
      setGraphSelection(ids)
      onSelectEntity(ids.length ? (keyById.get(ids[ids.length - 1]) ?? null) : null)
    },
    [keyById, onSelectEntity]
  )

  // View-only removal: hide the given nodes (and their edges) from the
  // canvas. If the currently-selected entity was among them, deselect it so
  // the findings panel doesn't keep showing a vanished node, and drop them
  // from the graph's own multi-selection too. The underlying NER data and
  // any exports of it are untouched — removed nodes return on the next fetch
  // (see the `nodes`-identity effect above).
  const handleDeleteNodes = useCallback(
    (ids: string[]) => {
      setRemovedIds((prev) => {
        const next = new Set(prev)
        for (const id of ids) next.add(id)
        return next
      })
      setGraphSelection((prev) => prev.filter((id) => !ids.includes(id)))
      if (selectedKey && ids.some((id) => keyById.get(id) === selectedKey)) {
        onSelectEntity(null)
      }
    },
    [keyById, selectedKey, onSelectEntity]
  )

  if (!isLoading && nodes.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No entity relationships to graph for this collection.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {(nodeCount != null && onNodeCountChange) || fg.nodes.length > 0 ? (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-border pt-2">
          {nodeCount != null && onNodeCountChange && (
            <>
              <GraphTopKControl
                value={nodeCount}
                max={nodeCountMax ?? nodeCount}
                onChange={onNodeCountChange}
              />
              {onResetNodeCount && (
                <button
                  type="button"
                  onClick={onResetNodeCount}
                  aria-label="Reset node count"
                  title="Reset the node count to the deploy default"
                  className="h-7 px-2 rounded-md border border-border text-xs"
                >
                  Reset count
                </button>
              )}
            </>
          )}
          {fg.nodes.length > 0 && (
            <>
              <Button
                type="button"
                variant="secondary"
                onClick={() =>
                  downloadText(
                    'docint-entity-graph.json',
                    toGraphJson(fg.nodes, fg.edges),
                    'application/json'
                  )
                }
              >
                Export JSON
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={() =>
                  downloadText(
                    'docint-entity-graph.graphml',
                    toGraphML(fg.nodes, fg.edges),
                    'application/xml'
                  )
                }
              >
                Export GraphML
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={() =>
                  downloadText(
                    'docint-entity-graph.html',
                    toGraphHtml({
                      title: 'Entity graph',
                      nodes: fg.nodes,
                      edges: fg.edges,
                      positions: apiRef.current?.getPositions() ?? {},
                      nodeStyles,
                      edgeStyles: ENTITY_EDGE_STYLES,
                      legend
                    }),
                    'text/html'
                  )
                }
              >
                Export HTML
              </Button>
            </>
          )}
        </div>
      ) : null}

      <ForceGraph
        apiRef={apiRef}
        nodes={fg.nodes}
        edges={fg.edges}
        nodeStyles={nodeStyles}
        edgeStyles={ENTITY_EDGE_STYLES}
        selectedIds={graphSelection}
        onSelectionChange={handleSelectionChange}
        onDeleteNodes={handleDeleteNodes}
        statusText={
          isLoading
            ? 'Building entity graph…'
            : 'Scroll to zoom, drag to move, click a node to inspect.'
        }
        legend={legend}
      />
    </div>
  )
}
