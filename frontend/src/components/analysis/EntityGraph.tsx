import { useCallback, useMemo, useState } from 'react'
import { ForceGraph } from '@infra/ui'
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

  // Node id -> docint selection key (`${text}::${type}`), used both to derive
  // `selectedIds` from `selectedKey` and to translate a ForceGraph selection
  // back into a key for `onSelectEntity`.
  const keyById = useMemo(() => {
    const m = new Map<string, string>()
    for (const n of nodes) m.set(n.id, keyForNode(n))
    return m
  }, [nodes, keyForNode])

  const selectedIds = useMemo(() => {
    if (!selectedKey) return []
    for (const [id, key] of keyById) {
      if (key === selectedKey) return [id]
    }
    return []
  }, [keyById, selectedKey])

  // Docint's findings panel is single-entity: a marquee/shift-click multi
  // selection has no representation there, so a non-empty selection resolves
  // to its most-recently-added id. An empty selection (background click)
  // clears it, same as chorus.
  const handleSelectionChange = useCallback(
    (ids: string[]) => {
      if (ids.length === 0) {
        onSelectEntity(null)
        return
      }
      const key = keyById.get(ids[ids.length - 1])
      if (key) onSelectEntity(key)
    },
    [keyById, onSelectEntity]
  )

  // View-only removal: hide the given nodes (and their edges) from the
  // canvas. If the currently-selected entity was among them, deselect it so
  // the findings panel doesn't keep showing a vanished node. The underlying
  // NER data and any exports of it are untouched — removed nodes return on
  // the next fetch (see the `nodes`-identity effect above).
  const handleDeleteNodes = useCallback(
    (ids: string[]) => {
      setRemovedIds((prev) => {
        const next = new Set(prev)
        for (const id of ids) next.add(id)
        return next
      })
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
      {nodeCount != null && onNodeCountChange && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-border pt-2">
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
        </div>
      )}

      <ForceGraph
        nodes={fg.nodes}
        edges={fg.edges}
        nodeStyles={nodeStyles}
        edgeStyles={ENTITY_EDGE_STYLES}
        selectedIds={selectedIds}
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
