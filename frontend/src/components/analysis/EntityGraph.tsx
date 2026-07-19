import { useCallback, useMemo } from 'react'
import { ForceGraph } from '@infra/ui'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'
import { ENTITY_EDGE_STYLES, legendForNodes, nodeStylesForTypes, toEntityForceGraph } from '@/lib/entityGraphElements'
import { GraphTopKControl } from './GraphTopKControl'

interface Props {
  nodes: NerGraphNode[]
  edges: NerGraphEdge[]
  selectedKey: string | null
  onSelectEntity: (key: string) => void
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
  const fg = useMemo(() => toEntityForceGraph(nodes, edges), [nodes, edges])

  const nodeStyles = useMemo(() => {
    const types = new Set(nodes.map((n) => n.type || 'Unlabeled'))
    return nodeStylesForTypes(types)
  }, [nodes])

  const legend = useMemo(() => legendForNodes(nodes), [nodes])

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
  // selection has no representation there, so an empty selection (background
  // click) is a no-op — the panel keeps showing whatever was last selected —
  // and any non-empty selection resolves to its most-recently-added id.
  const handleSelectionChange = useCallback(
    (ids: string[]) => {
      if (ids.length === 0) return
      const key = keyById.get(ids[ids.length - 1])
      if (key) onSelectEntity(key)
    },
    [keyById, onSelectEntity]
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
