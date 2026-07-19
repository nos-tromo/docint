/**
 * Pure mapper: NER graph payload → `@infra/ui` `ForceGraph` props.
 *
 * No React, no rendering runtime — converts the flat entity/edge arrays the
 * NER graph endpoint returns into the shape the shared `ForceGraph` primitive
 * consumes (ADR 0016 in chorus, where the primitive was extracted from). Kept
 * as a small lib file (mirroring chorus's `networkElements.ts`/
 * `socialElements.ts`) rather than inlined in the component so the mapping —
 * and the type→color palette — stay independently testable.
 */

import type { ForceGraphEdge, ForceGraphEdgeStyle, ForceGraphNode, ForceGraphNodeStyle } from '@infra/ui'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'

// A small categorical palette; entity types map onto it by a stable hash so a
// given label keeps its color across renders. Identical to the palette the
// local force-graph engine used, so colors are stable across the migration.
const PALETTE = [
  '#34d399', // emerald (brand accent family)
  '#60a5fa', // blue
  '#f472b6', // pink
  '#fbbf24', // amber
  '#a78bfa', // violet
  '#22d3ee', // cyan
  '#fb7185', // rose
  '#a3e635', // lime
  '#f59e0b', // orange
  '#818cf8' // indigo
]

/**
 * Hash an entity type onto the categorical palette. Deterministic — the same
 * type string always resolves to the same color.
 */
export function colorForType(type: string): string {
  let hash = 0
  for (let i = 0; i < type.length; i++) hash = (hash * 31 + type.charCodeAt(i)) | 0
  return PALETTE[Math.abs(hash) % PALETTE.length]
}

/** Edge styling, keyed by the NER graph's edge `kind`. */
export const ENTITY_EDGE_STYLES: Record<string, ForceGraphEdgeStyle> = {
  relation: { opacity: 0.85 },
  cooccurrence: { dashed: true, opacity: 0.4 }
}

/**
 * Build a `ForceGraph` `nodeStyles` map covering every type present in
 * `types`, each colored via {@link colorForType}.
 */
export function nodeStylesForTypes(types: Iterable<string>): Record<string, ForceGraphNodeStyle> {
  const styles: Record<string, ForceGraphNodeStyle> = {}
  for (const type of types) styles[type] = { color: colorForType(type) }
  return styles
}

/**
 * Convert the NER graph payload into `ForceGraph` node/edge arrays.
 *
 * Node kind: `type || 'Unlabeled'`. Node size: `mentions` (the primitive
 * scales radius via `sqrt(size)`, capped, matching the old
 * `radiusForMentions` formula).
 *
 * Edges pass through as `{source, target, kind, weight}` — `kind` is
 * `'relation'` or `'cooccurrence'`.
 */
export function toEntityForceGraph(
  nodes: NerGraphNode[],
  edges: NerGraphEdge[]
): { nodes: ForceGraphNode[]; edges: ForceGraphEdge[] } {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      label: n.text,
      kind: n.type || 'Unlabeled',
      size: n.mentions
    })),
    edges: edges.map((e) => ({
      source: e.source,
      target: e.target,
      kind: e.kind,
      weight: e.weight
    }))
  }
}

/**
 * Legend entries for the top-8 entity types by node count (matches the old
 * component's legend computation).
 */
export function legendForNodes(nodes: NerGraphNode[]): Array<{ kind: string; label: string }> {
  const counts = new Map<string, number>()
  for (const n of nodes) {
    const t = n.type || 'Unlabeled'
    counts.set(t, (counts.get(t) ?? 0) + 1)
  }
  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([type]) => ({ kind: type, label: type }))
}
