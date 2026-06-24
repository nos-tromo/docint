import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'
import {
  createForceSimulation,
  phyllotaxisSeed,
  type ForceLink,
  type ForceNode
} from '@/lib/forceGraph'

const WIDTH = 960
const HEIGHT = 620
const CENTER_X = WIDTH / 2
const CENTER_Y = HEIGHT / 2
const MIN_ZOOM = 0.25
const MAX_ZOOM = 4
const DRAG_THRESHOLD = 4 // px of movement before a press counts as a drag

// A small categorical palette; entity types map onto it by a stable hash so a
// given label keeps its color across renders.
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

function colorForType(type: string): string {
  let hash = 0
  for (let i = 0; i < type.length; i++) hash = (hash * 31 + type.charCodeAt(i)) | 0
  return PALETTE[Math.abs(hash) % PALETTE.length]
}

function radiusForMentions(mentions: number): number {
  return Math.min(34, 7 + Math.sqrt(Math.max(1, mentions)) * 2.4)
}

interface NodeMeta {
  text: string
  type: string
  mentions: number
  color: string
}

interface View {
  x: number
  y: number
  k: number
}

interface Props {
  nodes: NerGraphNode[]
  edges: NerGraphEdge[]
  selectedKey: string | null
  onSelectEntity: (key: string) => void
  /** Map a graph node to the `${text}::${type}` selection key used elsewhere. */
  keyForNode: (n: NerGraphNode) => string
  isLoading?: boolean
}

/**
 * Interactive, force-directed entity graph. Nodes are draggable (with
 * collision), the canvas zooms (wheel) and pans (background drag), and a click
 * selects an entity — surfacing its findings in the panel below, mirroring the
 * dropdown picker. Rendering is plain SVG over the dependency-free
 * {@link createForceSimulation} layout.
 */
export function EntityGraph({
  nodes,
  edges,
  selectedKey,
  onSelectEntity,
  keyForNode,
  isLoading
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)

  // Build the simulation (and a node-id → display-meta map) whenever the graph
  // payload changes. Seed positions deterministically so layouts are stable.
  const { sim, meta, selectableKeyById } = useMemo(() => {
    const seeds = phyllotaxisSeed(nodes.length, CENTER_X, CENTER_Y, 30)
    const metaById = new Map<string, NodeMeta>()
    const keyById = new Map<string, string>()
    const simNodes: ForceNode[] = nodes.map((n, i) => {
      metaById.set(n.id, {
        text: n.text,
        type: n.type,
        mentions: n.mentions,
        color: colorForType(n.type || 'Unlabeled')
      })
      keyById.set(n.id, keyForNode(n))
      return { id: n.id, x: seeds[i].x, y: seeds[i].y, vx: 0, vy: 0, r: radiusForMentions(n.mentions) }
    })
    const simLinks: ForceLink[] = edges.map((e) => ({
      source: e.source,
      target: e.target,
      weight: e.weight
    }))
    return {
      sim: createForceSimulation(simNodes, simLinks, { centerX: CENTER_X, centerY: CENTER_Y }),
      meta: metaById,
      selectableKeyById: keyById
    }
  }, [nodes, edges, keyForNode])

  // `frame` is bumped each animation step purely to re-render at the sim's
  // current positions; the authoritative state lives on the mutated sim nodes.
  const [, setFrame] = useState(0)
  const [view, setView] = useState<View>({ x: 0, y: 0, k: 1 })
  const draggingNodeRef = useRef<string | null>(null)
  const dragStartRef = useRef<{ x: number; y: number } | null>(null)
  const movedRef = useRef(false)
  const panRef = useRef<{ startX: number; startY: number; view: View } | null>(null)
  const rafRef = useRef(0)
  const runningRef = useRef(false)

  // Reset the viewport whenever a fresh graph is built.
  useEffect(() => {
    setView({ x: 0, y: 0, k: 1 })
  }, [sim])

  // Start (or resume) the animation loop. It ticks until the layout settles and
  // no node is being dragged, then idles — drag handlers call this again to
  // re-kick it so neighbors spring while a node is moved. Guarded for
  // environments without requestAnimationFrame (tests render the seed layout).
  const runLoop = useCallback(() => {
    if (typeof requestAnimationFrame !== 'function' || runningRef.current) return
    runningRef.current = true
    const step = () => {
      // A few ticks per frame settles the layout in well under a second.
      for (let i = 0; i < 3; i++) sim.tick()
      setFrame((f) => (f + 1) % 1_000_000)
      if (!sim.isSettled() || draggingNodeRef.current) {
        rafRef.current = requestAnimationFrame(step)
      } else {
        runningRef.current = false
        rafRef.current = 0
      }
    }
    rafRef.current = requestAnimationFrame(step)
  }, [sim])

  useEffect(() => {
    runLoop()
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      runningRef.current = false
    }
  }, [runLoop])

  /** Convert client (screen) coords to layout-space coords. */
  const screenToLayout = useCallback(
    (clientX: number, clientY: number): { x: number; y: number } => {
      const svg = svgRef.current
      if (!svg) return { x: 0, y: 0 }
      const rect = svg.getBoundingClientRect()
      const w = rect.width || WIDTH
      const h = rect.height || HEIGHT
      const px = ((clientX - rect.left) / w) * WIDTH
      const py = ((clientY - rect.top) / h) * HEIGHT
      return { x: (px - view.x) / view.k, y: (py - view.y) / view.k }
    },
    [view]
  )

  const onWheel = useCallback(
    (e: React.WheelEvent<SVGSVGElement>) => {
      e.preventDefault()
      const svg = svgRef.current
      if (!svg) return
      const rect = svg.getBoundingClientRect()
      const w = rect.width || WIDTH
      const h = rect.height || HEIGHT
      const px = ((e.clientX - rect.left) / w) * WIDTH
      const py = ((e.clientY - rect.top) / h) * HEIGHT
      setView((v) => {
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
        const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.k * factor))
        // Keep the layout point under the cursor stationary.
        const lx = (px - v.x) / v.k
        const ly = (py - v.y) / v.k
        return { k, x: px - lx * k, y: py - ly * k }
      })
    },
    []
  )

  const zoomBy = useCallback((factor: number) => {
    setView((v) => {
      const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.k * factor))
      // Zoom about the canvas center.
      const lx = (CENTER_X - v.x) / v.k
      const ly = (CENTER_Y - v.y) / v.k
      return { k, x: CENTER_X - lx * k, y: CENTER_Y - ly * k }
    })
  }, [])

  // --- background pan ---
  const onBackgroundPointerDown = useCallback(
    (e: React.PointerEvent<SVGRectElement>) => {
      e.currentTarget.setPointerCapture?.(e.pointerId)
      panRef.current = { startX: e.clientX, startY: e.clientY, view }
    },
    [view]
  )

  const onBackgroundPointerMove = useCallback((e: React.PointerEvent<SVGRectElement>) => {
    const pan = panRef.current
    if (!pan) return
    const svg = svgRef.current
    const rect = svg?.getBoundingClientRect()
    const w = rect?.width || WIDTH
    const h = rect?.height || HEIGHT
    const dx = ((e.clientX - pan.startX) / w) * WIDTH
    const dy = ((e.clientY - pan.startY) / h) * HEIGHT
    setView({ k: pan.view.k, x: pan.view.x + dx, y: pan.view.y + dy })
  }, [])

  const onBackgroundPointerUp = useCallback((e: React.PointerEvent<SVGRectElement>) => {
    e.currentTarget.releasePointerCapture?.(e.pointerId)
    panRef.current = null
  }, [])

  // --- node drag ---
  const onNodePointerDown = useCallback(
    (e: React.PointerEvent, id: string) => {
      e.stopPropagation()
      ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
      draggingNodeRef.current = id
      movedRef.current = false
      dragStartRef.current = { x: e.clientX, y: e.clientY }
      const p = screenToLayout(e.clientX, e.clientY)
      sim.fixNode(id, p.x, p.y)
      sim.reheat(0.3)
      runLoop()
    },
    [runLoop, screenToLayout, sim]
  )

  const onNodePointerMove = useCallback(
    (e: React.PointerEvent, id: string) => {
      if (draggingNodeRef.current !== id) return
      // Ignore sub-threshold jitter so a plain click is not read as a drag.
      const start = dragStartRef.current
      if (start && Math.hypot(e.clientX - start.x, e.clientY - start.y) > DRAG_THRESHOLD) {
        movedRef.current = true
      }
      const p = screenToLayout(e.clientX, e.clientY)
      sim.fixNode(id, p.x, p.y)
      sim.reheat(0.3)
      runLoop()
    },
    [runLoop, screenToLayout, sim]
  )

  const onNodePointerUp = useCallback(
    (e: React.PointerEvent, id: string) => {
      ;(e.currentTarget as Element).releasePointerCapture?.(e.pointerId)
      if (draggingNodeRef.current === id) {
        draggingNodeRef.current = null
        sim.releaseNode(id)
      }
    },
    [sim]
  )

  const handleSelect = useCallback(
    (id: string) => {
      // Suppress the click that ends a drag gesture.
      if (movedRef.current) {
        movedRef.current = false
        return
      }
      const key = selectableKeyById.get(id)
      if (key) onSelectEntity(key)
    },
    [onSelectEntity, selectableKeyById]
  )

  // Selected node id (graph nodes key by cluster id; match via selection key).
  const selectedId = useMemo(() => {
    if (!selectedKey) return null
    for (const [id, key] of selectableKeyById) if (key === selectedKey) return id
    return null
  }, [selectableKeyById, selectedKey])

  // Highlight neighbors of the selected node so its relations stand out.
  const neighborIds = useMemo(() => {
    if (!selectedId) return null
    const set = new Set<string>()
    for (const e of edges) {
      if (e.source === selectedId) set.add(e.target)
      else if (e.target === selectedId) set.add(e.source)
    }
    return set
  }, [edges, selectedId])

  const legendTypes = useMemo(() => {
    const counts = new Map<string, number>()
    for (const n of nodes) {
      const t = n.type || 'Unlabeled'
      counts.set(t, (counts.get(t) ?? 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([type]) => ({ type, color: colorForType(type) }))
  }, [nodes])

  if (!isLoading && nodes.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No entity relationships to graph for this collection.
      </div>
    )
  }

  const transform = `translate(${view.x} ${view.y}) scale(${view.k})`

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground">
          {isLoading
            ? 'Building entity graph…'
            : `${nodes.length} entit${nodes.length === 1 ? 'y' : 'ies'} · ${edges.length} link${edges.length === 1 ? '' : 's'}. Scroll to zoom, drag to move, click a node to inspect.`}
        </p>
        <div className="flex items-center gap-1">
          <button
            type="button"
            aria-label="Zoom in"
            onClick={() => zoomBy(1.25)}
            className="h-7 w-7 rounded-md border border-border text-sm leading-none"
          >
            +
          </button>
          <button
            type="button"
            aria-label="Zoom out"
            onClick={() => zoomBy(1 / 1.25)}
            className="h-7 w-7 rounded-md border border-border text-sm leading-none"
          >
            −
          </button>
          <button
            type="button"
            onClick={() => setView({ x: 0, y: 0, k: 1 })}
            className="h-7 px-2 rounded-md border border-border text-xs"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="relative rounded-md border border-border bg-zinc-950 overflow-hidden">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full h-[60vh] touch-none select-none"
          role="application"
          aria-label="Entity relationship graph"
          onWheel={onWheel}
        >
          {/* Background capture rect for panning. */}
          <rect
            x={0}
            y={0}
            width={WIDTH}
            height={HEIGHT}
            fill="transparent"
            onPointerDown={onBackgroundPointerDown}
            onPointerMove={onBackgroundPointerMove}
            onPointerUp={onBackgroundPointerUp}
          />
          <g transform={transform}>
            {edges.map((e, i) => {
              const a = sim.nodeById(e.source)
              const b = sim.nodeById(e.target)
              if (!a || !b) return null
              const incident = selectedId != null && (e.source === selectedId || e.target === selectedId)
              const dimmed = selectedId != null && !incident
              return (
                <line
                  key={`${e.source}->${e.target}:${e.label}:${i}`}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={incident ? '#52525b' : '#3f3f46'}
                  strokeOpacity={dimmed ? 0.15 : e.kind === 'relation' ? 0.85 : 0.4}
                  strokeWidth={Math.min(4, 0.6 + Math.log2(e.weight + 1)) / view.k}
                  strokeDasharray={e.kind === 'cooccurrence' ? `${4 / view.k} ${3 / view.k}` : undefined}
                />
              )
            })}
            {nodes.map((n) => {
              const sn = sim.nodeById(n.id)
              if (!sn) return null
              const m = meta.get(n.id)!
              const isSelected = n.id === selectedId
              const isNeighbor = neighborIds?.has(n.id) ?? false
              const dimmed = selectedId != null && !isSelected && !isNeighbor
              const r = sn.r
              return (
                <g
                  key={n.id}
                  transform={`translate(${sn.x} ${sn.y})`}
                  role="button"
                  tabIndex={0}
                  aria-label={`${m.text} (${m.type}, ${m.mentions} mentions)`}
                  aria-pressed={isSelected}
                  className="cursor-pointer outline-none"
                  opacity={dimmed ? 0.35 : 1}
                  onPointerDown={(e) => onNodePointerDown(e, n.id)}
                  onPointerMove={(e) => onNodePointerMove(e, n.id)}
                  onPointerUp={(e) => onNodePointerUp(e, n.id)}
                  onClick={() => handleSelect(n.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      const key = selectableKeyById.get(n.id)
                      if (key) onSelectEntity(key)
                    }
                  }}
                >
                  <circle
                    r={r}
                    fill={m.color}
                    fillOpacity={isSelected ? 1 : 0.85}
                    stroke={isSelected ? '#fafafa' : '#18181b'}
                    strokeWidth={(isSelected ? 3 : 1.5) / view.k}
                  />
                  <text
                    y={r + 11 / view.k}
                    textAnchor="middle"
                    fontSize={11 / view.k}
                    fill="#d4d4d8"
                    className="pointer-events-none"
                  >
                    {m.text.length > 24 ? `${m.text.slice(0, 23)}…` : m.text}
                  </text>
                </g>
              )
            })}
          </g>
        </svg>

        {legendTypes.length > 0 && (
          <div className="absolute right-2 top-2 max-w-[12rem] rounded-md border border-border bg-zinc-900/90 p-2 text-xs space-y-1">
            <div className="uppercase tracking-wide text-[10px] text-muted-foreground">Types</div>
            <ul className="space-y-0.5">
              {legendTypes.map(({ type, color }) => (
                <li key={type} className="flex items-center gap-1.5">
                  <span
                    aria-hidden="true"
                    className="inline-block h-2.5 w-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <span className="truncate">{type}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
