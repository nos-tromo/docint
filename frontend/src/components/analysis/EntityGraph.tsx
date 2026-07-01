import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'
import { cn } from '@/lib/cn'
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
// Edge-length ("spread") slider: bounds + the base spacing it scales. The bases
// mirror forceGraph's DEFAULTS (linkDistance 70, repulsion 320); a multiplier of
// 1 reproduces today's density, higher values push nodes farther apart.
const MIN_SPREAD = 0.5
const MAX_SPREAD = 3
const BASE_LINK_DISTANCE = 70
const BASE_REPULSION = 320

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

// Two-diagonal-arrows "expand to corners" glyph (lucide maximize-2).
function ExpandIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="15 3 21 3 21 9" />
      <polyline points="9 21 3 21 3 15" />
      <line x1="21" y1="3" x2="14" y2="10" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
  )
}

// Arrows pointing inward "collapse" glyph (lucide minimize-2).
function CollapseIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polyline points="4 14 10 14 10 20" />
      <polyline points="20 10 14 10 14 4" />
      <line x1="14" y1="10" x2="21" y2="3" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </svg>
  )
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
  const svgRef = useRef<SVGSVGElement | null>(null)

  // Edge-count (degree) filter: hide any node with fewer than `minDegree`
  // incident edges. Default 0 shows every node.
  const [minDegree, setMinDegree] = useState(0)

  // Edge-length multiplier from the "Edge length" slider; 1 = today's density.
  // Applied to the live sim's link rest-length + repulsion (see effect below).
  const [spread, setSpread] = useState(1)

  // In-app "maximize": grows the whole panel to a full-window overlay.
  const [isMaximized, setIsMaximized] = useState(false)

  // Incident-edge count per node id, from the full edge set.
  const degreeById = useMemo(() => {
    const deg = new Map<string, number>()
    for (const e of edges) {
      deg.set(e.source, (deg.get(e.source) ?? 0) + 1)
      deg.set(e.target, (deg.get(e.target) ?? 0) + 1)
    }
    return deg
  }, [edges])

  const maxDegree = useMemo(() => {
    let m = 0
    for (const n of nodes) m = Math.max(m, degreeById.get(n.id) ?? 0)
    return m
  }, [nodes, degreeById])

  // Keep the threshold valid when a smaller graph loads (e.g. a merge-mode
  // switch) so it never strands the view on an all-filtered, empty graph.
  useEffect(() => {
    setMinDegree((d) => Math.min(d, maxDegree))
  }, [maxDegree])

  const visibleNodes = useMemo(
    () =>
      minDegree <= 0
        ? nodes
        : nodes.filter((n) => (degreeById.get(n.id) ?? 0) >= minDegree),
    [nodes, degreeById, minDegree]
  )

  const visibleEdges = useMemo(() => {
    if (minDegree <= 0) return edges
    const ids = new Set(visibleNodes.map((n) => n.id))
    return edges.filter((e) => ids.has(e.source) && ids.has(e.target))
  }, [edges, visibleNodes, minDegree])

  // Build the simulation (and a node-id → display-meta map) over the currently
  // visible nodes/edges. Seed positions deterministically so layouts are stable.
  const { sim, meta, selectableKeyById } = useMemo(() => {
    const seeds = phyllotaxisSeed(visibleNodes.length, CENTER_X, CENTER_Y, 30)
    const metaById = new Map<string, NodeMeta>()
    const keyById = new Map<string, string>()
    const simNodes: ForceNode[] = visibleNodes.map((n, i) => {
      metaById.set(n.id, {
        text: n.text,
        type: n.type,
        mentions: n.mentions,
        color: colorForType(n.type || 'Unlabeled')
      })
      keyById.set(n.id, keyForNode(n))
      return { id: n.id, x: seeds[i].x, y: seeds[i].y, vx: 0, vy: 0, r: radiusForMentions(n.mentions) }
    })
    const simLinks: ForceLink[] = visibleEdges.map((e) => ({
      source: e.source,
      target: e.target,
      weight: e.weight
    }))
    return {
      sim: createForceSimulation(simNodes, simLinks, { centerX: CENTER_X, centerY: CENTER_Y }),
      meta: metaById,
      selectableKeyById: keyById
    }
  }, [visibleNodes, visibleEdges, keyForNode])

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

  // Apply the edge-length multiplier to the live simulation (no reseed): scale
  // link rest-length and repulsion together so clusters open up, not just
  // directly-linked pairs. Re-runs on a fresh graph (sim) and on slider moves.
  useEffect(() => {
    sim.setOptions({
      linkDistance: BASE_LINK_DISTANCE * spread,
      repulsion: BASE_REPULSION * spread
    })
    sim.reheat()
    runLoop()
  }, [sim, spread, runLoop])

  // While maximized: Escape exits, and body scroll is locked so the page behind
  // can't scroll under the overlay. Both are torn down on collapse/unmount.
  useEffect(() => {
    if (!isMaximized) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsMaximized(false)
    }
    window.addEventListener('keydown', onKeyDown)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = prevOverflow
    }
  }, [isMaximized])

  // If the graph empties (e.g. collection switch) the early return below would
  // bypass the overlay while leaving body scroll locked — auto-collapse to keep
  // the teardown effect honest.
  useEffect(() => {
    if (nodes.length === 0) setIsMaximized(false)
  }, [nodes.length])

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

  // Wheel-zoom, bound as a *non-passive* native listener via a ref callback
  // rather than React's synthetic `onWheel`. React registers wheel listeners as
  // passive, so a synthetic handler's preventDefault() is a no-op and the page
  // scrolls behind the graph while zooming; a directly-attached
  // { passive: false } listener lets us cancel that default scroll. The
  // ref-callback form (with React 19's cleanup return) re-binds correctly if the
  // <svg> unmounts and remounts — e.g. an initially-empty collection later fills
  // in. setView's functional updater keeps the handler free of stale view state,
  // so it never needs re-binding on its own.
  const setSvgRef = useCallback((svg: SVGSVGElement | null) => {
    svgRef.current = svg
    if (!svg) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
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
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => {
      svg.removeEventListener('wheel', onWheel)
      svgRef.current = null
    }
  }, [])

  const zoomBy = useCallback((factor: number) => {
    setView((v) => {
      const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.k * factor))
      // Zoom about the canvas center.
      const lx = (CENTER_X - v.x) / v.k
      const ly = (CENTER_Y - v.y) / v.k
      return { k, x: CENTER_X - lx * k, y: CENTER_Y - ly * k }
    })
  }, [])

  // Reset every graph control back to its default, not just the viewport:
  // edge filter (min edges → 0), edge length (→ 1x), and zoom/pan (→ home).
  const resetControls = useCallback(() => {
    setMinDegree(0)
    setSpread(1)
    setView({ x: 0, y: 0, k: 1 })
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
    for (const e of visibleEdges) {
      if (e.source === selectedId) set.add(e.target)
      else if (e.target === selectedId) set.add(e.source)
    }
    return set
  }, [visibleEdges, selectedId])

  const legendTypes = useMemo(() => {
    const counts = new Map<string, number>()
    for (const n of visibleNodes) {
      const t = n.type || 'Unlabeled'
      counts.set(t, (counts.get(t) ?? 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([type]) => ({ type, color: colorForType(type) }))
  }, [visibleNodes])

  if (!isLoading && nodes.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">
        No entity relationships to graph for this collection.
      </div>
    )
  }

  const transform = `translate(${view.x} ${view.y}) scale(${view.k})`

  return (
    <div
      data-maximized={isMaximized}
      className={cn(
        'space-y-2',
        isMaximized && 'fixed inset-0 z-50 flex flex-col bg-zinc-950 p-4'
      )}
    >
      <div className="space-y-2">
        <p className="text-sm text-muted-foreground">
          {isLoading
            ? 'Building entity graph…'
            : `${visibleNodes.length} entit${visibleNodes.length === 1 ? 'y' : 'ies'} · ${visibleEdges.length} link${visibleEdges.length === 1 ? '' : 's'}${minDegree > 0 ? ` · ≥${minDegree} edge${minDegree === 1 ? '' : 's'}` : ''}. Scroll to zoom, drag to move, click a node to inspect.`}
        </p>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-border pt-2">
          <div className="flex items-center gap-1" role="group" aria-label="Minimum edges per node">
            <span className="text-xs text-muted-foreground">Min edges</span>
            <button
              type="button"
              aria-label="Decrease minimum edges"
              disabled={minDegree <= 0}
              onClick={() => setMinDegree((d) => Math.max(0, d - 1))}
              className="h-7 w-7 rounded-md border border-border text-sm leading-none disabled:opacity-40"
            >
              −
            </button>
            <span aria-live="polite" className="w-5 text-center text-xs tabular-nums">
              {minDegree}
            </span>
            <button
              type="button"
              aria-label="Increase minimum edges"
              disabled={minDegree >= maxDegree}
              onClick={() => setMinDegree((d) => Math.min(maxDegree, d + 1))}
              className="h-7 w-7 rounded-md border border-border text-sm leading-none disabled:opacity-40"
            >
              +
            </button>
          </div>

          <span aria-hidden="true" className="h-5 border-l border-border" />

          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Edge length</span>
            <input
              type="range"
              min={MIN_SPREAD}
              max={MAX_SPREAD}
              step={0.1}
              value={spread}
              onChange={(e) => setSpread(Number(e.target.value))}
              aria-label="Edge length"
              title="Spread nodes apart to de-clutter a dense graph"
              className="w-28 cursor-pointer accent-primary"
            />
          </div>

          <span aria-hidden="true" className="h-5 border-l border-border" />

          <div className="flex items-center gap-1" role="group" aria-label="Zoom">
            <span className="text-xs text-muted-foreground">Zoom</span>
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
          </div>

          <span aria-hidden="true" className="h-5 border-l border-border" />

          {/* Standalone (not part of the Zoom group): resets every control. */}
          <button
            type="button"
            onClick={resetControls}
            title="Reset min edges, edge length, and zoom"
            className="h-7 px-2 rounded-md border border-border text-xs"
          >
            Reset
          </button>
        </div>
      </div>

      <div
        className={cn(
          'relative rounded-md border border-border bg-zinc-950 overflow-hidden',
          isMaximized && 'flex-1 min-h-0'
        )}
      >
        <svg
          ref={setSvgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className={cn('w-full touch-none select-none', isMaximized ? 'h-full' : 'h-[60vh]')}
          role="application"
          aria-label="Entity relationship graph"
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
            {visibleEdges.map((e, i) => {
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
            {visibleNodes.map((n) => {
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

        <button
          type="button"
          aria-label={isMaximized ? 'Collapse graph' : 'Expand graph'}
          aria-pressed={isMaximized}
          title={isMaximized ? 'Collapse graph' : 'Expand graph'}
          onClick={() => setIsMaximized((m) => !m)}
          className="absolute left-2 top-2 z-10 rounded-md border border-border bg-zinc-900/90 p-1.5 text-muted-foreground hover:text-foreground"
        >
          {isMaximized ? <CollapseIcon /> : <ExpandIcon />}
        </button>

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
