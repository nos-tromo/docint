/**
 * A tiny, dependency-free force-directed graph simulation.
 *
 * docint ships no graph library (airgap-safe, minimal deps), so the entity
 * graph runs this self-contained layout instead. It is intentionally a small
 * O(n²) solver — fine for the ≤ a few-hundred entity nodes the NER graph ever
 * shows — modelled on d3-force's structure (velocity-Verlet integration with
 * many-body repulsion, link springs, a gentle centering pull and pairwise
 * collision). The simulation is deterministic given its inputs (callers seed
 * positions, never `Math.random`), which keeps it unit-testable.
 */

export interface ForceNode {
  id: string
  x: number
  y: number
  /** Velocity components, integrated each tick. */
  vx: number
  vy: number
  /** Collision radius (node draw radius + padding handled by the solver). */
  r: number
  /** When set, the node is pinned here (used while a user drags it). */
  fx?: number | null
  fy?: number | null
}

export interface ForceLink {
  source: string
  target: string
  weight: number
}

export interface SimulationOptions {
  /** Layout-space center the nodes are gently pulled toward. */
  centerX: number
  centerY: number
  /** Rest length of a link spring. */
  linkDistance: number
  /** Many-body repulsion magnitude (applied as `repulsion·alpha / dist²`). */
  repulsion: number
  /** Link spring stiffness in `[0, 1]`. */
  linkStrength: number
  /** Positional pull toward the center in `[0, 1]`. */
  centerStrength: number
  /** Fraction of velocity retained each tick (friction); lower = more damping. */
  velocityDecay: number
  /** Extra gap enforced between node edges during collision. */
  collidePadding: number
}

const DEFAULTS: SimulationOptions = {
  centerX: 0,
  centerY: 0,
  linkDistance: 70,
  repulsion: 320,
  linkStrength: 0.08,
  centerStrength: 0.04,
  velocityDecay: 0.6,
  collidePadding: 6
}

const ALPHA_DECAY = 0.0228
const ALPHA_MIN = 0.001
// Deterministic, position-derived jitter so coincident nodes still separate
// without reaching for Math.random (keeps the solver reproducible).
const EPS = 1e-6

export interface ForceSimulation {
  readonly nodes: ForceNode[]
  /** Current cooling factor; the layout is settled once it dips below `alphaMin`. */
  alpha: number
  /** Advance the layout by one step. Returns the post-step alpha. */
  tick(): number
  /** True while the layout is still meaningfully moving. */
  isSettled(): boolean
  /** Re-heat the layout (e.g. after a drag) so it re-settles. */
  reheat(value?: number): void
  /** Pin a node to a layout-space position (drag start / move). */
  fixNode(id: string, x: number, y: number): void
  /** Release a previously pinned node (drag end). */
  releaseNode(id: string): void
  nodeById(id: string): ForceNode | undefined
}

/**
 * Build a force simulation over `nodes`/`links`. Node objects are mutated in
 * place on every {@link ForceSimulation.tick}, so the caller can read their
 * `x`/`y` straight after ticking to render.
 *
 * @param nodes Seed nodes (positions pre-assigned by the caller).
 * @param links Undirected links referencing node ids.
 * @param options Partial overrides for the layout constants.
 * @returns A live simulation handle.
 */
export function createForceSimulation(
  nodes: ForceNode[],
  links: ForceLink[],
  options: Partial<SimulationOptions> = {}
): ForceSimulation {
  const opts: SimulationOptions = { ...DEFAULTS, ...options }
  const byId = new Map<string, ForceNode>()
  for (const n of nodes) byId.set(n.id, n)

  // Keep only links whose endpoints both exist, resolved to node refs once.
  const edges = links
    .map((l) => ({ s: byId.get(l.source), t: byId.get(l.target), weight: l.weight }))
    .filter((e): e is { s: ForceNode; t: ForceNode; weight: number } => !!e.s && !!e.t && e.s !== e.t)

  let alpha = 1

  function applyFixed(n: ForceNode): void {
    if (n.fx != null) {
      n.x = n.fx
      n.vx = 0
    }
    if (n.fy != null) {
      n.y = n.fy
      n.vy = 0
    }
  }

  function tick(): number {
    alpha += (0 - alpha) * ALPHA_DECAY

    // Many-body repulsion (every unordered pair).
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i]
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j]
        let dx = b.x - a.x
        let dy = b.y - a.y
        let d2 = dx * dx + dy * dy
        if (d2 < EPS) {
          // Coincident: nudge along a deterministic, index-derived direction.
          dx = ((i - j) % 7) + 0.5
          dy = ((i + j) % 5) - 2
          d2 = dx * dx + dy * dy
        }
        const force = (opts.repulsion * alpha) / d2
        const dist = Math.sqrt(d2)
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        a.vx -= fx
        a.vy -= fy
        b.vx += fx
        b.vy += fy
      }
    }

    // Link springs pull endpoints toward `linkDistance`.
    for (const e of edges) {
      let dx = e.t.x - e.s.x
      let dy = e.t.y - e.s.y
      let dist = Math.sqrt(dx * dx + dy * dy)
      if (dist < EPS) {
        dx = 1
        dy = 0
        dist = 1
      }
      const desired = opts.linkDistance
      const k = ((dist - desired) / dist) * alpha * opts.linkStrength
      const sx = dx * k
      const sy = dy * k
      e.s.vx += sx
      e.s.vy += sy
      e.t.vx -= sx
      e.t.vy -= sy
    }

    // Gentle centering so disconnected components stay on screen.
    for (const n of nodes) {
      n.vx += (opts.centerX - n.x) * opts.centerStrength * alpha
      n.vy += (opts.centerY - n.y) * opts.centerStrength * alpha
    }

    // Pairwise collision resolution (velocity nudge, à la d3-force).
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i]
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j]
        const minDist = a.r + b.r + opts.collidePadding
        let dx = b.x - a.x
        let dy = b.y - a.y
        let d2 = dx * dx + dy * dy
        if (d2 >= minDist * minDist) continue
        if (d2 < EPS) {
          dx = ((i - j) % 3) + 0.5
          dy = ((i + j) % 3) - 1
          d2 = dx * dx + dy * dy
        }
        const dist = Math.sqrt(d2)
        const overlap = (minDist - dist) / dist
        const sx = dx * overlap * 0.5
        const sy = dy * overlap * 0.5
        a.vx -= sx
        a.vy -= sy
        b.vx += sx
        b.vy += sy
      }
    }

    // Integrate with friction; pinned nodes ignore their velocity.
    for (const n of nodes) {
      if (n.fx != null || n.fy != null) {
        applyFixed(n)
        continue
      }
      n.vx *= opts.velocityDecay
      n.vy *= opts.velocityDecay
      n.x += n.vx
      n.y += n.vy
    }

    return alpha
  }

  return {
    nodes,
    get alpha() {
      return alpha
    },
    set alpha(v: number) {
      alpha = v
    },
    tick,
    isSettled: () => alpha < ALPHA_MIN,
    reheat: (value = 0.6) => {
      alpha = Math.max(alpha, value)
    },
    fixNode: (id, x, y) => {
      const n = byId.get(id)
      if (!n) return
      n.fx = x
      n.fy = y
      n.x = x
      n.y = y
    },
    releaseNode: (id) => {
      const n = byId.get(id)
      if (!n) return
      n.fx = null
      n.fy = null
    },
    nodeById: (id) => byId.get(id)
  }
}

/**
 * Deterministically seed initial node positions on a phyllotaxis spiral
 * around a center. A spiral spreads nodes evenly so the simulation starts far
 * from a degenerate all-overlapping state — and being seedless keeps layouts
 * reproducible across renders and tests.
 *
 * @param count Number of positions to generate.
 * @param centerX Spiral center x.
 * @param centerY Spiral center y.
 * @param spacing Radial spacing constant.
 * @returns `count` `{x, y}` seed positions.
 */
export function phyllotaxisSeed(
  count: number,
  centerX: number,
  centerY: number,
  spacing = 24
): Array<{ x: number; y: number }> {
  const golden = Math.PI * (3 - Math.sqrt(5))
  const out: Array<{ x: number; y: number }> = []
  for (let i = 0; i < count; i++) {
    const radius = spacing * Math.sqrt(i + 0.5)
    const theta = i * golden
    out.push({ x: centerX + radius * Math.cos(theta), y: centerY + radius * Math.sin(theta) })
  }
  return out
}
