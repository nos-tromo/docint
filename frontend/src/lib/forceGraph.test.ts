import { describe, expect, it } from 'vitest'
import { createForceSimulation, phyllotaxisSeed, type ForceNode } from './forceGraph'

function node(id: string, x: number, y: number): ForceNode {
  return { id, x, y, vx: 0, vy: 0, r: 8 }
}

function dist(a: ForceNode, b: ForceNode): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function settle(sim: ReturnType<typeof createForceSimulation>, maxTicks = 600): void {
  for (let i = 0; i < maxTicks && !sim.isSettled(); i++) sim.tick()
}

describe('createForceSimulation', () => {
  it('pulls two linked nodes toward the link distance', () => {
    const nodes = [node('a', -200, 0), node('b', 200, 0)]
    const sim = createForceSimulation(nodes, [{ source: 'a', target: 'b', weight: 1 }], {
      centerX: 0,
      centerY: 0,
      linkDistance: 70
    })
    const before = dist(nodes[0], nodes[1])
    settle(sim)
    const after = dist(nodes[0], nodes[1])
    // Started far apart (400); the spring + centering must reel them in toward
    // the rest length without collapsing them on top of each other.
    expect(after).toBeLessThan(before)
    expect(after).toBeGreaterThan(20)
    expect(after).toBeLessThan(160)
  })

  it('separates coincident nodes via collision instead of stacking them', () => {
    const nodes = [node('a', 0, 0), node('b', 0, 0)]
    const sim = createForceSimulation(nodes, [])
    settle(sim)
    // Collision radius is r+r+padding = 8+8+6 = 22; they must end clearly apart.
    expect(dist(nodes[0], nodes[1])).toBeGreaterThan(15)
    expect(Number.isFinite(nodes[0].x)).toBe(true)
    expect(Number.isFinite(nodes[1].y)).toBe(true)
  })

  it('keeps a pinned node fixed while others move', () => {
    const nodes = [node('a', 50, 50), node('b', 60, 55)]
    const sim = createForceSimulation(nodes, [])
    sim.fixNode('a', 50, 50)
    settle(sim, 100)
    expect(nodes[0].x).toBe(50)
    expect(nodes[0].y).toBe(50)
    // The free node should have been pushed off the pinned one.
    expect(dist(nodes[0], nodes[1])).toBeGreaterThan(15)
  })

  it('cools monotonically and reports settled, and reheat revives it', () => {
    const sim = createForceSimulation([node('a', 0, 0)], [])
    const first = sim.alpha
    sim.tick()
    expect(sim.alpha).toBeLessThan(first)
    settle(sim)
    expect(sim.isSettled()).toBe(true)
    sim.reheat()
    expect(sim.isSettled()).toBe(false)
  })

  it('ignores links whose endpoints are missing', () => {
    const nodes = [node('a', 0, 0)]
    const sim = createForceSimulation(nodes, [{ source: 'a', target: 'ghost', weight: 1 }])
    expect(() => settle(sim, 50)).not.toThrow()
    expect(Number.isFinite(nodes[0].x)).toBe(true)
  })
})

describe('phyllotaxisSeed', () => {
  it('produces the requested count of distinct, centered positions', () => {
    const seeds = phyllotaxisSeed(25, 100, 100)
    expect(seeds).toHaveLength(25)
    // First point sits near the center; later points spiral outward.
    expect(Math.hypot(seeds[0].x - 100, seeds[0].y - 100)).toBeLessThan(40)
    expect(Math.hypot(seeds[24].x - 100, seeds[24].y - 100)).toBeGreaterThan(
      Math.hypot(seeds[1].x - 100, seeds[1].y - 100)
    )
  })

  it('is deterministic across calls', () => {
    expect(phyllotaxisSeed(10, 0, 0)).toEqual(phyllotaxisSeed(10, 0, 0))
  })
})
