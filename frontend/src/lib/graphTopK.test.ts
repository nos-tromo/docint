import { describe, it, expect } from 'vitest'
import { resolveGraphTopK } from './graphTopK'

const cfg = { graph_top_k: 80, graph_max_top_k: 500, collection_timeout: 120 }

describe('resolveGraphTopK', () => {
  it('uses the server default when nothing is stored', () => {
    expect(resolveGraphTopK(null, cfg)).toBe(80)
  })
  it('uses the stored value when present', () => {
    expect(resolveGraphTopK(250, cfg)).toBe(250)
  })
  it('clamps above the ceiling', () => {
    expect(resolveGraphTopK(9999, cfg)).toBe(500)
  })
  it('clamps below 1', () => {
    expect(resolveGraphTopK(0, cfg)).toBe(1)
  })
  it('falls back to 80/500 when config has not loaded', () => {
    expect(resolveGraphTopK(null, undefined)).toBe(80)
    expect(resolveGraphTopK(9999, undefined)).toBe(500)
  })
})
