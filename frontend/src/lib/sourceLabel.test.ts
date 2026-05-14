import { describe, it, expect } from 'vitest'
import { sourceLabel } from './sourceLabel'

describe('sourceLabel', () => {
  it('uses filename + page when present', () => {
    expect(sourceLabel({ filename: 'a.pdf', page: 12 } as any)).toBe('a.pdf · p. 12')
  })
  it('uses filename + row when no page', () => {
    expect(sourceLabel({ filename: 'a.csv', row: 3 } as any)).toBe('a.csv · row 3')
  })
  it('falls back to filename', () => {
    expect(sourceLabel({ filename: 'x.txt' } as any)).toBe('x.txt')
  })
  it('treats page 0 as a real value (not a falsy fall-through)', () => {
    expect(sourceLabel({ filename: 'a.pdf', page: 0 } as any)).toBe('a.pdf · p. 0')
  })
})
