import { describe, it, expect } from 'vitest'
import { sourceLabel } from './sourceLabel'

describe('sourceLabel', () => {
  it('uses filename + page_label when present', () => {
    expect(sourceLabel({ filename: 'a.pdf', page_label: '12' } as any)).toBe('a.pdf · p. 12')
  })
  it('uses filename + row_label when no page', () => {
    expect(sourceLabel({ filename: 'a.csv', row_label: 'r3' } as any)).toBe('a.csv · row r3')
  })
  it('falls back to filename', () => {
    expect(sourceLabel({ filename: 'x.txt' } as any)).toBe('x.txt')
  })
})
