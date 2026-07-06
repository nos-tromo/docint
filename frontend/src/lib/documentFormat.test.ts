import { describe, it, expect } from 'vitest'
import { mimeLabel, unitsLabel, shortHash } from './documentFormat'

describe('mimeLabel', () => {
  it('maps known MIME types to curated labels', () => {
    expect(mimeLabel('image/jpeg')).toBe('JPEG')
    expect(mimeLabel('text/csv')).toBe('CSV')
    expect(mimeLabel('application/x-ndjson')).toBe('NDJSON')
    expect(mimeLabel('application/pdf')).toBe('PDF')
    expect(
      mimeLabel(
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      )
    ).toBe('DOCX')
  })

  it('is case-insensitive and ignores parameters', () => {
    expect(mimeLabel('IMAGE/JPEG')).toBe('JPEG')
    expect(mimeLabel('text/csv; charset=utf-8')).toBe('CSV')
  })

  it('falls back to a cleaned, capped subtype for unknown types', () => {
    expect(mimeLabel('application/x-something')).toBe('SOMETHING')
    expect(mimeLabel('application/vnd.superlongvendortypename')).toBe('SUPERLONGVEN')
  })

  it('returns an em dash for missing MIME', () => {
    expect(mimeLabel(undefined)).toBe('—')
    expect(mimeLabel(null)).toBe('—')
    expect(mimeLabel('')).toBe('—')
  })
})

describe('unitsLabel', () => {
  it('prefers pages when present', () => {
    expect(unitsLabel({ page_count: 12, row_count: 5 })).toEqual({ text: '12 pg', sort: 12 })
  })

  it('uses rows when there are no pages, pluralizing correctly', () => {
    expect(unitsLabel({ page_count: 0, row_count: 138 })).toEqual({ text: '138 rows', sort: 138 })
    expect(unitsLabel({ row_count: 1 })).toEqual({ text: '1 row', sort: 1 })
  })

  it('returns an em dash instead of a misleading zero (e.g. images)', () => {
    expect(unitsLabel({ page_count: 0, row_count: 0 })).toEqual({ text: '—', sort: 0 })
    expect(unitsLabel({})).toEqual({ text: '—', sort: 0 })
  })
})

describe('shortHash', () => {
  it('keeps the leading characters', () => {
    expect(shortHash('abd4fc7803e1e8d1c7e2b92e8d8fb9d6')).toBe('abd4fc78')
    expect(shortHash('abcdef', 4)).toBe('abcd')
  })

  it('returns an em dash for missing hashes', () => {
    expect(shortHash(undefined)).toBe('—')
    expect(shortHash('')).toBe('—')
  })
})
