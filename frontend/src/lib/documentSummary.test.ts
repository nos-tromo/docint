import { describe, it, expect } from 'vitest'
import { summarizeDocuments } from './documentSummary'
import type { DocumentRecord } from '@/api/types'

function doc(overrides: Partial<DocumentRecord>): DocumentRecord {
  return { filename: 'f', file_hash: 'h', ...overrides }
}

describe('summarizeDocuments', () => {
  it('returns empty aggregates for no documents', () => {
    expect(summarizeDocuments([])).toEqual({
      documentCount: 0,
      nodeCount: 0,
      fileTypes: [],
      entityTypes: []
    })
  })

  it('sums nodes and counts documents', () => {
    const summary = summarizeDocuments([
      doc({ node_count: 1 }),
      doc({ node_count: 138 }),
      doc({}) // missing node_count counts as 0
    ])
    expect(summary.documentCount).toBe(3)
    expect(summary.nodeCount).toBe(139)
  })

  it('tallies file types by label, most common first', () => {
    const summary = summarizeDocuments([
      doc({ mimetype: 'image/jpeg' }),
      doc({ mimetype: 'image/jpeg' }),
      doc({ mimetype: 'text/csv' })
    ])
    expect(summary.fileTypes).toEqual([
      { label: 'JPEG', count: 2 },
      { label: 'CSV', count: 1 }
    ])
  })

  it('collects the sorted distinct union of entity types', () => {
    const summary = summarizeDocuments([
      doc({ entity_types: ['person', 'org'] }),
      doc({ entity_types: ['org', 'loc'] }),
      doc({ entity_types: [] })
    ])
    expect(summary.entityTypes).toEqual(['loc', 'org', 'person'])
  })
})
