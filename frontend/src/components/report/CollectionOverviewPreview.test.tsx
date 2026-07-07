import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { CollectionOverviewPreview } from './CollectionOverviewPreview'
import type { CollectionOverviewSnapshot } from '@/api/types'

const overview: CollectionOverviewSnapshot = {
  collection: 'c1',
  captured_at: '2026-07-06T10:00:00Z',
  document_count: 1,
  node_count: 6,
  file_types: [{ label: 'PDF', count: 1 }],
  entity_types: ['ORG'],
  documents: [
    { filename: 'a.pdf', type_label: 'PDF', page_count: 4, row_count: null, node_count: 6, file_hash: '0123456789abcdef' }
  ]
}

describe('CollectionOverviewPreview', () => {
  it('renders the strip and the manifest row with a truncated hash', () => {
    render(<CollectionOverviewPreview overview={overview} />)
    expect(screen.getByText('a.pdf')).toBeInTheDocument()
    expect(screen.getByText('0123456789ab')).toBeInTheDocument()
    expect(screen.queryByText('0123456789abcdef')).not.toBeInTheDocument()
  })

  it('pluralizes the count strip (singular at 1, plural otherwise)', () => {
    // fixture: document_count=1, file_types=1, entity_types=1 → singular; node_count=6 → plural
    const { container } = render(<CollectionOverviewPreview overview={overview} />)
    const text = container.textContent ?? ''
    expect(text).toContain('1 document ·')
    expect(text).toContain('6 nodes ·')
    expect(text).toContain('1 file type ·')
    expect(text).toContain('1 entity type')
    expect(text).not.toContain('1 documents')
    expect(text).not.toContain('1 file types')
    expect(text).not.toContain('1 entity types')
  })
})
