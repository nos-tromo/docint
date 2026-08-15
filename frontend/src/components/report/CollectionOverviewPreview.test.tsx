import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
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
  it('keeps the manifest behind its heading until asked', () => {
    render(<CollectionOverviewPreview overview={overview} />)
    // The counts are the summary and stay put; the table is the detail. A
    // sixteen-row manifest opening by default pushed the report's own findings
    // off the screen.
    expect(screen.getByText(/1 document ·/)).toBeInTheDocument()
    expect(screen.queryByText('a.pdf')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /document overview/i })).toHaveAttribute(
      'aria-expanded',
      'false'
    )
  })

  it('renders the strip and the manifest row with a truncated hash once opened', async () => {
    render(<CollectionOverviewPreview overview={overview} />)
    await userEvent.click(screen.getByRole('button', { name: /document overview/i }))
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
