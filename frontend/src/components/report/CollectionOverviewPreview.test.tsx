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
  it('arrives folded, with its totals on the bar and the manifest clipped', () => {
    render(<CollectionOverviewPreview overview={overview} />)
    const bar = screen.getByRole('button', { name: /document overview/i })
    // A sixteen-row manifest opening by default pushed the report's own
    // findings off the screen, so it arrives shut — but not blank: the totals
    // ride on the bar and the first rows peek out below it, clipped rather
    // than unmounted, so a folded section still says what is in it.
    expect(bar).toHaveAttribute('aria-expanded', 'false')
    // The totals sit beside the control on the bar, not inside it — the
    // disclosure is icon-only and names itself after the section.
    expect(screen.getByText(/1 document ·/)).toBeInTheDocument()
    const panel = document.getElementById(bar.getAttribute('aria-controls') ?? '')
    expect(panel).not.toBeNull()
    expect(panel!.style.maxHeight).not.toBe('')
  })

  it('renders the strip and the manifest row with a truncated hash once opened', async () => {
    render(<CollectionOverviewPreview overview={overview} />)
    const bar = screen.getByRole('button', { name: /document overview/i })
    await userEvent.click(bar)
    // Opened, the cap comes off entirely — the manifest scrolls in its own box
    // rather than being held to a peek.
    const panel = document.getElementById(bar.getAttribute('aria-controls') ?? '')
    expect(panel!.style.maxHeight).toBe('')
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
