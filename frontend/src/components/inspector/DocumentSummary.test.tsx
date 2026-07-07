import { describe, it, expect } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { DocumentSummary } from './DocumentSummary'
import type { DocumentRecord } from '@/api/types'

const DOCS: DocumentRecord[] = [
  { filename: 'a.jpg', file_hash: 'h1', mimetype: 'image/jpeg', node_count: 1, entity_types: ['loc', 'org', 'person'] },
  { filename: 'b.csv', file_hash: 'h2', mimetype: 'text/csv', node_count: 138, entity_types: ['date', 'event'] }
]

/** Locate a KpiCard by its label and return a scope for asserting its value/hint. */
function card(label: string) {
  return within(screen.getByText(label).parentElement as HTMLElement)
}

describe('DocumentSummary', () => {
  it('uses the true document count and marks aggregates as partial when paginated', () => {
    render(<DocumentSummary docs={DOCS} totalCount={7} partial />)

    expect(card('Documents').getByText('7')).toBeInTheDocument()
    expect(card('Nodes').getByText('139+')).toBeInTheDocument()
    expect(screen.getByText('across loaded documents')).toBeInTheDocument()
  })

  it('breaks down file types and lists distinct entity types', () => {
    render(<DocumentSummary docs={DOCS} totalCount={2} />)

    // Equal counts tie-break alphabetically: CSV before JPEG.
    expect(card('File types').getByText('2')).toBeInTheDocument()
    expect(screen.getByText('1 CSV · 1 JPEG')).toBeInTheDocument()

    expect(card('Entity types').getByText('5')).toBeInTheDocument()
    expect(screen.getByText('date, event, loc, org, person')).toBeInTheDocument()
  })

  it('falls back to the loaded count and omits the partial marker when complete', () => {
    render(<DocumentSummary docs={DOCS} />)

    expect(card('Documents').getByText('2')).toBeInTheDocument()
    expect(card('Nodes').getByText('139')).toBeInTheDocument()
    expect(screen.queryByText('across loaded documents')).not.toBeInTheDocument()
  })
})
