import { describe, it, expect } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { DocumentSummary } from './DocumentSummary'
import type { DocumentsSummary } from '@/api/types'

const SUMMARY: DocumentsSummary = {
  document_count: 305,
  node_count: 305,
  file_types: [
    { label: 'JPEG', count: 304 },
    { label: 'PNG', count: 1 }
  ],
  entity_types: ['loc', 'org', 'person']
}

/** Locate a KpiCard by its label and return a scope for asserting its value/hint. */
function card(label: string) {
  return within(screen.getByText(label).parentElement as HTMLElement)
}

describe('DocumentSummary', () => {
  it('renders collection-wide totals and breakdowns from the server summary', () => {
    render(<DocumentSummary summary={SUMMARY} />)

    expect(card('Documents').getByText('305')).toBeInTheDocument()
    expect(card('Nodes').getByText('305')).toBeInTheDocument()
    // The image tally reflects the whole collection (304), not just loaded rows.
    expect(card('File types').getByText('2')).toBeInTheDocument()
    expect(screen.getByText('304 JPEG · 1 PNG')).toBeInTheDocument()

    expect(card('Entity types').getByText('3')).toBeInTheDocument()
    expect(screen.getByText('loc, org, person')).toBeInTheDocument()
  })

  it('shows placeholders while the summary is loading', () => {
    render(<DocumentSummary />)

    expect(card('Documents').getByText('—')).toBeInTheDocument()
    expect(card('File types').getByText('—')).toBeInTheDocument()
    expect(screen.getByText('none extracted')).toBeInTheDocument()
  })

  it('caps the file-type hint tail into a +N summary', () => {
    render(
      <DocumentSummary
        summary={{
          document_count: 11,
          node_count: 11,
          file_types: [
            { label: 'JPEG', count: 4 },
            { label: 'PNG', count: 3 },
            { label: 'PDF', count: 2 },
            { label: 'CSV', count: 1 },
            { label: 'TXT', count: 1 }
          ],
          entity_types: []
        }}
      />
    )

    // Five types: the first four are shown, the tail collapses into "+1".
    expect(screen.getByText('4 JPEG · 3 PNG · 2 PDF · 1 CSV · +1')).toBeInTheDocument()
    expect(card('Entity types').getByText('—')).toBeInTheDocument()
  })
})
