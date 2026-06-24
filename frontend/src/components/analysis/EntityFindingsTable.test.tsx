import { describe, expect, it, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { EntityFindingsTable } from './EntityFindingsTable'
import { useUiStore } from '@/stores/ui'
import type { NerEntityRow, NerSourceRow } from '@/api/types'

const selected: NerEntityRow = {
  text: 'Berlin',
  type: 'LOC',
  mentions: 4,
  variants: [{ text: 'Berlin-Mitte' }]
}

const findings: NerSourceRow[] = [
  {
    chunk_id: 'c1',
    filename: 'doc.pdf',
    page: 3,
    chunk_text: 'Berlin is the capital.',
    entities: [{ text: 'Berlin', type: 'LOC' }]
  },
  {
    chunk_id: 'c2',
    filename: 'doc.pdf',
    page: 7,
    chunk_text: 'Alice traveled to Berlin-Mitte.',
    entities: [{ text: 'Berlin-Mitte', type: 'LOC' }]
  }
]

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: null, previewModal: null })
})

describe('EntityFindingsTable', () => {
  it('shows the findings count and a table header for the selected entity', () => {
    render(<EntityFindingsTable selected={selected} findings={findings} collection="alpha" />)
    const heading = screen.getByText(/findings for/i).parentElement!
    expect(heading).toHaveTextContent('Berlin')
    expect(heading).toHaveTextContent(/2 chunks/i)
    // Column headers present (table layout, not an accordion).
    expect(screen.getByText('Metadata')).toBeInTheDocument()
    expect(screen.getByText('Source')).toBeInTheDocument()
  })

  it('renders one row per finding with its chunk text inline', () => {
    render(<EntityFindingsTable selected={selected} findings={findings} collection="alpha" />)
    expect(screen.getAllByTestId('entity-finding-row')).toHaveLength(2)
    // "Berlin" is highlighted (a <mark>), so assert on the trailing segment.
    expect(screen.getByText(/is the capital/)).toBeInTheDocument()
  })

  it('renders the CSV download link with the selected entity in the query string', () => {
    render(<EntityFindingsTable selected={selected} findings={findings} collection="alpha" />)
    const link = screen.getByRole('link', { name: 'CSV' })
    const href = link.getAttribute('href') ?? ''
    expect(href).toContain('/collections/alpha/export/ner-sources.csv')
    expect(href).toContain('entity_text=Berlin')
    expect(href).toContain('entity_type=LOC')
    expect(link).toHaveAttribute('download')
  })

  it('prompts to pick an entity when none is selected', () => {
    render(<EntityFindingsTable selected={null} findings={[]} collection="alpha" />)
    expect(screen.getByText(/pick an entity/i)).toBeInTheDocument()
  })

  it('shows an empty state when the selected entity has no matched chunks', () => {
    render(<EntityFindingsTable selected={selected} findings={[]} collection="alpha" />)
    expect(screen.getByText(/no chunks were matched/i)).toBeInTheDocument()
  })
})
