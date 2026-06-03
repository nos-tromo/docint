import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntityInspector } from './EntityInspector'
import type { NerEntityRow, NerSourceRow } from '@/api/types'

const entities: NerEntityRow[] = [
  { text: 'Berlin', type: 'LOC', mentions: 4, variants: [{ text: 'Berlin-Mitte' }] },
  { text: 'Alice', type: 'PER', mentions: 2 }
]

const berlinFindings: NerSourceRow[] = [
  {
    chunk_id: 'c1',
    filename: 'doc.pdf',
    page: 3,
    chunk_text: 'Berlin is the capital. Berlin-Mitte is a district.',
    entities: [{ text: 'Berlin', type: 'LOC' }],
    reference_metadata: { author: 'Bob', timestamp: '2026-01-01' }
  },
  {
    chunk_id: 'c2',
    filename: 'doc.pdf',
    page: 7,
    chunk_text: 'Alice traveled to Berlin-Mitte.',
    entities: [
      { text: 'Alice', type: 'PER' },
      { text: 'Berlin-Mitte', type: 'LOC' }
    ]
  }
]

const keyOf = (e: NerEntityRow) => `${e.text}::${e.type}`

describe('EntityInspector', () => {
  it('renders the entity dropdown options labelled with mentions and type', () => {
    render(
      <EntityInspector
        entities={entities}
        selectedKey="Berlin::LOC"
        onSelectEntity={() => {}}
        findings={berlinFindings}
        collection="alpha"
        keyOf={keyOf}
      />
    )
    const dropdown = screen.getByLabelText(/^entity$/i) as HTMLSelectElement
    expect(Array.from(dropdown.options).map((o) => o.value)).toEqual([
      'Berlin::LOC',
      'Alice::PER'
    ])
    expect(dropdown.options[0].textContent).toMatch(/Berlin \[LOC\] · 4/)
  })

  it('shows the findings count for the selected entity', () => {
    render(
      <EntityInspector
        entities={entities}
        selectedKey="Berlin::LOC"
        onSelectEntity={() => {}}
        findings={berlinFindings}
        collection="alpha"
        keyOf={keyOf}
      />
    )
    const heading = screen.getByText(/findings for/i).parentElement!
    expect(heading).toHaveTextContent('Berlin')
    expect(heading).toHaveTextContent(/2 chunks/i)
  })

  it('calls onSelectEntity with the new key when the user changes entities', async () => {
    const onSelectEntity = vi.fn()
    render(
      <EntityInspector
        entities={entities}
        selectedKey="Berlin::LOC"
        onSelectEntity={onSelectEntity}
        findings={berlinFindings}
        collection="alpha"
        keyOf={keyOf}
      />
    )
    const dropdown = screen.getByLabelText(/^entity$/i)
    await userEvent.selectOptions(dropdown, 'Alice::PER')
    expect(onSelectEntity).toHaveBeenCalledWith('Alice::PER')
  })

  it('renders the CSV download link with the selected entity in the query string', () => {
    render(
      <EntityInspector
        entities={entities}
        selectedKey="Berlin::LOC"
        onSelectEntity={() => {}}
        findings={berlinFindings}
        collection="alpha"
        keyOf={keyOf}
      />
    )
    const link = screen.getByRole('link', { name: 'CSV' })
    const href = link.getAttribute('href') ?? ''
    expect(href).toContain('/collections/alpha/export/ner-sources.csv')
    expect(href).toContain('entity_text=Berlin')
    expect(href).toContain('entity_type=LOC')
    expect(link).toHaveAttribute('download')
  })

  it('falls back to a helpful message when the collection has no entities', () => {
    render(
      <EntityInspector
        entities={[]}
        selectedKey={null}
        onSelectEntity={() => {}}
        findings={[]}
        collection="alpha"
        keyOf={keyOf}
      />
    )
    expect(screen.getByText(/no entities found/i)).toBeInTheDocument()
  })
})
