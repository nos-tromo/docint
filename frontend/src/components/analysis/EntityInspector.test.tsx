import { describe, it, expect } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntityInspector } from './EntityInspector'
import type { NerEntityRow, NerSourceRow } from '@/api/types'

const entities: NerEntityRow[] = [
  { text: 'Berlin', type: 'LOC', mentions: 4, variants: [{ text: 'Berlin-Mitte' }] },
  { text: 'Alice', type: 'PER', mentions: 2 }
]

const sources: NerSourceRow[] = [
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
  },
  {
    chunk_id: 'c3',
    filename: 'doc.pdf',
    page: 9,
    chunk_text: 'Alice met Bob.',
    entities: [{ text: 'Alice', type: 'PER' }]
  }
]

describe('EntityInspector', () => {
  it('picks the first entity by default and lists chunks where it appears (incl. variants)', () => {
    render(<EntityInspector entities={entities} sources={sources} />)
    // Default selection is the first entity (Berlin). Chunks c1 and c2 both
    // reference Berlin/Berlin-Mitte; c3 does not.
    const heading = screen.getByText(/findings for/i).parentElement!
    expect(heading).toHaveTextContent('Berlin')
    expect(heading).toHaveTextContent(/2 chunks/i)
    expect(screen.getByText(/Chunk 1:/)).toBeInTheDocument()
    expect(screen.getByText(/Chunk 2:/)).toBeInTheDocument()
    expect(screen.queryByText(/Chunk 3:/)).not.toBeInTheDocument()
  })

  it('switches findings when the user picks a different entity', async () => {
    render(<EntityInspector entities={entities} sources={sources} />)
    const entityDropdown = screen.getByLabelText(/^entity$/i)
    await userEvent.selectOptions(entityDropdown, 'Alice::PER')
    const heading = screen.getByText(/findings for/i).parentElement!
    expect(heading).toHaveTextContent('Alice')
    expect(heading).toHaveTextContent(/2 chunks/i)
  })

  it('filters the entity dropdown by type', async () => {
    render(<EntityInspector entities={entities} sources={sources} />)
    const typeDropdown = screen.getByLabelText(/entity category/i)
    await userEvent.selectOptions(typeDropdown, 'PER')
    const entityDropdown = screen.getByLabelText(/^entity$/i) as HTMLSelectElement
    expect(
      Array.from(entityDropdown.options).map((o) => o.value)
    ).toEqual(['Alice::PER'])
  })

  it('shows reference metadata and highlights the entity text when a finding is expanded', async () => {
    render(<EntityInspector entities={entities} sources={sources} />)
    // First finding auto-opens when there's exactly one match — here there
    // are two, so expand chunk 1 manually.
    await userEvent.click(screen.getByText(/Chunk 1:/))
    expect(screen.getByText(/Author/)).toBeInTheDocument()
    expect(screen.getByText('Bob')).toBeInTheDocument()
    // The matching "Berlin" inside the chunk body should be wrapped in <mark>.
    const findingBlock = screen.getByText(/Chunk 1:/).closest('div')!
    const mark = within(findingBlock.parentElement!).getAllByText('Berlin')
      .find((el) => el.tagName === 'MARK')
    expect(mark).toBeDefined()
  })

  it('falls back to a helpful message when the collection has no entities', () => {
    render(<EntityInspector entities={[]} sources={[]} />)
    expect(screen.getByText(/no entities found/i)).toBeInTheDocument()
  })
})
