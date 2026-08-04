import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { Source } from '@/api/types'
import { SourcePills } from './SourcePills'

function renderPills(sources: Partial<Source>[]) {
  const qc = new QueryClient()
  return render(
    <QueryClientProvider client={qc}>
      <SourcePills sources={sources as Source[]} />
    </QueryClientProvider>
  )
}

describe('SourcePills', () => {
  it('renders nothing without sources', () => {
    const { container } = renderPills([])
    expect(container).toBeEmptyDOMElement()
  })

  it('renders one pill per source with its label and citation number', () => {
    renderPills([
      { id: 's1', filename: 'a.pdf', page: 3, citation_index: 1 },
      { id: 's2', filename: 'b.csv', row: 7, citation_index: 2 }
    ])

    const pills = screen.getAllByTestId('source-pill')
    expect(pills).toHaveLength(2)
    expect(pills[0]).toHaveTextContent('a.pdf · page 3')
    expect(pills[0]).toHaveTextContent('1')
    expect(pills[1]).toHaveTextContent('b.csv · row 7')
    expect(pills[1]).toHaveTextContent('2')
  })

  it('leaves a pill unnumbered when the generator never saw the source', () => {
    renderPills([{ id: 's1', filename: 'shot.png' }])
    expect(screen.getByTestId('source-pill')).not.toHaveTextContent(/\d/)
  })

  it('shows no detail card until a pill is clicked', () => {
    renderPills([
      {
        id: 's1',
        filename: 'a.pdf',
        text: 'chunk text',
        reference_metadata: { author: 'alice' }
      }
    ])
    expect(screen.queryByText('chunk text')).not.toBeInTheDocument()
  })

  it('expands the clicked pill into its detail card, already open', async () => {
    renderPills([
      {
        id: 's1',
        filename: 'a.pdf',
        text: 'chunk text',
        reference_metadata: { author: 'alice' }
      }
    ])

    await userEvent.click(screen.getByTestId('source-pill'))

    // One click on the pill lands with the card details expanded — the
    // snippet-backed Translate control and the metadata are visible without
    // a second click on the card header.
    expect(screen.getByText('alice')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /translate/i })).toBeInTheDocument()
  })

  it('collapses the card when the same pill is clicked again', async () => {
    renderPills([{ id: 's1', filename: 'a.pdf', text: 'chunk text' }])

    await userEvent.click(screen.getByTestId('source-pill'))
    expect(screen.getByText('chunk text')).toBeInTheDocument()

    await userEvent.click(screen.getByTestId('source-pill'))
    expect(screen.queryByText('chunk text')).not.toBeInTheDocument()
  })

  it('switches the card when another pill is clicked', async () => {
    renderPills([
      { id: 's1', filename: 'a.pdf', text: 'first chunk' },
      { id: 's2', filename: 'b.pdf', text: 'second chunk' }
    ])

    const pills = screen.getAllByTestId('source-pill')
    await userEvent.click(pills[0])
    expect(screen.getByText('first chunk')).toBeInTheDocument()

    await userEvent.click(pills[1])
    expect(screen.queryByText('first chunk')).not.toBeInTheDocument()
    expect(screen.getByText('second chunk')).toBeInTheDocument()
  })
})
