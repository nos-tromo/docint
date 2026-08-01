import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Citation } from './Citation'

describe('Citation', () => {
  it('offers a Translate control when the source has text', async () => {
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <Citation source={{ id: 's1', filename: 'f.pdf', text: 'Hola mundo' } as never} />
      </QueryClientProvider>
    )
    // The snippet (and the TranslateControl mounted next to it) only renders
    // once the citation card is expanded.
    await userEvent.click(screen.getByRole('button', { name: 'f.pdf' }))
    expect(screen.getByText('Hola mundo')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /translate/i })).toBeInTheDocument()
  })

  it('does not display the relevance score', () => {
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <Citation source={{ id: 's1', filename: 'f.pdf', text: 'Hola', score: 0.842 } as never} />
      </QueryClientProvider>
    )
    expect(screen.queryByText('0.842')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'f.pdf' })).toBeInTheDocument()
  })

  it('caps the metadata value track so unbreakable values cannot overflow the card', async () => {
    // Same regression class as the analysis tables: a plain `1fr` value track
    // is floored at min-content, so an unbreakable URL widens the dl past the
    // card (break-word does not reduce min-content).
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <Citation
          source={
            {
              id: 's1',
              filename: 'f.pdf',
              text: 'Hola',
              reference_metadata: { url: 'https://example.invalid/' + 'x'.repeat(160) },
            } as never
          }
        />
      </QueryClientProvider>
    )
    await userEvent.click(screen.getByRole('button', { name: 'f.pdf' }))
    const dl = document.querySelector('dl')
    expect(dl?.className).toContain('grid-cols-[auto_minmax(0,1fr)]')
  })
})
