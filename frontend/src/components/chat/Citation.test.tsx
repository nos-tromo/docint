import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Citation } from './Citation'
import { useUiStore } from '@/stores/ui'

describe('Citation', () => {
  it('offers a Translate control when the source has text', async () => {
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <Citation source={{ id: 's1', filename: 'f.pdf', text: 'Hola mundo' } as never} />
      </QueryClientProvider>
    )
    // The details are shown as soon as they mount: the pill that opened them
    // was the click.
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
    expect(screen.getByText('Hola')).toBeInTheDocument()
  })

  it('repeats neither the number nor the label the pill above carries', () => {
    // The pill is the details' heading (SourcePills tests its number and label).
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <Citation source={{ id: 's1', filename: 'f.pdf', text: 'Hola', citation_index: 3 } as never} />
      </QueryClientProvider>
    )
    const detail = screen.getByTestId('source-detail')
    expect(detail).not.toHaveTextContent('f.pdf')
    expect(detail).not.toHaveTextContent('3')
    // A quiet block on a left rule, not a boxed card.
    expect(detail).toHaveClass('border-l-2')
    expect(detail).not.toHaveClass('rounded-md')
  })

  it('says so in one line when the source carries no passage', () => {
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <Citation source={{ id: 's1', filename: 'f.txt' } as never} />
      </QueryClientProvider>
    )
    expect(screen.getByTestId('source-detail')).toHaveTextContent('No excerpt is stored for this source.')
    expect(screen.queryByRole('button', { name: /translate/i })).not.toBeInTheDocument()
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
    const dl = document.querySelector('dl')
    expect(dl?.className).toContain('grid-cols-[auto_minmax(0,1fr)]')
  })

  it('opens the preview dialog from the details instead of leaving the app', async () => {
    // The old full-width "Open original" anchor cost a row and navigated
    // away; the preview is an icon action at the details' right edge.
    useUiStore.setState({ selectedCollection: 'docs', previewModal: null })
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <Citation source={{ id: 's1', filename: 'f.pdf', file_hash: 'h1' } as never} />
      </QueryClientProvider>
    )

    await userEvent.click(screen.getByRole('button', { name: /preview/i }))

    expect(useUiStore.getState().previewModal).toEqual({
      collection: 'docs',
      file_hash: 'h1',
      filename: 'f.pdf'
    })
    expect(screen.queryByRole('link', { name: /original/i })).not.toBeInTheDocument()
  })

  it('omits the preview action when the source has no stored file', () => {
    useUiStore.setState({ selectedCollection: 'docs', previewModal: null })
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <Citation source={{ id: 's1', filename: 'f.pdf' } as never} />
      </QueryClientProvider>
    )
    expect(screen.queryByRole('button', { name: /preview/i })).not.toBeInTheDocument()
  })
})
