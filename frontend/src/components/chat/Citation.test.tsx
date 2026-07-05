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
})
