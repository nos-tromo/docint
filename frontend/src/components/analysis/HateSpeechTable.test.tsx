import { describe, it, expect } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { HateSpeechTable } from './HateSpeechTable'
import type { HateSpeechRow } from '@/api/types'

const rows: HateSpeechRow[] = [
  {
    chunk_id: 'h1',
    filename: 'rant.txt',
    page: 2,
    category: 'harassment',
    confidence: 'high',
    reason: 'Targets a protected group.',
    chunk_text: 'Body of the flagged passage.',
    reference_metadata: {
      author: 'Carol',
      network: 'docs',
      timestamp: '2026-02-14'
    }
  }
]

describe('HateSpeechTable', () => {
  it('renders one row per finding with category, source and chunk text inline', () => {
    render(<HateSpeechTable rows={rows} collection="alpha" />)
    const row = screen.getByTestId('hate-speech-row')
    expect(within(row).getByText(/harassment/i)).toBeInTheDocument()
    expect(within(row).getByText('rant.txt')).toBeInTheDocument()
    // Chunk text is shown inline — no expansion required.
    expect(within(row).getByText(/Body of the flagged passage/)).toBeInTheDocument()
  })

  it('flattens reason, confidence and reference metadata into the metadata column', () => {
    render(<HateSpeechTable rows={rows} collection="alpha" />)
    const dl = screen.getByText(/^Reason$/).closest('dl') as HTMLElement
    expect(within(dl).getByText(/Targets a protected group/)).toBeInTheDocument()
    expect(within(dl).getByText(/^Confidence$/)).toBeInTheDocument()
    expect(within(dl).getByText('high')).toBeInTheDocument()
    expect(within(dl).getByText(/^Author$/)).toBeInTheDocument()
    expect(within(dl).getByText('Carol')).toBeInTheDocument()
    expect(within(dl).getByText(/^Network$/)).toBeInTheDocument()
    expect(within(dl).getByText('docs')).toBeInTheDocument()
  })

  it('shows the empty state when nothing was flagged', () => {
    render(<HateSpeechTable rows={[]} collection="alpha" />)
    expect(screen.getByText(/no flagged content/i)).toBeInTheDocument()
  })

  it('renders a streaming CSV download link to the right collection-scoped endpoint', () => {
    render(<HateSpeechTable rows={rows} collection="alpha" />)
    const link = screen.getByRole('link', { name: 'CSV' })
    expect(link).toHaveAttribute(
      'href',
      expect.stringContaining('/collections/alpha/export/hate-speech.csv')
    )
    expect(link).toHaveAttribute('download')
  })
})
