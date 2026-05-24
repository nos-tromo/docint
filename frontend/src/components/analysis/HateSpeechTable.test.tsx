import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
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
  it('renders a summary line for each finding', () => {
    render(<HateSpeechTable rows={rows} collection="alpha" />)
    expect(screen.getByText(/harassment/i)).toBeInTheDocument()
    expect(screen.getByText(/Targets a protected group/)).toBeInTheDocument()
  })

  it('reveals the chunk text and reference metadata when a row is expanded', async () => {
    render(<HateSpeechTable rows={rows} collection="alpha" />)
    await userEvent.click(screen.getByText(/harassment/i))
    expect(screen.getByText(/Body of the flagged passage/)).toBeInTheDocument()
    expect(screen.getByText('Carol')).toBeInTheDocument()
    expect(screen.getByText(/^Author$/)).toBeInTheDocument()
    expect(screen.getByText(/^Network$/)).toBeInTheDocument()
    expect(screen.getByText('docs')).toBeInTheDocument()
    expect(screen.getByText('high')).toBeInTheDocument()
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
