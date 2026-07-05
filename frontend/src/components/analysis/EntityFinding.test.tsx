import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { EntityFinding } from './EntityFinding'
import { useUiStore } from '@/stores/ui'
import { entityFindingSnapshot } from '@/lib/reportSnapshots'
import type { NerSourceRow } from '@/api/types'

const GRID = '2.5rem 1fr 1fr 2fr 6rem'

beforeEach(() => {
  useUiStore.setState({
    selectedCollection: 'test-collection',
    currentSessionId: null,
    previewModal: null
  })
})

describe('EntityFinding', () => {
  it('shows source, locator and reference metadata inline (no expansion needed)', () => {
    const source: NerSourceRow = {
      chunk_id: 'chunk-42',
      filename: 'paper.pdf',
      page: 7,
      file_hash: 'abc1234567890def1234567890',
      filetype: 'application/pdf',
      source: 'core_pdf',
      score: 0.84,
      chunk_text: 'The quick brown fox.',
      reference_metadata: { author: 'Bob' },
      entities: [{ text: 'fox', type: 'ANIMAL', score: 0.91 }]
    }
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <EntityFinding
          index={1}
          source={source}
          highlightTerms={['fox']}
          selectedTypeLower="animal"
          gridTemplate={GRID}
        />
      </QueryClientProvider>
    )
    // Source column.
    expect(screen.getByText('paper.pdf')).toBeInTheDocument()
    expect(screen.getByText('page 7')).toBeInTheDocument()
    // Metadata is collapsed into one column (a single dl).
    const dl = screen.getByText(/^Score$/).closest('dl') as HTMLElement
    expect(within(dl).getByText('0.840')).toBeInTheDocument()
    expect(within(dl).getByText(/^Filetype$/)).toBeInTheDocument()
    expect(within(dl).getByText('application/pdf')).toBeInTheDocument()
    expect(within(dl).getByText(/^Chunk ID$/)).toBeInTheDocument()
    expect(within(dl).getByText('chunk-42')).toBeInTheDocument()
    expect(within(dl).getByText(/^File hash$/)).toBeInTheDocument()
    expect(within(dl).getByText(/^Author$/)).toBeInTheDocument()
    // Chunk text is rendered inline.
    expect(screen.getByText('The quick brown')).toBeInTheDocument()
  })

  it('renders matched mentions with score and excludes mismatched types', () => {
    const source: NerSourceRow = {
      chunk_id: 'c1',
      filename: 'a.pdf',
      page: 1,
      chunk_text: 'Berlin Berlin Berlin.',
      entities: [
        { text: 'Berlin', type: 'LOC', score: 0.77 },
        { text: 'Berlin', type: 'LOC', score: 0.81 },
        { text: 'Paris', type: 'LOC', score: 0.5 }
      ]
    }
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <EntityFinding
          index={1}
          source={source}
          highlightTerms={['Berlin']}
          selectedTypeLower="loc"
          gridTemplate={GRID}
        />
      </QueryClientProvider>
    )
    const mentions = screen.getByLabelText(/matched mentions/i)
    expect(within(mentions).getAllByText('Berlin')).toHaveLength(2)
    expect(within(mentions).queryByText('Paris')).not.toBeInTheDocument()
    expect(within(mentions).getByText(/0\.770/)).toBeInTheDocument()
    expect(within(mentions).getByText(/0\.810/)).toBeInTheDocument()
  })

  it('renders an Open original link only when collection and file_hash are present', () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { rerender } = render(
      <QueryClientProvider client={qc}>
        <EntityFinding
          index={1}
          source={{ chunk_id: 'c', filename: 'a.pdf', file_hash: 'hash', chunk_text: 't', entities: [] }}
          highlightTerms={[]}
          gridTemplate={GRID}
        />
      </QueryClientProvider>
    )
    expect(screen.getByText(/open original/i)).toBeInTheDocument()
    rerender(
      <QueryClientProvider client={qc}>
        <EntityFinding
          index={1}
          source={{ chunk_id: 'c', filename: 'a.pdf', chunk_text: 't', entities: [] }}
          highlightTerms={[]}
          gridTemplate={GRID}
        />
      </QueryClientProvider>
    )
    expect(screen.queryByText(/open original/i)).not.toBeInTheDocument()
  })

  it('clamps long chunk text behind a Show more / Show less toggle', async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <EntityFinding
          index={1}
          source={{ chunk_id: 'c', filename: 'a.pdf', chunk_text: 'x'.repeat(300), entities: [] }}
          highlightTerms={[]}
          gridTemplate={GRID}
        />
      </QueryClientProvider>
    )
    await userEvent.click(screen.getByRole('button', { name: /show more/i }))
    expect(screen.getByRole('button', { name: /show less/i })).toBeInTheDocument()
  })

  it('exposes the Add to report control when report context is provided', () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <EntityFinding
          index={1}
          source={{ chunk_id: 'c', filename: 'a.pdf', chunk_text: 't', entities: [] }}
          highlightTerms={[]}
          entityLabel="fox [ANIMAL]"
          reportDedupeKeys={new Set()}
          gridTemplate={GRID}
        />
      </QueryClientProvider>
    )
    expect(screen.getByRole('button', { name: /\+ report/i })).toBeInTheDocument()
  })

  it('includes the translation in the report snapshot after translating', () => {
    const snap = entityFindingSnapshot(
      { chunk_id: 'c1', chunk_text: 'orig' } as never,
      'ACME',
      { text: 'übersetzt', target_lang: 'de', model: 'm' }
    )
    expect(snap.snapshot.translation).toEqual({ text: 'übersetzt', target_lang: 'de', model: 'm' })
  })
})
