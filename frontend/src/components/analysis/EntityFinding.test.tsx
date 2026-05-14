import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntityFinding } from './EntityFinding'
import { useUiStore } from '@/stores/ui'
import type { NerSourceRow } from '@/api/types'

beforeEach(() => {
  useUiStore.setState({
    selectedCollection: 'test-collection',
    currentSessionId: null,
    previewModal: null
  })
})

describe('EntityFinding', () => {
  it('shows file/location/score metadata even when reference_metadata is empty (PDF/CSV)', async () => {
    const source: NerSourceRow = {
      chunk_id: 'chunk-42',
      filename: 'paper.pdf',
      page: 7,
      file_hash: 'abc1234567890def1234567890',
      filetype: 'application/pdf',
      source: 'core_pdf',
      score: 0.84,
      chunk_text: 'The quick brown fox.',
      entities: [{ text: 'fox', type: 'ANIMAL', score: 0.91 }]
    }
    render(
      <EntityFinding
        index={1}
        source={source}
        highlightTerms={['fox']}
        selectedTypeLower="animal"
        defaultOpen
      />
    )
    const dl = screen.getByText(/^File$/).closest('dl') as HTMLElement
    expect(dl).toBeTruthy()
    expect(within(dl).getByText('paper.pdf')).toBeInTheDocument()
    expect(within(dl).getByText(/^Location$/)).toBeInTheDocument()
    expect(within(dl).getByText('page 7')).toBeInTheDocument()
    expect(within(dl).getByText(/^Score$/)).toBeInTheDocument()
    expect(within(dl).getByText('0.840')).toBeInTheDocument()
    expect(within(dl).getByText(/^Filetype$/)).toBeInTheDocument()
    expect(within(dl).getByText('application/pdf')).toBeInTheDocument()
    expect(within(dl).getByText(/^Chunk ID$/)).toBeInTheDocument()
    expect(within(dl).getByText('chunk-42')).toBeInTheDocument()
    expect(within(dl).getByText(/^File hash$/)).toBeInTheDocument()
  })

  it('renders the matched mentions chip list with per-mention score', () => {
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
    render(
      <EntityFinding
        index={1}
        source={source}
        highlightTerms={['Berlin']}
        selectedTypeLower="loc"
        defaultOpen
      />
    )
    const mentionsList = screen.getByText(/matched mentions/i).parentElement!
      .querySelector('ul') as HTMLElement
    expect(mentionsList).toBeTruthy()
    expect(within(mentionsList).getAllByText('Berlin')).toHaveLength(2)
    expect(within(mentionsList).queryByText('Paris')).not.toBeInTheDocument()
    expect(within(mentionsList).getByText(/0\.770/)).toBeInTheDocument()
    expect(within(mentionsList).getByText(/0\.810/)).toBeInTheDocument()
  })

  it('renders an Open original link only when collection and file_hash are present', async () => {
    const { rerender } = render(
      <EntityFinding
        index={1}
        source={{
          chunk_id: 'c',
          filename: 'a.pdf',
          file_hash: 'hash',
          chunk_text: 't',
          entities: []
        }}
        highlightTerms={[]}
        defaultOpen
      />
    )
    expect(screen.getByText(/open original/i)).toBeInTheDocument()
    // Without file_hash the link must be hidden.
    rerender(
      <EntityFinding
        index={1}
        source={{ chunk_id: 'c', filename: 'a.pdf', chunk_text: 't', entities: [] }}
        highlightTerms={[]}
        defaultOpen
      />
    )
    expect(screen.queryByText(/open original/i)).not.toBeInTheDocument()
  })

  it('toggles the detail block on click', async () => {
    render(
      <EntityFinding
        index={1}
        source={{ chunk_id: 'c', filename: 'a.pdf', chunk_text: 'body', entities: [] }}
        highlightTerms={[]}
      />
    )
    expect(screen.queryByText('body')).not.toBeInTheDocument()
    await userEvent.click(screen.getByText(/Chunk 1:/))
    expect(screen.getByText('body')).toBeInTheDocument()
  })
})
