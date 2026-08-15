import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ChatTurn, type ChatTurnData } from './ChatTurn'
import type { ChatFinalEvent } from '@/api/types'
import { useUiStore } from '@/stores/ui'

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'docs', previewModal: null })
})

function renderTurn(turn: ChatTurnData) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <ChatTurn turn={turn} />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('ChatTurn', () => {
  it('surfaces the entities behind the answer above its sources', () => {
    const meta = {
      session_id: 's',
      sources: [
        {
          filename: 'handbook.pdf',
          page: 26,
          text: 'Alice Weber met the Acme Corp board.',
          entities: [
            { text: 'Alice Weber', type: 'PER' },
            { text: 'Acme Corp', type: 'ORG' }
          ]
        }
      ]
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice Weber.', done: true, meta })

    const pills = screen.getAllByTestId('answer-entity')
    expect(pills.map((p) => p.textContent)).toEqual(
      expect.arrayContaining([expect.stringContaining('Alice Weber')])
    )

    // Order matters: the entities summarize the answer, so they belong
    // between it and the evidence, not after the citation list.
    const entities = screen.getByTestId('answer-entities')
    const sources = screen.getByText(/handbook\.pdf/)
    expect(entities.compareDocumentPosition(sources)).toBe(Node.DOCUMENT_POSITION_FOLLOWING)
  })

  it('renders the sources as pills matching the entity presentation', () => {
    const meta = {
      session_id: 's',
      sources: [
        { id: 's1', filename: 'handbook.pdf', page: 26, text: 'first', citation_index: 1 },
        { id: 's2', filename: 'notes.md', text: 'second', citation_index: 2 }
      ]
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice Weber.', done: true, meta })

    const pills = screen.getAllByTestId('source-pill')
    expect(pills).toHaveLength(2)
    expect(pills[0]).toHaveTextContent('handbook.pdf · page 26')
    // Compact by default: no detail card until a pill is clicked.
    expect(screen.queryByText('first')).not.toBeInTheDocument()
  })

  it('renders no entity row for an answer whose sources carry none', () => {
    const meta = {
      session_id: 's',
      sources: [{ filename: 'handbook.pdf', page: 26, text: 'no entities here' }]
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Nobody.', done: true, meta })

    expect(screen.queryByTestId('answer-entities')).not.toBeInTheDocument()
  })

  it('stays quiet when the server confirms it answered from the selection', () => {
    const meta = {
      session_id: 's',
      sources: [],
      retrieval_mode: 'scoped',
      scoped_chunk_count: 2
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta, scopeRequested: 2 })

    expect(screen.queryByTestId('scope-not-applied')).not.toBeInTheDocument()
  })

  it('flags a turn that asked for a scope and got an unscoped answer', () => {
    // Without this the two are indistinguishable in the transcript, which is
    // how a dropped scope came to be presented as hand-picked evidence.
    const meta = {
      session_id: 's',
      sources: [],
      retrieval_mode: 'rewrite_compact'
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Something else.', done: true, meta, scopeRequested: 2 })

    expect(screen.getByTestId('scope-not-applied')).toBeInTheDocument()
  })

  it('names the corrective retry a reloaded session replays', () => {
    // The live retry frame is long gone by then; the provenance has to come
    // off the persisted turn or a reloaded transcript would present a second
    // attempt as though it were the original answer.
    const meta = {
      session_id: 's',
      sources: [],
      retried: true,
      retry_query: 'Security Council resolutions'
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'what did the UN say?', assistant: 'Three resolutions.', done: true, meta })

    expect(screen.getByTestId('retry-notice')).toHaveTextContent(/Security Council resolutions/)
  })

  it('shows no retry notice on an ordinary answer', () => {
    const meta = { session_id: 's', sources: [] } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta })

    expect(screen.queryByTestId('retry-notice')).not.toBeInTheDocument()
  })
})
