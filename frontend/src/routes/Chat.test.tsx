import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useNavigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Chat, chatReducer } from './Chat'
import type { ChatFinalEvent } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import { useChatUiStore } from '@/stores/chatUi'
import { useSearchUiStore } from '@/stores/searchUi'

function bodyFromString(s: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(c) {
      c.enqueue(enc.encode(s))
      c.close()
    }
  })
}

function renderChat() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/chat']}>
        <Routes>
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: null, currentSessionId: null, previewModal: null })
  useChatUiStore.setState({ drafts: {}, sidePanelOpen: true })
  useSearchUiStore.setState({ drafts: {}, queries: {}, scopes: {}, filtersOpen: false })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Chat SSE handling', () => {
  it('renders streamed tokens from untyped SSE frames and finalizes on the metadata envelope', async () => {
    // Mirrors the backend's actual /stream_query output: every frame is
    // `data: {...}` with no `event:` line, so each event surfaces as the
    // SSE default ('message'). The discriminator must be the payload
    // shape, not the event name.
    const frames =
      'data: {"token":"Hello"}\n\n' +
      'data: {"token":" world"}\n\n' +
      'data: {"response":"Hello world","sources":[],"session_id":"sess-1"}\n\n'

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    const textarea = await screen.findByPlaceholderText(/ask something/i)
    await userEvent.type(textarea, 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText(/Hello world/)).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(useUiStore.getState().currentSessionId).toBe('sess-1')
    })
  })

  it('sends the selected collection in the /stream_query request body', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue({
        ok: true,
        status: 200,
        body: bodyFromString(
          'data: {"response":"ok","sources":[],"session_id":"s"}\n\n'
        )
      })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'test-collection' })

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      const streamCall = fetchMock.mock.calls.find(([u]) =>
        String(u).includes('/stream_query')
      )
      expect(streamCall).toBeDefined()
      expect(JSON.parse(streamCall![1].body)).toMatchObject({
        question: 'hi',
        collection: 'test-collection'
      })
    })
  })

  it('marks the turn done on an untyped error frame instead of waiting forever, never rendering the raw field', async () => {
    // The backend's `error` field is a static protocol flag post-D2, not
    // prose — the UI must show catalog copy instead of the field itself.
    const frames = 'data: {"error":"boom: internal-host:9000 unreachable"}\n\n'

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    const textarea = await screen.findByPlaceholderText(/ask something/i)
    await userEvent.type(textarea, 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText(/\(no answer\)/i)).toBeInTheDocument()
    })
    // An error frame without a code is still the backend reporting a
    // failure, so it gets the reported-failure copy — just with no token
    // to append. Only a dropped connection may say the stream ended.
    expect(screen.getByText(/answer could not be generated/i)).toBeInTheDocument()
    expect(screen.queryByText(/boom/)).not.toBeInTheDocument()
  })

  it('surfaces a backend-likely-crashed message when the stream throws (e.g., OOM kill)', async () => {
    // Reader.read() rejects with a TypeError mid-stream — the same shape
    // the browser fetch surfaces when nginx closes the upstream because
    // the backend died. The chat reducer must convert this into a static,
    // generic message — never relaying the raw transport error verbatim.
    const aborting = new ReadableStream<Uint8Array>({
      start(c) {
        const enc = new TextEncoder()
        c.enqueue(enc.encode('data: {"token":"hi "}\n\n'))
        c.error(new TypeError('network error'))
      }
    })
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: aborting })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(
        screen.getByText(/stream ended unexpectedly/i)
      ).toBeInTheDocument()
    })
    // The raw transport error is never rendered — only static, generic copy.
    expect(screen.queryByText(/network error/)).not.toBeInTheDocument()
  })

  it('Enter submits, Shift+Enter inserts a newline', async () => {
    const frames =
      'data: {"token":"ok"}\n\n' +
      'data: {"response":"ok","sources":[],"session_id":"s"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    const textarea = (await screen.findByPlaceholderText(/ask something/i)) as HTMLTextAreaElement

    await userEvent.type(textarea, 'line1{Shift>}{Enter}{/Shift}line2')
    expect(textarea.value).toBe('line1\nline2')

    await userEvent.type(textarea, '{Enter}')
    await waitFor(() => {
      expect(textarea.value).toBe('')
    })
    await waitFor(() => {
      expect(screen.getByText(/^ok$/)).toBeInTheDocument()
    })
  })

  it('renders markdown bold in assistant output', async () => {
    const frames =
      'data: {"token":"**bold**"}\n\n' +
      'data: {"response":"**bold** answer","sources":[],"session_id":"s"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      const strong = screen.getByText('bold')
      expect(strong.tagName).toBe('STRONG')
    })
  })

  it('drops the image-only artifact duplicate that lacks file_hash', async () => {
    const sources = [
      { filename: 'img.jpg', file_hash: 'abc', score: 0.9, text: 'A' },
      { filename: 'img.jpg', score: 0.8, text: 'B' }
    ]
    const final = {
      response: 'see source',
      sources,
      session_id: 's'
    }
    const frames =
      'data: {"token":"see source"}\n\n' + `data: ${JSON.stringify(final)}\n\n`
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )
    useUiStore.setState({ selectedCollection: 'test-collection' })

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      // The sibling without file_hash 404s on preview — keep only the
      // resolvable one.
      const matches = screen.getAllByText('img.jpg')
      expect(matches).toHaveLength(1)
    })
  })

  it('keeps multiple distinct chunks from the same file', async () => {
    const sources = [
      {
        filename: 'doc.pdf',
        file_hash: 'h1',
        page: 3,
        score: 0.9,
        text: 'first reference snippet'
      },
      {
        filename: 'doc.pdf',
        file_hash: 'h1',
        page: 7,
        score: 0.85,
        text: 'second reference snippet'
      },
      {
        filename: 'transcript.txt',
        file_hash: 'h2',
        score: 0.8,
        text: 'segment one'
      },
      {
        filename: 'transcript.txt',
        file_hash: 'h2',
        score: 0.78,
        text: 'segment two'
      }
    ]
    const final = { response: 'see sources', sources, session_id: 's' }
    const frames =
      'data: {"token":"see sources"}\n\n' + `data: ${JSON.stringify(final)}\n\n`
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )
    useUiStore.setState({ selectedCollection: 'test-collection' })

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText('doc.pdf · page 3')).toBeInTheDocument()
      expect(screen.getByText('doc.pdf · page 7')).toBeInTheDocument()
      // Two text-file chunks with no page/row are kept because their
      // chunk text differs.
      expect(screen.getAllByText('transcript.txt')).toHaveLength(2)
    })
  })

  it('surfaces a collection-mismatch message on a 409', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 409, body: null }))
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 'sess-x' })

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText(/belongs to a different collection/i)).toBeInTheDocument()
    })
  })

  it('prompts to select a collection on a 400', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 400, body: null }))

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText(/select a collection before chatting/i)).toBeInTheDocument()
    })
  })
})

function renderChatWithSession(sessionId: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[`/chat/${sessionId}`]}>
        <Routes>
          <Route path="/chat/:sessionId" element={<Chat />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

function mockHistoryFetch(messages: unknown[]) {
  // /sessions/{id}/history is the only request fired during restoration;
  // anything else (e.g. /collections/list) is ignored — the chat renders
  // without it.
  vi.stubGlobal(
    'fetch',
    vi.fn((req: RequestInfo | URL) => {
      const u = typeof req === 'string' ? req : req.toString()
      if (u.includes('/sessions/') && u.includes('/history')) {
        return Promise.resolve(
          new Response(JSON.stringify({ messages }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
          })
        )
      }
      return Promise.resolve(new Response('null', { status: 200 }))
    })
  )
}

describe('Chat session-history validation restoration', () => {
  it('renders the validation banner state from restored session history', async () => {
    mockHistoryFetch([
      { role: 'user', content: 'hi' },
      {
        role: 'assistant',
        content: 'hello world',
        sources: [],
        validation_checked: true,
        validation_mismatch: false,
        validation_reason: null
      }
    ])

    renderChatWithSession('sess-restored')

    await waitFor(() => {
      expect(screen.getByText(/response validation passed/i)).toBeInTheDocument()
    })
    expect(screen.queryByText(/response not validated/i)).toBeNull()
  })

  it('falls back to "Response not validated" only for legacy restored messages without validation fields', async () => {
    mockHistoryFetch([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'legacy answer', sources: [] }
    ])

    renderChatWithSession('sess-legacy')

    await waitFor(() => {
      expect(screen.getByText(/response not validated/i)).toBeInTheDocument()
    })
    expect(screen.queryByText(/response validation passed/i)).toBeNull()
  })
})

describe('Chat drafts', () => {
  it('restores the draft for the open session on mount', async () => {
    mockHistoryFetch([])
    useChatUiStore.getState().setDraft('sess-restored', 'half typed question')

    renderChatWithSession('sess-restored')

    expect(await screen.findByPlaceholderText(/ask something/i)).toHaveValue(
      'half typed question'
    )
  })

  it('clears the draft after sending', async () => {
    useChatUiStore.getState().setDraft('new', 'half typed question')
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: bodyFromString('data: {"response":"ok","sources":[],"session_id":"s"}\n\n')
      })
    )

    renderChat()

    const textarea = await screen.findByPlaceholderText(/ask something/i)
    expect(textarea).toHaveValue('half typed question')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => expect(useChatUiStore.getState().drafts['new']).toBeUndefined())
  })
})

  it('appends the machine-readable code to the stream-error copy', async () => {
    const frames = 'data: {"error":"Internal server error","code":"generation_failed"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )
    renderChat()
    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => {
      expect(screen.getByText(/\(generation_failed\)/)).toBeInTheDocument()
    })
    // A coded error frame means the backend reported a failure and named
    // it — the stream did not end unexpectedly, so it must not say so.
    // That copy belongs to the transport path, which has no code at all.
    expect(screen.getByText(/answer could not be generated/i)).toBeInTheDocument()
    expect(screen.queryByText(/stream ended unexpectedly/i)).not.toBeInTheDocument()
  })

  it('shows actionable copy for a context_overflow stream error', async () => {
    const frames = 'data: {"error":"Internal server error","code":"context_overflow"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )
    renderChat()
    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    await waitFor(() => {
      expect(screen.getByText(/too large for the model/i)).toBeInTheDocument()
    })
    expect(screen.getByText(/\(context_overflow\)/)).toBeInTheDocument()
  })

describe('Chat side panel', () => {
  const INDEX_STATUS = {
    indexed: true,
    total: 4,
    with_search_text: 4,
    missing: 0,
    complete: true
  }

  function mockSearch(total: number) {
    vi.stubGlobal(
      'fetch',
      vi.fn((req: RequestInfo | URL) => {
        const u = typeof req === 'string' ? req : String(req)
        const body = u.includes('/search')
          ? {
              status: 'ok',
              hits: [],
              total,
              next_cursor: null,
              index_status: INDEX_STATUS
            }
          : { messages: [] }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => body,
          text: async () => JSON.stringify(body)
        })
      })
    )
  }

  it('shows the search panel by default, with the rail reporting it expanded', async () => {
    mockSearch(0)

    renderChat()

    expect(await screen.findByPlaceholderText(/keywords/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /hide the search panel/i })).toHaveAttribute(
      'aria-expanded',
      'true'
    )
  })

  it('collapses on the rail chevron and persists the choice', async () => {
    mockSearch(0)

    renderChat()

    await userEvent.click(
      await screen.findByRole('button', { name: /hide the search panel/i })
    )

    const rail = screen.getByRole('button', { name: /show the search panel/i })
    expect(rail).toHaveAttribute('aria-expanded', 'false')
    // Persisted in the chat UI store, so a reload does not silently reopen it.
    expect(useChatUiStore.getState().sidePanelOpen).toBe(false)
  })

  it('keeps the hit and active-filter counts on the rail while collapsed', async () => {
    // A panel that silently filters or scopes while hidden is a trap: the two
    // badges are the whole reason collapsing is safe.
    mockSearch(14)
    useUiStore.setState({ selectedCollection: 'docs' })
    useChatUiStore.setState({ sidePanelOpen: false })
    useSearchUiStore.setState({ queries: { new: 'partei' } })

    renderChat()

    expect(await screen.findByLabelText('14 search hits')).toHaveTextContent('14')
    expect(screen.getByLabelText('0 active filters')).toHaveTextContent('0')
  })
})

describe('Chat scope banner', () => {
  it('announces an active scope above the transcript and clears it', async () => {
    const fetchMock = vi.fn((req: RequestInfo | URL, init?: RequestInit) => {
      void init
      const u = typeof req === 'string' ? req : String(req)
      const body = u.includes('/scope') ? { chunk_ids: [] } : { messages: [] }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => body,
        text: async () => JSON.stringify(body)
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'docs', currentSessionId: 'sess-scoped' })
    useSearchUiStore.setState({
      scopes: { 'sess-scoped': { tokens: { p1: 10, p2: 20 }, usableTokens: 22000, missing: 1 } }
    })

    renderChatWithSession('sess-scoped')

    expect(await screen.findByTestId('scope-banner')).toHaveTextContent('Scoped to 2 chunks')
    expect(screen.getByTestId('scope-missing')).toHaveTextContent('1 of 2 chunks no longer exist')

    await userEvent.click(screen.getByRole('button', { name: /^clear$/i }))

    await waitFor(() => expect(screen.queryByTestId('scope-banner')).toBeNull())
    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/scope'))
      expect(call).toBeDefined()
      expect(call![1]!.method).toBe('DELETE')
    })
  })

  it('shows no banner while the session is unscoped', async () => {
    mockHistoryFetch([])

    renderChat()

    await screen.findByPlaceholderText(/ask something/i)
    expect(screen.queryByTestId('scope-banner')).toBeNull()
  })

  it('carries a pre-session selection onto the id the backend mints', async () => {
    // The session id only exists after the first turn, so chunks picked
    // beforehand have nowhere to be written. Dropping them would delete the
    // evidence the user selected in order to ask about it.
    const scopeBody = { chunk_ids: ['p1'], est_tokens: 40, usable_tokens: 22000, missing: 0 }
    const fetchMock = vi.fn((req: RequestInfo | URL, init?: RequestInit) => {
      void init
      const u = typeof req === 'string' ? req : String(req)
      if (u.includes('/stream_query')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          body: bodyFromString('data: {"response":"ok","sources":[],"session_id":"sess-new"}\n\n')
        })
      }
      const body = u.includes('/scope') ? scopeBody : { messages: [] }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => body,
        text: async () => JSON.stringify(body)
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'docs' })
    useSearchUiStore.setState({
      scopes: { new: { tokens: { p1: 40 }, usableTokens: 0, missing: 0 } }
    })

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/scope'))
      expect(call).toBeDefined()
      expect(call![1]!.method).toBe('PUT')
      expect(JSON.parse(String(call![1]!.body))).toEqual({ chunk_ids: ['p1'] })
    })
    await waitFor(() => {
      const state = useSearchUiStore.getState()
      expect(state.scopes['new']).toBeUndefined()
      expect(state.scopes['sess-new']?.usableTokens).toBe(22000)
    })
  })

  it('sends the pinned selection with the very question it scopes', async () => {
    // The scope endpoint needs a session row, and that row is minted by this
    // request — so the selection can only reach the server by riding along
    // with it. Installing it afterwards left the first answer unscoped while
    // the banner already claimed it was scoped.
    const fetchMock = vi.fn((req: RequestInfo | URL, init?: RequestInit) => {
      void init
      const u = typeof req === 'string' ? req : String(req)
      if (u.includes('/stream_query')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          body: bodyFromString(
            'data: {"response":"ok","sources":[],"session_id":"sess-new","retrieval_mode":"scoped","scoped_chunk_count":2}\n\n'
          )
        })
      }
      const body = u.includes('/scope')
        ? { chunk_ids: ['p1', 'p2'], est_tokens: 50, usable_tokens: 22000, missing: 0 }
        : { messages: [] }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => body,
        text: async () => JSON.stringify(body)
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'docs' })
    useSearchUiStore.setState({
      scopes: { new: { tokens: { p1: 40, p2: 10 }, usableTokens: 0, missing: 0 } }
    })

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/stream_query'))
      expect(call).toBeDefined()
      expect(JSON.parse(String(call![1]!.body)).scope_chunk_ids).toEqual(['p1', 'p2'])
    })
  })

  it('says so when an answer did not come from the pinned selection', async () => {
    // The guardrail for exactly the failure above: if the server does not
    // confirm it answered from the selection, the turn must not pass as
    // scoped just because the banner says the chat is.
    const fetchMock = vi.fn((req: RequestInfo | URL) => {
      const u = typeof req === 'string' ? req : String(req)
      if (u.includes('/stream_query')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          body: bodyFromString('data: {"response":"ok","sources":[],"session_id":"sess-new"}\n\n')
        })
      }
      const body = u.includes('/scope')
        ? { chunk_ids: ['p1'], est_tokens: 40, usable_tokens: 22000, missing: 0 }
        : { messages: [] }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => body,
        text: async () => JSON.stringify(body)
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'docs' })
    useSearchUiStore.setState({
      scopes: { new: { tokens: { p1: 40 }, usableTokens: 0, missing: 0 } }
    })

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    expect(await screen.findByTestId('scope-not-applied')).toBeInTheDocument()
  })
})

describe('Chat session switching', () => {
  function NavigateButton({ to }: { to: string }) {
    const navigate = useNavigate()
    return (
      <button type="button" onClick={() => navigate(to)}>
        go
      </button>
    )
  }

  // Both chat routes render the same `Chat` element, so React Router reuses
  // the component instance across the navigation — local state survives it.
  // That is what makes the transcript reset a real requirement rather than an
  // artifact of remounting.
  function renderChatAt(path: string) {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    return render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="/chat" element={<Chat />} />
            <Route path="/chat/:sessionId" element={<Chat />} />
          </Routes>
          <NavigateButton to="/chat" />
        </MemoryRouter>
      </QueryClientProvider>
    )
  }

  it('clears the restored transcript when the user starts a new session', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          messages: [
            { role: 'user', content: 'earlier question' },
            { role: 'assistant', content: 'earlier answer', sources: [] }
          ]
        })
      })
    )

    renderChatAt('/chat/sess-old')
    expect(await screen.findByText('earlier answer')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'go' }))

    await waitFor(() => {
      expect(screen.queryByText('earlier answer')).not.toBeInTheDocument()
    })
  })

  it('ignores stream frames that arrive with no open turn', () => {
    // The session reset empties the transcript, so a frame from the stream
    // being torn down has no turn to fold into. Folding into `undefined`
    // would throw out of the reducer and take the screen down.
    const empty = { turns: [], inflight: true }

    expect(chatReducer(empty, { type: 'token', token: 'x' })).toEqual(empty)
    expect(
      chatReducer(empty, { type: 'finalize', meta: { session_id: 's' } as ChatFinalEvent })
    ).toEqual({ turns: [], inflight: false })
    expect(chatReducer(empty, { type: 'fail', error: 'boom' })).toEqual({
      turns: [],
      inflight: false
    })
  })
})
