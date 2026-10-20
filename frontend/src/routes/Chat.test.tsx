import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useNavigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Chat, chatReducer } from './Chat'
import type { ChatFinalEvent } from '@/api/types'
import { chatTranscriptToText } from '@/lib/exports'
import { useChatFiltersStore } from '@/stores/chatFilters'
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

  it('sends the reasoning toggle in the /stream_query request body', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: bodyFromString('data: {"response":"ok","sources":[],"session_id":"s"}\n\n')
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'test-collection' })
    useChatFiltersStore.setState({ reasoning: true })

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      const streamCall = fetchMock.mock.calls.find(([u]) => String(u).includes('/stream_query'))
      expect(streamCall).toBeDefined()
      // Sent as an explicit boolean either way: the server's `null` means
      // "use the env default", which is not what a toggle the user can see
      // should ever mean.
      expect(JSON.parse(streamCall![1].body)).toMatchObject({ reasoning: true })
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

  it('finishes a turn whose stream ends with neither an answer nor an error', async () => {
    // A backend shut down gracefully mid-answer closes the stream cleanly, so
    // nothing above fires; the turn must not stay open with Send held busy.
    const frames = 'data: {"token":"The council met"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    expect(await screen.findByText(/stream ended unexpectedly/i)).toBeInTheDocument()
    expect(screen.getByText(/The council met/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /send/i })).not.toHaveAttribute('aria-busy')
    // A lookup carries no judgement, so it gets no assessment caveat.
    expect(screen.queryByText(/did not finish/i)).not.toBeInTheDocument()
  })

  it('keeps a finished answer when the connection drops after its final envelope', async () => {
    const enc = new TextEncoder()
    let stream!: ReadableStreamDefaultController<Uint8Array>
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        stream = controller
      }
    })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, body }))

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    stream.enqueue(enc.encode('data: {"response":"Delivered.","sources":[],"session_id":"sess-1"}\n\n'))
    expect(await screen.findByText('Delivered.')).toBeInTheDocument()

    await act(async () => {
      stream.error(new TypeError('network error'))
      await new Promise((resolve) => setTimeout(resolve, 20))
    })

    expect(screen.queryByText(/stream ended unexpectedly/i)).not.toBeInTheDocument()
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
  // A session's history is served only through the collection it is pinned to.
  beforeEach(() => {
    useUiStore.setState({ selectedCollection: 'docs' })
  })

  it('reads the history through the active collection', async () => {
    mockHistoryFetch([])
    renderChatWithSession('sess-restored')
    await waitFor(() => {
      const urls = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.map((c) => String(c[0]))
      expect(urls.some((u) => u.includes('/sessions/sess-restored/history?collection=docs'))).toBe(true)
    })
  })

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

  it('restores the assessment notice from session history', async () => {
    mockHistoryFetch([
      { role: 'user', content: 'What could be improved?' },
      {
        role: 'assistant',
        content: 'Fact [1].',
        sources: [],
        validation_checked: true,
        validation_mismatch: false,
        validation_reason: null,
        answer_mode: 'assessment',
        agent: {
          steps: 0,
          actions: [{ action: 'seed', detail: '', found: 2 }],
          reason: null,
          answer: 'split',
          shortened: 0,
          evidence_chars: 9
        }
      }
    ])

    renderChatWithSession('sess-restored')

    await waitFor(() => {
      expect(screen.getByTestId('assessment-notice')).toBeInTheDocument()
    })
    expect(screen.getByText(/validation passed for the source-backed part/i)).toBeInTheDocument()
  })

  it('reloaded turns keep their target chip', async () => {
    // The final frame that carried the target is long gone on a reload; the
    // history route is the only place it survives, so the mapping must copy it.
    mockHistoryFetch([
      { role: 'user', content: 'what is at the gate?' },
      { role: 'assistant', content: 'A barrier.', sources: [], retrieval_target: 'visual' },
      { role: 'user', content: 'who signed the notice?' },
      { role: 'assistant', content: 'The clerk.', sources: [], retrieval_target: 'documents' },
      { role: 'user', content: 'and the date?' },
      { role: 'assistant', content: 'In March.', sources: [], retrieval_target: 'all' },
      { role: 'user', content: 'anything else?' },
      // Recorded before the target was stored, or answered from a selection.
      { role: 'assistant', content: 'No.', sources: [] }
    ])

    renderChatWithSession('sess-targets')

    expect(await screen.findByText('Answered from images only')).toBeInTheDocument()
    expect(screen.getByText('Answered from documents only')).toBeInTheDocument()
    expect(screen.getAllByTestId('retrieval-target-used')).toHaveLength(2)
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

describe('Chat retrieval controls', () => {
  it('keeps the metadata filters and the retrieval mode on the chat, not in the search panel', async () => {
    // They govern what any answer retrieves against — whether or not the
    // search panel is open, and whether or not anything was ever searched.
    // Mounted in the panel they read as controls over the keyword index.
    useChatUiStore.setState({ drafts: {}, sidePanelOpen: false })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) }))

    renderChat()

    const filters = await screen.findByRole('button', { name: /filters/i })
    const mode = screen.getByRole('button', { name: /retrieval/i })
    // The panel is collapsed, so anything still living inside it is hidden.
    expect(filters).toBeVisible()
    expect(mode).toBeVisible()
    expect(screen.queryByTestId('search-panel')?.contains(filters)).toBeFalsy()
  })
})

describe('Chat retrieval target', () => {
  const picker = () => screen.getByRole('button', { name: /answer from/i })

  beforeEach(() => {
    useChatFiltersStore.getState().reset()
  })

  /** Open a chat, narrow it to documents, and let one turn mint its session. */
  async function answerOneTurnFromDocuments() {
    const frames =
      'data: {"token":"Answer"}\n\n' +
      'data: {"response":"Answer","sources":[],"session_id":"sess-1","retrieval_target":"documents"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )
    useUiStore.setState({ selectedCollection: 'docs' })

    renderChat()

    await userEvent.click(picker())
    expect(picker()).toHaveAccessibleName(/documents/i)
    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(await screen.findByText('Answer')).toBeInTheDocument()
    await waitFor(() => {
      expect(useUiStore.getState().currentSessionId).toBe('sess-1')
    })
    // The target belongs to the chat, so it holds for as long as the chat does.
    expect(picker()).toHaveAccessibleName(/documents/i)
  }

  it('a new chat starts at all', async () => {
    // One click on the picker used to outlive the chat it was made in (and a
    // reload besides), so a later chat answered from documents alone.
    await answerOneTurnFromDocuments()

    await userEvent.click(screen.getByRole('button', { name: /new chat/i }))

    await waitFor(() => {
      expect(picker()).toHaveAccessibleName(/everything/i)
    })
    expect(useChatFiltersStore.getState().retrievalTarget).toBe('all')
  })

  it('a collection switch starts the next chat at all', async () => {
    // The sidebar drops the open chat through the store rather than through
    // the New chat button, and takes the target with it.
    await answerOneTurnFromDocuments()

    act(() => {
      useUiStore.getState().setSelectedCollection('other')
    })

    await waitFor(() => {
      expect(picker()).toHaveAccessibleName(/everything/i)
    })
    expect(useChatFiltersStore.getState().retrievalTarget).toBe('all')
  })

  it('New chat resets the target in a chat that never minted a session', async () => {
    // With no session there is none to clear, so nothing the effect on the
    // effective session watches changes: the button resets the target itself.
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) }))

    renderChat()

    await userEvent.click(picker())
    expect(picker()).toHaveAccessibleName(/documents/i)

    await userEvent.click(screen.getByRole('button', { name: /new chat/i }))

    await waitFor(() => {
      expect(picker()).toHaveAccessibleName(/everything/i)
    })
    expect(useChatFiltersStore.getState().retrievalTarget).toBe('all')
  })

  it('New chat resets the target after a stateless turn, which mints no session either', async () => {
    const frames =
      'data: {"token":"Answer"}\n\n' +
      'data: {"response":"Answer","sources":[],"session_id":null,"retrieval_target":"documents"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )
    useUiStore.setState({ selectedCollection: 'docs' })
    useChatFiltersStore.setState({ retrievalMode: 'stateless' })

    renderChat()

    await userEvent.click(picker())
    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(await screen.findByText('Answer')).toBeInTheDocument()
    expect(useUiStore.getState().currentSessionId).toBeNull()
    expect(picker()).toHaveAccessibleName(/documents/i)

    await userEvent.click(screen.getByRole('button', { name: /new chat/i }))

    await waitFor(() => {
      expect(picker()).toHaveAccessibleName(/everything/i)
    })
    expect(useChatFiltersStore.getState().retrievalTarget).toBe('all')
  })

  it('does not carry a leftover target into a chat opened fresh', () => {
    useChatFiltersStore.setState({ retrievalTarget: 'visual' })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) }))

    renderChat()

    expect(picker()).toHaveAccessibleName(/everything/i)
    expect(useChatFiltersStore.getState().retrievalTarget).toBe('all')
  })
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

  it('is only a toggle while collapsed — every detail lives in the panel', async () => {
    mockSearch(14)
    useUiStore.setState({ selectedCollection: 'docs' })
    useChatUiStore.setState({ sidePanelOpen: false })
    useSearchUiStore.setState({ queries: { new: 'partei' } })

    renderChat()

    const toggle = await screen.findByRole('button', { name: /show the search panel/i })
    // No hit count, no filter count: a slim rail is not a dashboard.
    expect(toggle).toHaveTextContent('')
    expect(screen.queryByText('14')).toBeNull()
  })

  it('tints the collapsed toggle while a scope is live', async () => {
    // The one thing worth signalling from a shut panel: answers are coming
    // from hand-picked chunks. Everything else is only a readout.
    mockSearch(14)
    useUiStore.setState({ selectedCollection: 'docs' })
    useChatUiStore.setState({ sidePanelOpen: false })
    useSearchUiStore.setState({
      queries: { new: 'partei' },
      scopes: { new: { tokens: { p1: 1200 }, usableTokens: 22000, missing: 0 } }
    })

    renderChat()

    const toggle = await screen.findByRole('button', { name: /show the search panel/i })
    expect(toggle).toHaveAttribute('data-scoped', 'true')
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

  function BackButton() {
    const navigate = useNavigate()
    return (
      <button type="button" onClick={() => navigate(-1)}>
        back
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
          <BackButton />
        </MemoryRouter>
      </QueryClientProvider>
    )
  }

  it("shows none of a chat's answers when Back reopens it under another collection", async () => {
    // The sidebar leaves /chat/:id on a collection switch, but Back returns to
    // it, and any answer shown there could be added to the new collection's
    // report. The server refuses the history outside its collection.
    const urls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string) => {
        const url = String(u)
        urls.push(url)
        if (url.includes('/history') && url.includes('collection=other')) {
          const body = JSON.stringify({ detail: 'Session not found.' })
          return { ok: false, status: 404, json: async () => JSON.parse(body), text: async () => body }
        }
        return {
          ok: true,
          status: 200,
          headers: new Headers({ 'content-type': 'application/json' }),
          json: async () => ({
            messages: [
              { role: 'user', content: 'earlier question' },
              { role: 'assistant', content: 'earlier answer', sources: [] }
            ]
          })
        }
      })
    )
    useUiStore.setState({ selectedCollection: 'docs' })

    renderChatAt('/chat/sess-old')
    expect(await screen.findByText('earlier answer')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'go' }))
    await waitFor(() => expect(screen.queryByText('earlier answer')).not.toBeInTheDocument())

    act(() => useUiStore.getState().setSelectedCollection('other'))
    await userEvent.click(screen.getByRole('button', { name: 'back' }))

    await waitFor(() =>
      expect(urls.some((u) => u.includes('/sessions/sess-old/history?collection=other'))).toBe(true)
    )
    expect(screen.queryByText('earlier answer')).not.toBeInTheDocument()
  })

  it('clears the restored transcript when the user starts a new session', async () => {
    useUiStore.setState({ selectedCollection: 'docs' })
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

  it('clears a transcript whose session was minted on this page, not by the URL', async () => {
    // The one case the effect above cannot see: a chat started at /chat gets
    // its id from the first answer and never gains a :sessionId segment, so
    // "New chat" changes nothing the route watches. Before the reset keyed on
    // the effective session, the finished answer stayed on screen and the
    // click looked like a no-op — from the very button that sits under it.
    const frames =
      'data: {"token":"Answer"}\n\n' +
      'data: {"response":"Answer","sources":[],"session_id":"sess-1"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChatAt('/chat')

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(await screen.findByText('Answer')).toBeInTheDocument()
    await waitFor(() => {
      expect(useUiStore.getState().currentSessionId).toBe('sess-1')
    })

    await userEvent.click(screen.getByRole('button', { name: /new chat/i }))

    await waitFor(() => {
      expect(screen.queryByText('Answer')).not.toBeInTheDocument()
    })
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })

  it('leaves the next turn alone when an abandoned stream breaks afterwards', async () => {
    // Switching sessions aborts the request in flight. Its stream failing later
    // belongs to a turn that is gone, never to whatever turn is last by then.
    const enc = new TextEncoder()
    let abandoned!: ReadableStreamDefaultController<Uint8Array>
    const first = new ReadableStream<Uint8Array>({
      start(controller) {
        abandoned = controller
      }
    })
    const second =
      'data: {"token":"Second answer"}\n\n' +
      'data: {"response":"Second answer","sources":[],"session_id":"sess-2"}\n\n'
    let streams = 0
    vi.stubGlobal(
      'fetch',
      vi.fn((req: RequestInfo | URL) => {
        const isStream = String(req).includes('/stream_query')
        if (isStream) streams += 1
        const body = isStream && streams === 1 ? first : bodyFromString(isStream ? second : '')
        return Promise.resolve({ ok: true, status: 200, body })
      })
    )

    renderChatAt('/chat/sess-old')

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'first')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    abandoned.enqueue(enc.encode('data: {"token":"First partial"}\n\n'))
    expect(await screen.findByText('First partial')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'go' }))
    await waitFor(() => {
      expect(screen.queryByText('First partial')).not.toBeInTheDocument()
    })
    await userEvent.type(screen.getByPlaceholderText(/ask something/i), 'second')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    expect(await screen.findByText('Second answer')).toBeInTheDocument()

    await act(async () => {
      abandoned.error(new TypeError('network error'))
      await new Promise((resolve) => setTimeout(resolve, 20))
    })

    expect(screen.queryByText(/stream ended unexpectedly/i)).not.toBeInTheDocument()
  })

  it('ignores stream frames that arrive with no open turn', () => {
    // The session reset empties the transcript, so a frame from the stream
    // being torn down has no turn to fold into. Folding into `undefined`
    // would throw out of the reducer and take the screen down.
    const empty = { turns: [], inflight: true }

    expect(chatReducer(empty, { type: 'token', token: 'x' })).toEqual(empty)
    expect(chatReducer(empty, { type: 'retry', query: 'q' })).toEqual(empty)
    expect(
      chatReducer(empty, { type: 'agent', step: { step: 1, action: 'search', detail: 'q', found: 1 } })
    ).toEqual(empty)
    expect(
      chatReducer(empty, { type: 'finalize', meta: { session_id: 's' } as ChatFinalEvent })
    ).toEqual({ turns: [], inflight: false })
    expect(chatReducer(empty, { type: 'fail', error: 'boom', note: 'Cut off.' })).toEqual({
      turns: [],
      inflight: false
    })
  })

  it('replaces the rejected answer and names the retry when the backend re-answers', async () => {
    // The retry frame lands between two token streams: the first answer was
    // rejected, so it must be discarded rather than spliced onto the second.
    const frames =
      'data: {"token":"Evidence insufficient."}\n\n' +
      'data: {"retry":{"query":"Security Council resolutions"}}\n\n' +
      'data: {"token":"The Council adopted three resolutions."}\n\n' +
      'data: {"response":"The Council adopted three resolutions.","sources":[],' +
      '"session_id":"sess-1","retried":true,"retry_query":"Security Council resolutions"}\n\n'

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText(/The Council adopted three resolutions/)).toBeInTheDocument()
    })
    expect(screen.queryByText(/Evidence insufficient/)).not.toBeInTheDocument()
    const notice = screen.getByTestId('retry-notice')
    expect(notice).toHaveTextContent(/Security Council resolutions/)
  })

  it('keeps the retry notice after the final envelope arrives', async () => {
    const frames =
      'data: {"retry":{"query":"reformulated"}}\n\n' +
      'data: {"token":"A better answer."}\n\n' +
      'data: {"response":"A better answer.","sources":[],"session_id":"sess-1",' +
      '"retried":true,"retry_query":"reformulated"}\n\n'

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText(/A better answer/)).toBeInTheDocument()
    })
    expect(screen.getByTestId('retry-notice')).toHaveTextContent(/reformulated/)
  })

  it('restores the delivered answer when a retry is abandoned mid-stream', async () => {
    // The backend gave up on the retry and reported the first answer instead.
    // Without preferring the envelope's own text the turn would render blank —
    // the retry frame cleared the tokens and no replacement ever arrived.
    const frames =
      'data: {"token":"Evidence insufficient."}\n\n' +
      'data: {"retry":{"query":"reformulated"}}\n\n' +
      'data: {"response":"Evidence insufficient.","sources":[],"session_id":"sess-1"}\n\n'

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText(/Evidence insufficient/)).toBeInTheDocument()
    })
  })

  it('folds research steps into the open turn without finishing it', () => {
    const open = { turns: [{ user: 'q', assistant: '', done: false, meta: null }], inflight: true }
    const step = { step: 1, action: 'search' as const, detail: 'evaluation datasets', found: 1 }

    const next = chatReducer(open, { type: 'agent', step })

    expect(next.inflight).toBe(true)
    expect(next.turns[0].done).toBe(false)
    expect(next.turns[0].agentSteps).toEqual([step])
  })

  it('keeps successive research steps on the open turn in arrival order', () => {
    const open = { turns: [{ user: 'q', assistant: '', done: false, meta: null }], inflight: true }
    const first = { step: 0, action: 'seed' as const, detail: '', found: 2 }
    const second = { step: 1, action: 'search' as const, detail: 'evaluation datasets', found: 1 }

    const next = chatReducer(chatReducer(open, { type: 'agent', step: first }), { type: 'agent', step: second })

    expect(next.turns[0].agentSteps).toEqual([first, second])
    expect(next.turns[0].done).toBe(false)
    expect(next.inflight).toBe(true)
  })

  it('heads the partial text of an assessment that fails with the interrupted note', () => {
    // The label that marks the judgement arrives with the finished answer, so a
    // cut-off answer carries the caveat in its text: the transcript and every
    // other place the text goes keep it.
    const seed = { step: 0, action: 'seed' as const, detail: '', found: 2 }
    const open = {
      turns: [{ user: 'q', assistant: 'A second dataset would help.', done: false, meta: null, agentSteps: [seed] }],
      inflight: true
    }

    const next = chatReducer(open, { type: 'fail', error: 'boom', note: 'Cut off.' })

    expect(next.turns[0].assistant).toBe('*Cut off.*\n\nA second dataset would help.')
    expect(next.turns[0]).toMatchObject({ done: true, error: 'boom' })
    expect(next.inflight).toBe(false)
    expect(chatTranscriptToText(next.turns)).toContain('*Cut off.*')
  })

  it('adds no interrupted note to a lookup, to an assessment with no text yet, or twice', () => {
    const seed = { step: 0, action: 'seed' as const, detail: '', found: 2 }
    const turn = (assistant: string, agentSteps?: (typeof seed)[]) => ({
      turns: [{ user: 'q', assistant, done: false, meta: null, agentSteps }],
      inflight: true
    })
    const fail = { type: 'fail' as const, error: 'boom', note: 'Cut off.' }

    expect(chatReducer(turn('Partial'), fail).turns[0].assistant).toBe('Partial')
    expect(chatReducer(turn('  ', [seed]), fail).turns[0].assistant).toBe('  ')
    expect(chatReducer(chatReducer(turn('Partial', [seed]), fail), fail).turns[0].assistant).toBe(
      '*Cut off.*\n\nPartial'
    )
  })

  it('marks an assessment that an error frame cuts off mid-answer', async () => {
    const frames =
      'data: {"agent":{"step":0,"action":"seed","detail":"","found":2}}\n\n' +
      'data: {"agent":{"step":1,"action":"answer","detail":"","found":2}}\n\n' +
      'data: {"token":"A second dataset would help."}\n\n' +
      'data: {"error":"Internal server error"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'What could be improved?')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    expect(await screen.findByText(/did not finish/i)).toBeInTheDocument()
    expect(screen.getByText(/A second dataset would help/)).toBeInTheDocument()
    expect(screen.getByText(/answer could not be generated/i)).toBeInTheDocument()
  })

  it('marks an assessment whose stream breaks off mid-answer', async () => {
    const enc = new TextEncoder()
    let stream!: ReadableStreamDefaultController<Uint8Array>
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        stream = controller
      }
    })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, body }))

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'What could be improved?')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    stream.enqueue(enc.encode('data: {"agent":{"step":0,"action":"seed","detail":"","found":2}}\n\n'))
    stream.enqueue(enc.encode('data: {"token":"A second dataset would help."}\n\n'))
    expect(await screen.findByText(/A second dataset would help/)).toBeInTheDocument()

    stream.error(new TypeError('network error'))

    expect(await screen.findByText(/did not finish/i)).toBeInTheDocument()
    expect(screen.getByText(/stream ended unexpectedly/i)).toBeInTheDocument()
  })

  it('marks an assessment whose stream ends before its final envelope', async () => {
    const frames =
      'data: {"agent":{"step":0,"action":"seed","detail":"","found":2}}\n\n' +
      'data: {"token":"A second dataset would help."}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'What could be improved?')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    expect(await screen.findByText(/did not finish/i)).toBeInTheDocument()
    expect(screen.getByText(/stream ended unexpectedly/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /send/i })).not.toHaveAttribute('aria-busy')
  })

  it('streams an assessment turn through its research steps to the final notice', async () => {
    const trail =
      '{"steps":1,"actions":[{"action":"seed","detail":"","found":2},' +
      '{"action":"search","detail":"evaluation datasets","found":1}],' +
      '"reason":null,"answer":"split","shortened":0,"evidence_chars":9}'
    const frames =
      'data: {"agent":{"step":0,"action":"seed","detail":"","found":2}}\n\n' +
      'data: {"agent":{"step":1,"action":"search","detail":"evaluation datasets","found":1}}\n\n' +
      'data: {"token":"Fact [1]."}\n\n' +
      `data: {"response":"Fact [1].","sources":[],"session_id":"sess-1","answer_mode":"assessment","agent":${trail}}\n\n`

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'What could be improved?')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByTestId('assessment-notice')).toBeInTheDocument()
    })
    expect(screen.getByText(/Fact \[1\]\./)).toBeInTheDocument()
    expect(useUiStore.getState().currentSessionId).toBe('sess-1')
  })

  it('keeps the turn open while research steps stream, until the final envelope', async () => {
    // A step frame taken for the final envelope would finish the turn and
    // re-enable Send mid-research, yet leave the same end state behind.
    const enc = new TextEncoder()
    let stream!: ReadableStreamDefaultController<Uint8Array>
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        stream = controller
      }
    })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, body }))

    renderChat()

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'What could be improved?')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))
    stream.enqueue(enc.encode('data: {"agent":{"step":0,"action":"seed","detail":"","found":2}}\n\n'))

    expect(await screen.findByTestId('agent-trail-live')).toHaveTextContent('Initial search: 2 passages')
    expect(screen.getByRole('button', { name: /send/i })).toHaveAttribute('aria-busy', 'true')

    stream.enqueue(
      enc.encode(
        'data: {"response":"Fact [1].","sources":[],"session_id":"sess-1","answer_mode":"assessment",' +
          '"agent":{"steps":0,"actions":[{"action":"seed","detail":"","found":2}],"reason":null,' +
          '"answer":"split","shortened":0,"evidence_chars":9}}\n\n'
      )
    )
    stream.close()

    expect(await screen.findByTestId('assessment-notice')).toBeInTheDocument()
    expect(screen.queryByTestId('agent-trail-live')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /send/i })).not.toHaveAttribute('aria-busy')
  })

  it('sends the retrieval target with the question it applies to', async () => {
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
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ messages: [] }),
        text: async () => '{"messages":[]}'
      })
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'docs' })

    renderChat()
    // Picked inside the open chat: a chat opened fresh starts from everything.
    act(() => {
      useChatFiltersStore.getState().setRetrievalTarget('visual')
    })

    await userEvent.type(await screen.findByPlaceholderText(/ask something/i), 'hi')
    await userEvent.click(screen.getByRole('button', { name: /send/i }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/stream_query'))
      expect(call).toBeDefined()
      expect(JSON.parse(String(call![1]!.body)).retrieval_target).toBe('visual')
    })
  })
})
