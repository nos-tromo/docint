import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Chat } from './Chat'
import { useUiStore } from '@/stores/ui'

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

  it('marks the turn done on an untyped error frame instead of waiting forever', async () => {
    const frames = 'data: {"error":"boom"}\n\n'

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
    expect(screen.getByText(/boom/)).toBeInTheDocument()
  })

  it('surfaces a backend-likely-crashed message when the stream throws (e.g., OOM kill)', async () => {
    // Reader.read() rejects with a TypeError mid-stream — the same shape
    // the browser fetch surfaces when nginx closes the upstream because
    // the backend died. The chat reducer must convert this into an
    // actionable message instead of relaying "network error" verbatim.
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
    // Underlying transport detail is preserved for forensics.
    expect(screen.getByText(/network error/)).toBeInTheDocument()
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
      expect(screen.getByText('doc.pdf · p. 3')).toBeInTheDocument()
      expect(screen.getByText('doc.pdf · p. 7')).toBeInTheDocument()
      // Two text-file chunks with no page/row are kept because their
      // chunk text differs.
      expect(screen.getAllByText('transcript.txt')).toHaveLength(2)
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
