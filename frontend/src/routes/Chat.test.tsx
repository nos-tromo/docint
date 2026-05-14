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
  })
})
