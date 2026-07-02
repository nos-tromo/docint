import { describe, it, expect, vi, afterEach } from 'vitest'
import { streamSummary, summarize } from './analysis'

afterEach(() => vi.restoreAllMocks())

function bodyFromString(s: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(c) {
      c.enqueue(enc.encode(s))
      c.close()
    }
  })
}

function lastUrl() {
  return String((fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0])
}

describe('summary api carries the selected collection', () => {
  it('summarize sends the collection query param', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ summary: '', sources: [] }),
        text: async () => '{}'
      })
    )
    await summarize(false, 'docs')
    expect(lastUrl()).toContain('/summarize')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('streamSummary sends both refresh and collection query params', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: bodyFromString('data: {"summary":"ok","sources":[]}\n\n')
      })
    )
    for await (const _ of streamSummary(true, 'docs')) {
      void _
    }
    expect(lastUrl()).toContain('/summarize/stream')
    expect(lastUrl()).toContain('refresh=true')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('streamSummary omits the query string when nothing is passed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: bodyFromString('data: {"summary":"ok","sources":[]}\n\n')
      })
    )
    for await (const _ of streamSummary()) {
      void _
    }
    expect(lastUrl()).not.toContain('?')
  })
})
