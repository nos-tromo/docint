import { describe, it, expect, vi, afterEach } from 'vitest'
import { translate } from './translate'

afterEach(() => vi.restoreAllMocks())

describe('translate', () => {
  it('POSTs the text to /translate and returns the parsed body', async () => {
    const calls: Array<{ url: string; method: string; body: unknown }> = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        calls.push({ url: String(u), method: init?.method ?? 'GET', body: init?.body })
        return {
          ok: true,
          status: 200,
          json: async () => ({ ok: true, translation: 'Hallo', model: 'm', target_lang: 'de' })
        }
      })
    )
    const res = await translate('Hello')
    expect(res.translation).toBe('Hallo')
    expect(calls[0].method).toBe('POST')
    expect(calls[0].url).toContain('/translate')
    expect(JSON.parse(String(calls[0].body))).toEqual({ text: 'Hello' })
  })
})
