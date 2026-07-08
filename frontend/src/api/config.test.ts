import { describe, it, expect, vi, afterEach } from 'vitest'
import { getConfig } from './config'

afterEach(() => {
  vi.restoreAllMocks()
})

function mockFetch(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body)
    })
  )
}

describe('config api', () => {
  it('getConfig GETs /config and returns the parsed body', async () => {
    const body = {
      graph_top_k: 80,
      graph_max_top_k: 500,
      collection_timeout: 120,
      max_upload_bytes: 1024 * 1024 * 1024
    }
    mockFetch(body)
    const cfg = await getConfig()
    const call = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(String(call[0])).toContain('/config')
    expect(cfg).toEqual(body)
  })
})
