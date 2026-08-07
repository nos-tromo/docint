import { describe, it, expect, vi, afterEach } from 'vitest'
import { summarize } from './analysis'

afterEach(() => vi.restoreAllMocks())

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

  it('summarize sends both refresh and collection query params', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 202,
        json: async () => ({ job_id: 'j1' }),
        text: async () => '{}'
      })
    )
    await summarize(true, 'docs')
    expect(lastUrl()).toContain('/summarize')
    expect(lastUrl()).toContain('refresh=true')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('summarize omits the query string when nothing is passed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ summary: 'ok', sources: [] }),
        text: async () => '{}'
      })
    )
    await summarize()
    expect(lastUrl()).not.toContain('?')
  })

  it('resolves the job_id shape on a 202 response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 202,
        json: async () => ({ job_id: 'j1' }),
        text: async () => '{}'
      })
    )
    const result = await summarize(true, 'docs')
    expect(result).toEqual({ job_id: 'j1' })
  })
})
