import { describe, expect, it, vi } from 'vitest'
import { fetchAllPages } from './fetchAllPages'
import type { Page } from '@/api/collections'

function page(items: number[], next: string | null): Page<number> {
  return { items, next_cursor: next }
}

describe('fetchAllPages', () => {
  it('walks every cursor page and concatenates the items', async () => {
    const fetchPage = vi
      .fn<(cursor: string | null) => Promise<Page<number>>>()
      .mockResolvedValueOnce(page([1, 2], 'c1'))
      .mockResolvedValueOnce(page([3, 4], 'c2'))
      .mockResolvedValueOnce(page([5], null))

    await expect(fetchAllPages(fetchPage)).resolves.toEqual([1, 2, 3, 4, 5])
    expect(fetchPage.mock.calls.map((c) => c[0])).toEqual([null, 'c1', 'c2'])
  })

  it('returns a single page unchanged', async () => {
    const fetchPage = vi.fn().mockResolvedValue(page([7], null))
    await expect(fetchAllPages(fetchPage)).resolves.toEqual([7])
    expect(fetchPage).toHaveBeenCalledTimes(1)
  })

  it('stops at maxItems so a huge collection cannot be walked forever', async () => {
    const fetchPage = vi.fn(async (cursor: string | null) => page([Number(cursor ?? 0)], String(Number(cursor ?? 0) + 1)))
    const items = await fetchAllPages(fetchPage, { maxItems: 3 })
    expect(items).toHaveLength(3)
    expect(fetchPage).toHaveBeenCalledTimes(3)
  })

  it('stops when the cursor repeats, rather than looping', async () => {
    // A server that keeps handing back the same cursor would otherwise spin
    // this loop until maxItems; treat a non-advancing cursor as the end.
    const fetchPage = vi.fn(async () => page([1], 'same'))
    const items = await fetchAllPages(fetchPage, { maxItems: 100 })
    expect(items).toEqual([1, 1])
    expect(fetchPage).toHaveBeenCalledTimes(2)
  })

  it('propagates a page failure instead of returning a short list', async () => {
    const fetchPage = vi
      .fn<(cursor: string | null) => Promise<Page<number>>>()
      .mockResolvedValueOnce(page([1], 'c1'))
      .mockRejectedValueOnce(new Error('boom'))
    await expect(fetchAllPages(fetchPage)).rejects.toThrow('boom')
  })
})
