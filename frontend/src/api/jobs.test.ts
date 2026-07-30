import { describe, expect, it, vi, afterEach } from 'vitest'
import { createIngestJob } from './jobs'

const mockFetch = (status: number, body: unknown) =>
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))
  )

afterEach(() => vi.unstubAllGlobals())

describe('createIngestJob', () => {
  it('returns the new job id', async () => {
    mockFetch(202, { job_id: 'abc' })
    await expect(createIngestJob({ collection: 'mydocs', hybrid: true })).resolves.toEqual({
      job_id: 'abc',
      adopted: false
    })
  })

  it('adopts the in-flight job on 409 instead of throwing', async () => {
    mockFetch(409, { detail: { message: 'Ingestion already in progress.', job_id: 'running-1' } })
    await expect(createIngestJob({ collection: 'mydocs', hybrid: true })).resolves.toEqual({
      job_id: 'running-1',
      adopted: true
    })
  })

  it('rethrows other errors', async () => {
    mockFetch(404, { detail: 'Collection not found' })
    await expect(createIngestJob({ collection: 'gone', hybrid: true })).rejects.toThrow()
  })
})
