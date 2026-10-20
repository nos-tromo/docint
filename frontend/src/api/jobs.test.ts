import { describe, expect, it, vi, afterEach } from 'vitest'
import { cancelIngestJob, createIngestJob, dismissIngestJob, getIngestJob, listIngestJobs } from './jobs'
import { setOwnerParam } from './client'

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

  it('adopts an in-flight job named at the top level of the body', async () => {
    mockFetch(409, { message: 'Ingestion already in progress.', job_id: 'running-2' })
    await expect(createIngestJob({ collection: 'mydocs', hybrid: true })).resolves.toEqual({
      job_id: 'running-2',
      adopted: true
    })
  })

  it('rethrows other errors', async () => {
    mockFetch(404, { detail: 'Collection not found' })
    await expect(createIngestJob({ collection: 'gone', hybrid: true })).rejects.toThrow()
  })
})

describe('job helpers', () => {
  afterEach(() => setOwnerParam(null))

  it('act under the owner a caller pinned, whichever one is selected now', async () => {
    setOwnerParam('bob')
    mockFetch(200, { jobs: [], ok: true })

    await listIngestJobs('alice')
    await getIngestJob('job-1', 'alice')
    await dismissIngestJob('job-1', 'alice')
    await cancelIngestJob('job-1', 'alice')

    const urls = vi.mocked(fetch).mock.calls.map(([input]) => String(input))
    expect(urls).toHaveLength(4)
    for (const sent of urls) {
      expect(sent).toContain('owner=alice')
      expect(sent).not.toContain('owner=bob')
    }
  })
})
