import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useIngestRunStore } from './ingestRun'
import { useIngestJobsStore } from './ingestJobs'
import { defaultT } from '@/i18n/defaultT'

vi.mock('@/api/ingest', () => ({
  streamIngestUploadBatched: async function* () {
    yield { event: 'start', data: { collection: 'mydocs', files: ['a.txt'] }, receivedAt: 0 }
    yield { event: 'file_saved', data: { filename: 'a.txt' }, receivedAt: 0 }
    return { anySaved: true, failures: [] }
  }
}))

const createIngestJob = vi.fn()
vi.mock('@/api/jobs', () => ({
  createIngestJob: (...args: unknown[]) => createIngestJob(...args),
  INGEST_JOB_EVENTS_PATH: '/ingest/jobs/events'
}))

const file = (name: string) => new File(['x'], name, { type: 'text/plain' })

beforeEach(() => {
  useIngestRunStore.getState().reset()
  useIngestJobsStore.getState().clear()
  createIngestJob.mockReset()
  createIngestJob.mockResolvedValue({ job_id: 'job-1', adopted: false })
})

describe('useIngestRunStore', () => {
  it('records upload events and adopts the returned job id', async () => {
    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('a.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    expect(useIngestRunStore.getState().uploadEvents).toHaveLength(2)
    expect(useIngestRunStore.getState().activeJobId).toBe('job-1')
    expect(useIngestRunStore.getState().uploading).toBe(false)
  })

  it('clears the picked files once the job is queued', async () => {
    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('a.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    expect(useIngestRunStore.getState().files).toEqual([])
  })

  it('refuses to start without a collection or files', async () => {
    await useIngestRunStore.getState().start(1000, defaultT)
    expect(createIngestJob).not.toHaveBeenCalled()
  })

  it('dedupes files by name on add', () => {
    const s = useIngestRunStore.getState()
    s.addFiles([file('a.txt')])
    s.addFiles([file('a.txt'), file('b.txt')])
    expect(useIngestRunStore.getState().files.map((f) => f.name)).toEqual(['a.txt', 'b.txt'])
  })

  it('does not persist File objects', () => {
    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('a.txt')])
    const persisted = JSON.parse(localStorage.getItem('docint-ingest-run') ?? '{}')
    expect(persisted.state.collection).toBe('mydocs')
    expect(persisted.state.files).toBeUndefined()
  })
})
