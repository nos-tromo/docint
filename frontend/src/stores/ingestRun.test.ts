import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useIngestRunStore } from './ingestRun'
import { useIngestJobsStore } from './ingestJobs'
import { defaultT } from '@/i18n/defaultT'

// A `vi.fn()` wrapper (not a fixed factory) so individual tests can override
// the upload outcome per-call with `mockImplementationOnce` — needed to
// exercise `start()`'s failure branches below.
const streamIngestUploadBatched = vi.fn()
vi.mock('@/api/ingest', () => ({
  streamIngestUploadBatched: (...args: unknown[]) => streamIngestUploadBatched(...args)
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

  streamIngestUploadBatched.mockReset()
  // Default: a clean single-file, single-batch success. Tests exercising a
  // failure branch override this with `mockImplementationOnce`.
  streamIngestUploadBatched.mockImplementation(async function* () {
    yield { event: 'start', data: { collection: 'mydocs', files: ['a.txt'] }, receivedAt: 0 }
    yield { event: 'file_saved', data: { filename: 'a.txt' }, receivedAt: 0 }
    return { anySaved: true, failures: [] }
  })
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

  it('reports the upload leg so the run starts where the timer did', async () => {
    // The job only exists from finalize on, so the leg before it is the one
    // stretch the server cannot measure. Anchored to the synthetic `start`
    // event — the same stamp the card's timer ticks from — not to "now".
    streamIngestUploadBatched.mockImplementationOnce(async function* () {
      yield {
        event: 'start',
        data: { collection: 'mydocs', files: ['a.txt'] },
        receivedAt: Date.now() - 2_000
      }
      yield { event: 'file_saved', data: { filename: 'a.txt' }, receivedAt: Date.now() }
      return { anySaved: true, failures: [] }
    })
    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('a.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    const payload = createIngestJob.mock.calls[0][0] as { upload_elapsed_ms?: number }
    expect(payload.upload_elapsed_ms).toBeGreaterThanOrEqual(2_000)
    expect(payload.upload_elapsed_ms).toBeLessThan(3_000)
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

  it('removes a file at the given index', () => {
    const s = useIngestRunStore.getState()
    s.addFiles([file('a.txt'), file('b.txt'), file('c.txt')])
    s.removeFile(1)
    expect(useIngestRunStore.getState().files.map((f) => f.name)).toEqual(['a.txt', 'c.txt'])
  })

  it('clears all picked files', () => {
    const s = useIngestRunStore.getState()
    s.addFiles([file('a.txt')])
    s.clearFiles()
    expect(useIngestRunStore.getState().files).toEqual([])
  })

  it('does not persist File objects', () => {
    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('a.txt')])
    const persisted = JSON.parse(localStorage.getItem('docint-ingest-run') ?? '{}')
    expect(persisted.state.collection).toBe('mydocs')
    expect(persisted.state.files).toBeUndefined()
  })

  it("surfaces the generator's own terminal error message on total upload failure, not a recomputed generic one", async () => {
    // Regression test: the generator already picks the more actionable
    // message (distinguishing a 413-too-large failure from a generic
    // rejection) and yields it as the last event before returning
    // `anySaved: false`. The store must reuse that message rather than
    // recomputing a different, less specific one from `failures` alone.
    const tooLargeMessage = 'Every file is over the per-upload limit; raise it or upload smaller files.'
    streamIngestUploadBatched.mockImplementationOnce(async function* () {
      yield { event: 'start', data: { collection: 'mydocs', files: ['big.pdf'] }, receivedAt: 0 }
      yield { event: 'warning', data: { message: 'batch 1/1 too large' }, receivedAt: 0 }
      yield { event: 'error', data: { message: tooLargeMessage }, receivedAt: 0 }
      return { anySaved: false, failures: [{ batch: 1, total: 1, files: ['big.pdf'], status: 413 }] }
    })

    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('big.pdf')])
    await useIngestRunStore.getState().start(1000, defaultT)

    expect(useIngestRunStore.getState().error).toBe(tooLargeMessage)
    expect(useIngestRunStore.getState().uploading).toBe(false)
    expect(createIngestJob).not.toHaveBeenCalled()
  })

  it('falls back to a generic rejection message when the failed stream yields no terminal error event', async () => {
    // Defensive fallback for a malformed/empty failure stream (should not
    // happen from the real generator, but the store must not crash or show
    // a blank error if it does).
    streamIngestUploadBatched.mockImplementationOnce(async function* () {
      yield { event: 'start', data: { collection: 'mydocs', files: ['big.pdf'] }, receivedAt: 0 }
      return { anySaved: false, failures: [{ batch: 1, total: 1, files: ['big.pdf'], status: 413 }] }
    })

    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('big.pdf')])
    await useIngestRunStore.getState().start(1000, defaultT)

    expect(useIngestRunStore.getState().error).toBe(defaultT('ingest.upload_failed_rejected', { count: 1 }))
    expect(useIngestRunStore.getState().uploading).toBe(false)
  })

  it('sets an error and clears uploading when the upload stream throws', async () => {
    streamIngestUploadBatched.mockImplementationOnce(async function* () {
      yield { event: 'start', data: { collection: 'mydocs', files: ['a.txt'] }, receivedAt: 0 }
      throw new Error('network dropped')
    })

    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('a.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    expect(useIngestRunStore.getState().error).toBe(defaultT('ingest.failed_default'))
    expect(useIngestRunStore.getState().uploading).toBe(false)
    expect(createIngestJob).not.toHaveBeenCalled()
    // The throw happened mid-upload — files stay put so the user can retry.
    expect(useIngestRunStore.getState().files.map((f) => f.name)).toEqual(['a.txt'])
  })

  it('leaves the picked files in place when createIngestJob fails after a successful upload', async () => {
    createIngestJob.mockReset()
    createIngestJob.mockRejectedValueOnce(new Error('finalize unreachable'))

    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('a.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    // Pins the retry-without-re-picking claim: the files that were already
    // staged server-side stay selected so the user can hit start() again.
    expect(useIngestRunStore.getState().files.map((f) => f.name)).toEqual(['a.txt'])
    expect(useIngestRunStore.getState().activeJobId).toBeNull()
    expect(useIngestRunStore.getState().uploading).toBe(false)
    expect(useIngestRunStore.getState().error).toBe(defaultT('ingest.failed_default'))
  })

  it('populates warnings and failedFiles on a partial-batch failure that still saves something', async () => {
    streamIngestUploadBatched.mockImplementationOnce(async function* () {
      yield { event: 'start', data: { collection: 'mydocs', files: ['a.txt', 'big.pdf'] }, receivedAt: 0 }
      yield { event: 'file_saved', data: { filename: 'a.txt' }, receivedAt: 0 }
      yield { event: 'warning', data: { message: 'batch 2/2 too large; upload it separately' }, receivedAt: 0 }
      return { anySaved: true, failures: [{ batch: 2, total: 2, files: ['big.pdf'], status: 413 }] }
    })

    const s = useIngestRunStore.getState()
    s.setCollection('mydocs')
    s.addFiles([file('a.txt'), file('big.pdf')])
    await useIngestRunStore.getState().start(1000, defaultT)

    expect(useIngestRunStore.getState().warnings).toEqual(['batch 2/2 too large; upload it separately'])
    expect(useIngestRunStore.getState().failedFiles).toEqual(['big.pdf'])
    expect(useIngestRunStore.getState().activeJobId).toBe('job-1')
  })
})
