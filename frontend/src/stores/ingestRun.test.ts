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

    // The upload leg's events move to the job they produced, so `uploadEvents`
    // only ever describes the upload currently in flight.
    expect(useIngestRunStore.getState().uploadEventsByJob['job-1']).toHaveLength(2)
    expect(useIngestRunStore.getState().uploadEvents).toEqual([])
    expect(useIngestRunStore.getState().trackedJobs).toEqual([
      { job_id: 'job-1', collection: 'mydocs' }
    ])
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
    expect(useIngestRunStore.getState().trackedJobs).toEqual([])
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
    expect(useIngestRunStore.getState().trackedJobs[0].job_id).toBe('job-1')
  })
})

describe('useIngestRunStore — tracking several jobs', () => {
  it('keeps every started job, newest first', async () => {
    const s = useIngestRunStore.getState()
    s.setCollection('first')
    s.addFiles([file('a.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    createIngestJob.mockResolvedValue({ job_id: 'job-2', adopted: false })
    useIngestRunStore.getState().setCollection('second')
    useIngestRunStore.getState().addFiles([file('b.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    expect(useIngestRunStore.getState().trackedJobs).toEqual([
      { job_id: 'job-2', collection: 'second' },
      { job_id: 'job-1', collection: 'first' }
    ])
  })

  it('files each run\'s upload events under its own job', async () => {
    const s = useIngestRunStore.getState()
    s.setCollection('first')
    s.addFiles([file('a.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    createIngestJob.mockResolvedValue({ job_id: 'job-2', adopted: false })
    useIngestRunStore.getState().setCollection('second')
    useIngestRunStore.getState().addFiles([file('b.txt')])
    await useIngestRunStore.getState().start(1000, defaultT)

    const byJob = useIngestRunStore.getState().uploadEventsByJob
    expect(Object.keys(byJob).sort()).toEqual(['job-1', 'job-2'])
  })

  it('re-tracking an adopted job id does not duplicate it', () => {
    const s = useIngestRunStore.getState()
    s.trackJob('job-1', 'mydocs')
    s.trackJob('job-1', 'mydocs')
    expect(useIngestRunStore.getState().trackedJobs).toHaveLength(1)
  })

  it('untracks one job and drops only its upload events', () => {
    const s = useIngestRunStore.getState()
    s.trackJob('job-1', 'first')
    s.trackJob('job-2', 'second')
    useIngestRunStore.setState({
      uploadEventsByJob: {
        'job-1': [{ event: 'start', data: {}, receivedAt: 0 }],
        'job-2': [{ event: 'start', data: {}, receivedAt: 0 }]
      }
    })
    s.untrackJob('job-1')

    expect(useIngestRunStore.getState().trackedJobs).toEqual([
      { job_id: 'job-2', collection: 'second' }
    ])
    expect(Object.keys(useIngestRunStore.getState().uploadEventsByJob)).toEqual(['job-2'])
  })

  it('records handled jobs without losing the earlier ones', () => {
    const s = useIngestRunStore.getState()
    s.markJobHandled('job-1')
    s.markJobHandled('job-2')
    expect(useIngestRunStore.getState().handledJobIds).toEqual(['job-1', 'job-2'])
  })

  it('bounds the handled-job list so it cannot grow forever', () => {
    const s = useIngestRunStore.getState()
    for (let i = 0; i < 60; i += 1) s.markJobHandled(`job-${i}`)
    const handled = useIngestRunStore.getState().handledJobIds
    expect(handled).toHaveLength(50)
    // The newest are the ones worth keeping: an old id is only consulted to
    // stop a completed job reading as interrupted.
    expect(handled[handled.length - 1]).toBe('job-59')
  })

  it('carries a v1 single-job state into the tracked list on migration', () => {
    localStorage.setItem(
      'docint-ingest-run',
      JSON.stringify({
        version: 1,
        state: {
          collection: 'edited',
          ner: true,
          hate: false,
          activeJobId: 'job-1',
          activeJobCollection: 'mydocs',
          handledJobId: 'job-0'
        }
      })
    )
    useIngestRunStore.persist.rehydrate()

    expect(useIngestRunStore.getState().trackedJobs).toEqual([
      { job_id: 'job-1', collection: 'mydocs' }
    ])
    expect(useIngestRunStore.getState().handledJobIds).toEqual(['job-0'])
    expect(useIngestRunStore.getState().ner).toBe(true)
  })
})
