import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { buildIngestFormData, streamIngestUploadBatched, type BatchFailure } from './ingest'
import { streamUpload, UploadHttpError, UploadStalledError } from './upload'
import { setOwnerParam } from './client'
import type { IngestEvent } from './types'
import { defaultT } from '@/i18n/defaultT'

vi.mock('./upload', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./upload')>()),
  streamUpload: vi.fn()
}))

const uploadMock = vi.mocked(streamUpload)

beforeEach(() => {
  uploadMock.mockReset()
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  setOwnerParam(null)
})

/** Build a File whose reported `size` is `size` bytes (no real allocation). */
function fileOfSize(name: string, size: number): File {
  const f = new File([new Uint8Array(0)], name)
  Object.defineProperty(f, 'size', { value: size })
  return f
}

type Frame = [event: string, data: Record<string, unknown>]

/** A `streamUpload` stand-in that replays `frames`, then throws `failure` when given one. */
function reply(frames: Frame[], failure?: unknown) {
  return async function* () {
    for (const [event, data] of frames) yield { event, data }
    if (failure) throw failure
  }
}

/** One fully staged upload batch: start → file_saved per name → upload_complete. */
function staged(...names: string[]) {
  return reply([
    ['start', { collection: 'c1', files: names }],
    ...names.map((filename): Frame => ['file_saved', { filename }]),
    ['upload_complete', { collection: 'c1', files_saved: names.length }]
  ])
}

/** The file names the n-th upload request carried. */
function sentNames(call: number): string[] {
  return ((uploadMock.mock.calls[call][1] as FormData).getAll('files') as File[]).map((f) => f.name)
}

/** Drain the generator, splitting its yielded events from its final return value. */
async function collect(
  gen: AsyncGenerator<IngestEvent, { anySaved: boolean; failures: BatchFailure[] }, unknown>
): Promise<{
  events: Array<{ event: string; data: Record<string, unknown> }>
  result: { anySaved: boolean; failures: BatchFailure[] }
}> {
  const events: Array<{ event: string; data: Record<string, unknown> }> = []
  let next = await gen.next()
  while (!next.done) {
    events.push({ event: next.value.event, data: next.value.data })
    next = await gen.next()
  }
  return { events, result: next.value }
}

describe('streamIngestUploadBatched', () => {
  it('stages every batch, normalised to one logical upload stream', async () => {
    uploadMock.mockImplementationOnce(staged('a')).mockImplementationOnce(staged('b'))

    // budget = floor(1000 * 0.9) = 900; two 500-byte files → two batches.
    const files = [fileOfSize('a', 500), fileOfSize('b', 500)]
    const { events, result } = await collect(streamIngestUploadBatched('c1', files, 1000))

    // Two staged uploads — no finalize call; queuing the job is the caller's job now.
    expect(uploadMock).toHaveBeenCalledTimes(2)
    // One synthetic start (all files) plus both file_saved forwarded.
    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved', 'file_saved'])
    expect(events[0].data.files).toEqual(['a', 'b'])
    expect(result).toEqual({ anySaved: true, failures: [] })
  })

  it('stages a small selection as a single batch', async () => {
    uploadMock.mockImplementationOnce(staged('a'))

    const { events, result } = await collect(
      streamIngestUploadBatched('c1', [fileOfSize('a', 100)], 1_000_000)
    )

    expect(uploadMock).toHaveBeenCalledTimes(1)
    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved'])
    expect(result).toEqual({ anySaved: true, failures: [] })
  })

  it('continues past a 413 batch and reports it as a partial failure', async () => {
    uploadMock.mockImplementationOnce(staged('a')).mockImplementationOnce(reply([], new UploadHttpError(413)))

    const files = [fileOfSize('a', 500), fileOfSize('big', 500)]
    const { events, result } = await collect(streamIngestUploadBatched('c1', files, 1000))

    // The bad batch surfaces as a warning; the good one still uploads.
    expect(uploadMock).toHaveBeenCalledTimes(2)
    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved', 'warning'])
    expect(String(events[2].data.message)).toContain('per-upload limit')
    expect(result.anySaved).toBe(true)
    expect(result.failures).toEqual([{ batch: 2, total: 2, files: ['big'], status: 413 }])
  })

  it('emits a terminal error when every batch fails to upload', async () => {
    uploadMock.mockImplementation(reply([], new UploadHttpError(413)))

    const files = [fileOfSize('a', 500), fileOfSize('b', 500)]
    const { events, result } = await collect(streamIngestUploadBatched('c1', files, 1000))

    expect(uploadMock).toHaveBeenCalledTimes(2)
    expect(events.map((e) => e.event)).toEqual(['start', 'warning', 'warning', 'error'])
    expect(String(events[3].data.message)).toContain('per-upload limit')
    expect(result.anySaved).toBe(false)
    expect(result.failures).toHaveLength(2)
  })

  it('names the failing file on a save_failed event when it matches the upload list', async () => {
    uploadMock.mockImplementationOnce(
      reply([
        ['start', { collection: 'c1', files: ['a.txt'] }],
        ['error', { message: 'Failed to save file.', code: 'save_failed', filename: 'a.txt' }]
      ])
    )

    const { events, result } = await collect(streamIngestUploadBatched('c1', [fileOfSize('a.txt', 10)], 1000))
    const err = events.find((e) => e.event === 'error')
    expect(err).toBeDefined()
    expect(String(err!.data.message)).toContain('a.txt')
    expect(String(err!.data.message)).toContain('(save_failed)')
    // A server-side save error is not something a retry cures; the file is
    // reported as missing instead of being silently dropped.
    expect(uploadMock).toHaveBeenCalledTimes(1)
    expect(result.failures).toEqual([{ batch: 1, total: 1, files: ['a.txt'] }])
  })

  it('never names a file the client did not upload on save_failed', async () => {
    uploadMock.mockImplementationOnce(
      reply([
        ['start', { collection: 'c1', files: ['a.txt'] }],
        ['error', { message: 'Failed to save file.', code: 'save_failed', filename: '../../etc/passwd' }]
      ])
    )

    const { events } = await collect(streamIngestUploadBatched('c1', [fileOfSize('a.txt', 10)], 1000))
    const err = events.find((e) => e.event === 'error')
    expect(err).toBeDefined()
    expect(String(err!.data.message)).not.toContain('passwd')
    expect(String(err!.data.message)).toBe('Ingestion failed. (save_failed)')
  })

  it('never puts enrichment flags into the staged upload body — /ingest/upload stages only', async () => {
    uploadMock.mockImplementationOnce(staged('a.txt'))

    // Enrichment travels solely on the `createIngestJob` (`/ingest/finalize`)
    // call the run controller makes once every batch is staged — this
    // function has no enrichment parameter at all.
    await collect(streamIngestUploadBatched('c1', [fileOfSize('a.txt', 10)], 1000))
    const uploadBody = uploadMock.mock.calls[0][1]
    expect(uploadBody.get('ner')).toBeNull()
    expect(uploadBody.get('hate_speech')).toBeNull()
    expect(uploadBody.get('defer_ingest')).toBeNull()
    expect([...uploadBody.keys()].sort()).toEqual(['collection', 'files'])
  })

  it('packs batches at 64 MiB even when the server accepts far more per request', async () => {
    uploadMock.mockImplementationOnce(staged('a')).mockImplementationOnce(staged('b'))
    const mib = 1024 * 1024

    await collect(streamIngestUploadBatched('c1', [fileOfSize('a', 40 * mib), fileOfSize('b', 40 * mib)], 1024 * mib))

    expect(uploadMock).toHaveBeenCalledTimes(2)
  })

  it('re-sends only the files a stalled batch had not saved yet', async () => {
    vi.useFakeTimers()
    uploadMock
      .mockImplementationOnce(reply([['file_saved', { filename: 'a' }]], new UploadStalledError()))
      .mockImplementationOnce(staged('b'))

    const run = collect(streamIngestUploadBatched('c1', [fileOfSize('a', 1), fileOfSize('b', 1)], 1000))
    await vi.runAllTimersAsync()
    const { events, result } = await run

    expect(sentNames(0)).toEqual(['a', 'b'])
    expect(sentNames(1)).toEqual(['b'])
    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved', 'warning', 'file_saved', 'warning_resolved'])
    expect(String(events[2].data.message)).toBe('Batch 1/1 did not go through — retrying (1/2).')
    // The retry worked, so its notice is retracted rather than left on screen.
    expect(events[4].data.messages).toEqual([events[2].data.message])
    expect(result).toEqual({ anySaved: true, failures: [] })
  })

  it('clears one pending file per save ack, so a same-named file still gets re-sent', async () => {
    vi.useFakeTimers()
    uploadMock
      .mockImplementationOnce(reply([['file_saved', { filename: 'a' }]], new UploadStalledError()))
      .mockImplementationOnce(staged('a'))

    const first = new File([new Uint8Array(1)], 'a')
    const second = new File([new Uint8Array(2)], 'a')
    const run = collect(streamIngestUploadBatched('c1', [first, second], 1000))
    await vi.runAllTimersAsync()
    const { result } = await run

    expect(uploadMock).toHaveBeenCalledTimes(2)
    const resent = (uploadMock.mock.calls[1][1] as FormData).getAll('files') as File[]
    expect(resent.map((f) => f.size)).toEqual([2])
    expect(result).toEqual({ anySaved: true, failures: [] })
  })

  it('sends every attempt under the owner the run started with', async () => {
    vi.useFakeTimers()
    uploadMock
      .mockImplementationOnce(reply([['file_saved', { filename: 'a' }]], new UploadStalledError()))
      .mockImplementationOnce(staged('b'))

    const run = collect(
      streamIngestUploadBatched('c1', [fileOfSize('a', 1), fileOfSize('b', 1)], 1000, undefined, defaultT, 'alice')
    )
    // An admin switches the sidebar to another user's collection mid-run.
    setOwnerParam('bob')
    await vi.runAllTimersAsync()
    await run

    expect(uploadMock.mock.calls.map((call) => call[3])).toEqual(['alice', 'alice'])
  })

  it('retries a gateway error and a stream that ended without upload_complete', async () => {
    vi.useFakeTimers()
    uploadMock
      .mockImplementationOnce(reply([], new UploadHttpError(502)))
      .mockImplementationOnce(reply([['file_saved', { filename: 'a' }]]))
      .mockImplementationOnce(staged('b'))

    const run = collect(streamIngestUploadBatched('c1', [fileOfSize('a', 1), fileOfSize('b', 1)], 1000))
    await vi.runAllTimersAsync()
    const { result } = await run

    expect(uploadMock).toHaveBeenCalledTimes(3)
    expect(sentNames(2)).toEqual(['b'])
    expect(result).toEqual({ anySaved: true, failures: [] })
  })

  it('gives up after the retries and reports exactly the files that never arrived', async () => {
    vi.useFakeTimers()
    uploadMock.mockImplementation(reply([], new UploadStalledError()))

    const run = collect(streamIngestUploadBatched('c1', [fileOfSize('a', 1)], 1000))
    await vi.runAllTimersAsync()
    const { events, result } = await run

    expect(uploadMock).toHaveBeenCalledTimes(3)
    expect(events.map((e) => e.event)).toEqual(['start', 'warning', 'warning', 'warning', 'error'])
    expect(String(events[3].data.message)).toBe(
      'Batch 1/1 ("a") stopped making progress and was skipped; other batches continued.'
    )
    expect(result.failures).toEqual([{ batch: 1, total: 1, files: ['a'], stalled: true }])
  })

  it('counts a batch done once upload_complete arrived, even if the stream then stalls', async () => {
    uploadMock.mockImplementationOnce(
      reply(
        [
          ['file_saved', { filename: 'a' }],
          ['upload_complete', { collection: 'c1', files_saved: 1 }]
        ],
        new UploadStalledError()
      )
    )

    const { result } = await collect(streamIngestUploadBatched('c1', [fileOfSize('a', 1)], 1000))

    expect(uploadMock).toHaveBeenCalledTimes(1)
    expect(result).toEqual({ anySaved: true, failures: [] })
  })
})

describe('buildIngestFormData', () => {
  it('uses webkitRelativePath as the upload filename when present', () => {
    const f = new File([new Uint8Array([1])], 'a.jpg', { type: 'image/jpeg' })
    Object.defineProperty(f, 'webkitRelativePath', { value: 'export/media/sub/a.jpg' })
    const fd = buildIngestFormData('c1', [f])
    const entries = fd.getAll('files') as File[]
    expect(entries[0].name).toBe('export/media/sub/a.jpg')
    expect(fd.get('collection')).toBe('c1')
  })

  it('falls back to the file name when webkitRelativePath is empty', () => {
    const f = new File([new Uint8Array([1])], 'b.png', { type: 'image/png' })
    const fd = buildIngestFormData('c1', [f])
    expect((fd.getAll('files') as File[])[0].name).toBe('b.png')
  })

  it('sends only collection and files — no defer_ingest, ner, or hate_speech', () => {
    // /ingest/upload stages only (Task 4 deleted these fields from the
    // backend); sending them would be dead payload the server discards.
    // Enrichment now travels solely on the createIngestJob call.
    const f = new File([new Uint8Array([1])], 'b.png', { type: 'image/png' })
    const fd = buildIngestFormData('c1', [f])
    expect(fd.get('defer_ingest')).toBeNull()
    expect(fd.get('ner')).toBeNull()
    expect(fd.get('hate_speech')).toBeNull()
    expect([...fd.keys()].sort()).toEqual(['collection', 'files'])
  })
})
