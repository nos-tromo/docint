import { describe, it, expect, vi, afterEach } from 'vitest'
import { buildIngestFormData, streamIngestUploadBatched, type BatchFailure } from './ingest'
import type { IngestEvent } from './types'

afterEach(() => vi.restoreAllMocks())

/** Build a File whose reported `size` is `size` bytes (no real allocation). */
function fileOfSize(name: string, size: number): File {
  const f = new File([new Uint8Array(0)], name)
  Object.defineProperty(f, 'size', { value: size })
  return f
}

/** A mock fetch Response streaming the given SSE frame text. */
function sseResponse(frames: string) {
  const enc = new TextEncoder()
  return {
    ok: true,
    status: 200,
    body: new ReadableStream<Uint8Array>({
      start(c) {
        c.enqueue(enc.encode(frames))
        c.close()
      }
    })
  }
}

/** SSE frames for one staged upload batch: start → file_saved → upload_complete. */
function stagedBatch(filename: string) {
  return sseResponse(
    `event: start\ndata: ${JSON.stringify({ collection: 'c1', files: [filename] })}\n\n` +
      `event: file_saved\ndata: ${JSON.stringify({ filename })}\n\n` +
      `event: upload_complete\ndata: ${JSON.stringify({ collection: 'c1', files_saved: 1 })}\n\n`
  )
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
    const fetchMock = vi.fn().mockResolvedValueOnce(stagedBatch('a')).mockResolvedValueOnce(stagedBatch('b'))
    vi.stubGlobal('fetch', fetchMock)

    // budget = floor(1000 * 0.9) = 900; two 500-byte files → two batches.
    const files = [fileOfSize('a', 500), fileOfSize('b', 500)]
    const { events, result } = await collect(streamIngestUploadBatched('c1', files, 1000))

    // Two staged uploads — no finalize call; queuing the job is the caller's job now.
    expect(fetchMock).toHaveBeenCalledTimes(2)
    // One synthetic start (all files) plus both file_saved forwarded.
    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved', 'file_saved'])
    expect(events[0].data.files).toEqual(['a', 'b'])
    expect(result).toEqual({ anySaved: true, failures: [] })
  })

  it('stages a small selection as a single batch', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(stagedBatch('a'))
    vi.stubGlobal('fetch', fetchMock)

    const { events, result } = await collect(
      streamIngestUploadBatched('c1', [fileOfSize('a', 100)], 1_000_000)
    )

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved'])
    expect(result).toEqual({ anySaved: true, failures: [] })
  })

  it('continues past a 413 batch and reports it as a partial failure', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(stagedBatch('a'))
      .mockResolvedValueOnce({ ok: false, status: 413, body: null })
    vi.stubGlobal('fetch', fetchMock)

    const files = [fileOfSize('a', 500), fileOfSize('big', 500)]
    const { events, result } = await collect(streamIngestUploadBatched('c1', files, 1000))

    // The bad batch surfaces as a warning; the good one still uploads.
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved', 'warning'])
    expect(String(events[2].data.message)).toContain('per-upload limit')
    expect(result.anySaved).toBe(true)
    expect(result.failures).toEqual([{ batch: 2, total: 2, files: ['big'], status: 413 }])
  })

  it('emits a terminal error when every batch fails to upload', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 413, body: null })
    vi.stubGlobal('fetch', fetchMock)

    const files = [fileOfSize('a', 500), fileOfSize('b', 500)]
    const { events, result } = await collect(streamIngestUploadBatched('c1', files, 1000))

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(events.map((e) => e.event)).toEqual(['start', 'warning', 'warning', 'error'])
    expect(String(events[3].data.message)).toContain('per-upload limit')
    expect(result.anySaved).toBe(false)
    expect(result.failures).toHaveLength(2)
  })

  it('names the failing file on a save_failed event when it matches the upload list', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce(
      sseResponse(
        `event: start\ndata: ${JSON.stringify({ collection: 'c1', files: ['a.txt'] })}\n\n` +
          `event: error\ndata: ${JSON.stringify({ message: 'Failed to save file.', code: 'save_failed', filename: 'a.txt' })}\n\n`
      )
    ) as unknown as typeof fetch

    const { events } = await collect(streamIngestUploadBatched('c1', [fileOfSize('a.txt', 10)], 1000))
    const err = events.find((e) => e.event === 'error')
    expect(err).toBeDefined()
    expect(String(err!.data.message)).toContain('a.txt')
    expect(String(err!.data.message)).toContain('(save_failed)')
  })

  it('never names a file the client did not upload on save_failed', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce(
      sseResponse(
        `event: start\ndata: ${JSON.stringify({ collection: 'c1', files: ['a.txt'] })}\n\n` +
          `event: error\ndata: ${JSON.stringify({ message: 'Failed to save file.', code: 'save_failed', filename: '../../etc/passwd' })}\n\n`
      )
    ) as unknown as typeof fetch

    const { events } = await collect(streamIngestUploadBatched('c1', [fileOfSize('a.txt', 10)], 1000))
    const err = events.find((e) => e.event === 'error')
    expect(err).toBeDefined()
    expect(String(err!.data.message)).not.toContain('passwd')
    expect(String(err!.data.message)).toBe('Ingestion failed. (save_failed)')
  })

  it('never forwards enrichment flags into the staged upload body — /ingest/upload stages only and discards them', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(stagedBatch('a.txt'))
    vi.stubGlobal('fetch', fetchMock)

    // The trailing options argument is accepted only for a still-compiling
    // caller's sake (see the function's docstring); it must not leak into
    // the wire payload even when a caller does pass one.
    await collect(
      streamIngestUploadBatched('c1', [fileOfSize('a.txt', 10)], 1000, undefined, undefined, {
        ner: false,
        hateSpeech: true
      })
    )
    const uploadBody = fetchMock.mock.calls[0][1].body as FormData
    expect(uploadBody.get('ner')).toBeNull()
    expect(uploadBody.get('hate_speech')).toBeNull()
    expect(uploadBody.get('defer_ingest')).toBeNull()
    expect([...uploadBody.keys()].sort()).toEqual(['collection', 'files'])
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
