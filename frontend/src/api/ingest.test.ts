import { describe, it, expect, vi, afterEach } from 'vitest'
import { buildIngestFormData, streamIngestUploadBatched } from './ingest'
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

/** SSE frames for one staged (defer_ingest) upload batch: start → file_saved → upload_complete. */
function stagedBatch(filename: string) {
  return sseResponse(
    `event: start\ndata: ${JSON.stringify({ collection: 'c1', files: [filename] })}\n\n` +
      `event: file_saved\ndata: ${JSON.stringify({ filename })}\n\n` +
      `event: upload_complete\ndata: ${JSON.stringify({ collection: 'c1', files_saved: 1 })}\n\n`
  )
}

/** SSE frames for the /ingest/finalize ingestion pass. */
function finalizeStream(opts?: { empty?: boolean; error?: string; warning?: string }) {
  const started = `event: ingestion_started\ndata: ${JSON.stringify({ collection: 'c1' })}\n\n`
  if (opts?.error) {
    return sseResponse(started + `event: error\ndata: ${JSON.stringify({ message: opts.error })}\n\n`)
  }
  const warn = opts?.warning
    ? `event: warning\ndata: ${JSON.stringify({ message: opts.warning, collection: 'c1' })}\n\n`
    : ''
  return sseResponse(
    started +
      warn +
      `event: ingestion_complete\ndata: ${JSON.stringify({ collection: 'c1', empty: opts?.empty ?? false })}\n\n`
  )
}

async function collect(
  gen: AsyncGenerator<IngestEvent, void, unknown>
): Promise<Array<{ event: string; data: Record<string, unknown> }>> {
  const out: Array<{ event: string; data: Record<string, unknown> }> = []
  for await (const ev of gen) out.push({ event: ev.event, data: ev.data })
  return out
}

const lastCallUrl = (m: ReturnType<typeof vi.fn>): string => String(m.mock.calls.at(-1)?.[0])

describe('streamIngestUploadBatched', () => {
  it('stages each batch then finalises once, normalised to one logical stream', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(stagedBatch('a'))
      .mockResolvedValueOnce(stagedBatch('b'))
      .mockResolvedValueOnce(finalizeStream())
    vi.stubGlobal('fetch', fetchMock)

    // budget = floor(1000 * 0.9) = 900; two 500-byte files → two batches.
    const files = [fileOfSize('a', 500), fileOfSize('b', 500)]
    const events = await collect(streamIngestUploadBatched('c1', files, 1000))

    // Two staged uploads + one finalize.
    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(lastCallUrl(fetchMock)).toContain('/ingest/finalize')
    // Every upload batch must set defer_ingest so nothing ingests per batch.
    expect((fetchMock.mock.calls[0][1].body as FormData).get('defer_ingest')).toBe('true')
    // One synthetic start (all files), both file_saved forwarded, finalize's
    // ingestion_started forwarded, one synthetic terminal complete.
    expect(events.map((e) => e.event)).toEqual([
      'start',
      'file_saved',
      'file_saved',
      'ingestion_started',
      'ingestion_complete'
    ])
    expect(events[0].data.files).toEqual(['a', 'b'])
    expect(events.at(-1)?.data).toMatchObject({ collection: 'c1', empty: false })
    expect(events.at(-1)?.data.failed_files).toBeUndefined()
  })

  it('stages a small selection as a single batch then finalises', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(stagedBatch('a'))
      .mockResolvedValueOnce(finalizeStream())
    vi.stubGlobal('fetch', fetchMock)

    const events = await collect(streamIngestUploadBatched('c1', [fileOfSize('a', 100)], 1_000_000))

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(events.map((e) => e.event)).toEqual([
      'start',
      'file_saved',
      'ingestion_started',
      'ingestion_complete'
    ])
  })

  it('continues past a 413 batch, still finalises, and flags the partial failure', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(stagedBatch('a'))
      .mockResolvedValueOnce({ ok: false, status: 413, body: null })
      .mockResolvedValueOnce(finalizeStream())
    vi.stubGlobal('fetch', fetchMock)

    const files = [fileOfSize('a', 500), fileOfSize('big', 500)]
    const events = await collect(streamIngestUploadBatched('c1', files, 1000))

    // The bad batch surfaces as a warning; finalize still ingests the staged file.
    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(lastCallUrl(fetchMock)).toContain('/ingest/finalize')
    expect(events.map((e) => e.event)).toEqual([
      'start',
      'file_saved',
      'warning',
      'ingestion_started',
      'ingestion_complete'
    ])
    expect(String(events[2].data.message)).toContain('per-upload limit')
    expect(events.at(-1)?.data.failed_files).toEqual(['big'])
    expect(String(events.at(-1)?.data.failed_message)).toContain('big')
  })

  it('emits a terminal error and skips finalize when every batch fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 413, body: null })
    vi.stubGlobal('fetch', fetchMock)

    const files = [fileOfSize('a', 500), fileOfSize('b', 500)]
    const events = await collect(streamIngestUploadBatched('c1', files, 1000))

    // Nothing was staged → finalize is NOT called (still 2 fetches).
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(events.map((e) => e.event)).toEqual(['start', 'warning', 'warning', 'error'])
    expect(String(events[3].data.message)).toContain('per-upload limit')
  })

  it('surfaces a finalize failure as a terminal error', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(stagedBatch('a'))
      .mockResolvedValueOnce(finalizeStream({ error: 'Embedding endpoint unreachable' }))
    vi.stubGlobal('fetch', fetchMock)

    const events = await collect(streamIngestUploadBatched('c1', [fileOfSize('a', 100)], 1_000_000))

    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved', 'ingestion_started', 'error'])
    expect(String(events.at(-1)?.data.message)).toContain('Embedding endpoint unreachable')
  })

  it('completes as empty (with warning) when finalize finds nothing ingestable', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(stagedBatch('clip.mp4'))
      .mockResolvedValueOnce(finalizeStream({ empty: true, warning: 'No ingestable files found' }))
    vi.stubGlobal('fetch', fetchMock)

    const events = await collect(streamIngestUploadBatched('c1', [fileOfSize('clip.mp4', 100)], 1_000_000))

    expect(events.map((e) => e.event)).toEqual([
      'start',
      'file_saved',
      'ingestion_started',
      'warning',
      'ingestion_complete'
    ])
    expect(events.at(-1)?.data).toMatchObject({ empty: true })
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

  it('appends defer_ingest only when staging', () => {
    const f = new File([new Uint8Array([1])], 'b.png', { type: 'image/png' })
    expect(buildIngestFormData('c1', [f]).get('defer_ingest')).toBeNull()
    expect(buildIngestFormData('c1', [f], true).get('defer_ingest')).toBe('true')
  })
})
