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

/** SSE frames for one successful backend batch (start → file_saved → complete). */
function okBatch(filename: string, opts?: { empty?: boolean }) {
  return sseResponse(
    `event: start\ndata: ${JSON.stringify({ collection: 'c1', files: [filename] })}\n\n` +
      `event: file_saved\ndata: ${JSON.stringify({ filename })}\n\n` +
      `event: ingestion_complete\ndata: ${JSON.stringify({
        collection: 'c1',
        empty: opts?.empty ?? false
      })}\n\n`
  )
}

async function collect(
  gen: AsyncGenerator<IngestEvent, void, unknown>
): Promise<Array<{ event: string; data: Record<string, unknown> }>> {
  const out: Array<{ event: string; data: Record<string, unknown> }> = []
  for await (const ev of gen) out.push({ event: ev.event, data: ev.data })
  return out
}

describe('streamIngestUploadBatched', () => {
  it('splits a large selection into batches and normalises to one logical stream', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(okBatch('a')).mockResolvedValueOnce(okBatch('b'))
    vi.stubGlobal('fetch', fetchMock)

    // budget = floor(1000 * 0.9) = 900; two 500-byte files → two batches.
    const files = [fileOfSize('a', 500), fileOfSize('b', 500)]
    const events = await collect(streamIngestUploadBatched('c1', files, 1000))

    expect(fetchMock).toHaveBeenCalledTimes(2)
    // One synthetic start (all files), both file_saved forwarded, per-batch
    // starts + completes swallowed, one synthetic terminal complete.
    expect(events.map((e) => e.event)).toEqual([
      'start',
      'file_saved',
      'file_saved',
      'ingestion_complete'
    ])
    expect(events[0].data.files).toEqual(['a', 'b'])
    expect(events[3].data).toMatchObject({ collection: 'c1', empty: false })
    expect(events[3].data.failed_files).toBeUndefined()
  })

  it('uploads a small selection as a single batch', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(okBatch('a'))
    vi.stubGlobal('fetch', fetchMock)

    const events = await collect(
      streamIngestUploadBatched('c1', [fileOfSize('a', 100)], 1_000_000)
    )

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(events.map((e) => e.event)).toEqual(['start', 'file_saved', 'ingestion_complete'])
  })

  it('continues past a 413 batch and reports it as a partial failure', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(okBatch('a'))
      .mockResolvedValueOnce({ ok: false, status: 413, body: null })
    vi.stubGlobal('fetch', fetchMock)

    const files = [fileOfSize('a', 500), fileOfSize('big', 500)]
    const events = await collect(streamIngestUploadBatched('c1', files, 1000))

    // The bad batch surfaces as a warning; the good batch still commits.
    expect(events.map((e) => e.event)).toEqual([
      'start',
      'file_saved',
      'warning',
      'ingestion_complete'
    ])
    expect(String(events[2].data.message)).toContain('per-upload limit')
    expect(events[3].data.failed_files).toEqual(['big'])
    expect(String(events[3].data.failed_message)).toContain('big')
  })

  it('emits a terminal error when every batch fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 413, body: null })
    vi.stubGlobal('fetch', fetchMock)

    const files = [fileOfSize('a', 500), fileOfSize('b', 500)]
    const events = await collect(streamIngestUploadBatched('c1', files, 1000))

    expect(events.map((e) => e.event)).toEqual(['start', 'warning', 'warning', 'error'])
    expect(String(events[3].data.message)).toContain('per-upload limit')
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
})
