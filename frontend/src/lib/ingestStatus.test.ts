import { describe, it, expect } from 'vitest'
import type { IngestEvent } from '@/api/types'
import {
  deriveIngestStatus,
  formatBytes,
  formatDuration,
  parseProgressMessage
} from './ingestStatus'

function progress(message: string): IngestEvent {
  return { event: 'ingestion_progress', data: { message } }
}

describe('parseProgressMessage', () => {
  it('parses "Core pipeline processing PDF" stage messages', () => {
    const out = parseProgressMessage(
      'Core pipeline processing PDF (3/5): doc_science_complex.pdf'
    )
    expect(out).toEqual({
      kind: 'stage',
      label: 'Processing PDFs',
      current: 3,
      total: 5,
      item: 'doc_science_complex.pdf'
    })
  })

  it('parses "Core pipeline indexed N chunks" messages', () => {
    const out = parseProgressMessage(
      'Core pipeline indexed 61 chunks: doc_science_complex.pdf'
    )
    expect(out.kind).toBe('indexed')
    expect(out.current).toBe(61)
    expect(out.item).toBe('doc_science_complex.pdf')
  })

  it('parses entity-extraction progress', () => {
    const out = parseProgressMessage('Extracting entities: 200/703 chunks processed')
    expect(out).toEqual({
      kind: 'task',
      taskKey: 'entities',
      label: 'Entities',
      current: 200,
      total: 703
    })
  })

  it('parses hate-speech progress', () => {
    const out = parseProgressMessage(
      'Detecting hate speech: 150/703 chunks processed'
    )
    expect(out).toEqual({
      kind: 'task',
      taskKey: 'hate',
      label: 'Hate detection',
      current: 150,
      total: 703
    })
  })

  it('returns kind=unknown for unparseable input without throwing', () => {
    expect(() => parseProgressMessage('utterly unknown payload')).not.toThrow()
    expect(parseProgressMessage('utterly unknown payload')).toEqual({
      kind: 'unknown'
    })
    expect(parseProgressMessage('')).toEqual({ kind: 'unknown' })
    // The runtime guard tolerates non-string input even though the type
    // signature says string — defensive parsing for the SSE stream.
    expect(parseProgressMessage(undefined as unknown as string)).toEqual({
      kind: 'unknown'
    })
  })
})

describe('formatBytes', () => {
  it('handles zero and negative input', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(-5)).toBe('0 B')
  })

  it('keeps sub-KB values in bytes', () => {
    expect(formatBytes(1023)).toBe('1023 B')
  })

  it('promotes to KB at 1024', () => {
    expect(formatBytes(1024)).toBe('1.0 KB')
  })

  it('promotes to MB at 1 MiB', () => {
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB')
  })

  it('truncates rather than rounds for the decimal place', () => {
    expect(formatBytes(1_500_000)).toBe('1.4 MB')
  })
})

describe('formatDuration', () => {
  it('renders zero as 00:00', () => {
    expect(formatDuration(0)).toBe('00:00')
    expect(formatDuration(-100)).toBe('00:00')
  })

  it('renders minutes and seconds', () => {
    expect(formatDuration(65_000)).toBe('01:05')
  })

  it('rolls hours into the minutes column', () => {
    expect(formatDuration(3_725_000)).toBe('62:05')
  })
})

describe('deriveIngestStatus', () => {
  it('starts idle with empty event list', () => {
    expect(deriveIngestStatus([])).toEqual({
      phase: 'idle',
      totalFiles: 0,
      filesSaved: 0,
      tasks: [],
      indexed: 0,
      totalChunks: 0
    })
  })

  it('transitions to uploading on start and records totals + collection', () => {
    const events: IngestEvent[] = [
      {
        event: 'start',
        data: { collection: 'testdata-1', target_dir: '/tmp', files: ['a.pdf', 'b.pdf'] }
      }
    ]
    const status = deriveIngestStatus(events)
    expect(status.phase).toBe('uploading')
    expect(status.totalFiles).toBe(2)
    expect(status.collection).toBe('testdata-1')
    expect(typeof status.startedAt).toBe('number')
  })

  it('tracks upload progress and file_saved increments', () => {
    const sizes = { 'a.pdf': 2048, 'b.pdf': 4096 }
    const events: IngestEvent[] = [
      {
        event: 'start',
        data: { collection: 'c', target_dir: '/t', files: ['a.pdf', 'b.pdf'] }
      },
      { event: 'upload_progress', data: { filename: 'a.pdf', bytes_written: 1024 } },
      { event: 'file_saved', data: { filename: 'a.pdf', file_hash: 'h1', path: '/x' } },
      { event: 'upload_progress', data: { filename: 'b.pdf', bytes_written: 2048 } }
    ]
    const status = deriveIngestStatus(events, sizes)
    expect(status.filesSaved).toBe(1)
    expect(status.uploadingFile).toBe('b.pdf')
    expect(status.uploadingBytes).toBe(2048)
    expect(status.uploadingTotalBytes).toBe(4096)
  })

  it('switches to processing on ingestion_started and clears upload state', () => {
    const events: IngestEvent[] = [
      { event: 'start', data: { collection: 'c', target_dir: '/t', files: ['a.pdf'] } },
      { event: 'upload_progress', data: { filename: 'a.pdf', bytes_written: 1024 } },
      { event: 'ingestion_started', data: { collection: 'c' } }
    ]
    const status = deriveIngestStatus(events)
    expect(status.phase).toBe('processing')
    expect(status.uploadingFile).toBeUndefined()
    expect(status.uploadingBytes).toBeUndefined()
  })

  it('captures stage info from "processing PDF" progress', () => {
    const events: IngestEvent[] = [
      { event: 'start', data: { collection: 'c', target_dir: '/t', files: [] } },
      { event: 'ingestion_started', data: { collection: 'c' } },
      progress('Core pipeline processing PDF (3/5): foo.pdf')
    ]
    const status = deriveIngestStatus(events)
    expect(status.stage).toEqual({
      label: 'Processing PDFs',
      current: 3,
      total: 5,
      currentItem: 'foo.pdf'
    })
  })

  it('counts indexed events and bumps the stage current', () => {
    const events: IngestEvent[] = [
      { event: 'start', data: { collection: 'c', target_dir: '/t', files: [] } },
      { event: 'ingestion_started', data: { collection: 'c' } },
      progress('Core pipeline processing PDF (1/2): foo.pdf'),
      progress('Core pipeline indexed 42 chunks: foo.pdf')
    ]
    const status = deriveIngestStatus(events)
    expect(status.indexed).toBe(1)
    expect(status.totalChunks).toBe(42)
    expect(status.stage?.current).toBe(2)
  })

  it('records a single entity task entry across multiple progress events', () => {
    const events: IngestEvent[] = [
      { event: 'start', data: { collection: 'c', target_dir: '/t', files: [] } },
      { event: 'ingestion_started', data: { collection: 'c' } },
      progress('Extracting entities: 100/703 chunks processed')
    ]
    let status = deriveIngestStatus(events)
    expect(status.tasks).toHaveLength(1)
    expect(status.tasks[0]).toEqual({
      key: 'entities',
      label: 'Entities',
      current: 100,
      total: 703
    })

    const moreEvents: IngestEvent[] = [
      ...events,
      progress('Extracting entities: 200/703 chunks processed')
    ]
    status = deriveIngestStatus(moreEvents)
    expect(status.tasks).toHaveLength(1)
    expect(status.tasks[0].current).toBe(200)
  })

  it('tracks entities and hate detection concurrently in stable order', () => {
    const events: IngestEvent[] = [
      { event: 'start', data: { collection: 'c', target_dir: '/t', files: [] } },
      { event: 'ingestion_started', data: { collection: 'c' } },
      progress('Extracting entities: 200/703 chunks processed'),
      progress('Detecting hate speech: 150/703 chunks processed'),
      progress('Extracting entities: 450/703 chunks processed')
    ]
    const status = deriveIngestStatus(events)
    expect(status.tasks).toHaveLength(2)
    expect(status.tasks[0].key).toBe('entities')
    expect(status.tasks[0].current).toBe(450)
    expect(status.tasks[1].key).toBe('hate')
    expect(status.tasks[1].current).toBe(150)
  })

  it('ignores unknown progress messages without throwing', () => {
    const events: IngestEvent[] = [
      { event: 'start', data: { collection: 'c', target_dir: '/t', files: [] } },
      { event: 'ingestion_started', data: { collection: 'c' } },
      progress('Mystery line from a future backend version')
    ]
    expect(() => deriveIngestStatus(events)).not.toThrow()
    const status = deriveIngestStatus(events)
    expect(status.tasks).toEqual([])
    expect(status.stage).toBeUndefined()
  })

  it('marks ingestion_complete with a finishedAt timestamp', () => {
    const events: IngestEvent[] = [
      { event: 'start', data: { collection: 'c', target_dir: '/t', files: [] } },
      { event: 'ingestion_started', data: { collection: 'c' } },
      { event: 'ingestion_complete', data: { collection: 'c', data_dir: '/d' } }
    ]
    const status = deriveIngestStatus(events)
    expect(status.phase).toBe('complete')
    expect(typeof status.finishedAt).toBe('number')
  })

  it('records error events with phase=error and the message', () => {
    const events: IngestEvent[] = [
      { event: 'start', data: { collection: 'c', target_dir: '/t', files: [] } },
      { event: 'error', data: { message: 'boom' } }
    ]
    const status = deriveIngestStatus(events)
    expect(status.phase).toBe('error')
    expect(status.errorMessage).toBe('boom')
  })
})
