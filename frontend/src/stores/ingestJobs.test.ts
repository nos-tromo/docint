import { beforeEach, describe, expect, it } from 'vitest'
import { useIngestJobsStore, selectHasRunningJob } from './ingestJobs'
import type { IngestEvent } from '@/api/types'

const ev = (message: string): IngestEvent => ({
  event: 'ingestion_progress',
  data: { message },
  receivedAt: 0
})

beforeEach(() => useIngestJobsStore.getState().clear())

describe('useIngestJobsStore', () => {
  it('collapses consecutive same-kind progress events', () => {
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', ev('Extracting entities: 1/9 chunks processed'))
    appendEvent('job-1', ev('Extracting entities: 2/9 chunks processed'))
    appendEvent('job-1', ev('Extracting entities: 3/9 chunks processed'))

    const events = useIngestJobsStore.getState().events['job-1']
    expect(events).toHaveLength(1)
    expect((events[0].data as { message: string }).message).toContain('3/9')
  })

  it('keeps distinct progress kinds as separate entries', () => {
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', ev('Extracting entities: 1/9 chunks processed'))
    appendEvent('job-1', ev('Detecting hate speech: 1/9 chunks processed'))

    expect(useIngestJobsStore.getState().events['job-1']).toHaveLength(2)
  })

  it('collapses interleaved entities and hate frames to one entry each', () => {
    // The backend enriches both stages from one pool, so their counter frames
    // arrive interleaved rather than as two adjacent runs.
    const { appendEvent } = useIngestJobsStore.getState()
    for (let i = 1; i <= 9; i += 1) {
      appendEvent('job-1', ev(`Extracting entities: ${i}/9 chunks processed`))
      appendEvent('job-1', ev(`Detecting hate speech: ${i}/9 chunks processed`))
    }

    const events = useIngestJobsStore.getState().events['job-1']
    expect(events).toHaveLength(2)
    expect((events[0].data as { message: string }).message).toBe(
      'Extracting entities: 9/9 chunks processed'
    )
    expect((events[1].data as { message: string }).message).toBe(
      'Detecting hate speech: 9/9 chunks processed'
    )
  })

  it('does not merge per-file frames whose names differ only in digits', () => {
    // Digit masking gives "indexed 12 chunks: report_v1.pdf" and
    // "indexed 30 chunks: report_v2.pdf" the same kind, but they are distinct
    // files: only the enrichment counters may collapse across the trailing
    // run — everything else keeps adjacent-only collapsing.
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', ev('Core pipeline indexed 12 chunks: report_v1.pdf'))
    appendEvent('job-1', ev('Core pipeline processing PDF (2/2): report_v2.pdf'))
    appendEvent('job-1', ev('Core pipeline indexed 30 chunks: report_v2.pdf'))

    const events = useIngestJobsStore.getState().events['job-1']
    expect(events).toHaveLength(3)
    expect((events[0].data as { message: string }).message).toContain('report_v1.pdf')
  })

  it('does not collapse progress frames across a non-progress entry', () => {
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', ev('Extracting entities: 1/9 chunks processed'))
    appendEvent('job-1', { event: 'warning', data: { message: 'heads up' }, receivedAt: 0 })
    appendEvent('job-1', ev('Extracting entities: 2/9 chunks processed'))

    expect(useIngestJobsStore.getState().events['job-1']).toHaveLength(3)
  })

  it('resets a job log on ingestion_started so replays do not duplicate', () => {
    const { appendEvent } = useIngestJobsStore.getState()
    const started: IngestEvent = {
      event: 'ingestion_started',
      data: { collection: 'mydocs' },
      receivedAt: 0
    }
    const warn: IngestEvent = { event: 'warning', data: { message: 'heads up' }, receivedAt: 0 }

    appendEvent('job-1', started)
    appendEvent('job-1', warn)
    // A reconnect replays the same history from the top.
    appendEvent('job-1', started)
    appendEvent('job-1', warn)

    const events = useIngestJobsStore.getState().events['job-1']
    expect(events.map((e) => e.event)).toEqual(['ingestion_started', 'warning'])
  })

  it('collapses repeated summary_progress frames into one entry', () => {
    // progressKind() only recognised `ingestion_progress`, so every per-unit
    // frame of a summary build was appended: a 3,000-unit collection meant
    // 3,000 entries, an O(n) selectHasRunningJob scan per append (~4.5M
    // iterations) and a sidebar re-render per frame.
    const { appendEvent } = useIngestJobsStore.getState()
    for (let i = 1; i <= 50; i += 1) {
      appendEvent('job-1', {
        event: 'summary_progress',
        data: { job_id: 'job-1', message: `Summarizing ${i}/3000`, mapped: i, total_units: 3000 },
        receivedAt: i
      })
    }

    const events = useIngestJobsStore.getState().events['job-1']
    expect(events).toHaveLength(1)
    expect((events[0].data as { mapped: number }).mapped).toBe(50)
  })

  it('resets a job log on summary_started so replays do not duplicate', () => {
    const { appendEvent } = useIngestJobsStore.getState()
    const started: IngestEvent = {
      event: 'summary_started',
      data: { job_id: 'job-1', total_units: 4 },
      receivedAt: 0
    }
    const warn: IngestEvent = { event: 'warning', data: { message: 'heads up' }, receivedAt: 0 }

    appendEvent('job-1', started)
    appendEvent('job-1', warn)
    // A mid-build SSE reconnect replays the same history from the top.
    appendEvent('job-1', started)
    appendEvent('job-1', warn)

    const events = useIngestJobsStore.getState().events['job-1']
    expect(events.map((e) => e.event)).toEqual(['summary_started', 'warning'])
  })

  it('keeps jobs isolated from each other', () => {
    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', ev('one'))
    appendEvent('job-2', ev('two'))

    expect(useIngestJobsStore.getState().events['job-1']).toHaveLength(1)
    expect(useIngestJobsStore.getState().events['job-2']).toHaveLength(1)
  })

  it('drops a single job without touching the others', () => {
    const { appendEvent, dropJob } = useIngestJobsStore.getState()
    appendEvent('job-1', ev('one'))
    appendEvent('job-2', ev('two'))
    dropJob('job-1')

    expect(useIngestJobsStore.getState().events['job-1']).toBeUndefined()
    expect(useIngestJobsStore.getState().events['job-2']).toHaveLength(1)
  })

  it('treats a summary job as terminated by summary_completed, not stuck running', () => {
    // The owner-multiplexed stream carries summary-job frames through this
    // same store (no kind filter in useIngestJobStream.ts). Its terminal
    // event is summary_completed, not ingestion_complete — before this fix,
    // selectHasRunningJob only recognized the latter, so a finished summary
    // job would report as running forever and leave the sidebar badge stuck
    // on.
    const { appendEvent } = useIngestJobsStore.getState()
    const started: IngestEvent = {
      event: 'summary_started',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: 0
    }

    appendEvent('job-1', started)
    // Only the started frame so far — must still count as running so the
    // fix isn't a blanket "always false".
    expect(selectHasRunningJob(useIngestJobsStore.getState())).toBe(true)

    appendEvent('job-1', {
      event: 'summary_completed',
      data: { job_id: 'job-1' },
      receivedAt: 1
    })
    expect(selectHasRunningJob(useIngestJobsStore.getState())).toBe(false)
  })
})

describe('extract jobs share the store', () => {
  beforeEach(() => useIngestJobsStore.getState().clear())

  it('treats extract_started as the start of a fresh fold', () => {
    const store = useIngestJobsStore.getState()
    store.appendEvent('j1', { event: 'extract_started', data: {} } as IngestEvent)
    store.appendEvent('j1', { event: 'warning', data: { message: 'a warning' } } as IngestEvent)
    store.appendEvent('j1', { event: 'extract_started', data: {} } as IngestEvent)
    expect(useIngestJobsStore.getState().events.j1).toHaveLength(1)
  })

  it('stops counting an extract job as running once it completes', () => {
    const store = useIngestJobsStore.getState()
    store.appendEvent('j1', { event: 'extract_started', data: {} } as IngestEvent)
    expect(selectHasRunningJob(useIngestJobsStore.getState())).toBe(true)
    store.appendEvent('j1', { event: 'extract_completed', data: {} } as IngestEvent)
    expect(selectHasRunningJob(useIngestJobsStore.getState())).toBe(false)
  })
})
