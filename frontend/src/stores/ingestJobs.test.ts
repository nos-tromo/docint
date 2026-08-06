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
