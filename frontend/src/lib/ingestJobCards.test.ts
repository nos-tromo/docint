import { describe, expect, it } from 'vitest'
import { isFinishedCard, mergeJobCards } from './ingestJobCards'
import type { IngestJobSnapshot } from '@/api/types'

function snapshot(id: string, over: Partial<IngestJobSnapshot> = {}): IngestJobSnapshot {
  return {
    job_id: id,
    collection: `col-${id}`,
    status: 'running',
    message: null,
    error: null,
    empty: false,
    resolution: null,
    created_at: '',
    run_started_at: '',
    started_at: null,
    finished_at: null,
    ...over
  } as IngestJobSnapshot
}

describe('mergeJobCards', () => {
  it('lists a locally tracked job the server has not listed yet, first', () => {
    const cards = mergeJobCards(
      [{ job_id: 'new', collection: 'fresh' }],
      [snapshot('old')]
    )
    expect(cards.map((c) => c.jobId)).toEqual(['new', 'old'])
    expect(cards[0].listItem).toBeUndefined()
  })

  it('does not duplicate a job present in both sources', () => {
    const cards = mergeJobCards(
      [{ job_id: 'job-1', collection: 'mydocs' }],
      [snapshot('job-1')]
    )
    expect(cards).toHaveLength(1)
    expect(cards[0].listItem).toBeDefined()
  })

  it('keeps the server order for listed jobs', () => {
    const cards = mergeJobCards([], [snapshot('c'), snapshot('b'), snapshot('a')])
    expect(cards.map((c) => c.jobId)).toEqual(['c', 'b', 'a'])
  })

  it('falls back to the tracked name when a snapshot carries none', () => {
    const cards = mergeJobCards(
      [{ job_id: 'job-1', collection: 'mydocs' }],
      [snapshot('job-1', { collection: '' })]
    )
    expect(cards[0].collection).toBe('mydocs')
  })
})

describe('isFinishedCard', () => {
  it('reads a terminal frame', () => {
    expect(isFinishedCard({ jobId: 'job-1', collection: 'x' }, { 'job-1': true })).toBe(true)
  })

  it('reads a terminal snapshot status when no frame arrived', () => {
    const entry = { jobId: 'job-1', collection: 'x', listItem: snapshot('job-1', { status: 'completed' }) }
    expect(isFinishedCard(entry, {})).toBe(true)
  })

  it('leaves a running job alone', () => {
    const entry = { jobId: 'job-1', collection: 'x', listItem: snapshot('job-1') }
    expect(isFinishedCard(entry, {})).toBe(false)
  })
})
