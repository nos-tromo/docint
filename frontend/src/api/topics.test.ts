import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, setOwnerParam } from './client'
import {
  TOPIC_FIELD,
  getTopicChunksPage,
  getTopicDocuments,
  getTopicEvidence,
  getTopics,
  startTopics,
  topicValue,
  type TopicsOverview
} from './topics'

const OVERVIEW: TopicsOverview = {
  run_id: '20260102-030405-a1b2c3',
  created_at: '2026-01-02T03:04:05+00:00',
  updated_at: '2026-01-02T03:04:05+00:00',
  embed_model: 'embed-small',
  embed_model_changed: false,
  coverage: { total: 120, assigned: 100, unassigned: 20, pending: 0 },
  topics: [
    {
      id: 3,
      label: 'Shipping schedules',
      description: 'Delivery windows and carrier timetables.',
      keywords: [['shipping', 0.42]],
      size: 60,
      share: 0.5,
      origin: 'build',
      added_at: '2026-01-02T03:04:05+00:00',
      first_seen: null,
      last_seen: null
    }
  ]
}

const mockFetch = (status: number, body: unknown) =>
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      status === 204
        ? new Response(null, { status })
        : new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
    )
  )

/** Every request the stub saw, as a URL and its init. */
const requests = () =>
  vi.mocked(fetch).mock.calls.map(([input, init]) => ({ url: new URL(String(input), 'http://localhost'), init }))

beforeEach(() => setOwnerParam(null))
afterEach(() => {
  vi.unstubAllGlobals()
  setOwnerParam(null)
})

describe('topics API', () => {
  it('encodes the collection in every topics path', async () => {
    mockFetch(200, { job_id: 'j1' })
    const collection = 'q3 notes/v2?'

    await getTopics(collection)
    await startTopics(collection, 'build')
    await getTopicEvidence(collection, 7)
    await getTopicDocuments(collection, 7)
    await getTopicChunksPage({ collection, topicId: 7, cursor: 'c1', limit: 25 })

    const encoded = '/collections/q3%20notes%2Fv2%3F/topics'
    expect(requests().map(({ url }) => url.pathname.replace(/^.*(?=\/collections\/)/, ''))).toEqual([
      encoded,
      encoded,
      `${encoded}/7/evidence`,
      `${encoded}/7/documents`,
      `${encoded}/7/chunks`
    ])
  })

  it('returns null on 204', async () => {
    mockFetch(204, null)
    await expect(getTopics('docs')).resolves.toBeNull()
  })

  it('returns the overview of a collection that has a run', async () => {
    mockFetch(200, OVERVIEW)
    await expect(getTopics('docs')).resolves.toEqual(OVERVIEW)
  })

  it('carries the admin owner context on a read', async () => {
    mockFetch(200, OVERVIEW)
    setOwnerParam('other-operator')
    await getTopics('docs')
    expect(requests()[0].url.searchParams.get('owner')).toBe('other-operator')
  })

  it('posts the action it was asked for', async () => {
    mockFetch(202, { job_id: 'j1' })
    await expect(startTopics('docs', 'discover')).resolves.toEqual({ job_id: 'j1', adopted: false })
    const [{ init }] = requests()
    expect(init?.method).toBe('POST')
    expect(JSON.parse(String(init?.body))).toEqual({ action: 'discover' })
  })

  it('adopts an in-flight topics job on 409', async () => {
    mockFetch(409, { detail: { message: 'A topics run is already in progress.', job_id: 'j9' } })
    await expect(startTopics('docs', 'build')).resolves.toEqual({ job_id: 'j9', adopted: true })
  })

  it('rethrows a refusal that names no job, status intact', async () => {
    mockFetch(400, { detail: 'Build topics first.' })
    const failure = await startTopics('docs', 'align').catch((e: unknown) => e)
    expect(failure).toBeInstanceOf(ApiError)
    expect((failure as ApiError).status).toBe(400)
  })

  it('pages a topic’s chunks by cursor, fifty at a time unless asked otherwise', async () => {
    mockFetch(200, { items: [], next_cursor: null })

    await getTopicChunksPage({ collection: 'docs', topicId: 3, cursor: null })
    await getTopicChunksPage({ collection: 'docs', topicId: 3, cursor: 'c2', limit: 200 })

    const [first, second] = requests().map(({ url }) => url.searchParams)
    expect(first.get('cursor')).toBeNull()
    expect(first.get('limit')).toBe('50')
    expect(second.get('cursor')).toBe('c2')
    expect(second.get('limit')).toBe('200')
  })
})

describe('topicValue', () => {
  it('names a topic the way the backend stores it on a chunk', () => {
    expect(TOPIC_FIELD).toBe('docint_topic')
    expect(topicValue('20260102-030405-a1b2c3', 12)).toBe('20260102-030405-a1b2c3:12')
    expect(topicValue('20260102-030405-a1b2c3', -1)).toBe('20260102-030405-a1b2c3:-1')
  })
})
