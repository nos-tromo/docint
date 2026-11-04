import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PurposeBar, clearAllPurposes, setPurpose } from '@infra/ui'
import { ApiError, PURPOSE_SCOPE, apiDelete, apiGet, apiPatch, apiPost, apiPut, setOwnerParam } from './client'
import { streamSse, type SseEvent } from './sse'
import { UPLOAD_STALL_MS, streamUpload } from './upload'
import { FakeXhr, installFakeXhr } from '@/test/fakeXhr'

const HEADER = 'X-Audit-Purpose'

/** What the wrapper hands `fetch`: the resolved URL and its options. */
type FetchInit = { headers?: Record<string, string>; body?: unknown }
type FetchFn = (input: unknown, init?: FetchInit) => Promise<unknown>
type FetchMock = ReturnType<typeof vi.fn<FetchFn>>

/** Stub `fetch`: `answer` for every call, or the given answers once each, in order. */
function stubFetch(answer: (() => unknown) | null, ...once: unknown[]): FetchMock {
  const fn = answer ? vi.fn<FetchFn>(async () => answer()) : vi.fn<FetchFn>()
  for (const value of once) fn.mockResolvedValueOnce(value)
  vi.stubGlobal('fetch', fn)
  return fn
}

function respond(status: number, body: unknown = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

function sseBody(frames: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(c) {
      c.enqueue(enc.encode(frames))
      c.close()
    }
  })
}

/** The purpose header the nth `fetch` call carried. */
function sentPurpose(fetchMock: FetchMock, n = 0): string | undefined {
  return fetchMock.mock.calls[n][1]?.headers?.[HEADER]
}

async function drain(gen: AsyncGenerator<SseEvent, void, unknown>): Promise<SseEvent[]> {
  const out: SseEvent[] = []
  for await (const ev of gen) out.push(ev)
  return out
}

afterEach(() => {
  vi.useRealTimers()
  clearAllPurposes()
  sessionStorage.clear()
  setOwnerParam(null)
  vi.unstubAllGlobals()
})

describe('the stated purpose', () => {
  it('rides along on every request once stated', async () => {
    const fetchMock = stubFetch(() => respond(200, { ok: true }))
    setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001', reason: 'Review of the flagged postings' })

    await apiGet('/collections/documents', { collection: 'alpha' })
    await apiPost('/search', { question: 'Alice Example' })
    await apiPut('/sessions/s1/scope', { chunk_ids: [] })
    await apiPatch('/reports/1?collection=alpha', { title: 'Case' })
    await apiDelete('/sessions/s1')

    expect(fetchMock).toHaveBeenCalledTimes(5)
    for (let n = 0; n < 5; n++) {
      expect(sentPurpose(fetchMock, n)).toBe('case_ref=case-0001&reason=Review+of+the+flagged+postings')
    }
    // Merged into the request's own headers, never in their place.
    expect(fetchMock.mock.calls[1][1]?.headers?.['Content-Type']).toBe('application/json')
  })

  it('is absent while none is stated', async () => {
    const fetchMock = stubFetch(() => respond(200, { ok: true }))

    await apiGet('/collections/documents', { collection: 'alpha' })
    await apiPost('/search', { question: 'Alice Example' })

    expect(sentPurpose(fetchMock, 0)).toBeUndefined()
    expect(sentPurpose(fetchMock, 1)).toBeUndefined()
  })

  it('is never borrowed from another app sharing the origin', async () => {
    // Behind the gateway every app shares one origin, and so one sessionStorage.
    const fetchMock = stubFetch(() => respond(200, { ok: true }))
    setPurpose('chorus', { caseRef: 'case-0002' })

    await apiGet('/sessions/list', { collection: 'alpha' })

    expect(sentPurpose(fetchMock)).toBeUndefined()
  })

  it('streams a chat turn with the purpose', async () => {
    const fetchMock = stubFetch(() => ({ ok: true, status: 200, body: sseBody('event: done\ndata: {}\n\n') }))
    setPurpose(PURPOSE_SCOPE, { reason: 'Review' })

    const events = await drain(streamSse('/stream_query', { question: 'Who wrote it?' }))

    expect(events).toEqual([{ event: 'done', data: {} }])
    expect(sentPurpose(fetchMock)).toBe('reason=Review')
  })

  it('sets the purpose on an upload as a request header', async () => {
    installFakeXhr()
    setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })

    const done = drain(streamUpload('/ingest/upload', new FormData()))
    FakeXhr.instances[0].finish()
    await done

    expect(FakeXhr.instances[0].headers[HEADER]).toBe('case_ref=case-0001')
  })

  it('leaves an upload without the header while none is stated', async () => {
    installFakeXhr()

    const done = drain(streamUpload('/ingest/upload', new FormData()))
    FakeXhr.instances[0].finish()
    await done

    expect(FakeXhr.instances[0].headers).toEqual({})
  })
})

describe('a 428 for want of a purpose', () => {
  it('opens the purpose dialog and sends the request once more with the purpose given', async () => {
    const fetchMock = stubFetch(
      null,
      respond(428, { detail: 'A stated purpose is required for this request.' }),
      respond(200, { documents: [] })
    )
    const user = userEvent.setup()
    render(<PurposeBar scope={PURPOSE_SCOPE} required />)

    const pending = apiGet('/collections/documents', { collection: 'alpha' })
    await user.type(await screen.findByLabelText('Case reference'), 'case-0001{Enter}')

    await expect(pending).resolves.toEqual({ documents: [] })
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(sentPurpose(fetchMock, 0)).toBeUndefined()
    expect(sentPurpose(fetchMock, 1)).toBe('case_ref=case-0001')
    expect(String(fetchMock.mock.calls[1][0])).toBe(String(fetchMock.mock.calls[0][0]))
  })

  it('hands the refusal back, sending nothing more, when the dialog is cancelled', async () => {
    const fetchMock = stubFetch(() => respond(428, { detail: 'A stated purpose is required for this request.' }))
    const user = userEvent.setup()
    render(<PurposeBar scope={PURPOSE_SCOPE} required />)

    const pending = apiGet('/collections/documents', { collection: 'alpha' }).catch((e: unknown) => e)
    await screen.findByLabelText('Case reference')
    await user.keyboard('{Escape}')

    const err = await pending
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(428)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('sends again only once', async () => {
    const fetchMock = stubFetch(() => respond(428, { detail: 'A stated purpose is required for this request.' }))
    const user = userEvent.setup()
    render(<PurposeBar scope={PURPOSE_SCOPE} required />)

    const pending = apiPost('/search', { question: 'Alice Example' }).catch((e: unknown) => e)
    await user.type(await screen.findByLabelText('Reason'), 'Review{Enter}')

    expect(((await pending) as ApiError).status).toBe(428)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    // The body is built once and sent twice.
    expect(fetchMock.mock.calls[1][1]?.body).toBe(fetchMock.mock.calls[0][1]?.body)
  })

  it('resends to the owner the request was made for', async () => {
    const fetchMock = stubFetch(null, respond(428), respond(200, { ok: true }))
    const user = userEvent.setup()
    render(<PurposeBar scope={PURPOSE_SCOPE} />)
    setOwnerParam('jane.doe')

    const pending = apiGet('/collections/documents', { collection: 'alpha' })
    // An admin switching to someone else's collection while the dialog is open.
    setOwnerParam('bob')
    await user.type(await screen.findByLabelText('Case reference'), 'case-0001{Enter}')
    await pending

    expect(String(fetchMock.mock.calls[1][0])).toContain('owner=jane.doe')
    expect(String(fetchMock.mock.calls[1][0])).not.toContain('owner=bob')
  })

  it('fails as it is when no purpose bar is mounted to ask with', async () => {
    const fetchMock = stubFetch(() => respond(428))

    const err = await apiGet('/sessions/list', { collection: 'alpha' }).catch((e: unknown) => e)

    expect((err as ApiError).status).toBe(428)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('resends a chat stream with the purpose given', async () => {
    const fetchMock = stubFetch(null, respond(428), {
      ok: true,
      status: 200,
      body: sseBody('event: done\ndata: {"ok":true}\n\n')
    })
    const user = userEvent.setup()
    render(<PurposeBar scope={PURPOSE_SCOPE} />)

    const events = drain(streamSse('/stream_query', { question: 'Who wrote it?' }))
    await user.type(await screen.findByLabelText('Reason'), 'Review{Enter}')

    expect(await events).toEqual([{ event: 'done', data: { ok: true } }])
    expect(sentPurpose(fetchMock, 1)).toBe('reason=Review')
    expect(fetchMock.mock.calls[1][1]?.body).toBe(JSON.stringify({ question: 'Who wrote it?' }))
  })

  it('resends an upload with the same body and the purpose given', async () => {
    installFakeXhr()
    const user = userEvent.setup()
    render(<PurposeBar scope={PURPOSE_SCOPE} />)
    const body = new FormData()
    body.append('collection', 'alpha')

    const events = drain(streamUpload('/ingest/upload', body))
    const refused = FakeXhr.instances[0]
    refused.finish(428)
    await user.type(await screen.findByLabelText('Reason'), 'Review{Enter}')
    await waitFor(() => expect(FakeXhr.instances).toHaveLength(2))
    const resent = FakeXhr.instances[1]
    resent.receive('event: upload_complete\ndata: {"files_saved":1}\n\n')
    resent.finish()

    expect(await events).toEqual([{ event: 'upload_complete', data: { files_saved: 1 } }])
    expect(refused.headers[HEADER]).toBeUndefined()
    expect(resent.headers[HEADER]).toBe('reason=Review')
    expect(resent.body).toBe(body)
    expect(resent.url).toBe(refused.url)
  })

  it('counts the upload stall timeout per attempt, not across the time spent in the dialog', async () => {
    // Fake time that still moves on its own, so the dialog's own waits resolve.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    installFakeXhr()
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    render(<PurposeBar scope={PURPOSE_SCOPE} />)

    const events = drain(streamUpload('/ingest/upload', new FormData()))
    FakeXhr.instances[0].finish(428)
    const reason = await screen.findByLabelText('Reason')
    // The user takes longer to answer than one attempt may stall.
    await vi.advanceTimersByTimeAsync(UPLOAD_STALL_MS + 1_000)
    await user.type(reason, 'Review{Enter}')
    await waitFor(() => expect(FakeXhr.instances).toHaveLength(2))
    const resent = FakeXhr.instances[1]

    // The resend has a budget of its own: nothing it inherits has run out.
    expect(resent.aborted).toBe(false)
    resent.receive('event: upload_complete\ndata: {"files_saved":1}\n\n')
    resent.finish()
    expect(await events).toEqual([{ event: 'upload_complete', data: { files_saved: 1 } }])
  })
})

