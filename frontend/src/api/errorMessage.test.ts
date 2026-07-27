import { describe, expect, it } from 'vitest'
import { ApiError } from './client'
import { describeError, streamErrorText } from './errorMessage'

describe('describeError', () => {
  it('maps ApiError to the generic request key with the status', () => {
    expect(describeError(new ApiError(502, 'upstream exploded: /srv/secret'))).toEqual({
      key: 'common.error_request',
      vars: { status: 502 },
    })
  })
  it('maps network TypeError to the network key', () => {
    expect(describeError(new TypeError('Failed to fetch'))).toEqual({
      key: 'common.error_network',
    })
  })
  it('maps anything else to the unknown key', () => {
    expect(describeError('weird')).toEqual({ key: 'common.error_unknown' })
    expect(describeError(new Error('boom'))).toEqual({ key: 'common.error_unknown' })
  })
  it('never exposes detail in the ApiError message', () => {
    const e = new ApiError(500, { detail: 'stacktrace /etc/passwd' })
    expect(e.message).toBe('API 500')
  })
})

describe('streamErrorText', () => {
  const t = ((key: string, vars?: Record<string, string | number>) =>
    vars ? `${key}|${JSON.stringify(vars)}` : key) as Parameters<typeof streamErrorText>[0]

  it('maps a known code to its specific key and appends the token', () => {
    expect(streamErrorText(t, 'context_overflow', 'chat.error_stream_ended')).toBe(
      'chat.error_context_overflow (context_overflow)',
    )
  })
  it('falls back to the given key for unmapped valid codes, appending the token', () => {
    expect(streamErrorText(t, 'summary_failed', 'analysis.summary_failed')).toBe(
      'analysis.summary_failed (summary_failed)',
    )
  })
  it('ignores non-token codes entirely (prose can never render)', () => {
    expect(streamErrorText(t, 'Bad Prose! /etc/passwd', 'ingest.failed_default')).toBe('ingest.failed_default')
    expect(streamErrorText(t, 42, 'ingest.failed_default')).toBe('ingest.failed_default')
  })
  it('renders only the fallback when no code is present', () => {
    expect(streamErrorText(t, undefined, 'ingest.failed_default')).toBe('ingest.failed_default')
  })
})

describe('streamErrorText vars pass-through', () => {
  const t = ((key: string, vars?: Record<string, string | number>) =>
    vars ? `${key}|${JSON.stringify(vars)}` : key) as Parameters<typeof streamErrorText>[0]

  it('forwards vars to the catalog template', () => {
    expect(streamErrorText(t, 'save_failed', 'ingest.save_failed_file', { filename: 'a.txt' })).toBe(
      'ingest.save_failed_file|{"filename":"a.txt"} (save_failed)',
    )
  })
})
