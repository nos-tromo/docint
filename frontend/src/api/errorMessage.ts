import { ApiError } from './client'

export type ErrorKey = 'common.error_request' | 'common.error_unknown' | 'common.error_network'

export type ErrorDescriptor = {
  key: ErrorKey
  vars?: Record<string, string | number>
}

/** The only sanctioned path from a thrown error to user-visible text.
 *  Never render err.message, err.detail, or response bodies. */
export function describeError(err: unknown): ErrorDescriptor {
  if (err instanceof ApiError) {
    // Dev-only visibility; the body is generic post-backend-fix anyway.
    console.debug('API error detail', err.status, err.detail)
    return { key: 'common.error_request', vars: { status: err.status } }
  }
  if (err instanceof TypeError) {
    // fetch() network-level failure (server unreachable, DNS, CORS)
    return { key: 'common.error_network' }
  }
  return { key: 'common.error_unknown' }
}

import type { Strings } from '@/i18n'

/** Specific catalog copy for stream error codes that deserve their own
 *  message; every other valid code renders its fallback key's copy. */
const STREAM_ERROR_KEYS: Partial<Record<string, keyof Strings>> = {
  context_overflow: 'chat.error_context_overflow',
}

/** SSE error codes are a closed backend enum (see docint/core/errors.py);
 *  only strings matching this shape may ever reach the screen. */
const CODE_TOKEN = /^[a-z][a-z0-9_]{0,39}$/

/** Render a stream-error message from a backend `code` field.
 *
 *  The code is protocol, not prose: it is regex-validated before display and
 *  shown as a bare token — the triage counterpart of the HTTP status on
 *  request errors. Anything that is not a valid token renders the fallback
 *  catalog copy alone. */
export function streamErrorText(
  t: (key: keyof Strings, vars?: Record<string, string | number>) => string,
  code: unknown,
  fallbackKey: keyof Strings,
  vars?: Record<string, string | number>,
): string {
  const token = typeof code === 'string' && CODE_TOKEN.test(code) ? code : null
  const key = (token && STREAM_ERROR_KEYS[token]) || fallbackKey
  const base = t(key, vars)
  return token ? `${base} (${token})` : base
}
