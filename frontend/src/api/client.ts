/** API base: explicit VITE_API_BASE_URL wins; otherwise derive from
 *  Vite's base path so the SPA works under /docint/ and at root alike. */
export function apiBase(
  override: string | undefined = import.meta.env.VITE_API_BASE_URL,
  base: string = import.meta.env.BASE_URL,
): string {
  const raw = override ?? (base === '/' ? '' : base)
  return raw.replace(/\/+$/, '')
}

const BASE = apiBase()

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(`API ${status}`)
    this.name = 'ApiError'
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    // Read once, then parse: a body can only be consumed once, so
    // json()-then-text() throws on a non-JSON error page and loses the status.
    const body = await res.text()
    let detail: unknown = body
    try {
      detail = JSON.parse(body)
    } catch {
      // Not JSON — the raw text is the detail.
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export function url(path: string) {
  return `${BASE}${path}`
}

// Admin owner context: when an admin has a foreign collection selected, every
// request carries `owner=<that user>` so the backend resolves in their
// namespace (Principal.requested_owner). Null for non-admins/own collections.
let ownerParam: string | null = null

export function setOwnerParam(owner: string | null) {
  ownerParam = owner
}

export function getOwnerParam(): string | null {
  return ownerParam
}

export function withOwner(pathAndQuery: string): string {
  if (!ownerParam) return pathAndQuery
  const sep = pathAndQuery.includes('?') ? '&' : '?'
  return `${pathAndQuery}${sep}owner=${encodeURIComponent(ownerParam)}`
}

type QueryParams = Record<string, string | number | boolean | undefined>

function queryString(params?: QueryParams): string {
  if (!params) return ''
  const qs = Object.entries(params)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join('&')
  return qs ? `?${qs}` : ''
}

export async function apiGet<T>(path: string, params?: QueryParams): Promise<T> {
  return handle<T>(await fetch(url(withOwner(path + queryString(params)))))
}

/**
 * GET for an endpoint that answers 204 when the thing asked for simply does
 * not exist yet — distinct from 404, which stays an `ApiError`.
 *
 * `handle<T>` is deliberately left alone: teaching it about 204 would widen
 * every existing caller's return type to `T | null` for the sake of the one
 * endpoint that needs it (`GET /summarize`, where 204 means "nothing cached",
 * not "no such collection").
 */
export async function apiGetOrNull<T>(path: string, params?: QueryParams): Promise<T | null> {
  const res = await fetch(url(withOwner(path + queryString(params))))
  if (res.status === 204) return null
  return handle<T>(res)
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(
    await fetch(url(withOwner(path)), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
  )
}

export async function apiPut<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(
    await fetch(url(withOwner(path)), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
  )
}

export async function apiPatch<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(
    await fetch(url(withOwner(path)), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
  )
}

export async function apiDelete<T>(path: string): Promise<T> {
  return handle<T>(await fetch(url(withOwner(path)), { method: 'DELETE' }))
}
