import { PURPOSE_REQUIRED_STATUS, withPurpose } from '@infra/ui'

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

/**
 * The scope this app's stated purpose is kept under: the compose project
 * name, which is also the `service` the backend's audit events carry. The
 * header's purpose bar and every request below share it.
 */
export const PURPOSE_SCOPE = 'docint'

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(`API ${status}`)
    this.name = 'ApiError'
  }
}

/**
 * Whether an error is the backend refusing a request for want of a stated
 * purpose — what is left of a 428 once the user declined to state one.
 */
export function isPurposeRefusal(error: unknown): boolean {
  return error instanceof ApiError && error.status === PURPOSE_REQUIRED_STATUS
}

/**
 * The `ApiError` a non-2xx response stands for.
 *
 * Reads the body once, then parses: a body can only be consumed once, so
 * json()-then-text() throws on a non-JSON error page and loses the status.
 */
export async function apiErrorOf(res: Response): Promise<ApiError> {
  const body = await res.text()
  let detail: unknown = body
  try {
    detail = JSON.parse(body)
  } catch {
    // Not JSON — the raw text is the detail.
  }
  return new ApiError(res.status, detail)
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) throw await apiErrorOf(res)
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

/**
 * Add the admin owner context to a request path.
 *
 * @param pathAndQuery - The path, with any query string.
 * @param owner - The owner to send; defaults to the one selected now. A run
 *   spanning several requests passes the owner it started under, so a switch
 *   to another user's collection mid-run cannot move its later requests.
 * @returns The path, carrying `owner=` when there is one.
 */
export function withOwner(pathAndQuery: string, owner: string | null = ownerParam): string {
  if (!owner) return pathAndQuery
  const sep = pathAndQuery.includes('?') ? '&' : '?'
  return `${pathAndQuery}${sep}owner=${encodeURIComponent(owner)}`
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

/** Request options for {@link apiFetch}: `headers` as a plain object, so the purpose can be merged in. */
export type ApiRequestInit = Omit<RequestInit, 'headers'> & { headers?: Record<string, string> }

/**
 * Send one request to the API. Every request the SPA makes leaves through
 * here, the streaming and upload helpers included.
 *
 * The path is resolved against the sub-path base and the admin owner context
 * once, before anything is sent, so a resend goes where the first send went.
 * It is then sent through `withPurpose`: the purpose stated in the header's
 * purpose bar rides along as `X-Audit-Purpose`, and a `428` opens the bar's
 * dialog and sends the request once more with the purpose given there. The
 * body may therefore go twice, so it must be re-sendable — a string or
 * `FormData`, never a stream.
 *
 * @param path - API path, with any query string.
 * @param init - The `fetch` options.
 * @param owner - Owner to send instead of the one selected now (see `withOwner`).
 * @returns The response; the `428` itself when the user declined to state a purpose.
 */
export function apiFetch(path: string, init: ApiRequestInit = {}, owner?: string | null): Promise<Response> {
  const target = url(withOwner(path, owner))
  return withPurpose(PURPOSE_SCOPE, (purpose) =>
    fetch(target, { ...init, headers: { ...init.headers, ...purpose } })
  )
}

function jsonRequest(method: string, body: unknown): ApiRequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body)
  }
}

export async function apiGet<T>(path: string, params?: QueryParams, owner?: string | null): Promise<T> {
  return handle<T>(await apiFetch(path + queryString(params), {}, owner))
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
  const res = await apiFetch(path + queryString(params))
  if (res.status === 204) return null
  return handle<T>(res)
}

export async function apiPost<T>(path: string, body?: unknown, owner?: string | null): Promise<T> {
  return handle<T>(await apiFetch(path, jsonRequest('POST', body), owner))
}

export async function apiPut<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(await apiFetch(path, jsonRequest('PUT', body)))
}

export async function apiPatch<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(await apiFetch(path, jsonRequest('PATCH', body)))
}

export async function apiDelete<T>(path: string, owner?: string | null): Promise<T> {
  return handle<T>(await apiFetch(path, { method: 'DELETE' }, owner))
}
