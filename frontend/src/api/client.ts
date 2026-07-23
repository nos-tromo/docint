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
    super(`API error ${status}: ${JSON.stringify(detail)}`)
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: unknown
    try {
      detail = await res.json()
    } catch {
      detail = await res.text()
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export function url(path: string) {
  return `${BASE}${path}`
}

export async function apiGet<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const qs = params
    ? '?' +
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join('&')
    : ''
  return handle<T>(await fetch(url(path) + qs))
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(
    await fetch(url(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
  )
}

export async function apiPatch<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(
    await fetch(url(path), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
  )
}

export async function apiDelete<T>(path: string): Promise<T> {
  return handle<T>(await fetch(url(path), { method: 'DELETE' }))
}
