import { url } from './client'
import type { SseEvent } from './sse'

/**
 * Error thrown when an upload POST returns a non-OK HTTP status. Carries the
 * numeric `status` so callers can branch on it — notably 413 (Request Entity
 * Too Large, from nginx's `client_max_body_size`) which needs a size-specific
 * message rather than the generic "backend may have crashed" fallback.
 */
export class UploadHttpError extends Error {
  readonly status: number

  constructor(status: number) {
    super(`Upload failed: ${status}`)
    this.name = 'UploadHttpError'
    this.status = status
  }
}

export async function* streamUpload(
  path: string,
  formData: FormData,
  signal?: AbortSignal
): AsyncGenerator<SseEvent, void, unknown> {
  const res = await fetch(url(path), {
    method: 'POST',
    body: formData,
    signal
  })
  if (!res.ok || !res.body) {
    throw new UploadHttpError(res.status)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = 'message'
      const dataLines: string[] = []
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (dataLines.length === 0) continue
      const raw = dataLines.join('\n')
      try {
        yield { event, data: JSON.parse(raw) }
      } catch {
        yield { event, data: raw }
      }
    }
  }
}
