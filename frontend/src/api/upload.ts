import { url, withOwner } from './client'
import type { SseEvent } from './sse'

/**
 * How long an upload may go without sending a byte or receiving a frame
 * before it counts as stalled. Nothing else bounds the wait: a request the
 * browser never finishes sending reaches no server log and trips no proxy
 * timeout the page can see, so without this the run waits on it forever.
 */
export const UPLOAD_STALL_MS = 120_000

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

/** Error thrown when an upload made no progress for `UPLOAD_STALL_MS`; the request is aborted. */
export class UploadStalledError extends Error {
  constructor() {
    super('Upload stalled')
    this.name = 'UploadStalledError'
  }
}

/**
 * POST a multipart body and yield the SSE frames the server streams back.
 *
 * Built on `XMLHttpRequest` rather than `fetch` because only XHR reports
 * request-body progress: that is what tells a large upload still sending
 * apart from one that stopped, and it drives the stall watchdog.
 *
 * @param path - API path, resolved against the sub-path base and owner param.
 * @param formData - The multipart body.
 * @param signal - Optional abort signal cancelling the request.
 * @param owner - Owner to send instead of the one selected now (see `withOwner`).
 * @yields Parsed SSE frames, in arrival order.
 * @throws UploadHttpError on a non-OK status, UploadStalledError when nothing
 *   moved for `UPLOAD_STALL_MS`, a TypeError on a network failure, and an
 *   `AbortError` when `signal` aborts.
 */
export async function* streamUpload(
  path: string,
  formData: FormData,
  signal?: AbortSignal,
  owner?: string | null
): AsyncGenerator<SseEvent, void, unknown> {
  signal?.throwIfAborted()
  const xhr = new XMLHttpRequest()
  const frames: SseEvent[] = []
  let buffer = ''
  let seen = 0
  let done = false
  let error: unknown
  let wake: (() => void) | undefined
  let timer: ReturnType<typeof setTimeout> | undefined

  const notify = () => {
    wake?.()
    wake = undefined
  }
  const settle = (err?: unknown) => {
    if (done) return
    done = true
    error = err
    clearTimeout(timer)
    notify()
  }
  const arm = () => {
    clearTimeout(timer)
    timer = setTimeout(() => {
      settle(new UploadStalledError())
      xhr.abort()
    }, UPLOAD_STALL_MS)
  }
  const ok = () => xhr.status >= 200 && xhr.status < 300
  const readFrames = () => {
    buffer += xhr.responseText.slice(seen)
    seen = xhr.responseText.length
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
        frames.push({ event, data: JSON.parse(raw) })
      } catch {
        frames.push({ event, data: raw })
      }
    }
  }

  xhr.upload.onprogress = arm
  xhr.onprogress = () => {
    arm()
    if (ok()) readFrames()
    notify()
  }
  xhr.onload = () => {
    if (ok()) {
      readFrames()
      settle()
    } else {
      settle(new UploadHttpError(xhr.status))
    }
  }
  xhr.onerror = () => settle(new TypeError('Upload failed: network error'))
  xhr.onabort = () => settle(new DOMException('Upload aborted', 'AbortError'))
  const onAbort = () => xhr.abort()
  signal?.addEventListener('abort', onAbort)

  try {
    xhr.open('POST', url(withOwner(path, owner)))
    arm()
    xhr.send(formData)
    while (true) {
      const frame = frames.shift()
      if (frame) {
        yield frame
        continue
      }
      if (done) {
        if (error !== undefined) throw error
        return
      }
      await new Promise<void>((resolve) => {
        wake = resolve
      })
    }
  } finally {
    signal?.removeEventListener('abort', onAbort)
    if (!done) {
      // The consumer stopped reading early: release the request and its timer.
      clearTimeout(timer)
      xhr.abort()
    }
  }
}
