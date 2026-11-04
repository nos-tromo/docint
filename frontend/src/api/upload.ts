import { withPurpose } from '@infra/ui'
import { PURPOSE_SCOPE, url, withOwner } from './client'
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

/** One send of an upload: its status once known, then the frames it streams back. */
interface UploadAttempt {
  /** The response status; known once the attempt resolves. */
  readonly status: number
  /** Yield the frames as they arrive; throws once the request fails. */
  frames(): AsyncGenerator<SseEvent, void, unknown>
  /** Stop the request, if it is still running, and drop its timer and listener. */
  release(): void
}

/**
 * Send the multipart body once over `XMLHttpRequest`.
 *
 * Resolves as soon as the response status is known, so `withPurpose` can
 * decide whether to send again; the frames keep streaming into the attempt
 * meanwhile. Rejects when the request fails before then.
 *
 * @param target - The resolved URL.
 * @param formData - The multipart body; it may be sent twice.
 * @param headers - Request headers (the stated purpose).
 * @param signal - Optional abort signal cancelling the request.
 * @returns The attempt, once its status is known.
 */
function sendUpload(
  target: string,
  formData: FormData,
  headers: Record<string, string>,
  signal?: AbortSignal
): Promise<UploadAttempt> {
  return new Promise<UploadAttempt>((resolveHead, rejectHead) => {
    if (signal?.aborted) {
      rejectHead(new DOMException('Upload aborted', 'AbortError'))
      return
    }
    const xhr = new XMLHttpRequest()
    const frames: SseEvent[] = []
    let buffer = ''
    let seen = 0
    let done = false
    let headKnown = false
    let error: unknown
    let wake: (() => void) | undefined
    let timer: ReturnType<typeof setTimeout> | undefined

    const notify = () => {
      wake?.()
      wake = undefined
    }
    const onAbort = () => xhr.abort()
    const settle = (err?: unknown) => {
      if (done) return
      done = true
      error = err
      clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
      // Failed before the server answered: there is no attempt to hand back.
      if (!headKnown && err !== undefined) rejectHead(err)
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

    const attempt: UploadAttempt = {
      get status() {
        return xhr.status
      },
      async *frames() {
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
      },
      release() {
        signal?.removeEventListener('abort', onAbort)
        if (!done) {
          clearTimeout(timer)
          xhr.abort()
        }
      }
    }
    // The status is known from the first response byte; either event may come first.
    const headArrived = () => {
      if (headKnown || xhr.status === 0) return
      headKnown = true
      resolveHead(attempt)
    }

    xhr.upload.onprogress = arm
    xhr.onreadystatechange = () => {
      if (xhr.readyState >= 2) headArrived()
    }
    xhr.onprogress = () => {
      arm()
      headArrived()
      if (ok()) readFrames()
      notify()
    }
    xhr.onload = () => {
      headArrived()
      if (ok()) {
        readFrames()
        settle()
      } else {
        settle(new UploadHttpError(xhr.status))
      }
    }
    xhr.onerror = () => settle(new TypeError('Upload failed: network error'))
    xhr.onabort = () => settle(new DOMException('Upload aborted', 'AbortError'))
    signal?.addEventListener('abort', onAbort)

    xhr.open('POST', target)
    for (const [name, value] of Object.entries(headers)) xhr.setRequestHeader(name, value)
    arm()
    xhr.send(formData)
  })
}

/**
 * POST a multipart body and yield the SSE frames the server streams back.
 *
 * Built on `XMLHttpRequest` rather than `fetch` because only XHR reports
 * request-body progress: that is what tells a large upload still sending
 * apart from one that stopped, and it drives the stall watchdog. Like every
 * other request it carries the stated purpose, and a `428` opens the purpose
 * bar's dialog and sends the body once more (`withPurpose`).
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
  const target = url(withOwner(path, owner))
  let latest: UploadAttempt | undefined
  const attempt = await withPurpose(PURPOSE_SCOPE, async (headers) => {
    // A resend replaces the refused attempt, which has nothing more to say.
    latest?.release()
    latest = await sendUpload(target, formData, headers, signal)
    return latest
  })
  try {
    yield* attempt.frames()
  } finally {
    // The consumer may stop reading early: release the request and its timer.
    attempt.release()
  }
}
