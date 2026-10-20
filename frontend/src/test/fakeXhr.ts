import { vi } from 'vitest'

/** Minimal `XMLHttpRequest` stand-in: records the request; the test plays the network. */
export class FakeXhr {
  static instances: FakeXhr[] = []
  method = ''
  url = ''
  body: unknown = null
  status = 0
  responseText = ''
  aborted = false
  upload: { onprogress: (() => void) | null } = { onprogress: null }
  onprogress: (() => void) | null = null
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  onabort: (() => void) | null = null

  constructor() {
    FakeXhr.instances.push(this)
  }

  open(method: string, url: string) {
    this.method = method
    this.url = url
  }

  send(body: unknown) {
    this.body = body
  }

  abort() {
    if (this.aborted) return
    this.aborted = true
    this.onabort?.()
  }

  /** Report request-body bytes leaving the browser. */
  sent() {
    this.upload.onprogress?.()
  }

  /** Deliver a chunk of the response body. */
  receive(chunk: string, status = 200) {
    this.status = status
    this.responseText += chunk
    this.onprogress?.()
  }

  /** Complete the response. */
  finish(status = this.status || 200) {
    this.status = status
    this.onload?.()
  }

  /** Fail at the network level. */
  fail() {
    this.onerror?.()
  }
}

/** Replace the global `XMLHttpRequest` with `FakeXhr`; undo with `vi.unstubAllGlobals()`. */
export function installFakeXhr(): typeof FakeXhr {
  FakeXhr.instances = []
  vi.stubGlobal('XMLHttpRequest', FakeXhr)
  return FakeXhr
}
