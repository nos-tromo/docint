import { vi } from 'vitest'

/**
 * Stub what a fetched download touches, and record what it saved.
 *
 * `saveBlob` clicks a hidden anchor at an object URL, which happy-dom would
 * try to follow, and `openBlob` opens a tab. Undo with `vi.restoreAllMocks()`.
 *
 * @returns `saved()`, the anchors clicked so far (one per saved file, `download`
 *   naming it); `opened()`, the URLs new tabs were opened at; and `blobs`, every
 *   Blob handed out as an object URL, in order.
 */
export function captureDownloads() {
  const blobs: Blob[] = []
  vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => {
    blobs.push(blob as Blob)
    return `blob:docint/${blobs.length}`
  })
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
  const open = vi.spyOn(window, 'open').mockImplementation(() => ({ opener: window }) as unknown as Window)
  return {
    saved: () => click.mock.contexts as HTMLAnchorElement[],
    opened: () => open.mock.calls.map(([href]) => String(href)),
    open,
    blobs
  }
}

/** A file response as the API sends one: typed bytes, and the name in `Content-Disposition`. */
export function fileResponse(
  body: string,
  { type = 'application/octet-stream', filename, status = 200 }: { type?: string; filename?: string; status?: number } = {}
): Response {
  const headers: Record<string, string> = { 'Content-Type': type }
  if (filename) {
    headers['Content-Disposition'] =
      `attachment; filename="${filename.replace(/[^\x20-\x7e]/g, '_')}"; filename*=UTF-8''${encodeURIComponent(filename)}`
  }
  return new Response(body, { status, headers })
}
