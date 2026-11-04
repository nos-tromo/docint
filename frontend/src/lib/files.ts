/**
 * How long an object URL handed to the browser stays valid: long enough for a
 * download to start or a new tab to load it, after which it is only memory.
 */
const OBJECT_URL_TTL_MS = 60_000

/** Types a browser tab shows rather than saves. */
const VIEWABLE = /^(?:application\/(?:pdf|json)|image\/|text\/|audio\/|video\/)/i

/**
 * Save a Blob under `filename`, the way a download link would.
 *
 * @param blob - The file's bytes.
 * @param filename - The name to save it under.
 */
export function saveBlob(blob: Blob, filename: string): void {
  const href = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = href
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  // Not at once: a browser may still be reading the URL when click() returns.
  setTimeout(() => URL.revokeObjectURL(href), OBJECT_URL_TTL_MS)
}

/**
 * Show a Blob in a new tab, the way a `target="_blank"` link would.
 *
 * A type no tab can show is saved under `filename` instead — navigated to, it
 * would download under the object URL's random name. So is a Blob whose tab
 * the browser blocked, which happens when the fetch before it outlasted the
 * click's user activation. The tab is opened with an opener, so the object
 * URL resolves against this page, and is then cut loose from it, as
 * `rel="noopener"` would.
 *
 * @param blob - The file's bytes, typed as the server served them.
 * @param filename - The name to save it under where it cannot be shown.
 */
export function openBlob(blob: Blob, filename: string): void {
  if (!VIEWABLE.test(blob.type)) {
    saveBlob(blob, filename)
    return
  }
  const href = URL.createObjectURL(blob)
  const tab = window.open(href, '_blank')
  if (!tab) {
    URL.revokeObjectURL(href)
    saveBlob(blob, filename)
    return
  }
  tab.opener = null
  setTimeout(() => URL.revokeObjectURL(href), OBJECT_URL_TTL_MS)
}
