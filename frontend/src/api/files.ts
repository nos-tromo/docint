import { apiErrorOf, apiFetch } from './client'
import { openBlob, saveBlob } from '@/lib/files'

/** A file the API served: its bytes, and the name the server gave it. */
export interface FetchedFile {
  /** The bytes, typed as the server served them. */
  blob: Blob
  /** The name `Content-Disposition` gives the file, or null when it names none. */
  filename: string | null
}

/**
 * The filename a `Content-Disposition` header names: the UTF-8 `filename*`
 * form of RFC 6266 first, then the plain `filename`, as a browser picks.
 *
 * @param header - The header's value, or null.
 * @returns The filename, or null when the header names none.
 */
export function filenameFromDisposition(header: string | null): string | null {
  if (!header) return null
  const extended = /filename\*\s*=\s*([^;]+)/i.exec(header)?.[1]?.trim()
  const encoded = extended ? /^[^']*'[^']*'(.+)$/.exec(extended)?.[1] : undefined
  if (encoded) {
    try {
      return decodeURIComponent(encoded)
    } catch {
      // Not valid percent-encoding: fall back to the plain form.
    }
  }
  const plain = /filename\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^;\s]+))/i.exec(header)
  const name = plain?.[1] !== undefined ? plain[1].replace(/\\(.)/g, '$1') : plain?.[2]
  return name || null
}

/**
 * GET a file from the API through the purpose wrapper.
 *
 * A browser's own requests — `<a href>`, `<img src>`, `<iframe src>`, a new
 * tab — cannot carry `X-Audit-Purpose`, so every file the SPA downloads,
 * previews or shows from the API is fetched here, with the stated purpose and
 * the `428` handling every request has, and handed to the browser as a Blob.
 *
 * @param path - API path with its query string; `apiFetch` adds the base and the owner.
 * @param signal - Optional abort signal, for a fetch a newer one supersedes.
 * @returns The file.
 * @throws ApiError on a non-2xx answer — `428` when the user declined to state a purpose.
 */
export async function fetchFile(path: string, signal?: AbortSignal): Promise<FetchedFile> {
  const res = await apiFetch(path, { signal })
  if (!res.ok) throw await apiErrorOf(res)
  return { blob: await res.blob(), filename: filenameFromDisposition(res.headers.get('Content-Disposition')) }
}

/**
 * Download a file from the API, as a download link to it would.
 *
 * @param path - API path with its query string.
 * @param fallback - The name to save it under when the server gives none.
 */
export async function downloadFile(path: string, fallback: string): Promise<void> {
  const { blob, filename } = await fetchFile(path)
  saveBlob(blob, filename ?? fallback)
}

/**
 * Open a file from the API in a new tab, as a `target="_blank"` link to it
 * would; one the tab cannot show is saved instead (`openBlob`).
 *
 * @param path - API path with its query string.
 * @param fallback - The name to save it under when the server gives none.
 */
export async function viewFile(path: string, fallback: string): Promise<void> {
  const { blob, filename } = await fetchFile(path)
  openBlob(blob, filename ?? fallback)
}
