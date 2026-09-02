import { apiDelete, apiGet, apiPost, ApiError, getOwnerParam, url } from './client'
import type { ExtractRecord } from './types'

/** The three formats a single source can be downloaded in. */
export type SourceExtractFormat = 'md' | 'pdf' | 'zip'

/**
 * Queue a written extract of a collection, or of one source in it.
 *
 * A 409 means a build is already in flight for that collection — not an error
 * for the user, who has usually just re-submitted after a reload — so the
 * in-flight `job_id` is adopted, mirroring `createIngestJob`.
 *
 * @param collection - The caller's logical collection name.
 * @param target - One source to render, or undefined for the whole collection.
 * @returns The job id, and whether it was adopted from an in-flight run.
 */
export async function createExtract(
  collection: string,
  target?: string
): Promise<{ job_id: string; adopted: boolean }> {
  const path = `/collections/${encodeURIComponent(collection)}/extracts`
  try {
    const res = await apiPost<{ job_id: string }>(path, target ? { target } : {})
    return { job_id: res.job_id, adopted: false }
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) {
      const detail = e.detail as { detail?: { job_id?: string } } | { job_id?: string }
      const nested = (detail as { detail?: { job_id?: string } }).detail
      const jobId = nested?.job_id ?? (detail as { job_id?: string }).job_id
      if (jobId) return { job_id: jobId, adopted: true }
    }
    throw e
  }
}

/** List a collection's stored extracts, newest first. */
export const listExtracts = (collection: string) =>
  apiGet<{ extracts: ExtractRecord[] }>(`/collections/${encodeURIComponent(collection)}/extracts`)

/** Delete one stored extract. */
export const deleteExtract = (collection: string, extractId: string) =>
  apiDelete<{ ok: boolean }>(
    `/collections/${encodeURIComponent(collection)}/extracts/${encodeURIComponent(extractId)}`
  )

/** Append the admin owner context, as the other href builders do. */
function withOwnerQuery(path: string): string {
  const owner = getOwnerParam()
  return owner ? `${path}?owner=${encodeURIComponent(owner)}` : path
}

/**
 * Absolute URL of a stored bundle. Use as a download anchor's `href` so the
 * browser streams the archive natively.
 */
export function extractDownloadHref(collection: string, extractId: string): string {
  return url(
    withOwnerQuery(
      `/collections/${encodeURIComponent(collection)}/extracts/${encodeURIComponent(extractId)}/download`
    )
  )
}

/** Absolute URL of one source's extract in the given format. */
export function sourceExtractHref(
  collection: string,
  sourceId: string,
  fmt: SourceExtractFormat
): string {
  return url(
    withOwnerQuery(
      `/collections/${encodeURIComponent(collection)}/sources/${encodeURIComponent(sourceId)}/extract.${fmt}`
    )
  )
}
