import { apiDelete, apiGet, apiPost, ApiError, getOwnerParam, url } from './client'
import type { ExtractRecord } from './types'

/** The three formats a single source can be downloaded in. */
export type SourceExtractFormat = 'md' | 'pdf' | 'zip'

/**
 * The report chrome an extract is filed under: its case file and its operator.
 *
 * An extract is the appendix to a curated report, so it carries that report's
 * identity onto every page rather than inventing one of its own. Both are
 * optional — an extract built with no active report is simply unlabelled.
 */
export interface AppendixFields {
  reference_number?: string
  operator?: string
}

/** Drop the empty fields, so an unset value is absent rather than blank. */
function appendixEntries(appendix?: AppendixFields): [string, string][] {
  return Object.entries(appendix ?? {}).filter((entry): entry is [string, string] => !!entry[1])
}

/**
 * Queue a written extract of a collection, or of one source in it.
 *
 * A 409 means a build is already in flight for that collection — not an error
 * for the user, who has usually just re-submitted after a reload — so the
 * in-flight `job_id` is adopted, mirroring `createIngestJob`.
 *
 * @param collection - The caller's logical collection name.
 * @param target - One source to render, or undefined for the whole collection.
 * @param appendix - Case file and operator to print on the rendered PDF.
 * @returns The job id, and whether it was adopted from an in-flight run.
 */
export async function createExtract(
  collection: string,
  target?: string,
  appendix?: AppendixFields
): Promise<{ job_id: string; adopted: boolean }> {
  const path = `/collections/${encodeURIComponent(collection)}/extracts`
  const body = {
    ...(target ? { target } : {}),
    ...Object.fromEntries(appendixEntries(appendix))
  }
  try {
    const res = await apiPost<{ job_id: string }>(path, body)
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
function withOwnerQuery(path: string, extra: [string, string][] = []): string {
  const owner = getOwnerParam()
  const params = new URLSearchParams([...(owner ? [['owner', owner]] : []), ...extra])
  const query = params.toString()
  return query ? `${path}?${query}` : path
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
  fmt: SourceExtractFormat,
  appendix?: AppendixFields
): string {
  return url(
    withOwnerQuery(
      `/collections/${encodeURIComponent(collection)}/sources/${encodeURIComponent(sourceId)}/extract.${fmt}`,
      appendixEntries(appendix)
    )
  )
}
