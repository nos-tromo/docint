import { apiDelete, apiGet, getOwnerParam, url } from './client'
import { queueJob } from './jobs'
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
 * A 409 means a build is already in flight for that collection, whose
 * `job_id` is adopted (see {@link queueJob}).
 *
 * @param collection - The caller's logical collection name.
 * @param target - One source to render, or undefined for the whole collection.
 * @param appendix - Case file and operator to print on the rendered PDF.
 * @returns The job id, and whether it was adopted from an in-flight run.
 */
export function createExtract(
  collection: string,
  target?: string,
  appendix?: AppendixFields
): Promise<{ job_id: string; adopted: boolean }> {
  return queueJob(`/collections/${encodeURIComponent(collection)}/extracts`, {
    ...(target ? { target } : {}),
    ...Object.fromEntries(appendixEntries(appendix))
  })
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
