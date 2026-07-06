import { streamUpload } from './upload'
import type { IngestEvent } from './types'

export function buildIngestFormData(collection: string, files: File[]): FormData {
  const fd = new FormData()
  fd.append('collection', collection)
  for (const f of files) fd.append('files', f, f.webkitRelativePath || f.name)
  return fd
}

export async function* streamIngestUpload(
  collection: string,
  files: File[],
  signal?: AbortSignal
): AsyncGenerator<IngestEvent, void, unknown> {
  const fd = buildIngestFormData(collection, files)
  // Stamp each event with the moment it arrives — once, here, as it streams
  // in — so `deriveIngestStatus` can compute the elapsed timer purely from
  // `receivedAt` instead of reading the wall clock on every re-derivation.
  for await (const ev of streamUpload('/ingest/upload', fd, signal)) {
    yield { ...ev, receivedAt: Date.now() } as IngestEvent
  }
}

export const sourcePreviewUrl = (collection: string, file_hash: string) =>
  `/sources/preview?collection=${encodeURIComponent(collection)}&file_hash=${encodeURIComponent(file_hash)}`
