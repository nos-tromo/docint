import { streamUpload } from './upload'

export const streamIngestUpload = (
  collection: string,
  files: File[],
  signal?: AbortSignal
) => {
  const fd = new FormData()
  fd.append('collection', collection)
  for (const f of files) fd.append('files', f, f.name)
  return streamUpload('/ingest/upload', fd, signal)
}

export const sourcePreviewUrl = (collection: string, file_hash: string) =>
  `/sources/preview?collection=${encodeURIComponent(collection)}&file_hash=${encodeURIComponent(file_hash)}`
