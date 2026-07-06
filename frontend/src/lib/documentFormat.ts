import type { DocumentRecord } from '@/api/types'

/**
 * Human-friendly short labels for the MIME types docint actually ingests.
 * Anything not listed falls back to a cleaned-up subtype (see {@link mimeLabel}).
 */
const MIME_LABELS: Record<string, string> = {
  'image/jpeg': 'JPEG',
  'image/jpg': 'JPEG',
  'image/png': 'PNG',
  'image/webp': 'WEBP',
  'image/gif': 'GIF',
  'image/tiff': 'TIFF',
  'application/pdf': 'PDF',
  'text/csv': 'CSV',
  'application/json': 'JSON',
  'application/x-ndjson': 'NDJSON',
  'application/ld+json': 'JSON-LD',
  'text/plain': 'TXT',
  'text/markdown': 'MD',
  'text/rtf': 'RTF',
  'application/rtf': 'RTF',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
  'application/msword': 'DOC',
  'video/mp4': 'MP4',
  'audio/mpeg': 'MP3'
}

/**
 * Turn a raw MIME type into a compact, upper-cased label for the inspector.
 *
 * Known types map to a curated label; unknown ones degrade to the subtype with
 * `x-` / `vnd.` prefixes stripped and capped in length, so a verbose vendor MIME
 * never blows out the column.
 *
 * @param mimetype Raw MIME string (e.g. `application/x-ndjson`), or nullish.
 * @returns A short label such as `NDJSON`, or `—` when the MIME is missing.
 */
export function mimeLabel(mimetype?: string | null): string {
  if (!mimetype) return '—'
  const key = mimetype.trim().toLowerCase()
  if (!key) return '—'
  if (MIME_LABELS[key]) return MIME_LABELS[key]
  const subtype = key.split(';')[0]?.split('/')[1] ?? key
  const cleaned = subtype.replace(/^x-/, '').replace(/^vnd\./, '')
  if (!cleaned) return '—'
  const short = cleaned.length > 12 ? cleaned.slice(0, 12) : cleaned
  return short.toUpperCase()
}

/** Rendered units cell plus the numeric key used to sort the column. */
export interface UnitsLabel {
  text: string
  sort: number
}

/**
 * Describe a document's "size" using whichever count is meaningful for its kind.
 *
 * Pages win when present (PDFs, docx); otherwise rows (CSV, transcript segments).
 * Images and anything with neither return `—` instead of a misleading `0`.
 *
 * @param doc Document record (only `page_count` / `row_count` are read).
 * @returns The display text and a numeric `sort` key (`0` when there are no units).
 */
export function unitsLabel(doc: Pick<DocumentRecord, 'page_count' | 'row_count'>): UnitsLabel {
  const pages = doc.page_count ?? 0
  if (pages > 0) return { text: `${pages} pg`, sort: pages }
  const rows = doc.row_count ?? 0
  if (rows > 0) return { text: `${rows} ${rows === 1 ? 'row' : 'rows'}`, sort: rows }
  return { text: '—', sort: 0 }
}

/**
 * First `length` characters of a content hash, for a compact inspector cell.
 *
 * @param hash Full hex hash, or nullish.
 * @param length Number of leading characters to keep (default 8).
 * @returns The truncated hash, or `—` when the hash is missing.
 */
export function shortHash(hash?: string | null, length = 8): string {
  if (!hash) return '—'
  return hash.slice(0, length)
}
