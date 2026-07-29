import type { ReferenceMetadata } from '@/api/types'
import type { Strings } from '@/i18n'
import { defaultT } from '@/i18n/defaultT'

// Mirrors docint/utils/reference_metadata.py REFERENCE_METADATA_FIELDS and
// preserves its display order. The body-text fields (text/parent_text/
// anchor_text) are listed but the inline summary skips them by default,
// matching the Streamlit `reference_metadata_inline` behavior.
//
// `label` stays the English literal (read directly by referenceMetadata.test.ts
// and used as the fallback for a field with no catalog entry); the actual
// display label a caller sees comes from `LABEL_KEY` below, resolved through
// the caller's own `t`.
export const REFERENCE_METADATA_FIELDS: Array<{ key: keyof ReferenceMetadata; label: string }> = [
  { key: 'network', label: 'Network' },
  { key: 'type', label: 'Type' },
  { key: 'uuid', label: 'UUID' },
  { key: 'posting_uuid', label: 'Posting UUID' },
  { key: 'posting_id', label: 'Posting ID' },
  { key: 'media_id', label: 'Media ID' },
  { key: 'url', label: 'URL' },
  { key: 'posting_network', label: 'Posting Network' },
  { key: 'posting_author', label: 'Posting Author' },
  { key: 'posting_author_id', label: 'Posting Author ID' },
  { key: 'posting_vanity', label: 'Posting Vanity' },
  { key: 'posting_timestamp', label: 'Posting Timestamp' },
  { key: 'posting_url', label: 'Posting URL' },
  { key: 'posting_text', label: 'Posting Text' },
  { key: 'timestamp', label: 'Timestamp' },
  { key: 'author', label: 'Author' },
  { key: 'author_id', label: 'Author ID' },
  { key: 'vanity', label: 'Vanity' },
  { key: 'text', label: 'Text' },
  { key: 'text_id', label: 'Text ID' },
  { key: 'parent_text', label: 'Parent Text' },
  { key: 'anchor_text', label: 'Anchor Text' },
  { key: 'speaker', label: 'Speaker' },
  { key: 'language', label: 'Language' },
  { key: 'detected_language', label: 'Detected Language' },
  { key: 'source_file', label: 'Source File' }
]

const LABEL_KEY: Partial<Record<string, keyof Strings>> = {
  network: 'common.refmeta_network',
  type: 'common.refmeta_type',
  uuid: 'common.refmeta_uuid',
  posting_uuid: 'common.refmeta_posting_uuid',
  posting_id: 'common.refmeta_posting_id',
  media_id: 'common.refmeta_media_id',
  url: 'common.refmeta_url',
  posting_network: 'common.refmeta_posting_network',
  posting_author: 'common.refmeta_posting_author',
  posting_author_id: 'common.refmeta_posting_author_id',
  posting_vanity: 'common.refmeta_posting_vanity',
  posting_timestamp: 'common.refmeta_posting_timestamp',
  posting_url: 'common.refmeta_posting_url',
  posting_text: 'common.refmeta_posting_text',
  timestamp: 'common.refmeta_timestamp',
  author: 'common.refmeta_author',
  author_id: 'common.refmeta_author_id',
  vanity: 'common.refmeta_vanity',
  text: 'common.refmeta_text',
  text_id: 'common.refmeta_text_id',
  parent_text: 'common.refmeta_parent_text',
  anchor_text: 'common.refmeta_anchor_text',
  speaker: 'common.refmeta_speaker',
  language: 'common.refmeta_language',
  detected_language: 'common.refmeta_detected_language',
  source_file: 'common.refmeta_source_file'
}

const BODY_TEXT_KEYS = new Set(['text', 'parent_text', 'anchor_text'])

export interface MetadataPillItem {
  key: string
  label?: string
  value: string
  href?: string
}

// Display-only curation for the Analysis tables' pill cells. Opaque IDs stay
// available in CSV exports and report snapshots; they are only dropped here.
const PILL_EXCLUDED_KEYS = new Set([
  'uuid',
  'posting_uuid',
  'posting_id',
  'media_id',
  'author_id',
  'posting_author_id',
  'text_id',
  'posting_text'
])
const PILL_LABELED_KEYS = new Set(['author', 'posting_author', 'vanity', 'posting_vanity', 'speaker'])
const PILL_TIMESTAMP_KEYS = new Set(['timestamp', 'posting_timestamp'])
const PILL_URL_KEYS = new Set(['url', 'posting_url'])

// Reference metadata originates from ingested (untrusted) social exports, so
// a value only becomes a clickable href when it parses as an absolute
// http(s) URL — anything else (javascript:, data:, malformed text) falls
// back to a plain, non-linked value pill.
function safeHttpUrl(value: string): string | null {
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? value : null
  } catch {
    return null
  }
}

export function referenceMetadataPills(
  meta: ReferenceMetadata | undefined,
  t: (key: keyof Strings, vars?: Record<string, string | number>) => string = defaultT
): MetadataPillItem[] {
  if (!meta) return []
  const pills: MetadataPillItem[] = []
  for (const { key, label } of REFERENCE_METADATA_FIELDS) {
    const k = key as string
    if (BODY_TEXT_KEYS.has(k) || PILL_EXCLUDED_KEYS.has(k)) continue
    const raw = meta[key]
    if (raw === null || raw === undefined) continue
    const text = String(raw).trim()
    if (!text) continue
    if (PILL_URL_KEYS.has(k)) {
      const href = safeHttpUrl(text)
      if (href) {
        pills.push({ key: k, value: t('common.pill_open_link'), href })
      } else {
        pills.push({ key: k, value: text })
      }
      continue
    }
    if (PILL_TIMESTAMP_KEYS.has(k)) {
      pills.push({ key: k, value: text.replace(/\.\d+$/, '') })
      continue
    }
    if (PILL_LABELED_KEYS.has(k)) {
      const labelKey = LABEL_KEY[k]
      pills.push({ key: k, label: labelKey ? t(labelKey) : label, value: text })
      continue
    }
    pills.push({ key: k, value: text })
  }
  return pills
}

export function referenceMetadataItems(
  meta: ReferenceMetadata | undefined,
  options: { includeText?: boolean } = {},
  t: (key: keyof Strings, vars?: Record<string, string | number>) => string = defaultT
): Array<{ label: string; value: string }> {
  if (!meta) return []
  const { includeText = false } = options
  const items: Array<{ label: string; value: string }> = []
  for (const { key, label } of REFERENCE_METADATA_FIELDS) {
    if (!includeText && BODY_TEXT_KEYS.has(key as string)) continue
    const raw = meta[key]
    if (raw === null || raw === undefined) continue
    const text = String(raw).trim()
    if (!text) continue
    const labelKey = LABEL_KEY[key as string]
    items.push({ label: labelKey ? t(labelKey) : label, value: text })
  }
  return items
}

export function referenceMetadataValue(
  meta: ReferenceMetadata | undefined,
  key: string
): string {
  if (!meta) return ''
  const raw = (meta as Record<string, unknown>)[key]
  if (raw === null || raw === undefined) return ''
  return String(raw).trim()
}
