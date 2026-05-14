import type { ReferenceMetadata } from '@/api/types'

// Mirrors docint/utils/reference_metadata.py REFERENCE_METADATA_FIELDS and
// preserves its display order. The body-text fields (text/parent_text/
// anchor_text) are listed but the inline summary skips them by default,
// matching the Streamlit `reference_metadata_inline` behavior.
export const REFERENCE_METADATA_FIELDS: Array<{ key: keyof ReferenceMetadata; label: string }> = [
  { key: 'network', label: 'Network' },
  { key: 'type', label: 'Type' },
  { key: 'uuid', label: 'UUID' },
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
  { key: 'source_file', label: 'Source File' }
]

const BODY_TEXT_KEYS = new Set(['text', 'parent_text', 'anchor_text'])

export function referenceMetadataItems(
  meta: ReferenceMetadata | undefined,
  options: { includeText?: boolean } = {}
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
    items.push({ label, value: text })
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
