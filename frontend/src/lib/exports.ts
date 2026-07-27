import type { ChatTurnData } from '@/components/chat/ChatTurn'
import type {
  ChatFinalEvent,
  ReferenceMetadata,
  Source,
  SummaryResponse
} from '@/api/types'
import type { Strings } from '@/i18n'
import { defaultT } from '@/i18n/defaultT'
import {
  referenceMetadataItems,
  referenceMetadataValue
} from './referenceMetadata'

type Translate = (key: keyof Strings, vars?: Record<string, string | number>) => string

// Width matches docint/ui/analysis.py::TXT_EXPORT_SEPARATOR — the latest
// Streamlit export style. Kept exactly so cross-version diffing stays
// usable.
const TXT_EXPORT_SEPARATOR = '='.repeat(72)

// Ordered text sections emitted after the metadata block. `text` falls
// back to chunk_text/text when reference_metadata lacks an explicit
// value — without falling back to those keys *also* into the metadata
// list, which is what caused the body text to render twice in the
// previous frontend port. Labels reuse the shared referenceMetadata catalog
// keys — the exact same field names shown inline in citations/findings.
const TEXT_SECTIONS: Array<{ key: 'anchor_text' | 'parent_text' | 'text'; labelKey: keyof Strings }> = [
  { key: 'anchor_text', labelKey: 'common.refmeta_anchor_text' },
  { key: 'parent_text', labelKey: 'common.refmeta_parent_text' },
  { key: 'text', labelKey: 'common.refmeta_text' }
]

interface ChunkLike {
  filename?: string | null
  source_ref?: string | null
  page?: number | null
  row?: number | null
  chunk_id?: string | null
  chunk_text?: string | null
  text?: string | null
  reference_metadata?: ReferenceMetadata
}

function referenceExportSections(
  chunk: ChunkLike,
  t: Translate
): {
  metadataLines: string[]
  textSections: string[]
} {
  const metadataLines: string[] = []
  for (const { label, value } of referenceMetadataItems(
    chunk.reference_metadata,
    { includeText: false },
    t
  )) {
    metadataLines.push(`- ${label}: ${value}`)
  }

  const textSections: string[] = []
  for (const { key, labelKey } of TEXT_SECTIONS) {
    let value = referenceMetadataValue(chunk.reference_metadata, key)
    if (!value && key === 'text') {
      value = String(chunk.chunk_text ?? chunk.text ?? '').trim()
    }
    if (!value) continue
    const label = t(labelKey)
    textSections.push(label, '-'.repeat(label.length), value, '')
  }
  if (textSections.length > 0) textSections.pop()
  return { metadataLines, textSections }
}

function analysisChunkBlock(
  chunk: ChunkLike,
  index: number,
  extra: Array<[string, unknown]> = [],
  t: Translate = defaultT
): string[] {
  const source = String(chunk.filename ?? chunk.source_ref ?? t('export.unknown_source'))
  const hasPage = chunk.page !== null && chunk.page !== undefined
  const locationLabel = hasPage ? t('common.meta_page') : t('common.meta_row')
  const locationValue = hasPage ? chunk.page : (chunk.row ?? t('common.not_applicable'))
  const lines: string[] = [
    TXT_EXPORT_SEPARATOR,
    `[${index}] ${source}`,
    `- ${locationLabel}: ${locationValue}`,
    `- ${t('common.meta_chunk_id')}: ${chunk.chunk_id ?? t('common.not_applicable')}`
  ]
  for (const [label, value] of extra) {
    if (value === null || value === undefined) continue
    const text = String(value).trim()
    if (text) lines.push(`- ${label}: ${text}`)
  }
  const { metadataLines, textSections } = referenceExportSections(chunk, t)
  if (metadataLines.length > 0) {
    lines.push('')
    lines.push(...metadataLines)
  }
  if (textSections.length > 0) {
    lines.push('')
    lines.push(...textSections)
  }
  lines.push('')
  return lines
}

// ---------------------------------------------------------------------------
// Chat transcript
// ---------------------------------------------------------------------------

function sourceLines(src: Source, index: number, t: Translate = defaultT): string[] {
  const chunk: ChunkLike = {
    filename: src.filename,
    page: src.page,
    row: src.row,
    chunk_id: src.id,
    // Prefer full text over preview_text (which may be truncated).
    text: src.text ?? src.preview_text,
    reference_metadata: src.reference_metadata
  }
  const extra: Array<[string, unknown]> = []
  if (src.score !== null && src.score !== undefined) {
    extra.push([t('common.meta_score'), src.score.toFixed(3)])
  }
  return analysisChunkBlock(chunk, index, extra, t)
}

export function chatTranscriptToText(turns: ChatTurnData[], t: Translate = defaultT): string {
  const out: string[] = []
  for (const turn of turns) {
    out.push(`${t('export.user_prefix')}${turn.user}`, '')
    out.push(`${t('export.assistant_prefix')}${turn.assistant}`, '')
    const meta: ChatFinalEvent | null = turn.meta
    if (meta) {
      if (meta.validation_checked !== undefined || meta.validation_reason) {
        const parts = [
          `checked=${meta.validation_checked ?? ''}`,
          `mismatch=${meta.validation_mismatch ?? ''}`
        ]
        if (meta.validation_reason) parts.push(`reason=${meta.validation_reason}`)
        out.push(`${t('export.validation_prefix')}${parts.join(', ')}`, '')
      }
      const sources = meta.sources ?? []
      if (sources.length > 0) {
        out.push(t('export.sources_heading'))
        sources.forEach((s, i) => out.push(...sourceLines(s, i + 1, t)))
      }
    }
  }
  return out.join('\n').trimEnd() + '\n'
}

// ---------------------------------------------------------------------------
// Summary (markdown)
// ---------------------------------------------------------------------------

export function summaryToMarkdown(
  meta: SummaryResponse | null,
  text: string,
  t: Translate = defaultT
): string {
  const out: string[] = [`# ${t('report.default_summary')}`, '']
  const body = (meta?.summary ?? text ?? '').trim()
  out.push(body || t('export.empty_placeholder'))
  const sources = meta?.sources ?? []
  if (sources.length > 0) {
    out.push('', `## ${t('chat.sources')}`, '')
    sources.forEach((s, i) => out.push(...sourceLines(s, i + 1, t)))
  }
  return out.join('\n').trimEnd() + '\n'
}

// Entity-findings and hate-speech downloads are streamed directly from
// the backend (`/collections/{name}/export/*.csv`). Schemas live in
// `docint/utils/csv_stream.py`; the frontend uses anchor links built by
// `csvExportHref` in `src/api/collections.ts`.

// Re-export for tests/UI callers.
export { TXT_EXPORT_SEPARATOR }
