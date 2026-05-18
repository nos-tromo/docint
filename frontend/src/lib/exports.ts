import type { ChatTurnData } from '@/components/chat/ChatTurn'
import type {
  ChatFinalEvent,
  HateSpeechRow,
  NerEntityRow,
  NerSourceRow,
  ReferenceMetadata,
  Source,
  SummaryResponse
} from '@/api/types'
import {
  referenceMetadataItems,
  referenceMetadataValue
} from './referenceMetadata'
import { toCsv } from './csv'

// Width matches docint/ui/analysis.py::TXT_EXPORT_SEPARATOR — the latest
// Streamlit export style. Kept exactly so cross-version diffing stays
// usable.
const TXT_EXPORT_SEPARATOR = '='.repeat(72)

// Ordered text sections emitted after the metadata block. `text` falls
// back to chunk_text/text when reference_metadata lacks an explicit
// value — without falling back to those keys *also* into the metadata
// list, which is what caused the body text to render twice in the
// previous frontend port.
const TEXT_SECTIONS: Array<{ key: 'anchor_text' | 'parent_text' | 'text'; label: string }> = [
  { key: 'anchor_text', label: 'Anchor Text' },
  { key: 'parent_text', label: 'Parent Text' },
  { key: 'text', label: 'Text' }
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

function referenceExportSections(chunk: ChunkLike): {
  metadataLines: string[]
  textSections: string[]
} {
  const metadataLines: string[] = []
  for (const { label, value } of referenceMetadataItems(chunk.reference_metadata, {
    includeText: false
  })) {
    metadataLines.push(`- ${label}: ${value}`)
  }

  const textSections: string[] = []
  for (const { key, label } of TEXT_SECTIONS) {
    let value = referenceMetadataValue(chunk.reference_metadata, key)
    if (!value && key === 'text') {
      value = String(chunk.chunk_text ?? chunk.text ?? '').trim()
    }
    if (!value) continue
    textSections.push(label, '-'.repeat(label.length), value, '')
  }
  if (textSections.length > 0) textSections.pop()
  return { metadataLines, textSections }
}

function analysisChunkBlock(
  chunk: ChunkLike,
  index: number,
  extra: Array<[string, unknown]> = []
): string[] {
  const source = String(chunk.filename ?? chunk.source_ref ?? 'Unknown')
  const hasPage = chunk.page !== null && chunk.page !== undefined
  const locationLabel = hasPage ? 'Page' : 'Row'
  const locationValue = hasPage ? chunk.page : (chunk.row ?? 'n/a')
  const lines: string[] = [
    TXT_EXPORT_SEPARATOR,
    `[${index}] ${source}`,
    `- ${locationLabel}: ${locationValue}`,
    `- Chunk ID: ${chunk.chunk_id ?? 'n/a'}`
  ]
  for (const [label, value] of extra) {
    if (value === null || value === undefined) continue
    const text = String(value).trim()
    if (text) lines.push(`- ${label}: ${text}`)
  }
  const { metadataLines, textSections } = referenceExportSections(chunk)
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

function sourceLines(src: Source, index: number): string[] {
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
    extra.push(['Score', src.score.toFixed(3)])
  }
  return analysisChunkBlock(chunk, index, extra)
}

export function chatTranscriptToText(turns: ChatTurnData[]): string {
  const out: string[] = []
  for (const t of turns) {
    out.push(`USER: ${t.user}`, '')
    out.push(`ASSISTANT: ${t.assistant}`, '')
    const meta: ChatFinalEvent | null = t.meta
    if (meta) {
      if (meta.validation_checked !== undefined || meta.validation_reason) {
        const parts = [
          `checked=${meta.validation_checked ?? ''}`,
          `mismatch=${meta.validation_mismatch ?? ''}`
        ]
        if (meta.validation_reason) parts.push(`reason=${meta.validation_reason}`)
        out.push(`VALIDATION: ${parts.join(', ')}`, '')
      }
      const sources = meta.sources ?? []
      if (sources.length > 0) {
        out.push('SOURCES:')
        sources.forEach((s, i) => out.push(...sourceLines(s, i + 1)))
      }
    }
  }
  return out.join('\n').trimEnd() + '\n'
}

// ---------------------------------------------------------------------------
// Summary (markdown)
// ---------------------------------------------------------------------------

export function summaryToMarkdown(meta: SummaryResponse | null, text: string): string {
  const out: string[] = ['# Summary', '']
  const body = (meta?.summary ?? text ?? '').trim()
  out.push(body || '(empty)')
  const sources = meta?.sources ?? []
  if (sources.length > 0) {
    out.push('', '## Sources', '')
    sources.forEach((s, i) => out.push(...sourceLines(s, i + 1)))
  }
  return out.join('\n').trimEnd() + '\n'
}

// ---------------------------------------------------------------------------
// Entity / hate-speech (analysis-chunk format)
// ---------------------------------------------------------------------------

export function entityFindingsToText(
  entity: NerEntityRow,
  chunks: NerSourceRow[]
): string {
  const label = `${entity.text} [${entity.type || 'Unlabeled'}]`
  const out: string[] = [`Entity Findings: ${label}`, '']
  chunks.forEach((c, i) => out.push(...analysisChunkBlock(c, i + 1)))
  return out.join('\n').trimEnd() + '\n'
}

export function hateSpeechToText(rows: HateSpeechRow[]): string {
  const out: string[] = ['Hate-Speech Findings', '']
  rows.forEach((r, i) =>
    out.push(
      ...analysisChunkBlock(r, i + 1, [
        ['Category', r.category],
        ['Confidence', r.confidence],
        ['Reason', r.reason]
      ])
    )
  )
  return out.join('\n').trimEnd() + '\n'
}

// ---------------------------------------------------------------------------
// CSV (matches the latest analysis.py columns)
// ---------------------------------------------------------------------------

const ENTITY_CSV_COLUMNS = [
  'entity',
  'source',
  'page',
  'row',
  'chunk_id',
  'chunk_text',
  'network',
  'ref_type',
  'uuid',
  'timestamp',
  'author',
  'author_id',
  'vanity',
  'text_id',
  'anchor_text',
  'parent_text'
] as const

const HATE_CSV_COLUMNS = [
  'source',
  'page',
  'row',
  'chunk_id',
  'category',
  'confidence',
  'reason',
  'chunk_text',
  'network',
  'ref_type',
  'uuid',
  'timestamp',
  'author',
  'author_id',
  'vanity',
  'text_id',
  'anchor_text',
  'parent_text'
] as const

function refField(meta: ReferenceMetadata | undefined, key: string): unknown {
  if (!meta) return ''
  // `ref_type` is the CSV column name for the reference_metadata `type`
  // key (matches the Streamlit schema).
  const lookup = key === 'ref_type' ? 'type' : key
  const raw = (meta as Record<string, unknown>)[lookup]
  return raw ?? ''
}

export function entityFindingsToCsv(
  entity: NerEntityRow,
  chunks: NerSourceRow[]
): string {
  const label = `${entity.text} [${entity.type || 'Unlabeled'}]`
  const rows = chunks.map((c) => ({
    entity: label,
    source: c.filename ?? '',
    page: c.page ?? '',
    row: c.row ?? '',
    chunk_id: c.chunk_id ?? '',
    chunk_text: c.chunk_text ?? c.text ?? '',
    network: refField(c.reference_metadata, 'network'),
    ref_type: refField(c.reference_metadata, 'ref_type'),
    uuid: refField(c.reference_metadata, 'uuid'),
    timestamp: refField(c.reference_metadata, 'timestamp'),
    author: refField(c.reference_metadata, 'author'),
    author_id: refField(c.reference_metadata, 'author_id'),
    vanity: refField(c.reference_metadata, 'vanity'),
    text_id: refField(c.reference_metadata, 'text_id'),
    anchor_text: refField(c.reference_metadata, 'anchor_text'),
    parent_text: refField(c.reference_metadata, 'parent_text')
  })) as unknown as Array<Record<string, unknown>>
  return toCsv(rows, ENTITY_CSV_COLUMNS as unknown as string[])
}

export function hateSpeechToCsv(rows: HateSpeechRow[]): string {
  const out = rows.map((r) => ({
    source: r.filename ?? r.source_ref ?? '',
    page: r.page ?? '',
    row: r.row ?? '',
    chunk_id: r.chunk_id ?? '',
    category: r.category ?? '',
    confidence: r.confidence ?? '',
    reason: r.reason ?? '',
    chunk_text: r.chunk_text ?? r.text ?? '',
    network: refField(r.reference_metadata, 'network'),
    ref_type: refField(r.reference_metadata, 'ref_type'),
    uuid: refField(r.reference_metadata, 'uuid'),
    timestamp: refField(r.reference_metadata, 'timestamp'),
    author: refField(r.reference_metadata, 'author'),
    author_id: refField(r.reference_metadata, 'author_id'),
    vanity: refField(r.reference_metadata, 'vanity'),
    text_id: refField(r.reference_metadata, 'text_id'),
    anchor_text: refField(r.reference_metadata, 'anchor_text'),
    parent_text: refField(r.reference_metadata, 'parent_text')
  })) as unknown as Array<Record<string, unknown>>
  return toCsv(out, HATE_CSV_COLUMNS as unknown as string[])
}

// Re-export for tests/UI callers.
export { TXT_EXPORT_SEPARATOR }
