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

// CSV exports for entity findings and hate-speech findings are streamed
// directly from the backend (`/collections/{name}/export/*.csv`) so the
// browser no longer accumulates the whole collection in memory. The
// canonical schemas live in `docint/utils/csv_stream.py` and the frontend
// uses anchor links built by `csvExportHref` in `src/api/collections.ts`.

// Re-export for tests/UI callers.
export { TXT_EXPORT_SEPARATOR }
