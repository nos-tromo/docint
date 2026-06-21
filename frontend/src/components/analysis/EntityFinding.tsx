import { useState } from 'react'
import type { NerSourceRow } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import { sourcePreviewUrl } from '@/api/ingest'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { highlightSegments } from '@/lib/highlight'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { entityFindingSnapshot } from '@/lib/reportSnapshots'

interface Props {
  index: number
  source: NerSourceRow
  // Lowercase text candidates that identify the picked entity (and its
  // variants); used both for chunk filtering and to highlight matched
  // mentions inside the chunk body.
  highlightTerms: string[]
  selectedTypeLower?: string
  defaultOpen?: boolean
  // Report-builder context. When `reportDedupeKeys` is provided (the entity
  // is rendered inside a report-aware view), an "Add to report" control shows
  // and `entityLabel` becomes the report's entity column for this chunk.
  entityLabel?: string
  reportDedupeKeys?: Set<string>
}

function locationLabel(source: NerSourceRow): string {
  const parts: string[] = []
  if (source.filename) parts.push(source.filename)
  const loc: string[] = []
  if (source.page !== null && source.page !== undefined) loc.push(`page ${source.page}`)
  if (source.row !== null && source.row !== undefined) loc.push(`row ${source.row}`)
  if (loc.length > 0) parts.push(`(${loc.join(', ')})`)
  return parts.join(' ') || 'Unknown source'
}

function shortHash(hash: string | undefined | null): string {
  if (!hash) return ''
  return hash.length > 14 ? `${hash.slice(0, 10)}…${hash.slice(-4)}` : hash
}

function matchedMentions(
  source: NerSourceRow,
  terms: string[],
  typeLower?: string
): Array<{ text: string; type: string; score?: number | null }> {
  if (!source.entities) return []
  const lowerTerms = new Set(terms.map((t) => t.toLowerCase()))
  return source.entities.filter((ent) => {
    const txt = (ent.text ?? '').trim().toLowerCase()
    if (!txt) return false
    // Exclude only when both the requested type and the candidate type are
    // non-empty AND they disagree. Mirrors sourceContainsEntity's intent.
    const candType = (ent.type ?? '').toLowerCase()
    if (typeLower && candType && candType !== typeLower) {
      return false
    }
    return lowerTerms.has(txt)
  })
}

export function EntityFinding({
  index,
  source,
  highlightTerms,
  selectedTypeLower,
  defaultOpen = false,
  entityLabel,
  reportDedupeKeys
}: Props) {
  const [open, setOpen] = useState(defaultOpen)
  const reportItem = entityLabel != null ? entityFindingSnapshot(source, entityLabel) : null
  const inReport = reportItem != null && (reportDedupeKeys?.has(reportItem.dedupe_key) ?? false)
  const collection = useUiStore((s) => s.selectedCollection)
  const refMeta = referenceMetadataItems(source.reference_metadata)
  const chunkText = (source.chunk_text ?? source.text ?? '').trim()
  const segments = open ? highlightSegments(chunkText, highlightTerms) : []
  const mentions = matchedMentions(source, highlightTerms, selectedTypeLower)
  const previewHref =
    collection && source.file_hash ? sourcePreviewUrl(collection, source.file_hash) : null

  // Always-on locator rows so non-transcript sources (PDF/CSV with empty
  // reference_metadata) still surface the fields a user needs to inspect
  // a finding: file, page/row, score, chunk id, filetype, file hash.
  const locator: Array<{ label: string; value: string }> = []
  if (source.filename) locator.push({ label: 'File', value: source.filename })
  const loc: string[] = []
  if (source.page !== null && source.page !== undefined) loc.push(`page ${source.page}`)
  if (source.row !== null && source.row !== undefined) loc.push(`row ${source.row}`)
  if (loc.length > 0) locator.push({ label: 'Location', value: loc.join(', ') })
  if (source.score !== null && source.score !== undefined) {
    locator.push({ label: 'Score', value: source.score.toFixed(3) })
  }
  if (source.filetype) locator.push({ label: 'Filetype', value: String(source.filetype) })
  if (source.source) locator.push({ label: 'Reader', value: String(source.source) })
  if (source.chunk_id) locator.push({ label: 'Chunk ID', value: source.chunk_id })
  if (source.file_hash) locator.push({ label: 'File hash', value: shortHash(source.file_hash) })

  return (
    <div className="rounded-md border border-border bg-zinc-900">
      <div className="flex items-center gap-1 pr-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex-1 min-w-0 flex items-center justify-between gap-2 px-3 py-2 text-left text-sm"
        >
          <span className="truncate">
            <span className="text-muted-foreground mr-2">Chunk {index}:</span>
            {locationLabel(source)}
          </span>
          <span aria-hidden="true" className="text-muted-foreground text-xs shrink-0">
            {open ? '▾' : '▸'}
          </span>
        </button>
        {reportItem && reportDedupeKeys && <AddToReportButton item={reportItem} inReport={inReport} />}
      </div>
      {open && (
        <div className="border-t border-border px-3 py-3 space-y-3 text-sm">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            {locator.map(({ label, value }) => (
              <div key={label} className="contents">
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="break-words">{value}</dd>
              </div>
            ))}
            {refMeta.map(({ label, value }) => (
              <div key={`ref-${label}`} className="contents">
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="break-words">{value}</dd>
              </div>
            ))}
          </dl>

          {mentions.length > 0 && (
            <div className="text-xs">
              <div className="text-muted-foreground mb-1 uppercase tracking-wide">
                Matched mentions
              </div>
              <ul className="flex flex-wrap gap-1.5">
                {mentions.map((m, i) => (
                  <li
                    key={i}
                    className="inline-flex items-center gap-1 rounded-md border border-border bg-zinc-950 px-2 py-0.5"
                  >
                    <span>{m.text}</span>
                    {m.type && (
                      <span className="text-muted-foreground">· {m.type}</span>
                    )}
                    {typeof m.score === 'number' && (
                      <span className="text-muted-foreground">
                        · {m.score.toFixed(3)}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {chunkText ? (
            <div className="whitespace-pre-wrap leading-6 bg-zinc-950 rounded p-3">
              {segments.map((seg, i) =>
                seg.highlight ? (
                  <mark
                    key={i}
                    className="bg-yellow-300 text-zinc-950 rounded px-0.5"
                  >
                    {seg.text}
                  </mark>
                ) : (
                  <span key={i}>{seg.text}</span>
                )
              )}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              Chunk text unavailable for this record.
            </p>
          )}

          {previewHref && (
            <a
              href={previewHref}
              target="_blank"
              rel="noreferrer"
              className="inline-block text-xs text-blue-400 hover:text-blue-300"
            >
              Open original ↗
            </a>
          )}
        </div>
      )}
    </div>
  )
}
