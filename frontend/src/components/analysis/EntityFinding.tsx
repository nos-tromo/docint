import { useState } from 'react'
import type { NerSourceRow } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import { sourcePreviewUrl } from '@/api/ingest'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { highlightSegments } from '@/lib/highlight'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { TranslateControl } from '@/components/common/TranslateControl'
import { entityFindingSnapshot } from '@/lib/reportSnapshots'
import { cn } from '@/lib/cn'

// Length past which the chunk text is clamped behind a "Show more" toggle so
// rows stay scannable without hiding short findings.
const TEXT_CLAMP_CHARS = 240

interface Props {
  index: number
  source: NerSourceRow
  // Lowercase text candidates that identify the picked entity (and its
  // variants); used both for chunk filtering and to highlight matched
  // mentions inside the chunk body.
  highlightTerms: string[]
  selectedTypeLower?: string
  // Report-builder context. When `reportDedupeKeys` is provided (the row is
  // rendered inside a report-aware view), an "Add to report" control shows and
  // `entityLabel` becomes the report's entity column for this chunk.
  entityLabel?: string
  reportDedupeKeys?: Set<string>
  /** Shared CSS grid template so every row aligns with the table header. */
  gridTemplate: string
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

/**
 * One finding (chunk) rendered as a table row. The previously-collapsed
 * locator and reference-metadata fields are flattened into a single "Metadata"
 * cell shown inline, so a long entity's findings are scannable without
 * expanding each one. Only the (optionally long) chunk text stays behind a
 * per-row "Show more" toggle.
 */
export function EntityFinding({
  index,
  source,
  highlightTerms,
  selectedTypeLower,
  entityLabel,
  reportDedupeKeys,
  gridTemplate
}: Props) {
  const [expanded, setExpanded] = useState(false)
  const [translation, setTranslation] = useState<
    { text: string; target_lang: string; model: string } | null
  >(null)
  const reportItem =
    entityLabel != null ? entityFindingSnapshot(source, entityLabel, translation ?? undefined) : null
  const inReport = reportItem != null && (reportDedupeKeys?.has(reportItem.dedupe_key) ?? false)
  const collection = useUiStore((s) => s.selectedCollection)
  const refMeta = referenceMetadataItems(source.reference_metadata)
  const chunkText = (source.chunk_text ?? source.text ?? '').trim()
  const segments = highlightSegments(chunkText, highlightTerms)
  const mentions = matchedMentions(source, highlightTerms, selectedTypeLower)
  const previewHref =
    collection && source.file_hash ? sourcePreviewUrl(collection, source.file_hash) : null

  const locParts: string[] = []
  if (source.page !== null && source.page !== undefined) locParts.push(`page ${source.page}`)
  if (source.row !== null && source.row !== undefined) locParts.push(`row ${source.row}`)

  // Every secondary field collapses into the single Metadata column.
  const metadata: Array<{ label: string; value: string }> = []
  if (source.score !== null && source.score !== undefined) {
    metadata.push({ label: 'Score', value: source.score.toFixed(3) })
  }
  if (source.filetype) metadata.push({ label: 'Filetype', value: String(source.filetype) })
  if (source.source) metadata.push({ label: 'Reader', value: String(source.source) })
  if (source.chunk_id) metadata.push({ label: 'Chunk ID', value: source.chunk_id })
  if (source.file_hash) metadata.push({ label: 'File hash', value: shortHash(source.file_hash) })
  for (const item of refMeta) metadata.push(item)

  const canClamp = chunkText.length > TEXT_CLAMP_CHARS

  return (
    <div
      className="group grid items-start gap-3 border-b border-border px-3 py-2.5 text-sm hover:bg-zinc-900/40"
      style={{ gridTemplateColumns: gridTemplate }}
      data-testid="entity-finding-row"
    >
      <div className="text-xs text-muted-foreground tabular-nums pt-0.5">{index}</div>

      <div className="min-w-0 space-y-1">
        <div className="font-medium break-words">{source.filename || 'Unknown source'}</div>
        {locParts.length > 0 && (
          <div className="text-xs text-muted-foreground">{locParts.join(', ')}</div>
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

      <div className="min-w-0 space-y-1.5">
        {metadata.length > 0 ? (
          <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-xs">
            {metadata.map(({ label, value }) => (
              <div key={label} className="contents">
                <dt className="text-muted-foreground whitespace-nowrap">{label}</dt>
                <dd className="break-words">{value}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <span className="text-xs text-muted-foreground">—</span>
        )}
        {mentions.length > 0 && (
          <ul className="flex flex-wrap gap-1" aria-label="Matched mentions">
            {mentions.map((m, i) => (
              <li
                key={i}
                className="inline-flex items-center gap-1 rounded border border-border bg-zinc-950 px-1.5 py-0.5 text-[11px]"
              >
                <span>{m.text}</span>
                {m.type && <span className="text-muted-foreground">· {m.type}</span>}
                {typeof m.score === 'number' && (
                  <span className="text-muted-foreground">· {m.score.toFixed(3)}</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="min-w-0">
        {chunkText ? (
          <>
            <p
              className={cn(
                'whitespace-pre-wrap leading-6 break-words',
                canClamp && !expanded && 'line-clamp-4'
              )}
            >
              {segments.map((seg, i) =>
                seg.highlight ? (
                  <mark key={i} className="bg-yellow-300 text-zinc-950 rounded px-0.5">
                    {seg.text}
                  </mark>
                ) : (
                  <span key={i}>{seg.text}</span>
                )
              )}
            </p>
            {canClamp && (
              <button
                type="button"
                onClick={() => setExpanded((v) => !v)}
                className="mt-1 text-xs text-blue-400 hover:text-blue-300"
              >
                {expanded ? 'Show less' : 'Show more'}
              </button>
            )}
          </>
        ) : (
          <span className="text-xs text-muted-foreground">Chunk text unavailable.</span>
        )}
      </div>

      <div className="flex items-center justify-end gap-1">
        {chunkText && <TranslateControl text={chunkText} onTranslated={setTranslation} />}
        {reportItem && reportDedupeKeys && <AddToReportButton item={reportItem} inReport={inReport} />}
      </div>
    </div>
  )
}
