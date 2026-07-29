import { useState } from 'react'
import type { NerSourceRow } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import { sourcePreviewUrl } from '@/api/ingest'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { highlightSegments } from '@/lib/highlight'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { useTranslatable, type TranslationPayload } from '@/hooks/useTranslatable'
import { TranslateToggle } from '@/components/common/TranslateToggle'
import { ClampedText } from '@/components/common/ClampedText'
import { entityFindingSnapshot } from '@/lib/reportSnapshots'
import { useT } from '@/i18n/LanguageContext'

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
  const t = useT()
  const [translation, setTranslation] = useState<TranslationPayload | null>(null)
  const reportItem =
    entityLabel != null ? entityFindingSnapshot(source, entityLabel, translation ?? undefined) : null
  const inReport = reportItem != null && (reportDedupeKeys?.has(reportItem.dedupe_key) ?? false)
  const collection = useUiStore((s) => s.selectedCollection)
  const refMeta = referenceMetadataItems(source.reference_metadata, {}, t)
  const chunkText = (source.chunk_text ?? source.text ?? '').trim()
  const segments = highlightSegments(chunkText, highlightTerms)
  const translationState = useTranslatable(chunkText, setTranslation)
  const mentions = matchedMentions(source, highlightTerms, selectedTypeLower)
  const previewHref =
    collection && source.file_hash ? sourcePreviewUrl(collection, source.file_hash) : null

  const locParts: string[] = []
  if (source.page !== null && source.page !== undefined) {
    locParts.push(t('common.loc_page', { page: source.page }))
  }
  if (source.row !== null && source.row !== undefined) {
    locParts.push(t('common.loc_row', { row: source.row }))
  }

  // Every secondary field collapses into the single Metadata column.
  const metadata: Array<{ label: string; value: string }> = []
  if (source.score !== null && source.score !== undefined) {
    metadata.push({ label: t('common.meta_score'), value: source.score.toFixed(3) })
  }
  if (source.filetype) metadata.push({ label: t('common.meta_filetype'), value: String(source.filetype) })
  if (source.source) metadata.push({ label: t('common.meta_reader'), value: String(source.source) })
  if (source.chunk_id) metadata.push({ label: t('common.meta_chunk_id'), value: source.chunk_id })
  if (source.file_hash) {
    metadata.push({ label: t('common.meta_file_hash'), value: shortHash(source.file_hash) })
  }
  for (const item of refMeta) metadata.push(item)

  return (
    <div
      className="group grid items-start gap-3 border-b border-border px-3 py-2.5 text-sm hover:bg-muted/40"
      style={{ gridTemplateColumns: gridTemplate }}
      data-testid="entity-finding-row"
    >
      <div className="text-xs text-muted-foreground tabular-nums pt-0.5">{index}</div>

      <div className="min-w-0 space-y-1">
        <div className="font-medium break-words">{source.filename || t('common.unknown_source')}</div>
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
            {t('chat.open_original')}
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
          <ul className="flex flex-wrap gap-1" aria-label={t('entities.matched_mentions_aria')}>
            {mentions.map((m, i) => (
              <li
                key={i}
                className="inline-flex items-center gap-1 rounded border border-border bg-muted px-1.5 py-0.5 text-[11px]"
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
            {translationState.shown && (
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                {t('common.translation')}
              </div>
            )}
            <ClampedText length={(translationState.translation ?? chunkText).length}>
              {translationState.translation ??
                segments.map((seg, i) =>
                  seg.highlight ? (
                    <mark key={i} className="bg-yellow-300 text-zinc-950 rounded px-0.5">
                      {seg.text}
                    </mark>
                  ) : (
                    <span key={i}>{seg.text}</span>
                  )
                )}
            </ClampedText>
            {translationState.failed && (
              <div className="mt-1 text-[11px] text-muted-foreground">
                {t('common.translation_unavailable')}
              </div>
            )}
          </>
        ) : (
          <span className="text-xs text-muted-foreground">{t('common.chunk_text_unavailable')}</span>
        )}
      </div>

      <div className="flex items-center justify-end gap-1">
        {chunkText && (
          <TranslateToggle
            shown={translationState.shown}
            busy={translationState.busy}
            onClick={translationState.toggle}
          />
        )}
        {reportItem && reportDedupeKeys && <AddToReportButton item={reportItem} inReport={inReport} />}
      </div>
    </div>
  )
}
