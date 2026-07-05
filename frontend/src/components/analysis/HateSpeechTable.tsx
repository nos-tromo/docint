import { useRef, useState } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { csvExportHref } from '@/api/collections'
import type { HateSpeechRow } from '@/api/types'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { TranslateControl, type TranslationPayload } from '@/components/common/TranslateControl'
import { hateSpeechSnapshot } from '@/lib/reportSnapshots'
import { cn } from '@/lib/cn'

export type { HateSpeechRow }

// Shared column template for the header row and every body row. Metadata is a
// single column (reason / confidence / chunk id / reference metadata).
const HATE_GRID = '2.5rem 6.5rem minmax(8rem,0.8fr) minmax(9rem,1.1fr) minmax(12rem,1.8fr) 6rem'
const TEXT_CLAMP_CHARS = 240

interface Props {
  rows: HateSpeechRow[]
  isFetching?: boolean
  hasNextPage?: boolean
  onLoadMore?: () => void
  collection: string
  reportDedupeKeys?: Set<string>
}

function locationParts(r: HateSpeechRow): string {
  const parts: string[] = []
  if (r.page_label) parts.push(`page ${r.page_label}`)
  else if (r.page !== null && r.page !== undefined) parts.push(`page ${r.page}`)
  if (r.row !== null && r.row !== undefined) parts.push(`row ${r.row}`)
  return parts.join(', ')
}

/**
 * One flagged chunk rendered as a table row. The former accordion's hidden
 * fields (reason, confidence, chunk id, reference metadata) are flattened into
 * a single Metadata column shown inline, mirroring the entity findings table.
 */
function HateSpeechTableRow({
  row,
  index,
  reportDedupeKeys
}: {
  row: HateSpeechRow
  index: number
  reportDedupeKeys?: Set<string>
}) {
  const [expanded, setExpanded] = useState(false)
  const [translation, setTranslation] = useState<TranslationPayload | null>(null)
  const reportItem = hateSpeechSnapshot(row, translation ?? undefined)
  const inReport = reportDedupeKeys?.has(reportItem.dedupe_key) ?? false
  const refMeta = referenceMetadataItems(row.reference_metadata)
  const chunkText = (row.chunk_text ?? row.text ?? '').trim()
  const source = row.source_ref ?? row.filename ?? 'Unknown source'
  const location = locationParts(row)
  const category = (row.category ?? 'unknown').trim()
  const reason = (row.reason ?? '').trim()
  const canClamp = chunkText.length > TEXT_CLAMP_CHARS

  const metadata: Array<{ label: string; value: string }> = []
  if (reason) metadata.push({ label: 'Reason', value: reason })
  if (row.confidence) metadata.push({ label: 'Confidence', value: String(row.confidence) })
  if (row.chunk_id) metadata.push({ label: 'Chunk ID', value: row.chunk_id })
  for (const item of refMeta) metadata.push(item)

  return (
    <div
      className="group grid items-start gap-3 border-b border-border px-3 py-2.5 text-sm hover:bg-zinc-900/40"
      style={{ gridTemplateColumns: HATE_GRID }}
      data-testid="hate-speech-row"
    >
      <div className="text-xs text-muted-foreground tabular-nums pt-0.5">{index}</div>
      <div className="text-xs font-medium uppercase break-words pt-0.5">{category}</div>
      <div className="min-w-0 space-y-0.5">
        <div className="break-words">{source}</div>
        {location && <div className="text-xs text-muted-foreground">{location}</div>}
      </div>
      <div className="min-w-0">
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
              {chunkText}
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
        {reportDedupeKeys && <AddToReportButton item={reportItem} inReport={inReport} />}
      </div>
    </div>
  )
}

/**
 * Hate-speech findings as a virtualized table — one flagged chunk per row, all
 * secondary fields collapsed into a single Metadata column. Preserves the CSV
 * export and per-row "Add to report" control.
 */
export function HateSpeechTable({
  rows,
  isFetching,
  hasNextPage,
  onLoadMore,
  collection,
  reportDedupeKeys
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 120,
    overscan: 8
  })

  if (!rows.length) {
    return (
      <div className="text-sm text-muted-foreground">
        {isFetching ? 'Loading flagged content…' : 'No flagged content.'}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {rows.length} flagged chunk{rows.length === 1 ? '' : 's'}
          {hasNextPage ? '+' : ''}.
        </p>
        {collection && (
          <a
            href={csvExportHref(collection, 'hate-speech')}
            download
            className="px-3 py-1 rounded-md border border-border text-sm"
          >
            CSV
          </a>
        )}
      </div>
      <div className="rounded-md border border-border overflow-hidden">
        <div
          className="grid gap-3 px-3 py-2 bg-zinc-900 border-b border-border text-[11px] uppercase tracking-wide text-muted-foreground"
          style={{ gridTemplateColumns: HATE_GRID }}
        >
          <span>#</span>
          <span>Category</span>
          <span>Source</span>
          <span>Metadata</span>
          <span>Text</span>
          <span className="text-right">Report</span>
        </div>
        <div
          ref={scrollRef}
          className="max-h-[70vh] overflow-y-auto"
          data-testid="hate-speech-scroll"
        >
          <div className="relative" style={{ height: `${virtualizer.getTotalSize()}px` }}>
            {virtualizer.getVirtualItems().map((vRow) => {
              const r = rows[vRow.index]
              return (
                <div
                  key={r.chunk_id ?? vRow.index}
                  data-index={vRow.index}
                  ref={virtualizer.measureElement}
                  className="absolute left-0 right-0"
                  style={{ transform: `translateY(${vRow.start}px)` }}
                >
                  <HateSpeechTableRow
                    row={r}
                    index={vRow.index + 1}
                    reportDedupeKeys={reportDedupeKeys}
                  />
                </div>
              )
            })}
          </div>
          {hasNextPage && onLoadMore && (
            <div className="flex justify-center py-2">
              <button
                type="button"
                onClick={onLoadMore}
                disabled={isFetching}
                className="px-3 py-1 rounded-md border border-border text-sm disabled:opacity-50"
              >
                {isFetching ? 'Loading…' : 'Load more'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
