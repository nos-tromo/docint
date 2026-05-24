import { useRef, useState } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { csvExportHref } from '@/api/collections'
import type { HateSpeechRow } from '@/api/types'
import { referenceMetadataItems } from '@/lib/referenceMetadata'

export type { HateSpeechRow }

interface Props {
  rows: HateSpeechRow[]
  isFetching?: boolean
  hasNextPage?: boolean
  onLoadMore?: () => void
  collection: string
}

function locationParts(r: HateSpeechRow): string {
  const parts: string[] = []
  if (r.page_label) parts.push(`page ${r.page_label}`)
  else if (r.page !== null && r.page !== undefined) parts.push(`page ${r.page}`)
  if (r.row !== null && r.row !== undefined) parts.push(`row ${r.row}`)
  return parts.join(', ')
}

function HateSpeechRowDetail({ row, index }: { row: HateSpeechRow; index: number }) {
  const [open, setOpen] = useState(false)
  const meta = referenceMetadataItems(row.reference_metadata)
  const chunkText = (row.chunk_text ?? row.text ?? '').trim()
  const source = row.source_ref ?? row.filename ?? 'Unknown source'
  const location = locationParts(row)
  const category = (row.category ?? 'unknown').trim()
  const reason = (row.reason ?? '').trim()
  return (
    <div className="rounded-md border border-border bg-zinc-900">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-start justify-between gap-2 px-3 py-2 text-left text-sm"
      >
        <span className="min-w-0 flex-1">
          <span className="text-muted-foreground mr-2">{index}</span>
          <span className="uppercase text-xs font-medium mr-2">{category}</span>
          {reason && <span className="text-muted-foreground">{reason}</span>}
          <span className="block text-xs text-muted-foreground truncate mt-0.5">
            {source}
            {location && <> · {location}</>}
          </span>
        </span>
        <span aria-hidden="true" className="text-muted-foreground text-xs shrink-0 mt-0.5">
          {open ? '▾' : '▸'}
        </span>
      </button>
      {open && (
        <div className="border-t border-border px-3 py-3 space-y-3 text-sm">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            <dt className="text-muted-foreground">Source</dt>
            <dd className="break-words">{source}</dd>
            {location && (
              <>
                <dt className="text-muted-foreground">Location</dt>
                <dd>{location}</dd>
              </>
            )}
            {row.confidence && (
              <>
                <dt className="text-muted-foreground">Confidence</dt>
                <dd>{row.confidence}</dd>
              </>
            )}
            {row.chunk_id && (
              <>
                <dt className="text-muted-foreground">Chunk ID</dt>
                <dd className="break-all">{row.chunk_id}</dd>
              </>
            )}
            {meta.map(({ label, value }) => (
              <div key={label} className="contents">
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="break-words">{value}</dd>
              </div>
            ))}
          </dl>
          {chunkText ? (
            <div className="whitespace-pre-wrap leading-6 bg-zinc-950 rounded p-3">
              {chunkText}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              Chunk text unavailable for this record.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export function HateSpeechTable({
  rows,
  isFetching,
  hasNextPage,
  onLoadMore,
  collection
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 72,
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
          {hasNextPage ? '+' : ''}. Click a row for the full text and reference metadata.
        </p>
        {collection && (
          <div className="flex gap-2">
            <a
              href={csvExportHref(collection, 'hate-speech')}
              download
              className="px-3 py-1 rounded-md border border-border text-sm"
            >
              CSV
            </a>
          </div>
        )}
      </div>
      <div
        ref={scrollRef}
        className="max-h-[70vh] overflow-y-auto"
        data-testid="hate-speech-scroll"
      >
        <ul
          className="relative"
          style={{ height: `${virtualizer.getTotalSize()}px` }}
        >
          {virtualizer.getVirtualItems().map((vRow) => {
            const r = rows[vRow.index]
            return (
              <li
                key={r.chunk_id ?? vRow.index}
                data-index={vRow.index}
                ref={virtualizer.measureElement}
                className="absolute left-0 right-0 pb-2"
                style={{ transform: `translateY(${vRow.start}px)` }}
              >
                <HateSpeechRowDetail row={r} index={vRow.index + 1} />
              </li>
            )
          })}
        </ul>
        {hasNextPage && onLoadMore && (
          <div className="flex justify-center pt-2">
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
  )
}
