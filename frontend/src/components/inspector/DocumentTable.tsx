import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type Column,
  type RowData,
  type SortDirection,
  type SortingState,
  type ColumnDef
} from '@tanstack/react-table'
import { useMemo, useRef, useState, type ReactNode } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { Badge, CopyButton } from '@infra/ui'
import type { DocumentRecord } from '@/api/types'
import { csvExportHref } from '@/api/collections'
import { mimeLabel, shortHash, unitsLabel } from '@/lib/documentFormat'
import { cn } from '@/lib/cn'

// Per-column layout hints consumed by the shared grid renderer below.
declare module '@tanstack/react-table' {
  /* eslint-disable @typescript-eslint/no-unused-vars */
  interface ColumnMeta<TData extends RowData, TValue> {
    align?: 'right'
  }
  /* eslint-enable @typescript-eslint/no-unused-vars */
}

/**
 * One grid template drives both the header row and every body row, so columns
 * align by construction (the previous table-header / flex-body split did not).
 */
const GRID_COLUMNS = 'minmax(0,2.4fr) 72px 92px 72px minmax(0,1.8fr) 96px'

const COLUMNS: ColumnDef<DocumentRecord>[] = [
  {
    accessorKey: 'filename',
    header: 'Filename',
    cell: (c) => (
      <span className="block truncate font-mono text-[13px]" title={c.getValue<string>()}>
        {c.getValue<string>()}
      </span>
    )
  },
  {
    id: 'type',
    accessorFn: (r) => mimeLabel(r.mimetype),
    header: 'Type',
    cell: (c) => <span className="text-muted-foreground">{c.getValue<string>()}</span>
  },
  {
    id: 'units',
    accessorFn: (r) => unitsLabel(r).sort,
    header: 'Units',
    meta: { align: 'right' },
    cell: (c) => {
      const units = unitsLabel(c.row.original)
      return (
        <span className={cn('tabular-nums', units.text === '—' && 'text-muted-foreground')}>
          {units.text}
        </span>
      )
    }
  },
  {
    accessorKey: 'node_count',
    header: 'Nodes',
    meta: { align: 'right' },
    cell: (c) => <span className="tabular-nums">{c.getValue<number | undefined>() ?? 0}</span>
  },
  {
    id: 'entity_types',
    accessorFn: (r) => (r.entity_types ?? []).join(', '),
    header: 'Entities',
    enableSorting: false,
    cell: (c) => <EntityBadges types={c.row.original.entity_types ?? []} />
  },
  {
    accessorKey: 'file_hash',
    header: 'Hash',
    enableSorting: false,
    cell: (c) => {
      const hash = c.getValue<string>()
      return (
        <span className="flex items-center gap-1">
          <span className="font-mono text-xs text-muted-foreground">{shortHash(hash)}</span>
          <CopyButton text={hash} label={`Copy hash for ${c.row.original.filename}`} className="h-6 w-6" />
        </span>
      )
    }
  }
]

interface Props {
  docs: DocumentRecord[]
  isFetching?: boolean
  hasNextPage?: boolean
  onLoadMore?: () => void
  collection: string
}

/** Entity-type chips: sorted, first four, with a `+N` overflow marker. */
function EntityBadges({ types }: { types: string[] }) {
  if (types.length === 0) return <span className="text-muted-foreground">—</span>
  const sorted = [...types].sort((a, b) => a.localeCompare(b))
  const shown = sorted.slice(0, 4)
  const extra = sorted.length - shown.length
  return (
    <div className="flex flex-wrap items-center gap-1">
      {shown.map((t) => (
        <Badge key={t} variant="neutral">
          {t}
        </Badge>
      ))}
      {extra > 0 && (
        <span className="text-xs text-muted-foreground" title={sorted.join(', ')}>
          +{extra}
        </span>
      )}
    </div>
  )
}

/** Sort direction indicator; a faint dot when the column is sortable but unsorted. */
function SortGlyph({ dir }: { dir: false | SortDirection }) {
  if (dir === 'asc') return <span aria-hidden>↑</span>
  if (dir === 'desc') return <span aria-hidden>↓</span>
  return <span aria-hidden className="opacity-0 transition-opacity group-hover:opacity-40">↕</span>
}

function HeaderCell({ column, children }: { column: Column<DocumentRecord>; children: ReactNode }) {
  const align = column.columnDef.meta?.align
  if (!column.getCanSort()) {
    return <div className={cn('min-w-0', align === 'right' && 'text-right')}>{children}</div>
  }
  return (
    <div className={cn('min-w-0', align === 'right' && 'text-right')}>
      <button
        type="button"
        onClick={column.getToggleSortingHandler()}
        className={cn(
          'group inline-flex items-center gap-1 uppercase hover:text-foreground',
          align === 'right' && 'flex-row-reverse'
        )}
      >
        {children}
        <SortGlyph dir={column.getIsSorted()} />
      </button>
    </div>
  )
}

/** Read-only overview of a collection's documents, one aligned row each. */
export function DocumentTable({ docs, isFetching, hasNextPage, onLoadMore, collection }: Props) {
  const [sorting, setSorting] = useState<SortingState>([])
  const data = useMemo(() => docs, [docs])
  const table = useReactTable({
    data,
    columns: COLUMNS,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel()
  })

  const rows = table.getRowModel().rows
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 44,
    overscan: 12
  })

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {docs.length} document{docs.length === 1 ? '' : 's'}
          {hasNextPage ? '+' : ''}
          {isFetching ? ' · loading…' : ''}
        </p>
        {collection && (
          <a
            href={csvExportHref(collection, 'documents')}
            download
            className="rounded-md border border-border px-3 py-1 text-sm hover:bg-white/5"
          >
            Export CSV
          </a>
        )}
      </div>

      {docs.length === 0 && !isFetching ? (
        <div className="rounded-lg border border-dashed border-border bg-zinc-900/50 p-10 text-center">
          <p className="text-sm text-muted-foreground">No documents in this collection yet.</p>
          <p className="mt-1 text-xs text-muted-foreground">Ingest files to see them here.</p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-border bg-zinc-900">
          <div
            ref={scrollRef}
            className="max-h-[70vh] overflow-auto"
            data-testid="documents-scroll"
            role="table"
            aria-label="Documents"
          >
            <div role="rowgroup">
              {table.getHeaderGroups().map((hg) => (
                <div
                  key={hg.id}
                  role="row"
                  className="sticky top-0 z-10 grid gap-x-4 border-b border-border bg-zinc-900 px-4 py-2.5 text-xs font-medium uppercase tracking-wide text-muted-foreground"
                  style={{ gridTemplateColumns: GRID_COLUMNS }}
                >
                  {hg.headers.map((h) => (
                    <HeaderCell key={h.id} column={h.column}>
                      {flexRender(h.column.columnDef.header, h.getContext())}
                    </HeaderCell>
                  ))}
                </div>
              ))}
            </div>

            <div
              role="rowgroup"
              className="relative"
              style={{ height: `${virtualizer.getTotalSize()}px` }}
            >
              {virtualizer.getVirtualItems().map((vRow) => {
                const row = rows[vRow.index]
                return (
                  <div
                    key={row.id}
                    role="row"
                    data-index={vRow.index}
                    ref={virtualizer.measureElement}
                    className="absolute left-0 right-0 grid items-center gap-x-4 border-b border-border/60 px-4 py-2 text-sm hover:bg-white/5"
                    style={{ gridTemplateColumns: GRID_COLUMNS, transform: `translateY(${vRow.start}px)` }}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <div
                        key={cell.id}
                        role="cell"
                        className={cn(
                          'min-w-0',
                          cell.column.columnDef.meta?.align === 'right' && 'text-right'
                        )}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>
          </div>

          {hasNextPage && onLoadMore && (
            <div className="flex justify-center border-t border-border p-2">
              <button
                type="button"
                onClick={onLoadMore}
                disabled={isFetching}
                className="rounded-md border border-border px-3 py-1 text-sm hover:bg-white/5 disabled:opacity-50"
              >
                {isFetching ? 'Loading…' : 'Load more'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
