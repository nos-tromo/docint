import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
  type ColumnDef
} from '@tanstack/react-table'
import { useMemo, useRef, useState } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { DocumentRecord } from '@/api/types'
import { csvExportHref } from '@/api/collections'

const COLUMNS: ColumnDef<DocumentRecord>[] = [
  { accessorKey: 'filename', header: 'Filename' },
  { accessorKey: 'mimetype', header: 'MIME' },
  {
    accessorFn: (r) => r.page_count ?? r.row_count ?? 0,
    id: 'units',
    header: 'Pages/Rows'
  },
  { accessorKey: 'node_count', header: 'Nodes' },
  {
    accessorFn: (r) => (r.entity_types ?? []).join(', '),
    id: 'entity_types',
    header: 'Entity types'
  },
  { accessorKey: 'file_hash', header: 'Hash' }
]

interface Props {
  docs: DocumentRecord[]
  isFetching?: boolean
  hasNextPage?: boolean
  onLoadMore?: () => void
  collection: string
}

export function DocumentTable({
  docs,
  isFetching,
  hasNextPage,
  onLoadMore,
  collection
}: Props) {
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
    estimateSize: () => 36,
    overscan: 12
  })

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <p className="text-sm text-muted-foreground">
          {docs.length} document{docs.length === 1 ? '' : 's'}
          {hasNextPage ? '+' : ''}
          {isFetching ? ' (loading…)' : ''}
        </p>
        {collection && (
          <a
            href={csvExportHref(collection, 'documents')}
            download
            className="px-3 py-1 rounded-md border border-border text-sm"
          >
            CSV
          </a>
        )}
      </div>
      <div
        ref={scrollRef}
        className="max-h-[70vh] overflow-y-auto"
        data-testid="documents-scroll"
      >
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-muted-foreground">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((h) => (
                  <th
                    key={h.id}
                    onClick={h.column.getToggleSortingHandler()}
                    className="cursor-pointer py-2 select-none"
                  >
                    {flexRender(h.column.columnDef.header, h.getContext())}
                    {h.column.getIsSorted() === 'asc' && ' ↑'}
                    {h.column.getIsSorted() === 'desc' && ' ↓'}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody
            className="relative block"
            style={{ height: `${virtualizer.getTotalSize()}px` }}
          >
            {virtualizer.getVirtualItems().map((vRow) => {
              const row = rows[vRow.index]
              return (
                <tr
                  key={row.id}
                  data-index={vRow.index}
                  ref={virtualizer.measureElement}
                  className="border-t border-border absolute left-0 right-0 flex"
                  style={{ transform: `translateY(${vRow.start}px)` }}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="py-1 align-top flex-1 px-2">
                      {flexRender(
                        cell.column.columnDef.cell ?? (() => cell.getValue()),
                        cell.getContext()
                      )}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
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
