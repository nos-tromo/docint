import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
  type ColumnDef
} from '@tanstack/react-table'
import { useMemo, useState } from 'react'
import type { DocumentRecord } from '@/api/types'
import { downloadCsv, toCsv } from '@/lib/csv'

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

export function DocumentTable({ docs }: { docs: DocumentRecord[] }) {
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
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() =>
            downloadCsv(
              'documents.csv',
              toCsv(
                docs as unknown as Record<string, unknown>[],
                ['filename', 'mimetype', 'page_count', 'row_count', 'node_count', 'file_hash']
              )
            )
          }
          className="px-3 py-1 rounded-md border border-border text-sm"
        >
          CSV
        </button>
      </div>
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
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-t border-border">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="py-1 align-top">
                  {flexRender(
                    cell.column.columnDef.cell ?? (() => cell.getValue()),
                    cell.getContext()
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
