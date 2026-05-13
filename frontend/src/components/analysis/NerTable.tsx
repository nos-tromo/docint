import { useMemo, useState } from 'react'
import { downloadCsv, toCsv } from '@/lib/csv'

export interface EntityRow {
  text: string
  type: string
  count: number
}

export function NerTable({ rows }: { rows: EntityRow[] }) {
  const [filter, setFilter] = useState('')
  const [type, setType] = useState('')
  const types = useMemo(() => Array.from(new Set(rows.map((r) => r.type))).sort(), [rows])
  const filtered = rows.filter(
    (r) => (!type || r.type === type) && (!filter || r.text.toLowerCase().includes(filter.toLowerCase()))
  )

  return (
    <div className="space-y-3">
      <div className="flex gap-2 text-sm">
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="filter…"
          className="bg-zinc-900 border border-border rounded-md px-2 py-1 flex-1"
        />
        <select
          value={type}
          onChange={(e) => setType(e.target.value)}
          className="bg-zinc-900 border border-border rounded-md px-2 py-1"
        >
          <option value="">All types</option>
          {types.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => downloadCsv('entities.csv', toCsv(filtered as unknown as Record<string, unknown>[], ['text', 'type', 'count']))}
          className="px-3 py-1 rounded-md border border-border"
        >
          CSV
        </button>
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-muted-foreground">
          <tr>
            <th className="py-2">Text</th>
            <th>Type</th>
            <th className="text-right">Count</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((r, i) => (
            <tr key={i} className="border-t border-border">
              <td className="py-1">{r.text}</td>
              <td>{r.type}</td>
              <td className="text-right">{r.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
