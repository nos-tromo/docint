import { downloadCsv, toCsv } from '@/lib/csv'

export interface HateSpeechRow {
  filename: string
  page_label?: string | null
  text: string
  score?: number
}

export function HateSpeechTable({ rows }: { rows: HateSpeechRow[] }) {
  if (!rows.length) return <div className="text-sm text-muted-foreground">No flagged content.</div>
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() =>
            downloadCsv(
              'hate-speech.csv',
              toCsv(rows as unknown as Record<string, unknown>[], ['filename', 'page_label', 'score', 'text'])
            )
          }
          className="px-3 py-1 rounded-md border border-border text-sm"
        >
          CSV
        </button>
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-muted-foreground">
          <tr>
            <th className="py-2">File</th>
            <th>Page</th>
            <th>Score</th>
            <th>Text</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-border align-top">
              <td className="py-1">{r.filename}</td>
              <td>{r.page_label ?? ''}</td>
              <td>{r.score?.toFixed(3) ?? ''}</td>
              <td className="whitespace-pre-wrap">{r.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
