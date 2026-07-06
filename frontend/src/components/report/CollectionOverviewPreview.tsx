import type { CollectionOverviewSnapshot } from '@/api/types'

/** Pages-or-rows cell for a manifest row (em-dash when neither applies). */
function units(doc: { page_count: number; row_count: number | null }): string {
  if (doc.page_count > 0) return String(doc.page_count)
  if (doc.row_count && doc.row_count > 0) return String(doc.row_count)
  return '—'
}

/**
 * In-app preview of a report's frozen document-overview section: the count
 * strip plus the per-document manifest. The authoritative render is server-side
 * (the report exports); this mirrors it for the on-screen preview.
 */
export function CollectionOverviewPreview({ overview }: { overview: CollectionOverviewSnapshot }) {
  return (
    <div className="space-y-2">
      <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground">Document overview</h2>
      <div className="text-xs text-muted-foreground">
        {overview.document_count} documents · {overview.node_count} nodes · {overview.file_types.length} file types ·{' '}
        {overview.entity_types.length} entity types
      </div>
      <div className="overflow-auto rounded-md border border-border">
        <table className="w-full text-xs">
          <thead className="text-muted-foreground">
            <tr>
              <th className="text-left px-2 py-1 font-medium">Document</th>
              <th className="text-left px-2 py-1 font-medium">Type</th>
              <th className="text-right px-2 py-1 font-medium">Pages / rows</th>
              <th className="text-left px-2 py-1 font-medium">Hash</th>
            </tr>
          </thead>
          <tbody>
            {overview.documents.map((d) => (
              <tr key={d.file_hash || d.filename} className="border-t border-border">
                <td className="px-2 py-1 break-all">{d.filename}</td>
                <td className="px-2 py-1">{d.type_label}</td>
                <td className="px-2 py-1 text-right tabular-nums">{units(d)}</td>
                <td className="px-2 py-1 font-mono text-muted-foreground">{(d.file_hash || '—').slice(0, 12)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
