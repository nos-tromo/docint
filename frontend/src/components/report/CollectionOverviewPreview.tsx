import type { CollectionOverviewSnapshot } from '@/api/types'
import { ReportSection } from '@/components/report/ReportSection'
import { useT } from '@/i18n/LanguageContext'
import type { Strings } from '@/i18n'

/** Pages-or-rows cell for a manifest row (em-dash when neither applies). */
function units(doc: { page_count: number; row_count: number | null }): string {
  if (doc.page_count > 0) return String(doc.page_count)
  if (doc.row_count && doc.row_count > 0) return String(doc.row_count)
  return '—'
}

/**
 * "N noun" via the matching one/other catalog pair. `document` reuses the
 * shared `table.documents_*` pair (same wording as the collection's document
 * count elsewhere in the SPA); the other three are report-scoped keys.
 */
function count(
  n: number,
  base: 'document' | 'node' | 'filetype' | 'entitytype',
  t: (key: keyof Strings, vars?: Record<string, string | number>) => string
): string {
  if (base === 'document') {
    return t(n === 1 ? 'table.documents_one' : 'table.documents_other', { count: n })
  }
  const key = n === 1 ? (`report.count_${base}_one` as keyof Strings) : (`report.count_${base}_other` as keyof Strings)
  return t(key, { count: n })
}

/**
 * In-app preview of a report's frozen document-overview section: the count
 * strip plus the per-document manifest. The authoritative render is server-side
 * (the report exports); this mirrors it for the on-screen preview.
 */
export function CollectionOverviewPreview({ overview }: { overview: CollectionOverviewSnapshot }) {
  const t = useT()
  // One string rather than a row of spans: it is the bar's trailing text, and
  // the separators are what the count-strip test reads out of textContent.
  const countStrip = [
    count(overview.document_count, 'document', t),
    count(overview.node_count, 'node', t),
    count(overview.file_types.length, 'filetype', t),
    count(overview.entity_types.length, 'entitytype', t)
  ].join(' · ')
  return (
    // The manifest is the longest thing in a report and the least often read —
    // sixteen documents pushed the report's own findings off the screen. Its
    // totals ride on the bar, so a folded overview still says how much is
    // behind it, and they keep the bar's accessible name distinct from the
    // "Document overview" checkbox in the metadata row above.
    <ReportSection
      title={t('report.document_overview')}
      count={countStrip}
      defaultOpen={false}
    >
      <div className="max-h-[60vh] overflow-auto rounded-md border border-border">
        <table className="w-full text-xs">
          <thead className="text-muted-foreground">
            <tr>
              <th className="text-left px-2 py-1 font-medium">{t('report.col_document')}</th>
              <th className="text-left px-2 py-1 font-medium">{t('table.col_type')}</th>
              <th className="text-right px-2 py-1 font-medium">{t('report.col_pages_rows')}</th>
              <th className="text-left px-2 py-1 font-medium">{t('table.col_hash')}</th>
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
    </ReportSection>
  )
}
