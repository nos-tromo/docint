import { useId, useState } from 'react'
import { DisclosureButton } from '@infra/ui'
import type { Analysis, AnalysisChart, AnalysisSource, AnalysisSummary, AnalysisTable } from '@/api/types'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { useSeedSessionAnalysis, useSessionAnalysis } from '@/hooks/useSessions'
import { useT } from '@/i18n/LanguageContext'
import { isPngDataUri } from '@/lib/analysis'
import { analysisReportItem } from '@/lib/reportSnapshots'
import { useUiStore } from '@/stores/ui'

/** Rows of a table the card shows; the rest are counted. */
const PREVIEW_ROWS = 5

type Translate = ReturnType<typeof useT>

function isAnalysis(entry: Analysis | AnalysisSummary): entry is Analysis {
  return typeof (entry as Analysis).code === 'string'
}

/** Stored JSON, so a field that should be a list is trusted only as one. */
function list<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : []
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/**
 * The first rows of one table, and how many more it has.
 *
 * Every cell is text the sandbox wrote from retrieved passages, so it goes through React as text and nothing else.
 * The scroll area is reachable by keyboard and named by the table's caption: its name, else its number.
 */
function TablePreview({ table, n }: { table: AnalysisTable; n: number }) {
  const t = useT()
  const captionId = useId()
  const columns = list<string>(table.columns)
  const rows = list<string[]>(table.rows)
  const shown = rows.slice(0, PREVIEW_ROWS)
  const rowCount = typeof table.row_count === 'number' ? table.row_count : 0
  // The run may have had more rows than the stored table kept.
  const more = Math.max(rowCount, rows.length) - shown.length
  const cut = table.truncated === true || rows.length < rowCount
  return (
    <div className="space-y-1">
      <div
        role="region"
        aria-labelledby={captionId}
        tabIndex={0}
        className="overflow-x-auto rounded-md border border-border outline-none focus-visible:ring-1 focus-visible:ring-primary"
      >
        <table className="w-full text-xs">
          <caption id={captionId} className="break-words px-2 py-1 text-left text-xs text-muted-foreground">
            {text(table.name) || t('chat.analysis_table_n', { n })}
          </caption>
          <thead className="text-muted-foreground">
            <tr>
              {columns.map((column, c) => (
                // A header may repeat, or be empty (pandas' index column), and the table is never reordered.
                // eslint-disable-next-line @eslint-react/no-array-index-key
                <th key={c} scope="col" className="px-2 py-1 text-left font-medium">
                  {text(column)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row, r) => (
              // eslint-disable-next-line @eslint-react/no-array-index-key
              <tr key={r} className="border-t border-border">
                {list<string>(row).map((cell, c) => (
                  // eslint-disable-next-line @eslint-react/no-array-index-key
                  <td key={c} className="break-words px-2 py-1 align-top">
                    {text(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {more > 0 && (
        <div className="text-xs text-muted-foreground">
          {t(more === 1 ? 'chat.analysis_more_rows_one' : 'chat.analysis_more_rows', { count: more })}
        </div>
      )}
      {cut && <div className="text-xs text-muted-foreground">{t('chat.analysis_table_cut')}</div>}
    </div>
  )
}

/**
 * The charts that are PNG data URIs, each opening the preview dialog from its own bytes.
 *
 * A chart with no name is called by the analysis's title.
 */
function Charts({ charts, title }: { charts: AnalysisChart[]; title: string }) {
  const openPreview = useUiStore((s) => s.openPreview)
  const drawn = charts.filter((chart) => isPngDataUri(chart?.data_uri))
  if (drawn.length === 0) return null
  return (
    <div className="flex flex-wrap gap-3">
      {drawn.map((chart, i) => {
        const name = text(chart.name)
        const label = name || title
        return (
          // Two charts may share a name, and the list is never reordered.
          // eslint-disable-next-line @eslint-react/no-array-index-key
          <figure key={i} className="max-w-xs space-y-1">
            <button
              type="button"
              onClick={() => openPreview({ filename: label, data_uri: chart.data_uri })}
              className="block cursor-pointer overflow-hidden rounded border border-border outline-none hover:border-foreground/40 focus-visible:ring-1 focus-visible:ring-primary"
            >
              <img src={chart.data_uri} alt={label} className="h-40 w-auto max-w-full bg-white object-contain" />
            </button>
            {name && <figcaption className="break-words text-xs text-muted-foreground">{name}</figcaption>}
          </figure>
        )
      })}
    </div>
  )
}

function sourceText(source: AnalysisSource, t: Translate): string {
  const number = typeof source?.citation_index === 'number' ? `[${source.citation_index}] ` : ''
  const file = text(source?.filename) || t('common.unknown_source')
  const page = typeof source?.page === 'number' ? ` (${t('common.loc_page', { page: source.page })})` : ''
  return `${number}${file}${page}`
}

/** What the analysis was computed from and by which workbench job; null when it names neither. */
function provenanceLine(analysis: Analysis, t: Translate): string | null {
  const provenance = analysis.provenance
  const workbench = provenance?.workbench
  const version = text(workbench?.version)
  const job = text(workbench?.job_id)
  const sources = list<AnalysisSource>(provenance?.sources).map((source) => sourceText(source, t))
  if (sources.length > 0) {
    return t('chat.analysis_provenance', { sources: sources.join('; '), version: version || '?', job: job || '?' })
  }
  return version || job ? t('chat.analysis_provenance_workbench', { version: version || '?', job: job || '?' }) : null
}

/** The code behind a disclosure: the whole row is the button, its label is constant and `aria-expanded` says the rest. */
function CodeDisclosure({ code }: { code: string }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const codeId = useId()
  const label = t('chat.analysis_code')
  return (
    <div>
      <DisclosureButton
        expanded={open}
        controls={codeId}
        label={label}
        onClick={() => setOpen((value) => !value)}
        className="h-6 w-full flex-row-reverse justify-between text-xs"
      >
        <span>{label}</span>
      </DisclosureButton>
      {open && (
        <pre
          id={codeId}
          role="region"
          aria-label={label}
          tabIndex={0}
          className="mt-2 max-h-72 overflow-auto rounded-md border border-border bg-background p-2 text-xs outline-none focus-visible:ring-1 focus-visible:ring-primary"
        >
          <code>{code}</code>
        </pre>
      )}
    </div>
  )
}

/**
 * One analysis a turn's `run_python` run kept, shown below the answer: its tables, its charts, the code
 * behind them and what it was computed from.
 *
 * The final frame carries the analysis in full. A turn reloaded from history carries a summary, so the
 * card reads the analysis once through its own route. Adding it to a report sends its id alone, and
 * works from the summary too.
 *
 * Everything in it was written by code that ran over retrieved passages: it is shown as text, and a
 * chart only if it is a PNG data URI.
 *
 * @param props.entry - The analysis, or its summary from history.
 * @param props.index - The number the user sees it by, from 1, and the title of an untitled one.
 * @param props.sessionId - The session it belongs to; a summary is read through it.
 * @param props.inReport - Whether the active report already holds it.
 */
export function AnalysisCard({
  entry,
  index,
  sessionId,
  inReport
}: {
  entry: Analysis | AnalysisSummary
  index: number
  sessionId: string | null
  inReport: boolean
}) {
  const t = useT()
  const titleId = useId()
  const full = isAnalysis(entry)
  useSeedSessionAnalysis(sessionId, full ? entry : null)
  const read = useSessionAnalysis(sessionId, entry.id, !full)
  const analysis = full ? entry : (read.data ?? null)
  const title = text(analysis?.title).trim() || text(entry.title).trim() || t('chat.analysis_label', { n: index })
  const provenance = analysis ? provenanceLine(analysis, t) : null
  return (
    <section
      aria-labelledby={titleId}
      data-testid="analysis-card"
      className="mt-3 space-y-3 rounded-md border border-border bg-background px-3 py-3 text-sm"
    >
      <div className="flex items-start justify-between gap-2">
        <div id={titleId} className="min-w-0 break-words font-medium">
          {title}
        </div>
        <AddToReportButton item={analysisReportItem(entry.id)} inReport={inReport} />
      </div>
      {!analysis && read.isLoading && (
        <p role="status" className="text-xs text-muted-foreground">
          {t('chat.analysis_loading')}
        </p>
      )}
      {!analysis && read.isError && (
        <p role="alert" className="text-xs text-muted-foreground">
          {t('chat.analysis_failed')}
        </p>
      )}
      {!analysis && read.status === 'pending' && read.fetchStatus === 'idle' && (
        <p className="text-xs text-muted-foreground">{t('chat.analysis_no_detail')}</p>
      )}
      {analysis && (
        <>
          {list<AnalysisTable>(analysis.tables).map((table, i) => (
            // eslint-disable-next-line @eslint-react/no-array-index-key
            <TablePreview key={i} table={table} n={i + 1} />
          ))}
          <Charts charts={list<AnalysisChart>(analysis.charts)} title={title} />
          {text(analysis.code) && <CodeDisclosure code={analysis.code} />}
          {provenance && <p className="break-words text-xs text-muted-foreground">{provenance}</p>}
        </>
      )}
    </section>
  )
}
