import { Button } from '@infra/ui'
import { reportExportHref } from '@/api/reports'
import type { ArtifactType, ReportExportFormat, ReportItem } from '@/api/types'
import { CollectionOverviewPreview } from '@/components/report/CollectionOverviewPreview'
import {
  useCreateReport,
  useDeleteReport,
  useRefreshCollectionOverview,
  useRemoveReportItem,
  useReorderReportItems,
  useReport,
  useReports,
  useUpdateReport,
  useUpdateReportItem
} from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { cn } from '@/lib/cn'

// Summaries lead the document, matching the server renderer's SECTION_ORDER.
const SECTIONS: Array<{ type: ArtifactType; label: string }> = [
  { type: 'summary', label: 'Summaries' },
  { type: 'chat_answer', label: 'Chat answers' },
  { type: 'entity_finding', label: 'Entity findings' },
  { type: 'hate_speech_finding', label: 'Hate-speech findings' }
]

const EXPORTS: Array<{ format: ReportExportFormat; label: string; view?: boolean }> = [
  { format: 'pdf', label: 'PDF' },
  { format: 'md', label: 'Markdown' },
  { format: 'html', label: 'HTML', view: true },
  { format: 'zip', label: 'CSV' },
  { format: 'json', label: 'JSON' }
]

function str(snapshot: Record<string, unknown>, key: string): string {
  const v = snapshot[key]
  if (v == null) return ''
  return typeof v === 'string' ? v : String(v)
}

function truncate(text: string, n = 240): string {
  const t = text.trim()
  return t.length > n ? `${t.slice(0, n).trimEnd()} …` : t
}

function itemTitle(item: ReportItem): string {
  const s = item.snapshot
  switch (item.artifact_type) {
    case 'chat_answer':
      return truncate(str(s, 'user_text') || 'Chat answer', 120)
    case 'entity_finding':
      return str(s, 'entity_label') || 'Entity finding'
    case 'hate_speech_finding': {
      const cat = str(s, 'category')
      const conf = str(s, 'confidence')
      return [cat, conf && `(${conf})`].filter(Boolean).join(' ') || 'Hate-speech finding'
    }
    default:
      return str(s, 'collection') || 'Summary'
  }
}

function itemBody(item: ReportItem): string {
  const s = item.snapshot
  switch (item.artifact_type) {
    case 'chat_answer':
      return truncate(str(s, 'model_response'))
    case 'entity_finding':
      return truncate(str(s, 'chunk_text'))
    case 'hate_speech_finding':
      return truncate(str(s, 'reason') || str(s, 'chunk_text'))
    default:
      return truncate(str(s, 'text'))
  }
}

function itemSource(item: ReportItem): string {
  const s = item.snapshot
  const file = str(s, 'filename')
  const loc = str(s, 'page') ? `page ${str(s, 'page')}` : str(s, 'row') ? `row ${str(s, 'row')}` : ''
  return [file, loc].filter(Boolean).join(' · ')
}

export function Report() {
  const collection = useUiStore((s) => s.selectedCollection)
  const activeReportId = useReportStore((s) => s.activeReportId)
  const setActiveReportId = useReportStore((s) => s.setActiveReportId)

  const reports = useReports()
  const active = useReport(activeReportId)
  const createReport = useCreateReport()
  const updateReport = useUpdateReport()
  const deleteReport = useDeleteReport()
  const removeItem = useRemoveReportItem()
  const reorderItems = useReorderReportItems()
  const updateItem = useUpdateReportItem()
  const refreshOverview = useRefreshCollectionOverview()

  const report = active.data
  const items = report?.items ?? []

  const onCreate = async () => {
    try {
      const created = await createReport.mutateAsync({
        title: 'Untitled report',
        collection_name: collection ?? undefined
      })
      setActiveReportId(created.id)
    } catch {
      /* surfaced via createReport.isError below */
    }
  }

  const onDelete = (id: number) => {
    if (!confirm('Delete this report? This cannot be undone.')) return
    deleteReport.mutate(id, {
      onSuccess: () => {
        if (activeReportId === id) setActiveReportId(null)
      }
    })
  }

  // Swap an item with its same-type neighbor in display order, then persist the
  // full global id order (the renderer groups by type but orders by position).
  const move = (item: ReportItem, dir: -1 | 1) => {
    if (!report) return
    const section = items.filter((i) => i.artifact_type === item.artifact_type)
    const pos = section.findIndex((i) => i.id === item.id)
    const neighbor = section[pos + dir]
    if (!neighbor) return
    const ids = items.map((i) => i.id)
    const a = ids.indexOf(item.id)
    const b = ids.indexOf(neighbor.id)
    ;[ids[a], ids[b]] = [ids[b], ids[a]]
    reorderItems.mutate({ reportId: report.id, itemIds: ids })
  }

  const reportList = reports.data?.reports ?? []

  return (
    <div className="p-8 grid grid-cols-[18rem_1fr] gap-6 h-full">
      <aside className="flex flex-col gap-3 min-h-0">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">Reports</h1>
          <Button variant="primary" size="sm" onClick={onCreate} disabled={createReport.isPending}>
            + New
          </Button>
        </div>
        {createReport.isError && (
          <p className="text-xs text-red-400">Couldn’t create the report — is the backend reachable?</p>
        )}
        <ul className="flex-1 overflow-auto space-y-1">
          {reports.isError ? (
            <li className="px-2 py-1 text-sm text-red-400">Failed to load reports.</li>
          ) : reportList.length === 0 ? (
            <li className="px-2 py-1 text-sm text-muted-foreground">No reports yet.</li>
          ) : null}
          {reportList.map((r) => {
            const isActive = r.id === activeReportId
            return (
              <li key={r.id} className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setActiveReportId(r.id)}
                  className={cn(
                    'flex-1 text-left text-sm px-2 py-1.5 rounded-md truncate',
                    isActive ? 'bg-primary/10 text-primary' : 'hover:bg-zinc-900'
                  )}
                  title={r.title}
                >
                  {r.title}
                  <span className="ml-1 text-xs text-muted-foreground">({r.item_count})</span>
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(r.id)}
                  className="text-xs text-zinc-500 hover:text-red-400 px-1"
                  aria-label="Delete report"
                >
                  ×
                </button>
              </li>
            )
          })}
        </ul>
      </aside>

      <section className="flex flex-col min-h-0">
        {!activeReportId ? (
          <p className="text-sm text-muted-foreground">
            Select a report, or create one and add artifacts from Chat or Analysis.
          </p>
        ) : active.isError ? (
          <p className="text-sm text-muted-foreground">
            This report could not be loaded.{' '}
            <button type="button" className="underline" onClick={() => setActiveReportId(null)}>
              Clear selection
            </button>
          </p>
        ) : !report ? (
          <p className="text-sm text-muted-foreground">Loading report…</p>
        ) : (
          <>
            <div className="flex items-start justify-between gap-3 mb-4">
              <div className="min-w-0 flex-1 space-y-1.5">
                <input
                  key={report.id}
                  defaultValue={report.title}
                  onBlur={(e) => {
                    const title = e.target.value.trim()
                    if (title && title !== report.title) {
                      updateReport.mutate({ id: report.id, title })
                    }
                  }}
                  className="w-full bg-transparent text-2xl font-semibold outline-hidden border-b border-transparent focus:border-border"
                  aria-label="Report title"
                />
                <div className="flex flex-wrap gap-x-6 gap-y-1.5">
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="uppercase tracking-wide">Operator</span>
                    <input
                      key={`op-${report.id}`}
                      defaultValue={report.operator ?? ''}
                      placeholder="Bearbeiter/-in"
                      onBlur={(e) => {
                        const operator = e.target.value
                        if (operator !== (report.operator ?? '')) {
                          updateReport.mutate({ id: report.id, operator })
                        }
                      }}
                      className="bg-zinc-950 border border-border rounded px-2 py-1 text-xs text-foreground"
                      aria-label="Operator (Bearbeiter/-in)"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="uppercase tracking-wide">File reference</span>
                    <input
                      key={`ref-${report.id}`}
                      defaultValue={report.reference_number ?? ''}
                      placeholder="Aktenzeichen"
                      onBlur={(e) => {
                        const reference_number = e.target.value
                        if (reference_number !== (report.reference_number ?? '')) {
                          updateReport.mutate({ id: report.id, reference_number })
                        }
                      }}
                      className="bg-zinc-950 border border-border rounded px-2 py-1 text-xs text-foreground"
                      aria-label="File reference (Aktenzeichen)"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={report.show_toc ?? true}
                      onChange={(e) => updateReport.mutate({ id: report.id, show_toc: e.target.checked })}
                      className="accent-primary"
                      aria-label="Table of contents"
                    />
                    <span className="uppercase tracking-wide">Table of contents</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={report.show_collection_overview ?? true}
                      onChange={(e) =>
                        updateReport.mutate({ id: report.id, show_collection_overview: e.target.checked })
                      }
                      className="accent-primary"
                      aria-label="Document overview"
                    />
                    <span className="uppercase tracking-wide">Document overview</span>
                  </label>
                  {(report.show_collection_overview ?? true) && (
                    <button
                      type="button"
                      onClick={() => refreshOverview.mutate(report.id)}
                      disabled={refreshOverview.isPending}
                      className="text-xs text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground disabled:opacity-50"
                    >
                      {refreshOverview.isPending
                        ? 'Refreshing…'
                        : report.collection_overview?.captured_at
                          ? `Refresh overview (captured ${report.collection_overview.captured_at.slice(0, 10)})`
                          : 'Capture overview'}
                    </button>
                  )}
                </div>
              </div>

              {/* Export: a single Download button; formats expand on hover/focus. */}
              <div className="relative group shrink-0">
                <button
                  type="button"
                  className="px-3 py-1 rounded-md border border-border text-sm hover:bg-zinc-900"
                  aria-haspopup="menu"
                >
                  Download ▾
                </button>
                <div className="absolute right-0 top-full z-10 hidden pt-1 group-hover:block group-focus-within:block">
                  <div className="flex flex-col min-w-[11rem] rounded-md border border-border bg-zinc-950 p-1 shadow-lg">
                    {EXPORTS.map((e) => (
                      <a
                        key={e.format}
                        href={reportExportHref(report.id, e.format)}
                        {...(e.view ? { target: '_blank', rel: 'noreferrer' } : { download: true })}
                        className="block rounded px-3 py-1.5 text-sm hover:bg-zinc-800 whitespace-nowrap"
                        title={e.view ? 'Open in a new tab' : `Download ${e.label}`}
                      >
                        {e.label}
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {items.length === 0 && !report.collection_overview ? (
              <p className="text-sm text-muted-foreground">
                This report is empty. Use the “+ Report” control on a chat answer, entity finding, or
                hate-speech finding to add it here.
              </p>
            ) : (
              <div className="flex-1 overflow-auto space-y-6 pr-2">
                {SECTIONS.map(({ type, label }) => {
                  const sectionItems = items.filter((i) => i.artifact_type === type)
                  if (sectionItems.length === 0) return null
                  return (
                    <div key={type} className="space-y-2">
                      <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground">
                        {label} ({sectionItems.length})
                      </h2>
                      {sectionItems.map((item, si) => (
                        <div key={item.id} className="rounded-md border border-border bg-zinc-900 p-3 space-y-2">
                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0">
                              <div className="font-medium text-sm break-words">{itemTitle(item)}</div>
                              {itemSource(item) && (
                                <div className="text-xs text-muted-foreground">{itemSource(item)}</div>
                              )}
                            </div>
                            <div className="flex items-center gap-1 shrink-0">
                              <button
                                type="button"
                                onClick={() => move(item, -1)}
                                disabled={si === 0}
                                className="px-1.5 text-muted-foreground hover:text-foreground disabled:opacity-30"
                                aria-label="Move up"
                              >
                                ↑
                              </button>
                              <button
                                type="button"
                                onClick={() => move(item, 1)}
                                disabled={si === sectionItems.length - 1}
                                className="px-1.5 text-muted-foreground hover:text-foreground disabled:opacity-30"
                                aria-label="Move down"
                              >
                                ↓
                              </button>
                              <button
                                type="button"
                                onClick={() => removeItem.mutate({ reportId: report.id, itemId: item.id })}
                                className="px-1.5 text-zinc-500 hover:text-red-400"
                                aria-label="Remove item"
                              >
                                ×
                              </button>
                            </div>
                          </div>
                          {itemBody(item) && (
                            <p className="text-sm text-muted-foreground whitespace-pre-wrap break-words">
                              {itemBody(item)}
                            </p>
                          )}
                          <input
                            key={`note-${item.id}`}
                            defaultValue={item.note ?? ''}
                            placeholder="Add a note…"
                            onBlur={(e) => {
                              const note = e.target.value
                              if (note !== (item.note ?? '')) {
                                updateItem.mutate({ reportId: report.id, itemId: item.id, note: note || null })
                              }
                            }}
                            className="w-full bg-zinc-950 border border-border rounded px-2 py-1 text-xs"
                          />
                        </div>
                      ))}
                    </div>
                  )
                })}
                {(report.show_collection_overview ?? true) &&
                  report.collection_overview &&
                  report.collection_overview.documents.length > 0 && (
                    <CollectionOverviewPreview overview={report.collection_overview} />
                  )}
              </div>
            )}
          </>
        )}
      </section>
    </div>
  )
}
