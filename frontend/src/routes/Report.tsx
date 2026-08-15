import {
  ChevronDownIcon,
  DeleteButton,
  DownloadButton,
  MoveDownButton,
  MoveUpButton,
  NewButton,
  RemoveButton
} from '@infra/ui'
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
import { useWhoami } from '@/hooks/useWhoami'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'
import type { Strings } from '@/i18n'
import { hateCategoryLabel } from '@/lib/hateCategoryLabel'

type Translate = (key: keyof Strings, vars?: Record<string, string | number>) => string

// Summaries lead the document, matching the server renderer's SECTION_ORDER.
function sections(t: Translate): Array<{ type: ArtifactType; label: string }> {
  return [
    { type: 'summary', label: t('report.section_summaries') },
    { type: 'chat_answer', label: t('report.section_chat_answers') },
    { type: 'entity_finding', label: t('report.section_entity_findings') },
    { type: 'hate_speech_finding', label: t('report.section_hate_findings') }
  ]
}

function exportFormats(t: Translate): Array<{ format: ReportExportFormat; label: string; view?: boolean }> {
  return [
    { format: 'pdf', label: t('report.format_pdf') },
    { format: 'md', label: t('report.format_markdown') },
    { format: 'html', label: t('report.format_html'), view: true },
    { format: 'zip', label: t('report.format_csv') },
    { format: 'json', label: t('report.format_json') }
  ]
}

function str(snapshot: Record<string, unknown>, key: string): string {
  const v = snapshot[key]
  if (v == null) return ''
  return typeof v === 'string' ? v : String(v)
}

function truncate(text: string, n = 240): string {
  const t = text.trim()
  return t.length > n ? `${t.slice(0, n).trimEnd()} …` : t
}

function itemTitle(item: ReportItem, t: Translate): string {
  const s = item.snapshot
  switch (item.artifact_type) {
    case 'chat_answer':
      return truncate(str(s, 'user_text') || t('report.default_chat_answer'), 120)
    case 'entity_finding':
      return str(s, 'entity_label') || t('report.default_entity_finding')
    case 'hate_speech_finding': {
      // `category` is a frozen protocol value from the same fixed enum as
      // HateSpeechTable's — mapped through the same shared label lookup so a
      // category reads identically in Analysis and in Report (e.g. German
      // "Rasse", not "race"). `confidence` stays verbatim, mirroring
      // HateSpeechTable's treatment of that field.
      const cat = str(s, 'category')
      const conf = str(s, 'confidence')
      const label = cat ? hateCategoryLabel(cat, t) : ''
      return [label, conf && `(${conf})`].filter(Boolean).join(' ') || t('report.default_hate_finding')
    }
    default:
      return str(s, 'collection') || t('report.default_summary')
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

function itemSource(item: ReportItem, t: Translate): string {
  const s = item.snapshot
  const file = str(s, 'filename')
  const loc = str(s, 'page')
    ? t('common.loc_page', { page: str(s, 'page') })
    : str(s, 'row')
      ? t('common.loc_row', { row: str(s, 'row') })
      : ''
  return [file, loc].filter(Boolean).join(' · ')
}

export function Report() {
  const t = useT()
  const collection = useUiStore((s) => s.selectedCollection)
  const activeReportId = useReportStore((s) => s.activeReportId)
  const setActiveReportId = useReportStore((s) => s.setActiveReportId)

  const whoami = useWhoami()
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

  // Single source of truth for the overview preview: the snapshot to render, or
  // null. Both the empty-state guard and the render below key off this, so the
  // guard and the render can never diverge — an overview that is toggled off or
  // has no documents falls through to the "empty" message, never a blank area.
  const overviewSnapshot = report?.collection_overview ?? null
  const overviewToShow =
    (report?.show_collection_overview ?? true) && overviewSnapshot && (overviewSnapshot.documents?.length ?? 0) > 0
      ? overviewSnapshot
      : null

  const onCreate = async () => {
    try {
      const created = await createReport.mutateAsync({
        title: t('report.untitled_title'),
        collection_name: collection ?? undefined,
        // Create-time default only: the operator field stays editable, and an
        // unknown identity (dev without the gateway, or a failed /whoami) must
        // leave it empty rather than guess.
        operator: whoami.data?.display_name ?? whoami.data?.username
      })
      setActiveReportId(created.id)
    } catch {
      /* surfaced via createReport.isError below */
    }
  }

  const onDelete = (id: number) => {
    if (!confirm(t('report.delete_confirm'))) return
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
          <h1 className="text-xl font-semibold">{t('report.title')}</h1>
          <NewButton
            label={t('common.new_report')}
            onClick={onCreate}
            busy={createReport.isPending}
            // Also gated while /whoami is in flight: creating before the
            // identity resolves would store a report with no operator.
            disabled={whoami.isLoading}
          />
        </div>
        {createReport.isError && (
          <p className="text-xs text-[var(--status-red-fg)]">{t('report.create_error')}</p>
        )}
        <ul className="flex-1 overflow-auto space-y-1">
          {reports.isError ? (
            <li className="px-2 py-1 text-sm text-red-400">{t('report.load_error')}</li>
          ) : reportList.length === 0 ? (
            <li className="px-2 py-1 text-sm text-muted-foreground">{t('report.empty_list')}</li>
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
                    isActive ? 'bg-primary/10 text-primary' : 'hover:bg-muted'
                  )}
                  title={r.title}
                >
                  {r.title}
                  <span className="ml-1 text-xs text-muted-foreground">({r.item_count})</span>
                </button>
                <DeleteButton
                  label={t('report.delete_aria')}
                  onClick={() => onDelete(r.id)}
                  className="h-7"
                />
              </li>
            )
          })}
        </ul>
      </aside>

      <section className="flex flex-col min-h-0">
        {!activeReportId ? (
          <p className="text-sm text-muted-foreground">{t('report.select_hint')}</p>
        ) : active.isError ? (
          <p className="text-sm text-muted-foreground">
            {t('report.load_failed')}{' '}
            <button type="button" className="underline" onClick={() => setActiveReportId(null)}>
              {t('report.clear_selection')}
            </button>
          </p>
        ) : !report ? (
          <p className="text-sm text-muted-foreground">{t('report.loading')}</p>
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
                  aria-label={t('report.title_aria')}
                />
                <div className="flex flex-wrap gap-x-6 gap-y-1.5">
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="uppercase tracking-wide">{t('report.operator_label')}</span>
                    <input
                      key={`op-${report.id}`}
                      defaultValue={report.operator ?? ''}
                      placeholder={t('report.operator_label')}
                      onBlur={(e) => {
                        const operator = e.target.value
                        if (operator !== (report.operator ?? '')) {
                          updateReport.mutate({ id: report.id, operator })
                        }
                      }}
                      className="bg-muted border border-border rounded px-2 py-1 text-xs text-foreground"
                      aria-label={t('report.operator_label')}
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="uppercase tracking-wide">{t('report.reference_label')}</span>
                    <input
                      key={`ref-${report.id}`}
                      defaultValue={report.reference_number ?? ''}
                      placeholder={t('report.reference_label')}
                      onBlur={(e) => {
                        const reference_number = e.target.value
                        if (reference_number !== (report.reference_number ?? '')) {
                          updateReport.mutate({ id: report.id, reference_number })
                        }
                      }}
                      className="bg-muted border border-border rounded px-2 py-1 text-xs text-foreground"
                      aria-label={t('report.reference_label')}
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={report.show_toc ?? true}
                      onChange={(e) => updateReport.mutate({ id: report.id, show_toc: e.target.checked })}
                      className="accent-primary"
                      aria-label={t('report.toc_label')}
                    />
                    <span className="uppercase tracking-wide">{t('report.toc_label')}</span>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={report.show_collection_overview ?? true}
                      onChange={(e) =>
                        updateReport.mutate({ id: report.id, show_collection_overview: e.target.checked })
                      }
                      className="accent-primary"
                      aria-label={t('report.document_overview')}
                    />
                    <span className="uppercase tracking-wide">{t('report.document_overview')}</span>
                  </label>
                  {(report.show_collection_overview ?? true) && (
                    <button
                      type="button"
                      onClick={() => refreshOverview.mutate(report.id)}
                      disabled={refreshOverview.isPending}
                      className="text-xs text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground disabled:opacity-50"
                    >
                      {refreshOverview.isPending
                        ? t('report.refresh_overview_pending')
                        : report.collection_overview?.captured_at
                          ? t('report.refresh_overview_captured', {
                              date: report.collection_overview.captured_at.slice(0, 10)
                            })
                          : t('report.capture_overview')}
                    </button>
                  )}
                </div>
              </div>

              {/* Export: a single Download button; formats expand on hover/focus.
                  The caret stays — it is the only thing saying the icon opens a
                  list rather than downloading something on the spot. */}
              <div className="relative group shrink-0">
                <DownloadButton label={t('chat.download')} aria-haspopup="menu" className="gap-1 px-2">
                  <ChevronDownIcon className="h-3.5 w-3.5" />
                </DownloadButton>
                <div className="absolute right-0 top-full z-10 hidden pt-1 group-hover:block group-focus-within:block">
                  <div className="flex flex-col min-w-[11rem] rounded-md border border-border bg-muted p-1 shadow-lg">
                    {exportFormats(t).map((e) => (
                      <a
                        key={e.format}
                        href={reportExportHref(report.id, e.format)}
                        {...(e.view ? { target: '_blank', rel: 'noreferrer' } : { download: true })}
                        className="block rounded px-3 py-1.5 text-sm hover:bg-accent whitespace-nowrap"
                        title={e.view ? t('report.open_new_tab_title') : t('report.download_format_title', { label: e.label })}
                      >
                        {e.label}
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {items.length === 0 && overviewToShow === null ? (
              <p className="text-sm text-muted-foreground">
                {t('report.empty_report_hint')}
              </p>
            ) : (
              <div className="flex-1 overflow-auto space-y-6 pr-2">
                {sections(t).map(({ type, label }) => {
                  const sectionItems = items.filter((i) => i.artifact_type === type)
                  if (sectionItems.length === 0) return null
                  return (
                    <div key={type} className="space-y-2">
                      <h2 className="text-sm font-medium uppercase tracking-wide text-muted-foreground">
                        {label} ({sectionItems.length})
                      </h2>
                      {sectionItems.map((item, si) => (
                        <div key={item.id} className="rounded-md border border-border bg-muted p-3 space-y-2">
                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0">
                              <div className="font-medium text-sm break-words">{itemTitle(item, t)}</div>
                              {itemSource(item, t) && (
                                <div className="text-xs text-muted-foreground">{itemSource(item, t)}</div>
                              )}
                            </div>
                            <div className="flex items-center gap-1 shrink-0">
                              <MoveUpButton
                                label={t('report.move_up_aria')}
                                onClick={() => move(item, -1)}
                                disabled={si === 0}
                                className="h-7"
                              />
                              <MoveDownButton
                                label={t('report.move_down_aria')}
                                onClick={() => move(item, 1)}
                                disabled={si === sectionItems.length - 1}
                                className="h-7"
                              />
                              {/* × not trash: this takes the finding out of the
                                  report. The evidence itself is untouched. */}
                              <RemoveButton
                                label={t('report.remove_item_aria')}
                                onClick={() => removeItem.mutate({ reportId: report.id, itemId: item.id })}
                                className="h-7"
                              />
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
                            placeholder={t('report.note_placeholder')}
                            onBlur={(e) => {
                              const note = e.target.value
                              if (note !== (item.note ?? '')) {
                                updateItem.mutate({ reportId: report.id, itemId: item.id, note: note || null })
                              }
                            }}
                            className="w-full bg-muted border border-border rounded px-2 py-1 text-xs"
                          />
                        </div>
                      ))}
                    </div>
                  )
                })}
                {overviewToShow && <CollectionOverviewPreview overview={overviewToShow} />}
              </div>
            )}
          </>
        )}
      </section>
    </div>
  )
}
