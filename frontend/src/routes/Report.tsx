import {
  ChevronDownIcon,
  DeleteButton,
  DownloadButton,
  Menu,
  MenuItem,
  MoveDownButton,
  MoveUpButton,
  NewButton,
  RefreshButton,
  RemoveButton,
  SelectMenu,
  StatusIcon,
  type StatusIconStatus
} from '@infra/ui'
import { reportExportHref, reportPdfHref } from '@/api/reports'
import type {
  ArtifactType,
  ReportExportFormat,
  ReportItem,
  ReportSummary,
  SnapshotThumbnail
} from '@/api/types'
import { ImageTextParts } from '@/components/common/ImageTextParts'
import { CollectionOverviewPreview } from '@/components/report/CollectionOverviewPreview'
import { ReportSection } from '@/components/report/ReportSection'
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
import { useReportPdf, type ReportPdfPhase, type ReportPdfView } from '@/hooks/useReportPdf'
import { useWhoami } from '@/hooks/useWhoami'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'
import type { Strings } from '@/i18n'
import { hateCategoryLabel, hateConfidenceLabel } from '@/lib/hateCategoryLabel'
import { hasImageParts, type ImageParts } from '@/lib/reportSnapshots'

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

// The PDF is not among them: it is rendered by a background job, so its entry
// is built separately (`ExportMenu`).
function exportFormats(
  t: Translate
): Array<{ format: Exclude<ReportExportFormat, 'pdf'>; label: string; view?: boolean }> {
  return [
    { format: 'md', label: t('report.format_markdown') },
    { format: 'html', label: t('report.format_html'), view: true },
    { format: 'zip', label: t('report.format_csv') },
    { format: 'json', label: t('report.format_json') }
  ]
}

const PDF_MARKER: Partial<Record<ReportPdfPhase, StatusIconStatus>> = {
  queued: 'idle',
  running: 'running',
  failed: 'failed'
}

/** What the PDF render is doing, in words — or null when there is nothing to say. */
function pdfStatusText(view: ReportPdfView, t: Translate): string | null {
  switch (view.phase) {
    case 'queued':
      return t('report.pdf_queued')
    case 'running':
      if (view.stage === 'finishing') return t('report.pdf_finishing')
      return view.page != null ? t('report.pdf_rendering_page', { page: view.page }) : t('report.pdf_rendering')
    case 'failed':
      return t('report.pdf_failed')
    default:
      return null
  }
}

/**
 * The one-line account of a PDF render beside the export menu. The marker
 * carries the wording as its accessible name, so the visible copy beside it
 * is hidden from assistive tech rather than read out twice.
 */
function PdfStatusLine({ view, t }: { view: ReportPdfView; t: Translate }) {
  const text = pdfStatusText(view, t)
  const marker = PDF_MARKER[view.phase]
  if (!text || !marker) return null
  return (
    <p
      role={view.phase === 'failed' ? 'alert' : undefined}
      className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground"
    >
      <StatusIcon status={marker} label={text} />
      <span aria-hidden="true">{text}</span>
    </p>
  )
}

/**
 * Export menu for one report: a single download icon that opens the formats.
 *
 * The caret is what says the icon opens a list rather than downloading
 * something on the spot. It was a CSS hover disclosure until the federation
 * grew a shared `Menu` — hover is not a gesture a touch screen or a keyboard
 * has, and the panel had no `role="menu"`, no `aria-expanded` and no way to
 * dismiss it.
 *
 * The PDF entry is the exception among plain links. A large report renders
 * for longer than the gateway waits on one request, so the PDF is a background
 * job: the entry starts a render, the line beside the menu follows it, and the
 * file downloads when it is ready. Only a PDF matching the report as it stands
 * is linked directly; an edited report renders again.
 *
 * @param report - The report to export.
 * @param t - The active locale's translate function.
 * @returns The export control and the PDF's status line.
 */
function ExportMenu({ report, t }: { report: Pick<ReportSummary, 'id' | 'updated_at'>; t: Translate }) {
  const pdf = useReportPdf(report)
  const rendering = pdf.phase === 'queued' || pdf.phase === 'running'
  const pdfLabel = t('report.format_pdf')
  return (
    <>
      <PdfStatusLine view={pdf} t={t} />
      <Menu
        align="end"
        className="shrink-0"
        panelClassName="min-w-[11rem]"
        trigger={(props) => (
          <DownloadButton {...props} label={t('chat.download')} className="gap-1 px-2">
            <ChevronDownIcon className="h-3.5 w-3.5" />
          </DownloadButton>
        )}
      >
        {pdf.phase === 'ready' && pdf.pdf ? (
          <MenuItem
            href={reportPdfHref(report.id)}
            download={pdf.pdf.filename}
            hint={t('report.download_format_title', { label: pdfLabel })}
          >
            {pdfLabel}
          </MenuItem>
        ) : (
          <MenuItem
            onSelect={() => void pdf.start()}
            disabled={rendering || pdf.loading}
            hint={
              rendering
                ? (pdfStatusText(pdf, t) ?? undefined)
                : pdf.phase === 'stale'
                  ? t('report.pdf_stale_title')
                  : t('report.pdf_create_title')
            }
          >
            {pdfLabel}
          </MenuItem>
        )}
        {exportFormats(t).map((e) => (
          <MenuItem
            key={e.format}
            href={reportExportHref(report.id, e.format)}
            {...(e.view ? { target: '_blank' as const, rel: 'noreferrer' } : { download: true })}
            hint={
              e.view
                ? t('report.open_new_tab_title')
                : t('report.download_format_title', { label: e.label })
            }
          >
            {e.label}
          </MenuItem>
        ))}
      </Menu>
    </>
  )
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
      // `category` and `confidence` are frozen protocol values from the
      // hate-speech prompt's fixed enums, shown through the shared label
      // lookups: the category reads as it does in Analysis, and both use the
      // labels the exported report prints (German "Rasse", "hoch" — not "race",
      // "high").
      const cat = str(s, 'category')
      const conf = str(s, 'confidence')
      const label = cat ? hateCategoryLabel(cat, t) : ''
      return (
        [label, conf && `(${hateConfidenceLabel(conf, t)})`].filter(Boolean).join(' ') ||
        t('report.default_hate_finding')
      )
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

/**
 * The printed words, description and tags of a finding judged from an image,
 * when the snapshot froze them apart — shown labelled instead of the body. A
 * hate-speech finding keeps its reason as the body and shows them only when
 * it has none.
 */
function itemImageParts(item: ReportItem): ImageParts | null {
  const s = item.snapshot
  if (item.artifact_type !== 'entity_finding' && !(item.artifact_type === 'hate_speech_finding' && !str(s, 'reason'))) {
    return null
  }
  // Stored JSON, so only string tags are trusted.
  const tags = Array.isArray(s.image_tags) ? s.image_tags.filter((tag): tag is string => typeof tag === 'string') : []
  const parts = { ocr_text: str(s, 'ocr_text'), image_description: str(s, 'image_description'), image_tags: tags }
  return hasImageParts(parts) ? parts : null
}

/** An item's body: an image finding's labelled parts, else its text. */
function ItemBody({ item }: { item: ReportItem }) {
  const parts = itemImageParts(item)
  if (parts) {
    return (
      <div className="text-sm text-muted-foreground">
        <ImageTextParts parts={parts} />
      </div>
    )
  }
  const body = itemBody(item)
  return body ? <p className="text-sm text-muted-foreground whitespace-pre-wrap break-words">{body}</p> : null
}

function isRenderableThumbnail(value: unknown): value is SnapshotThumbnail {
  // Snapshots are stored JSON: only an actual inline image may reach an
  // <img src>, mirroring the server renderers' validation.
  if (typeof value !== 'object' || value === null) return false
  const uri = (value as { data_uri?: unknown }).data_uri
  return typeof uri === 'string' && uri.startsWith('data:image/')
}

interface EvidenceFigure {
  thumb: SnapshotThumbnail
  caption: string
  filename: string
}

/**
 * The frozen figures of one item, each captioned the way the exports caption
 * it: a chat answer can carry several images at once and side by side they are
 * indistinguishable, so the caption repeats the citation number the answer
 * cites. A finding holds one figure and already names its source above it.
 */
function itemFigures(item: ReportItem): EvidenceFigure[] {
  const s = item.snapshot
  if (item.artifact_type !== 'chat_answer') {
    return isRenderableThumbnail(s.thumbnail)
      ? [{ thumb: s.thumbnail, caption: '', filename: str(s, 'filename') }]
      : []
  }
  const sources = Array.isArray(s.sources) ? s.sources : []
  const figures: EvidenceFigure[] = []
  for (const src of sources) {
    const source = src as { thumbnail?: unknown; filename?: unknown; citation_index?: unknown }
    if (!isRenderableThumbnail(source.thumbnail)) continue
    const name = typeof source.filename === 'string' ? source.filename : ''
    const index = typeof source.citation_index === 'number' ? source.citation_index : null
    figures.push({
      thumb: source.thumbnail,
      caption: (index != null ? `[${index}] ${name}` : name).trim(),
      filename: name
    })
  }
  return figures
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

/**
 * One item's frozen evidence, laid out as a strip of captioned figures.
 * Fixed-height image boxes so several sit on a shared baseline, and the alt
 * text follows the frozen `kind` — a keyframe is not an image the way a
 * screenshot is, and the exports have always labeled the two differently.
 */
function EvidenceStrip({ figures, t }: { figures: EvidenceFigure[]; t: Translate }) {
  const openPreview = useUiStore((s) => s.openPreview)
  if (figures.length === 0) return null
  return (
    <div className="flex flex-wrap gap-3">
      {figures.map(({ thumb, caption, filename }) => (
        <figure key={filename} className="max-w-[12rem] space-y-1">
          {/* Enlarged from the frozen bytes the snapshot carries, never from
              the source store: a report is meant to outlive the collection it
              was drawn from, so its evidence must not need one. */}
          <button
            type="button"
            onClick={() =>
              openPreview({ filename: filename || t('common.unknown_source'), data_uri: thumb.data_uri })
            }
            className="block cursor-pointer rounded border border-border overflow-hidden hover:border-foreground/40 focus-visible:ring-1 focus-visible:ring-primary outline-none"
          >
            <img
              src={thumb.data_uri}
              alt={thumb.kind === 'video_keyframe' ? t('report.keyframe_alt') : t('report.thumbnail_alt')}
              className="h-28 w-auto max-w-full object-contain"
            />
          </button>
          {caption && (
            <figcaption className="text-xs text-muted-foreground break-words">{caption}</figcaption>
          )}
        </figure>
      ))}
    </div>
  )
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

  // Deletes the report on screen — the only one reachable now that the list is
  // a selector rather than a column of rows.
  const onDelete = (id: number) => {
    if (!confirm(t('report.delete_confirm'))) return
    deleteReport.mutate(id, { onSuccess: () => setActiveReportId(null) })
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
    <div className="p-8 flex flex-col gap-4 h-full min-h-0">
      {/* The visible heading is the selector below — it names the report on
          screen, which is worth more than the word "Reports" over a route the
          nav already labels. The landmark stays for screen-reader navigation,
          which every other route gets from PageHeader. */}
      <h1 className="sr-only">{t('report.title')}</h1>

      <div className="flex items-center gap-3">
        {/* The page title *is* the picker. It was a native <select> sized up
            to text-2xl — but a native popup inherits its control's font size,
            so macOS Chrome opened a 24px list that covered this header.
            SelectMenu draws its panel as a sibling of the trigger, so the
            trigger stays title-sized and the list stays text-sm.

            The item count rides inside the label rather than beside it:
            "Vorgang Alpha (1)" is what an operator reads as the report's name,
            on the closed title and in the list alike. */}
        <SelectMenu
          label={t('report.select_aria')}
          options={reportList.map((r) => ({
            value: String(r.id),
            label: `${r.title} (${r.item_count})`
          }))}
          value={activeReportId != null ? String(activeReportId) : null}
          onChange={(id) => setActiveReportId(Number(id))}
          placeholder={t('report.choose')}
          emptyLabel={t('report.empty_list')}
          className="min-w-0 max-w-[40rem]"
          triggerClassName="text-2xl font-semibold"
        />

        <div className="ml-auto flex items-center gap-2">
          <NewButton
            label={t('common.new_report')}
            onClick={onCreate}
            busy={createReport.isPending}
            // Also gated while /whoami is in flight: creating before the
            // identity resolves would store a report with no operator.
            disabled={whoami.isLoading}
          />
          {report && (
            <DeleteButton label={t('report.delete_aria')} onClick={() => onDelete(report.id)} />
          )}
          {report && <ExportMenu report={report} t={t} />}
        </div>
      </div>

      {reports.isError && (
        <p className="text-xs text-[var(--status-red-fg)]" role="alert">
          {t('report.load_error')}
        </p>
      )}
      {createReport.isError && (
        <p className="text-xs text-[var(--status-red-fg)]">{t('report.create_error')}</p>
      )}

      <section className="flex flex-col min-h-0 flex-1">
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
                <div className="flex flex-wrap gap-x-6 gap-y-1.5">
                  {/* Renaming lives here rather than on the title line, which
                      is now the selector. It joins the two fields that already
                      carry report identity, directly under a control showing
                      the same value. */}
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="uppercase tracking-wide">{t('report.title_label')}</span>
                    <input
                      key={`title-${report.id}`}
                      defaultValue={report.title}
                      placeholder={t('report.title_label')}
                      onBlur={(e) => {
                        const title = e.target.value.trim()
                        if (title && title !== report.title) {
                          updateReport.mutate({ id: report.id, title })
                        }
                      }}
                      className="bg-muted border border-border rounded px-2 py-1 text-xs text-foreground"
                      aria-label={t('report.title_aria')}
                    />
                  </label>
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
                    // The refresh icon, not a dotted-underline text link: an
                    // underline in a row of form fields reads as a link to
                    // somewhere else, and this rebuilds what is on screen. The
                    // label keeps carrying the captured date, so the tooltip
                    // and the accessible name still say when the snapshot is
                    // from — an icon alone would drop that.
                    <RefreshButton
                      label={
                        report.collection_overview?.captured_at
                          ? t('report.refresh_overview_captured', {
                              date: report.collection_overview.captured_at.slice(0, 10)
                            })
                          : t('report.capture_overview')
                      }
                      busy={refreshOverview.isPending}
                      onClick={() => refreshOverview.mutate(report.id)}
                    />
                  )}
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
                    <ReportSection key={type} title={label} count={`(${sectionItems.length})`}>
                      {/* The exports re-sort this section by posting time, so the
                          arrows below order only this list and the JSON/CSV. */}
                      {type === 'hate_speech_finding' && (
                        <p className="text-xs text-muted-foreground">{t('report.hate_export_order_hint')}</p>
                      )}
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
                          <ItemBody item={item} />
                          <EvidenceStrip figures={itemFigures(item)} t={t} />
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
                    </ReportSection>
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
