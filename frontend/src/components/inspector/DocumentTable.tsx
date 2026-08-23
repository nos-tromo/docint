import {
  createSortedRowModel,
  flexRender,
  rowSortingFeature,
  sortFn_alphanumeric,
  sortFn_text,
  tableFeatures,
  useTable,
  type Column,
  type SortDirection,
  type SortingState,
  type ColumnDef
} from '@tanstack/react-table'
import { useMemo, useRef, useState, type ReactNode } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import {
  Badge,
  ChevronDownIcon,
  ChevronUpIcon,
  ChevronsUpDownIcon,
  CopyButton,
  DownloadLink
} from '@infra/ui'
import type { DocumentRecord } from '@/api/types'
import { csvExportHref } from '@/api/collections'
import { mimeLabel, shortHash, unitsLabel } from '@/lib/documentFormat'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'
import { SourcePreviewAction } from '@/components/common/SourcePreviewAction'

/**
 * The table's feature set, stitched statically so only what this grid uses is
 * bundled: sorting plus its row model. `columnMeta` is the per-column layout
 * hint the grid renderer reads — a typed slot on this table rather than a
 * global `declare module`, so it cannot leak into another table's meta.
 *
 * `sortFns` registers the two built-ins the automatic per-column detection
 * picks for this data (filenames sort alphanumerically, labels as text);
 * numeric columns fall through to the always-present basic comparator. An
 * unregistered name would silently downgrade to that comparator too.
 */
const FEATURES = tableFeatures({
  rowSortingFeature,
  sortedRowModel: createSortedRowModel(),
  sortFns: { alphanumeric: sortFn_alphanumeric, text: sortFn_text },
  columnMeta: {} as { align?: 'right' }
})

type Features = typeof FEATURES

/**
 * One grid template drives both the header row and every body row, so columns
 * align by construction (the previous table-header / flex-body split did not).
 */
const GRID_COLUMNS = 'minmax(0,2.4fr) 72px 92px 72px minmax(0,1.8fr) 96px'

/** Column defs depend on the active locale (header labels, copy-button label), so they're built per-render rather than module-level. */
function buildColumns(t: ReturnType<typeof useT>): ColumnDef<Features, DocumentRecord>[] {
  return [
    {
      accessorKey: 'filename',
      header: t('table.col_filename'),
      cell: (c) => (
        <span className="flex min-w-0 items-center gap-1">
          <span className="min-w-0 flex-1 truncate font-mono text-[13px]" title={c.getValue<string>()}>
            {c.getValue<string>()}
          </span>
          <SourcePreviewAction
            fileHash={c.row.original.file_hash}
            filename={c.getValue<string>()}
          />
        </span>
      )
    },
    {
      id: 'type',
      accessorFn: (r) => mimeLabel(r.mimetype),
      header: t('table.col_type'),
      cell: (c) => <span className="text-muted-foreground">{c.getValue<string>()}</span>
    },
    {
      id: 'units',
      accessorFn: (r) => unitsLabel(r, t).sort,
      header: t('table.col_units'),
      meta: { align: 'right' },
      cell: (c) => {
        const units = unitsLabel(c.row.original, t)
        return (
          <span className={cn('tabular-nums', units.text === '—' && 'text-muted-foreground')}>
            {units.text}
          </span>
        )
      }
    },
    {
      accessorKey: 'node_count',
      header: t('table.col_nodes'),
      meta: { align: 'right' },
      cell: (c) => <span className="tabular-nums">{c.getValue<number | undefined>() ?? 0}</span>
    },
    {
      id: 'entity_types',
      accessorFn: (r) => (r.entity_types ?? []).join(', '),
      header: t('table.col_entities'),
      enableSorting: false,
      cell: (c) => <EntityBadges types={c.row.original.entity_types ?? []} />
    },
    {
      accessorKey: 'file_hash',
      header: t('table.col_hash'),
      enableSorting: false,
      cell: (c) => {
        const hash = c.getValue<string>()
        return (
          <span className="flex items-center gap-1">
            <span className="font-mono text-xs text-muted-foreground">{shortHash(hash)}</span>
            <CopyButton
              text={hash}
              label={t('table.copy_hash', { filename: c.row.original.filename })}
              copiedLabel={t('common.copied')}
              className="h-6 w-6"
            />
          </span>
        )
      }
    }
  ]
}

interface Props {
  docs: DocumentRecord[]
  isFetching?: boolean
  hasNextPage?: boolean
  onLoadMore?: () => void
  collection: string
}

/** Entity-type chips: sorted, first four, with a `+N` overflow marker. */
function EntityBadges({ types }: { types: string[] }) {
  if (types.length === 0) return <span className="text-muted-foreground">—</span>
  const sorted = [...types].sort((a, b) => a.localeCompare(b))
  const shown = sorted.slice(0, 4)
  const extra = sorted.length - shown.length
  return (
    <div className="flex flex-wrap items-center gap-1">
      {shown.map((t) => (
        <Badge key={t} variant="neutral">
          {t}
        </Badge>
      ))}
      {extra > 0 && (
        <span className="text-xs text-muted-foreground" title={sorted.join(', ')}>
          +{extra}
        </span>
      )}
    </div>
  )
}

/**
 * Sort direction indicator, faint until hovered when the column is sortable but
 * unsorted. Drawn, not typed: `↑`/`↓`/`↕` render from whatever font the OS
 * falls back to, and these sit inline with the header text at every size.
 */
function SortIndicator({ dir }: { dir: false | SortDirection }) {
  if (dir === 'asc') return <ChevronUpIcon className="h-3.5 w-3.5" />
  if (dir === 'desc') return <ChevronDownIcon className="h-3.5 w-3.5" />
  return (
    <ChevronsUpDownIcon className="h-3.5 w-3.5 opacity-0 transition-opacity group-hover:opacity-40" />
  )
}

function HeaderCell({
  column,
  children
}: {
  column: Column<Features, DocumentRecord>
  children: ReactNode
}) {
  const align = column.columnDef.meta?.align
  if (!column.getCanSort()) {
    return <div className={cn('min-w-0', align === 'right' && 'text-right')}>{children}</div>
  }
  return (
    <div className={cn('min-w-0', align === 'right' && 'text-right')}>
      <button
        type="button"
        onClick={column.getToggleSortingHandler()}
        className={cn(
          'group inline-flex items-center gap-1 uppercase hover:text-foreground',
          align === 'right' && 'flex-row-reverse'
        )}
      >
        {children}
        <SortIndicator dir={column.getIsSorted()} />
      </button>
    </div>
  )
}

/** Read-only overview of a collection's documents, one aligned row each. */
export function DocumentTable({ docs, isFetching, hasNextPage, onLoadMore, collection }: Props) {
  const t = useT()
  const [sorting, setSorting] = useState<SortingState>([])
  const data = useMemo(() => docs, [docs])
  const columns = useMemo(() => buildColumns(t), [t])
  const table = useTable({
    features: FEATURES,
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting
  })

  const rows = table.getRowModel().rows
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 44,
    overscan: 12
  })

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {t(docs.length === 1 ? 'table.documents_one' : 'table.documents_other', { count: docs.length })}
          {hasNextPage ? '+' : ''}
          {isFetching ? ` ${t('table.loading_suffix')}` : ''}
        </p>
        {collection && (
          <DownloadLink href={csvExportHref(collection, 'documents')} label={t('table.export_csv')} />
        )}
      </div>

      {docs.length === 0 && !isFetching ? (
        <div className="rounded-lg border border-dashed border-border bg-muted/50 p-10 text-center">
          <p className="text-sm text-muted-foreground">{t('table.empty_title')}</p>
          <p className="mt-1 text-xs text-muted-foreground">{t('table.empty_hint')}</p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-border bg-muted">
          <div
            ref={scrollRef}
            className="max-h-[70vh] overflow-auto"
            data-testid="documents-scroll"
            role="table"
            aria-label={t('table.aria_documents')}
          >
            <div role="rowgroup">
              {table.getHeaderGroups().map((hg) => (
                <div
                  key={hg.id}
                  role="row"
                  className="sticky top-0 z-10 grid gap-x-4 border-b border-border bg-muted px-4 py-2.5 text-xs font-medium uppercase tracking-wide text-muted-foreground"
                  style={{ gridTemplateColumns: GRID_COLUMNS }}
                >
                  {hg.headers.map((h) => (
                    <HeaderCell key={h.id} column={h.column}>
                      {flexRender(h.column.columnDef.header, h.getContext())}
                    </HeaderCell>
                  ))}
                </div>
              ))}
            </div>

            <div
              role="rowgroup"
              className="relative"
              style={{ height: `${virtualizer.getTotalSize()}px` }}
            >
              {virtualizer.getVirtualItems().map((vRow) => {
                const row = rows[vRow.index]
                return (
                  <div
                    key={row.id}
                    role="row"
                    data-index={vRow.index}
                    ref={virtualizer.measureElement}
                    className="group absolute left-0 right-0 grid items-center gap-x-4 border-b border-border/60 px-4 py-2 text-sm hover:bg-white/5"
                    style={{ gridTemplateColumns: GRID_COLUMNS, transform: `translateY(${vRow.start}px)` }}
                  >
                    {/* Every column is always shown here, so these are the
                        visible cells; `getVisibleCells` belongs to the column
                        visibility feature this table does not register. */}
                    {row.getAllCells().map((cell) => (
                      <div
                        key={cell.id}
                        role="cell"
                        className={cn(
                          'min-w-0',
                          cell.column.columnDef.meta?.align === 'right' && 'text-right'
                        )}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>
          </div>

          {hasNextPage && onLoadMore && (
            <div className="flex justify-center border-t border-border p-2">
              <button
                type="button"
                onClick={onLoadMore}
                disabled={isFetching}
                className="rounded-md border border-border px-3 py-1 text-sm hover:bg-white/5 disabled:opacity-50"
              >
                {isFetching ? t('common.loading_ellipsis') : t('table.load_more')}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
