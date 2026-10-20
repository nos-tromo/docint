import { Banner, Button } from '@infra/ui'
import { useAssignReport, useUnassignedReports } from '@/hooks/useReports'
import { useLang, useT } from '@/i18n/LanguageContext'
import { formatDate } from '@/lib/formatDate'

/**
 * The caller's reports kept from before every report named its collection.
 *
 * Such a report belongs to no collection, so no collection lists it or opens
 * it. This lists them and moves one into the active collection. The server
 * assigns only a report without a collection, so nothing here can move a
 * report between collections. Renders nothing once there are none.
 *
 * @param collection - The active collection, or null while none is active.
 */
export function UnassignedReports({ collection }: { collection: string | null }) {
  const t = useT()
  const lang = useLang()
  const unassigned = useUnassignedReports()
  const assign = useAssignReport()
  const reports = unassigned.data?.reports ?? []
  if (reports.length === 0) return null

  return (
    <Banner variant="info" className="space-y-2">
      <h2 className="font-medium">{t('report.unassigned_title')}</h2>
      <p className="text-xs text-muted-foreground">
        {collection ? t('report.unassigned_hint', { collection }) : t('report.unassigned_select_collection')}
      </p>
      <ul className="space-y-1">
        {reports.map((r) => (
          <li key={r.id} className="flex items-center justify-between gap-3">
            <span className="min-w-0 truncate">
              {`${r.title} (${r.item_count})`}
              {r.updated_at && (
                <>
                  {' '}
                  <span className="ml-1 text-xs text-muted-foreground">{formatDate(r.updated_at, lang)}</span>
                </>
              )}
            </span>
            {collection && (
              <Button
                size="sm"
                variant="secondary"
                className="shrink-0 whitespace-nowrap"
                disabled={assign.isPending}
                onClick={() => assign.mutate(r.id)}
              >
                {t('report.unassigned_move', { collection })}
              </Button>
            )}
          </li>
        ))}
      </ul>
      {assign.isError && (
        <p className="text-xs text-[var(--status-red-fg)]" role="alert">
          {t('report.unassigned_move_error')}
        </p>
      )}
    </Banner>
  )
}
