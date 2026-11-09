import { Button } from '@infra/ui'
import { useT } from '@/i18n/LanguageContext'

/**
 * The end of a paged list: one centred "Load more" under a rule.
 *
 * The same control at every list end, so a reader recognises it; it sits
 * below a list's scroll area rather than inside it, where it would only show
 * once the list had been scrolled to the bottom.
 *
 * @param onClick - Fetch the next page.
 * @param busy - A page is loading; the button holds and says so.
 */
export function LoadMore({ onClick, busy = false }: { onClick: () => void; busy?: boolean }) {
  const t = useT()
  return (
    <div className="flex justify-center border-t border-border p-2">
      <Button variant="secondary" size="sm" disabled={busy} onClick={onClick}>
        {busy ? t('common.loading_ellipsis') : t('table.load_more')}
      </Button>
    </div>
  )
}
