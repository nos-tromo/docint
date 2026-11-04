import type { ReactNode } from 'react'
import { DownloadIcon, IconButton, WarningIcon } from '@infra/ui'
import { useFileAction } from '@/hooks/useFileAction'
import { useT } from '@/i18n/LanguageContext'

/**
 * Download a file the API serves, through the purpose wrapper.
 *
 * What a `DownloadLink` was before: the same icon and quiet look, but a
 * button, because a link's request cannot carry the stated purpose and would
 * be refused wherever one is required. While the file is fetched it shows a
 * spinner; a failed download turns it into a warning that says so, since
 * there is no browser download bar to report it.
 *
 * @param props.label - Names the download (aria-label and tooltip).
 * @param props.download - Fetches and saves the file, e.g. `() => downloadFile(path, name)`.
 */
export function FileDownloadButton({
  label,
  download,
  className,
  children
}: {
  label: string
  download: () => Promise<void>
  className?: string
  children?: ReactNode
}) {
  const t = useT()
  const { run, busy, failed } = useFileAction()
  return (
    <>
      <IconButton
        icon={failed ? <WarningIcon /> : <DownloadIcon />}
        label={label}
        hint={failed ? t('common.download_failed') : undefined}
        variant={failed ? 'danger' : 'ghost'}
        busy={busy}
        onClick={() => void run(download)}
        className={className}
      >
        {children}
      </IconButton>
      {failed && (
        <span role="status" className="sr-only">
          {t('common.download_failed')}
        </span>
      )}
    </>
  )
}
