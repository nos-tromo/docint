import { useMemo, type ReactNode } from 'react'
import { AppShell, PurposeBar, type PurposeBarLabels } from '@infra/ui'
import { Sidebar } from './Sidebar'
import { PreviewDialog } from '@/components/common/PreviewDialog'
import { PURPOSE_SCOPE } from '@/api/client'
import { useT } from '@/i18n/LanguageContext'
import { useConfig } from '@/hooks/useConfig'
import { useWhoami } from '@/hooks/useWhoami'
import { useVersion } from '@/hooks/useVersion'
import { useIngestJobStream } from '@/hooks/useIngestJobStream'

/** The purpose bar's strings in the active locale. */
function usePurposeLabels(): PurposeBarLabels {
  const t = useT()
  return useMemo(
    () => ({
      purpose: t('purpose.name'),
      empty: t('purpose.empty'),
      required: t('purpose.required'),
      title: t('purpose.title'),
      description: t('purpose.description'),
      requested: t('purpose.requested'),
      caseRef: t('purpose.case_ref'),
      reason: t('purpose.reason'),
      save: t('purpose.save'),
      cancel: t('purpose.cancel'),
      clear: t('purpose.clear'),
      errorLength: t('purpose.error_length'),
      errorControl: t('purpose.error_control'),
      errorHeader: t('purpose.error_header'),
      errorEmpty: t('purpose.error_empty')
    }),
    [t]
  )
}

export function Shell({ children }: { children: ReactNode }) {
  const t = useT()
  useIngestJobStream()
  const { data: config } = useConfig()
  const { data: whoami } = useWhoami()
  const { data: version } = useVersion()
  const purposeLabels = usePurposeLabels()
  return (
    <>
      <AppShell
        title="docint"
        // The backend's trusted-header principal (X-Auth-User) is resolved
        // server-side and echoed back via the authenticated GET /whoami —
        // undefined while loading or on a fetch error, so the header simply
        // omits the user block rather than showing a stale/wrong identity.
        // display_name (the gateway's decorative X-Auth-Name) is preferred
        // when present; falls back to username until that header is deployed.
        user={whoami?.display_name ?? whoami?.username}
        version={version?.version ? `v${version.version}` : undefined}
        homeLabel={t('appHeader.home')}
        themeLabels={{
          system: t('appHeader.theme_system'),
          light: t('appHeader.theme_light'),
          dark: t('appHeader.theme_dark')
        }}
        signOutLabel={t('appHeader.sign_out')}
        // Where the purpose every request carries is stated (`apiFetch`), and
        // the dialog a 428 opens. The backend says which actions it refuses
        // without one; any at all make the empty bar a warning.
        headerActions={
          <PurposeBar
            scope={PURPOSE_SCOPE}
            required={(config?.require_purpose?.length ?? 0) > 0}
            labels={purposeLabels}
          />
        }
        sidebar={<Sidebar />}
      >
        {children}
      </AppShell>
      {/* One dialog for the whole app: any control can open a document
          preview through the ui store without owning modal state. */}
      <PreviewDialog />
    </>
  )
}
