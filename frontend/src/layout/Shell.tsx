import type { ReactNode } from 'react'
import { AppHeader } from '@infra/ui'
import { Sidebar } from './Sidebar'
import { useT } from '@/i18n/LanguageContext'
import { useWhoami } from '@/hooks/useWhoami'
import { useVersion } from '@/hooks/useVersion'

export function Shell({ children }: { children: ReactNode }) {
  const t = useT()
  const { data: whoami } = useWhoami()
  const { data: version } = useVersion()
  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <AppHeader
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
      />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex-1 min-w-0 overflow-auto">{children}</main>
      </div>
    </div>
  )
}
