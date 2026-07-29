import type { ReactNode } from 'react'
import { AppHeader } from '@infra/ui'
import { Sidebar } from './Sidebar'
import { useT } from '@/i18n/LanguageContext'

export function Shell({ children }: { children: ReactNode }) {
  const t = useT()
  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <AppHeader
        title="docint"
        // No identity source: X-Auth-User is a trusted header the edge
        // gateway injects server-side for the backend only, never surfaced
        // to the SPA — the header hides the user block when undefined.
        user={undefined}
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
