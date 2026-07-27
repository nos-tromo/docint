import { createContext, useContext, useEffect } from 'react'
import { Spinner } from '@infra/ui'
import { useConfig } from '@/hooks/useConfig'
import { catalogs, format } from './index'
import type { Lang, Strings } from './index'

// Default 'en' means components (and their tests) work without a provider.
const LanguageContext = createContext<Lang>('en')

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const { data, isLoading } = useConfig()
  const language: Lang = data?.language === 'de' ? 'de' : 'en'
  // Keep the document's lang attribute in sync so a de deployment never
  // renders German text inside a lang="en" document.
  useEffect(() => {
    document.documentElement.lang = language
  }, [language])
  // Block only the very first paint so a de deployment never flashes English;
  // a failed config fetch falls through to 'en' — it can never blank the UI.
  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Spinner label="…" />
      </div>
    )
  }
  return <LanguageContext.Provider value={language}>{children}</LanguageContext.Provider>
}

export function useLang(): Lang {
  return useContext(LanguageContext)
}

export function useT(): (
  key: keyof Strings,
  vars?: Record<string, string | number>,
) => string {
  const lang = useContext(LanguageContext)
  return (key, vars) => format(catalogs[lang][key], vars)
}

export { LanguageContext }
