import { QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { LanguageProvider } from './i18n/LanguageContext'
import { queryClient } from './api/queryClient'
import { Shell } from '@/layout/Shell'
import { Router } from './routes/Router'

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <LanguageProvider>
        <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/+$/, '')}>
          <Shell>
            <Router />
          </Shell>
        </BrowserRouter>
      </LanguageProvider>
    </QueryClientProvider>
  )
}
