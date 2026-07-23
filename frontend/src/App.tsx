import { QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { queryClient } from './api/queryClient'
import { Router } from './routes/Router'

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/+$/, '')}>
        <Router />
      </BrowserRouter>
    </QueryClientProvider>
  )
}
