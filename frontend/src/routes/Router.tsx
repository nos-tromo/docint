import { Route, Routes } from 'react-router-dom'
import { Shell } from '@/layout/Shell'
import { Dashboard } from './Dashboard'
import { Chat } from './Chat'
import { Ingest } from './Ingest'
import { Analysis } from './Analysis'
import { Inspector } from './Inspector'
import { Report } from './Report'

export function Router() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/chat/:sessionId" element={<Chat />} />
        <Route path="/ingest" element={<Ingest />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/inspector" element={<Inspector />} />
        <Route path="/report" element={<Report />} />
      </Routes>
    </Shell>
  )
}
