import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ReportState {
  // The report new artifacts are added to. Persisted: a stale id simply 404s
  // and the UI clears it, unlike `selectedCollection` which must start fresh.
  activeReportId: number | null
  setActiveReportId: (id: number | null) => void
}

export const useReportStore = create<ReportState>()(
  persist(
    (set) => ({
      activeReportId: null,
      setActiveReportId: (id) => set({ activeReportId: id })
    }),
    { name: 'docint-report' }
  )
)
