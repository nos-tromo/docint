import { describe, it, expect, beforeEach } from 'vitest'
import { useReportStore } from './report'

beforeEach(() => {
  localStorage.clear()
  useReportStore.setState({ activeReportId: null })
})

describe('useReportStore', () => {
  it('sets and clears the active report id', () => {
    useReportStore.getState().setActiveReportId(7)
    expect(useReportStore.getState().activeReportId).toBe(7)
    useReportStore.getState().setActiveReportId(null)
    expect(useReportStore.getState().activeReportId).toBeNull()
  })

  it('persists the active report id across reloads', () => {
    useReportStore.getState().setActiveReportId(42)
    const persisted = JSON.parse(localStorage.getItem('docint-report')!).state
    expect(persisted.activeReportId).toBe(42)
  })
})
