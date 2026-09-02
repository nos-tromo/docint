import { useReport } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import type { AppendixFields } from '@/api/extracts'

/**
 * The case file and operator an extract is filed under.
 *
 * An extract is the appendix to a curated report, so it carries that report's
 * own identity rather than inventing one: the active report's case file and
 * operator go on every page of the rendered PDF, exactly as they appear on the
 * report itself.
 *
 * With no active report both are absent — deliberately, and for the reason the
 * Report screen leaves its own operator field empty rather than guessing: an
 * appendix that named a different operator than the report it belongs to would
 * be worse than one that names none. The CLI's `--operator` covers the offline
 * case, where there is no report to inherit from.
 *
 * Shared by the panel's "Build extract" and the per-source download so the two
 * cannot label the same collection differently.
 */
export function useAppendixFields(): AppendixFields {
  const activeReportId = useReportStore((s) => s.activeReportId)
  const report = useReport(activeReportId)
  return {
    reference_number: report.data?.reference_number ?? undefined,
    operator: report.data?.operator ?? undefined
  }
}

/** The active report's title, for the line naming what an extract appends to. */
export function useAppendixReportTitle(): string | null {
  const activeReportId = useReportStore((s) => s.activeReportId)
  const report = useReport(activeReportId)
  return report.data?.title ?? null
}
