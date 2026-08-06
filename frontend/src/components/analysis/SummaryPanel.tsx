import { useEffect, useReducer, useRef } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Button, useTheme } from '@infra/ui'
import { cn } from '@/lib/cn'
import { summarize } from '@/api/analysis'
import { streamSseGet } from '@/api/sse'
import { INGEST_JOB_EVENTS_PATH } from '@/api/jobs'
import { ApiError } from '@/api/client'
import { describeError, streamErrorText } from '@/api/errorMessage'
import { SourcePills } from '@/components/chat/SourcePills'
import { ValidationBanner } from '@/components/chat/ValidationBanner'
import { downloadText } from '@/lib/csv'
import { summaryToMarkdown } from '@/lib/exports'
import { CoverageBanner } from './CoverageBanner'
import type { SummaryResponse } from '@/api/types'
import { useUiStore } from '@/stores/ui'
import { summarySnapshot } from '@/lib/reportSnapshots'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { useT } from '@/i18n/LanguageContext'

interface State {
  text: string
  done: boolean
  busy: boolean
  building: boolean
  jobId: string | null
  mapped: number | null
  totalUnits: number | null
  meta: SummaryResponse | null
  error: string | null
}
type Action =
  | { type: 'start' }
  | { type: 'building'; jobId: string }
  | { type: 'progress'; mapped: number | null; totalUnits: number | null }
  | { type: 'done'; meta: SummaryResponse }
  | { type: 'fail'; error: string }

const initialState: State = {
  text: '',
  done: false,
  busy: false,
  building: false,
  jobId: null,
  mapped: null,
  totalUnits: null,
  meta: null,
  error: null
}

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'start':
      return { ...initialState, busy: true }
    case 'building':
      return { ...s, building: true, jobId: a.jobId, mapped: null, totalUnits: null }
    case 'progress':
      return { ...s, mapped: a.mapped, totalUnits: a.totalUnits }
    case 'done':
      return {
        ...s,
        busy: false,
        building: false,
        done: true,
        meta: a.meta,
        text: a.meta.summary || s.text
      }
    case 'fail':
      return { ...s, busy: false, building: false, done: true, error: a.error }
  }
}

/** Extract the in-flight `job_id` a 409 `ApiError` carries, mirroring
 *  `createIngestJob`'s nested-detail unwrap. */
function inFlightJobId(e: ApiError): string | null {
  const detail = e.detail as { detail?: { job_id?: string } } | { job_id?: string } | undefined
  const nested = (detail as { detail?: { job_id?: string } } | undefined)?.detail
  const jobId = nested?.job_id ?? (detail as { job_id?: string } | undefined)?.job_id
  return typeof jobId === 'string' ? jobId : null
}

export function SummaryPanel({ reportDedupeKeys }: { reportDedupeKeys?: Set<string> }) {
  const t = useT()
  // See ChatTurn.tsx: `prose-invert` only belongs on a dark background.
  const { resolved } = useTheme()
  const collection = useUiStore((s) => s.selectedCollection)
  const [state, dispatch] = useReducer(reducer, initialState)
  // Tracks the in-flight build's SSE subscription so it can be torn down on
  // unmount as well as on a terminal frame — mirrors `useIngestJobStream`'s
  // AbortController idiom.
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      controllerRef.current?.abort()
      controllerRef.current = null
    }
  }, [])

  const finish = async () => {
    try {
      const result = await summarize(false, collection ?? undefined)
      if ('summary' in result) {
        dispatch({ type: 'done', meta: result })
      } else {
        // The rebuild just completed, so a cache-hit is expected here; a
        // second 202/409 would mean the job registry disagrees with the
        // completion frame we just observed.
        dispatch({ type: 'fail', error: t('analysis.summary_failed') })
      }
    } catch (e) {
      const d = describeError(e)
      dispatch({ type: 'fail', error: t(d.key, d.vars) })
    }
  }

  const runBuild = async (jobId: string) => {
    dispatch({ type: 'building', jobId })
    const controller = new AbortController()
    controllerRef.current = controller
    try {
      for await (const frame of streamSseGet(INGEST_JOB_EVENTS_PATH, controller.signal)) {
        const data = (frame.data ?? {}) as Record<string, unknown>
        // Every frame on this owner-multiplexed stream is tagged with
        // job_id; frames for other jobs (including ingest jobs) must be
        // ignored so a concurrent run never resolves this panel.
        if (data.job_id !== jobId) continue
        if (frame.event === 'summary_progress') {
          const mapped = typeof data.mapped === 'number' ? data.mapped : null
          const totalUnits = typeof data.total_units === 'number' ? data.total_units : null
          dispatch({ type: 'progress', mapped, totalUnits })
          continue
        }
        if (frame.event === 'summary_completed') {
          controller.abort()
          controllerRef.current = null
          await finish()
          return
        }
        if (frame.event === 'error') {
          controller.abort()
          controllerRef.current = null
          dispatch({
            type: 'fail',
            error: streamErrorText(t, data.code, 'analysis.summary_failed')
          })
          return
        }
        // summary_started: no-op, the panel already entered `building`.
      }
    } catch (e) {
      if (controller.signal.aborted) return // intentional teardown above
      const d = describeError(e)
      dispatch({ type: 'fail', error: t(d.key, d.vars) })
    }
  }

  const generate = async (refresh: boolean) => {
    dispatch({ type: 'start' })
    try {
      const result = await summarize(refresh, collection ?? undefined)
      if ('summary' in result) {
        dispatch({ type: 'done', meta: result })
        return
      }
      await runBuild(result.job_id)
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        const jobId = inFlightJobId(e)
        if (jobId) {
          await runBuild(jobId)
          return
        }
      }
      const d = describeError(e)
      dispatch({ type: 'fail', error: t(d.key, d.vars) })
    }
  }

  const reportItem = state.text && collection ? summarySnapshot({ collection, text: state.text }) : null
  const inReport = reportItem != null && (reportDedupeKeys?.has(reportItem.dedupe_key) ?? false)
  const progressPct =
    state.totalUnits != null && state.totalUnits > 0
      ? Math.min(100, Math.round(((state.mapped ?? 0) / state.totalUnits) * 100))
      : null

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <Button variant="primary" onClick={() => generate(false)} disabled={state.busy}>
          {state.busy ? t('analysis.summary_generating') : t('analysis.summary_generate')}
        </Button>
        <Button variant="secondary" onClick={() => generate(true)} disabled={state.busy}>
          {t('analysis.summary_refresh')}
        </Button>
        {state.text && (
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="secondary"
              onClick={() =>
                downloadText(
                  'summary.md',
                  summaryToMarkdown(state.meta, state.text, t),
                  'text/markdown;charset=utf-8'
                )
              }
            >
              {t('analysis.summary_download_md')}
            </Button>
            {reportItem && reportDedupeKeys && <AddToReportButton item={reportItem} inReport={inReport} />}
          </div>
        )}
      </div>

      {state.building && (
        <div className="space-y-1.5" data-testid="summary-build-progress">
          <div className="text-sm text-muted-foreground">
            {progressPct !== null && state.mapped !== null && state.totalUnits !== null
              ? t('analysis.summary_building_progress', {
                  mapped: state.mapped,
                  total: state.totalUnits
                })
              : t('analysis.summary_building')}
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-all"
              style={{ width: progressPct !== null ? `${progressPct}%` : '100%' }}
            />
          </div>
        </div>
      )}

      {state.error && <div className="text-red-400 text-sm">{state.error}</div>}
      {state.text && (
        <div className="rounded-md border border-border bg-muted p-4 text-sm">
          <div
            className={cn(
              'prose prose-sm max-w-none prose-p:my-2 prose-pre:bg-muted prose-code:before:content-none prose-code:after:content-none',
              resolved === 'dark' && 'prose-invert'
            )}
          >
            <Markdown remarkPlugins={[remarkGfm]}>{state.text}</Markdown>
          </div>
        </div>
      )}
      {state.meta && <ValidationBanner v={state.meta} />}
      {state.meta?.summary_diagnostics && (
        <CoverageBanner d={state.meta.summary_diagnostics} />
      )}
      {state.meta?.sources && state.meta.sources.length > 0 && (
        <div className="space-y-1.5">
          <div className="text-xs uppercase text-muted-foreground">{t('analysis.summary_sources')}</div>
          <SourcePills sources={state.meta.sources} />
        </div>
      )}
    </div>
  )
}
