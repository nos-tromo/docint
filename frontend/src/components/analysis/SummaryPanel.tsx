import { useEffect, useReducer, useRef } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Button, DownloadButton, RefreshButton, useTheme } from '@infra/ui'
import { cn } from '@/lib/cn'
import { cachedSummary, summarize } from '@/api/analysis'
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

/**
 * What the panel is showing, independent of whether work is in flight.
 *
 * `busy`/`building` say whether a build is running; `phase` says what the
 * operator can see and therefore which control to offer. Keeping them apart is
 * what lets a rebuild run *over* a summary that stays on screen.
 */
type Phase = 'probing' | 'empty' | 'ready' | 'failed'

interface State {
  phase: Phase
  text: string
  busy: boolean
  building: boolean
  jobId: string | null
  mapped: number | null
  totalUnits: number | null
  meta: SummaryResponse | null
  error: string | null
}
type Action =
  | { type: 'reset' }
  | { type: 'probed'; meta: SummaryResponse | null }
  | { type: 'probe_failed'; error: string }
  | { type: 'start' }
  | { type: 'building'; jobId: string }
  | { type: 'progress'; mapped: number | null; totalUnits: number | null }
  | { type: 'done'; meta: SummaryResponse }
  | { type: 'fail'; error: string }

const initialState: State = {
  phase: 'probing',
  text: '',
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
    case 'reset':
      return initialState
    case 'probed':
      // The probe is slower than a click. If the operator already pressed
      // Create or Refresh, this answer is stale by definition and must not
      // stomp the build it lost the race to.
      if (s.busy || s.building) return s
      return a.meta && a.meta.summary.trim()
        ? { ...s, phase: 'ready', meta: a.meta, text: a.meta.summary, error: null }
        : { ...s, phase: 'empty', meta: null, text: '', error: null }
    case 'probe_failed':
      // A probe that could not run is not a summary that failed to build: the
      // create action stays offered, so a transport blip cannot lock the panel.
      if (s.busy || s.building) return s
      return { ...s, phase: 'empty', error: a.error }
    case 'start':
      // Deliberately not `{ ...initialState, busy: true }`, which is what this
      // was: that blanked a perfectly good summary the moment Refresh was
      // pressed, leaving an empty panel for the minutes a rebuild takes. Keep
      // what is on screen and reset only the run's own fields; the text is
      // replaced in one step when the new one lands.
      return { ...s, busy: true, error: null, building: false, jobId: null, mapped: null, totalUnits: null }
    case 'building':
      return { ...s, building: true, jobId: a.jobId, mapped: null, totalUnits: null }
    case 'progress':
      return { ...s, mapped: a.mapped, totalUnits: a.totalUnits }
    case 'done': {
      const text = a.meta.summary || s.text
      return {
        ...s,
        busy: false,
        building: false,
        error: null,
        meta: a.meta,
        text,
        // A build that legitimately produced nothing (an empty collection)
        // goes back to offering the create action rather than sitting on a
        // blank "ready" panel.
        phase: text ? 'ready' : 'empty'
      }
    }
    case 'fail':
      // A refresh that failed over an existing summary keeps showing it and
      // adds the error line. Only a failure with nothing to show is a failed
      // panel.
      return { ...s, busy: false, building: false, error: a.error, phase: s.text ? 'ready' : 'failed' }
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
  // `useT()` returns a fresh function on every render, so `t` cannot be a
  // dependency below — it would re-run the probe on every render, and each run
  // may cost a server-side validation pass. A ref is the stable handle.
  const tRef = useRef(t)
  tRef.current = t

  // Read what is already cached, and never build. The panel is mounted only
  // while its tab is open, so this runs on every visit — hence the GET, whose
  // handler has no queue branch at all: `summarize(false, …)` would answer a
  // cache miss by starting a minutes-long build nobody asked for.
  useEffect(() => {
    // Switching collections abandons any build this panel was following. The
    // events stream is owner-multiplexed, so an un-torn-down subscription
    // would keep dispatching the previous collection's frames into the new
    // collection's state.
    controllerRef.current?.abort()
    controllerRef.current = null
    dispatch({ type: 'reset' })
    if (!collection) return
    let cancelled = false
    void (async () => {
      try {
        const cached = await cachedSummary(collection)
        if (!cancelled) dispatch({ type: 'probed', meta: cached })
      } catch (e) {
        if (cancelled) return
        const d = describeError(e)
        dispatch({ type: 'probe_failed', error: tRef.current(d.key, d.vars) })
      }
    })()
    return () => {
      cancelled = true
      controllerRef.current?.abort()
      controllerRef.current = null
    }
  }, [collection])

  // `finish`/`runBuild` are mutually recursive: after a `summary_completed`
  // frame, `finish` refetches and — if the refetch itself carries a fresh
  // `job_id` — re-attaches via `runBuild`, which calls back into `finish`
  // once *that* build completes. `attempt` bounds the recursion to a single
  // re-attach so a persistently disagreeing job registry still surfaces as
  // a failure instead of bouncing forever.
  const finish = async (attempt = 0) => {
    try {
      const result = await summarize(false, collection ?? undefined)
      if ('summary' in result) {
        dispatch({ type: 'done', meta: result })
        return
      }
      if (attempt >= 1) {
        // Second consecutive job_id: the bound is reached, so this is no
        // longer explainable by the ordinary revision-bump race below —
        // report it like any other unexpected state.
        dispatch({
          type: 'fail',
          error: streamErrorText(t, 'summary_requeued', 'analysis.summary_failed')
        })
        return
      }
      // A concurrent ingest can bump the collection's summary revision
      // mid-build; `build_tree_summary`'s compare-and-set cache write
      // (`_store_cached_collection_summary`) then deliberately skips
      // caching a build made stale by that bump. The build itself
      // succeeded — nothing failed — but with no cache entry to serve, this
      // refetch legitimately 202s with the server's own freshly-queued
      // rebuild. Follow it instead of reporting a false failure.
      await runBuild(result.job_id, attempt + 1)
    } catch (e) {
      const d = describeError(e)
      dispatch({ type: 'fail', error: t(d.key, d.vars) })
    }
  }

  const runBuild = async (jobId: string, finishAttempt = 0) => {
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
          await finish(finishAttempt)
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
      <div className="flex items-center gap-2">
        {state.phase === 'probing' && (
          <span className="text-sm text-muted-foreground">{t('common.loading_ellipsis')}</span>
        )}
        {(state.phase === 'empty' || state.phase === 'failed') && (
          // Text, not an icon: an empty panel has to say what its button will
          // make. A drawing alone cannot introduce something not yet on screen.
          <Button variant="primary" onClick={() => generate(false)} disabled={state.busy}>
            {state.busy ? t('analysis.summary_generating') : t('analysis.summary_generate')}
          </Button>
        )}
        {state.phase === 'ready' && (
          // The summary is already on screen, so the control that rebuilds it
          // is chrome. Its label stays "Aktualisieren" —
          // `analysis.coverage_partial_detail` tells the reader to click that
          // word, and an icon whose name no longer says it would leave the
          // sentence pointing at nothing.
          <RefreshButton
            label={t('analysis.summary_refresh')}
            busy={state.busy}
            onClick={() => generate(true)}
          />
        )}
        {state.text && (
          <div className="ml-auto flex items-center gap-2">
            <DownloadButton
              label={t('analysis.summary_download_md')}
              onClick={() =>
                downloadText(
                  'summary.md',
                  summaryToMarkdown(state.meta, state.text, t),
                  'text/markdown;charset=utf-8'
                )
              }
            />
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
