import { useRef, useState } from 'react'
import { translate } from '@/api/translate'
import { useConfig } from '@/hooks/useConfig'
import { ADD_ALL_CONFIRM_THRESHOLD, ADD_ALL_MAX_ITEMS_FALLBACK } from '@/hooks/useReports'
import { useTranslationsStore } from '@/stores/translations'
import { useT } from '@/i18n/LanguageContext'

/**
 * How many translate calls may be in flight at once.
 *
 * `POST /translate` is a synchronous FastAPI route, so every in-flight call
 * holds one worker thread of a pool the whole API shares, and nothing on the
 * server rate-limits it. Three is enough to keep a section moving without a
 * bulk run starving unrelated requests.
 */
export const TRANSLATE_ALL_CONCURRENCY = 3

/**
 * How many consecutive failures end a run.
 *
 * A translate failure is fail-soft and cheap on its own, but a dead or
 * misconfigured model fails *every* call — and a section is thousands of them.
 * Stopping after three in a row reports the outage in seconds instead of
 * grinding through the whole corpus to say the same thing, the same budget
 * `DocumentOcrEngine` keeps for the same reason.
 */
export const TRANSLATE_ALL_MAX_CONSECUTIVE_FAILURES = 3

export type TranslateAllStatus =
  | 'idle'
  | 'fetching'
  | 'translating'
  | 'done'
  | 'stopped'
  | 'failed'
  | 'too_many'

export interface TranslateAllOutcome {
  status: TranslateAllStatus
  /** Snippets this run set out to translate (already-known ones excluded). */
  total: number
  /** Snippets translated so far. */
  done: number
  /** Snippets the model could not translate. */
  failed: number
  /** Rows skipped because their text was already translated this session. */
  skipped: number
}

/**
 * Translate every finding of an Analysis section in one action.
 *
 * The section-wide counterpart of the per-row Translate toggle. A corpus in a
 * language the investigator does not read is not made readable one hover at a
 * time, and a report built from it is only as readable as the findings that
 * were translated before they went in — so the whole section is translated
 * up front, and "Add all" then carries every translation into its snapshots.
 *
 * The caller supplies `fetchAll` — the same page walk its "Add all" uses, so
 * "all" means every finding the section's filter matches rather than the rows
 * paged in — and `textOf`, the same `chunkTextOf` derivation the store is
 * keyed by, so a row finds back exactly what this run filed for it.
 *
 * Translations land in the shared translations store, not here: that is what
 * makes them visible to the rows, to the per-row snapshot builder and to the
 * batch add alike. Texts the store already holds are never re-sent, which is
 * what makes a re-run after a stop, a failure, or a few manual clicks cost
 * only the remainder.
 *
 * Deliberately N small client calls rather than a batch endpoint: translation
 * is one model round-trip per snippet either way, and a request that ran for
 * the length of a whole section would be a background job with none of the
 * machinery a background job needs. Here the work is visible, stoppable, and
 * resumable, and a failure costs one snippet.
 */
export function useTranslateAll<Row>(params: {
  /** Walk the section's cursor pages, stopping after `maxItems` rows. */
  fetchAll: (maxItems: number) => Promise<Row[]>
  /** The row's canonical text — the translations store's key. */
  textOf: (row: Row) => string
}) {
  const t = useT()
  const { data: config } = useConfig()
  // The same ceiling "Add all" refuses at: a section too large to add is a
  // section there is no point translating for a report.
  const cap = Math.max(1, Math.trunc(config?.report_batch_max_items ?? ADD_ALL_MAX_ITEMS_FALLBACK))
  const [outcome, setOutcome] = useState<TranslateAllOutcome>({
    status: 'idle',
    total: 0,
    done: 0,
    failed: 0,
    skipped: 0
  })
  const stopRef = useRef(false)

  const run = async (): Promise<void> => {
    if (outcome.status === 'fetching' || outcome.status === 'translating') return
    stopRef.current = false
    setOutcome({ status: 'fetching', total: 0, done: 0, failed: 0, skipped: 0 })
    try {
      const rows = await params.fetchAll(cap + 1)
      // One row past the cap means the walk stopped early, so how much of the
      // section is missing is unknowable — the same refusal "Add all" makes.
      if (rows.length > cap) {
        setOutcome({ status: 'too_many', total: 0, done: 0, failed: 0, skipped: 0 })
        return
      }
      // Read the store once, outside React: `put` replaces the whole map, so a
      // subscribed hook would re-render this section's header per translation.
      const known = useTranslationsStore.getState().byText
      const queue: string[] = []
      const seen = new Set<string>()
      for (const row of rows) {
        const text = params.textOf(row)
        // Identical text is one translation, not one per row that carries it.
        if (!text || seen.has(text)) continue
        seen.add(text)
        if (!known[text]) queue.push(text)
      }
      const skipped = seen.size - queue.length
      if (queue.length === 0) {
        setOutcome({ status: 'done', total: 0, done: 0, failed: 0, skipped })
        return
      }
      if (
        queue.length > ADD_ALL_CONFIRM_THRESHOLD &&
        !window.confirm(t('common.translate_all_confirm', { count: queue.length }))
      ) {
        setOutcome({ status: 'idle', total: 0, done: 0, failed: 0, skipped: 0 })
        return
      }

      const total = queue.length
      setOutcome({ status: 'translating', total, done: 0, failed: 0, skipped })
      let next = 0
      let done = 0
      let failed = 0
      let consecutive = 0
      let broke = false
      const worker = async (): Promise<void> => {
        while (!broke && !stopRef.current && next < queue.length) {
          const text = queue[next++]
          try {
            const res = await translate(text)
            if (res.ok && res.translation != null) {
              useTranslationsStore.getState().put(text, {
                text: res.translation,
                target_lang: res.target_lang,
                model: res.model
              })
              done += 1
              consecutive = 0
            } else {
              // Fail-soft `ok: false` — the endpoint answered, the model did not.
              failed += 1
              consecutive += 1
            }
          } catch {
            failed += 1
            consecutive += 1
          }
          if (consecutive >= TRANSLATE_ALL_MAX_CONSECUTIVE_FAILURES) broke = true
          setOutcome({ status: 'translating', total, done, failed, skipped })
        }
      }
      await Promise.all(
        Array.from({ length: Math.min(TRANSLATE_ALL_CONCURRENCY, queue.length) }, () => worker())
      )
      const status: TranslateAllStatus = broke ? 'failed' : stopRef.current ? 'stopped' : 'done'
      setOutcome({ status, total, done, failed, skipped })
    } catch (e) {
      console.error('Translate all failed', e)
      setOutcome({ status: 'failed', total: 0, done: 0, failed: 0, skipped: 0 })
    }
  }

  /**
   * Stop issuing new translate calls.
   *
   * Calls already in flight run to completion and still file their result —
   * `apiPost` carries no abort signal, and throwing away an answer the model
   * already produced would only make the resume more expensive.
   */
  const stop = () => {
    stopRef.current = true
  }

  const reset = () => setOutcome({ status: 'idle', total: 0, done: 0, failed: 0, skipped: 0 })

  return { run, stop, reset, cap, ...outcome }
}
