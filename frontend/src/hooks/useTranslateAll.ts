import { useRef, useState } from 'react'
import { translate } from '@/api/translate'
import { useConfig } from '@/hooks/useConfig'
import { ADD_ALL_CONFIRM_THRESHOLD, ADD_ALL_MAX_ITEMS_FALLBACK } from '@/hooks/useReports'
import { useTranslationsStore } from '@/stores/translations'
import { useT } from '@/i18n/LanguageContext'

/**
 * Calls in flight at once. `/translate` is a sync route, so each one holds a
 * thread of the pool the whole API shares, and nothing there rate-limits it.
 */
export const TRANSLATE_ALL_CONCURRENCY = 3

/**
 * Consecutive failures that end a run. A dead model fails every call and a
 * section is thousands of them, so the outage is reported in seconds — the
 * same budget `DocumentOcrEngine` keeps.
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
  done: number
  failed: number
  /** Rows skipped because their text was already translated this session. */
  skipped: number
}

/**
 * Translate every finding of an Analysis section in one action.
 *
 * `fetchAll` is the section's own page walk, so this reaches rows never
 * rendered; results go to the shared store, and texts it already holds are
 * never re-sent, so a re-run costs only the remainder. N small client calls
 * rather than a batch endpoint: one round-trip per snippet either way.
 */
export function useTranslateAll<Row>(params: {
  /** Walk the section's cursor pages, stopping after `maxItems` rows. */
  fetchAll: (maxItems: number) => Promise<Row[]>
  /** The row's canonical text — the translations store's key. */
  textOf: (row: Row) => string
}) {
  const t = useT()
  const { data: config } = useConfig()
  // The ceiling "Add all" refuses at: too large to add is too large to bother
  // translating for a report.
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
      // A row past the cap means the walk stopped early, so how much is
      // missing is unknowable — the refusal "Add all" makes.
      if (rows.length > cap) {
        setOutcome({ status: 'too_many', total: 0, done: 0, failed: 0, skipped: 0 })
        return
      }
      // Read outside React: `put` replaces the whole map, so subscribing here
      // would re-render the section header once per translation.
      const known = useTranslationsStore.getState().byText
      const queue: string[] = []
      const seen = new Set<string>()
      for (const row of rows) {
        const text = params.textOf(row)
        // Identical text is one translation, not one per row carrying it.
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
              // Fail-soft `ok: false`: the endpoint answered, the model did not.
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
   * Stop issuing new calls. Ones in flight finish and still file their result:
   * `apiPost` has no abort signal, and discarding a paid-for answer would only
   * make the resume dearer.
   */
  const stop = () => {
    stopRef.current = true
  }

  const reset = () => setOutcome({ status: 'idle', total: 0, done: 0, failed: 0, skipped: 0 })

  return { run, stop, reset, cap, ...outcome }
}
