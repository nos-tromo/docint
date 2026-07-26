import { en } from './en'
import { format } from './index'
import type { Strings } from './index'

/**
 * English-only `t` for pure lib modules (no React, no `useT()`) that need a
 * translate function as an optional parameter so they stay callable —
 * including from existing unit tests — without a `LanguageContext`. Real UI
 * call sites always pass their own `useT()` result; this is only the
 * fallback default.
 */
export const defaultT = (key: keyof Strings, vars?: Record<string, string | number>): string =>
  format(en[key], vars)
