import type { AppConfig } from '@/api/types'

/** Fallback default node count before the backend config has loaded. */
const FALLBACK_DEFAULT = 80
/** Fallback ceiling before the backend config has loaded. */
const FALLBACK_MAX = 500

/**
 * Resolve the effective graph node count from the user's stored choice and the
 * deploy-time config. A `null` `stored` means "use the server default". The
 * result is always an integer clamped to `[1, graph_max_top_k]`, so a stale
 * persisted value can never exceed a lowered ceiling.
 */
export function resolveGraphTopK(stored: number | null, cfg: AppConfig | undefined): number {
  const max = cfg?.graph_max_top_k ?? FALLBACK_MAX
  const fallback = cfg?.graph_top_k ?? FALLBACK_DEFAULT
  const value = stored ?? fallback
  return Math.min(Math.max(1, Math.trunc(value)), max)
}
