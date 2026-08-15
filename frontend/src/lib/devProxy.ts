/**
 * Which request paths the dev server hands to the backend, and the one that it
 * must not.
 *
 * Imported by `vite.config.ts` rather than declared there so the rule below can
 * be tested — it encodes a collision that is invisible until someone reloads
 * the wrong screen, and the same rule has to hold in `frontend/nginx/default.conf`
 * for production (see its `$ingest_spa_page` map).
 */

/** Path prefixes proxied to the backend, relative to the `/docint/` base. */
export const API_PREFIXES = [
  'collections',
  'config',
  'version',
  'health',
  'sessions',
  'reports',
  'sources',
  'query',
  'search',
  'stream_query',
  'summarize',
  'ingest',
  'agent',
  'translate',
  'whoami'
]

/**
 * Paths that are both an SPA route and a backend endpoint, split by method.
 *
 * `/ingest` is the only one: the SPA's ingest screen lives there, and so does
 * `POST /ingest`, the CLI/batch endpoint that ingests the server's own
 * `DATA_PATH`. Everything *under* `/ingest/` — upload, finalize, jobs, the SSE
 * stream — is API-only.
 *
 * A route added here needs the same carve-out in the production nginx config;
 * neither layer can infer it from the other.
 */
const BASE = '/docint'

const METHOD_SPLIT_PAGES = new Set([`${BASE}/ingest`, `${BASE}/ingest/`])

/**
 * The file the dev server should serve instead of proxying, or `undefined` to
 * proxy as usual.
 *
 * Vite's `bypass` contract: a returned string is served as that path.
 *
 * @param url - The request URL as received, before any rewrite.
 * @param method - The HTTP method.
 * @returns `'/index.html'` for an SPA navigation, otherwise `undefined`.
 */
export function spaShellBypass(
  url: string | undefined,
  method: string | undefined
): string | undefined {
  const path = (url ?? '').split('?')[0]
  if (!METHOD_SPLIT_PAGES.has(path)) return undefined
  // Only a navigation wants the app. POST is the API on the same path, and
  // anything else (DELETE, PUT) is a client error the backend should answer,
  // not a page.
  // Inside the base, not a bare `/index.html`: Vite's base middleware
  // redirects anything outside `/docint/` to it, which would answer a reload
  // of the ingest screen with a 302 to the dashboard — the app, but not the
  // page that was asked for.
  return method === 'GET' || method === 'HEAD' ? `${BASE}/index.html` : undefined
}
