import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { API_PREFIXES, spaShellBypass } from './devProxy'

describe('spaShellBypass', () => {
  it('serves the app for a navigation to the ingest screen', () => {
    // The bug it exists for: `/ingest` matched the API proxy prefix, so
    // loading or reloading the ingest screen was answered by FastAPI instead
    // of the app — reachable only by clicking through from another route.
    // Inside the base: a bare `/index.html` gets redirected to `/docint/` by
    // Vite's own base middleware, which lands on the dashboard instead.
    expect(spaShellBypass('/docint/ingest', 'GET')).toBe('/docint/index.html')
    expect(spaShellBypass('/docint/ingest/', 'GET')).toBe('/docint/index.html')
    expect(spaShellBypass('/docint/ingest', 'HEAD')).toBe('/docint/index.html')
  })

  it('leaves the CLI ingest endpoint on the same path alone', () => {
    // `POST /ingest` ingests the server's own DATA_PATH. Same path, different
    // method, different service — which is the whole reason this is a function
    // and not a shorter prefix.
    expect(spaShellBypass('/docint/ingest', 'POST')).toBeUndefined()
  })

  it('leaves every path under /ingest/ to the backend', () => {
    for (const path of [
      '/docint/ingest/upload',
      '/docint/ingest/finalize',
      '/docint/ingest/jobs',
      '/docint/ingest/jobs/events'
    ]) {
      expect(spaShellBypass(path, 'GET'), path).toBeUndefined()
      expect(spaShellBypass(path, 'POST'), path).toBeUndefined()
    }
  })

  it('ignores the query string when deciding', () => {
    expect(spaShellBypass('/docint/ingest?collection=docs', 'GET')).toBe('/docint/index.html')
  })

  it('does not claim other API prefixes', () => {
    for (const prefix of API_PREFIXES.filter((p) => p !== 'ingest')) {
      expect(spaShellBypass(`/docint/${prefix}`, 'GET'), prefix).toBeUndefined()
    }
  })
})

describe('the production proxy agrees with the dev one', () => {
  // Two files have to encode the same rule and neither can infer it from the
  // other, so this asserts the nginx side still carries its half. Without it,
  // the collision comes back in production only — the place nobody is looking.
  // vitest runs with the frontend directory as cwd.
  const nginxConf = readFileSync(resolve(process.cwd(), 'nginx/default.conf'), 'utf8')

  it('splits the ingest path by method in nginx too', () => {
    expect(nginxConf).toContain('map "$request_method:$uri" $ingest_spa_page')
    expect(nginxConf).toMatch(/"GET:\/ingest"\s+1;/)
    expect(nginxConf).toMatch(/"HEAD:\/ingest"\s+1;/)
    // The carve-out has to fire inside the location that would otherwise
    // proxy, and land on the shell.
    expect(nginxConf).toContain('error_page 418 = @spa_shell')
    expect(nginxConf).toContain('if ($ingest_spa_page)')
    expect(nginxConf).toContain('location @spa_shell')
  })

  it('still proxies every API prefix the dev server proxies', () => {
    // A prefix added to the dev proxy but not to nginx works in dev and 404s
    // in production, which is how this class of bug hides. Three routings
    // count as proxied: the shared JSON regex, a dedicated exact-match
    // location (the SSE endpoints have their own, for buffering), or a
    // dedicated prefix location (ingest, for large uploads).
    const [apiRegex = ''] = nginxConf.match(/location ~ \^\/\([a-z_|.\\]+\)\(\/\|\$\)/) ?? []
    expect(apiRegex).not.toBe('')
    for (const prefix of API_PREFIXES) {
      const routed =
        apiRegex.includes(`|${prefix}|`) ||
        apiRegex.includes(`(${prefix}|`) ||
        apiRegex.includes(`|${prefix})`) ||
        nginxConf.includes(`location = /${prefix}`) ||
        nginxConf.includes(`location ~ ^/${prefix}(`)
      expect(routed, `${prefix} is proxied in dev but not routed in nginx`).toBe(true)
    }
  })
})
