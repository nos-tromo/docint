import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { API_PREFIXES, spaShellBypass } from './src/lib/devProxy'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const BACKEND = 'http://localhost:8000'

export default defineConfig({
  base: '/docint/',
  plugins: [react()],
  resolve: {
    alias: { '@': resolve(__dirname, './src') }
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: Object.fromEntries(
      API_PREFIXES.map((p) => [
        `/docint/${p}`,
        {
          target: BACKEND,
          changeOrigin: true,
          rewrite: (path: string) => path.replace(/^\/docint/, ''),
          // `/docint/ingest` is both an SPA route and a backend endpoint, so
          // the prefix match above would send a page load to the API. Nginx
          // splits the same path by method in production; this is that rule
          // for the dev server.
          bypass: (req: { url?: string; method?: string }) =>
            spaShellBypass(req.url, req.method),
        },
      ]),
    )
  },
  test: {
    globals: true,
    environment: 'happy-dom',
    setupFiles: ['./src/test/setup.ts']
  }
})
