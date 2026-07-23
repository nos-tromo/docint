import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const BACKEND = 'http://localhost:8000'

const API_PREFIXES = ['collections', 'config', 'version', 'sessions', 'reports', 'sources', 'query', 'stream_query', 'summarize', 'ingest', 'agent', 'translate']

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
        { target: BACKEND, changeOrigin: true, rewrite: (path: string) => path.replace(/^\/docint/, '') },
      ]),
    )
  },
  test: {
    globals: true,
    environment: 'happy-dom',
    setupFiles: ['./src/test/setup.ts']
  }
})
