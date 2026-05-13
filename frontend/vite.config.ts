/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': resolve(__dirname, './src') }
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/collections': 'http://localhost:8000',
      '/sessions': 'http://localhost:8000',
      '/sources': 'http://localhost:8000',
      '/query': 'http://localhost:8000',
      '/stream_query': 'http://localhost:8000',
      '/summarize': 'http://localhost:8000',
      '/ingest': 'http://localhost:8000',
      '/agent': 'http://localhost:8000'
    }
  },
  test: {
    globals: true,
    environment: 'happy-dom',
    setupFiles: ['./src/test/setup.ts']
  }
})
