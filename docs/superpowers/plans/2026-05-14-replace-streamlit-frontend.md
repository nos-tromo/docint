# Replace Streamlit Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `docint/ui/` (Streamlit) with a Vite + React + TypeScript SPA served as static assets by FastAPI, preserving every feature and adding native dark theme + fine-grained reactivity.

**Architecture:** Single-page app under `frontend/`. In dev, Vite serves on `:5173` and proxies `/api` to FastAPI on `:8000`. In prod, `vite build` writes to `frontend/dist/` and FastAPI mounts it via `StaticFiles(html=True)` at `/`. State is split: TanStack Query for server state, Zustand for UI state, component-local reducers for in-flight streams.

**Tech Stack:** Vite, React 18, TypeScript, Tailwind CSS, shadcn/ui (Radix), TanStack Query v5, Zustand, React Router v6, TanStack Table v8, Recharts, React Hook Form + Zod, JSZip, Vitest + Testing Library.

**Source spec:** `docs/superpowers/specs/2026-05-13-replace-streamlit-frontend-design.md`

---

## Phase 0 — Frontend scaffold

### Task 0.1: Create the Vite + React + TypeScript project

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/.gitignore`
- Create: `frontend/.npmrc`

- [ ] **Step 1: Create the directory and initialize pnpm**

```bash
mkdir -p /Users/himarc/dev/nos-tromo/docint/frontend
cd /Users/himarc/dev/nos-tromo/docint/frontend
corepack enable
corepack prepare pnpm@9.12.0 --activate
```

Expected: `pnpm --version` prints `9.12.0`.

- [ ] **Step 2: Write `frontend/package.json`**

```json
{
  "name": "docint-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint . --ext .ts,.tsx",
    "typecheck": "tsc -b --noEmit"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.27.0",
    "@tanstack/react-query": "^5.59.0",
    "@tanstack/react-table": "^8.20.5",
    "zustand": "^5.0.0",
    "recharts": "^2.13.0",
    "react-hook-form": "^7.53.0",
    "zod": "^3.23.8",
    "@hookform/resolvers": "^3.9.0",
    "jszip": "^3.10.1",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.5.4"
  },
  "devDependencies": {
    "@types/react": "^18.3.11",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.2",
    "typescript": "^5.6.2",
    "vite": "^5.4.8",
    "vitest": "^2.1.2",
    "@testing-library/react": "^16.0.1",
    "@testing-library/jest-dom": "^6.5.0",
    "@testing-library/user-event": "^14.5.2",
    "happy-dom": "^15.7.4",
    "tailwindcss": "^3.4.13",
    "postcss": "^8.4.47",
    "autoprefixer": "^10.4.20",
    "eslint": "^9.12.0",
    "@typescript-eslint/parser": "^8.8.1",
    "@typescript-eslint/eslint-plugin": "^8.8.1",
    "eslint-plugin-react": "^7.37.1",
    "eslint-plugin-react-hooks": "^5.0.0",
    "openapi-typescript": "^7.4.1"
  },
  "packageManager": "pnpm@9.12.0"
}
```

- [ ] **Step 3: Write `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [ ] **Step 4: Write `frontend/tsconfig.node.json`**

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 5: Write `frontend/vite.config.ts`**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') }
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
```

- [ ] **Step 6: Write `frontend/index.html`**

```html
<!doctype html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>DocInt — Document Intelligence</title>
  </head>
  <body class="bg-zinc-950 text-zinc-100">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Write `frontend/src/main.tsx`**

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/globals.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

- [ ] **Step 8: Write `frontend/src/App.tsx`**

```tsx
export default function App() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <h1 className="text-2xl font-semibold">DocInt</h1>
    </div>
  )
}
```

- [ ] **Step 9: Write `frontend/.gitignore`**

```
node_modules
dist
.vite
*.log
coverage
```

- [ ] **Step 10: Write `frontend/.npmrc`**

```
auto-install-peers=true
strict-peer-dependencies=false
```

- [ ] **Step 11: Install dependencies**

Run: `cd /Users/himarc/dev/nos-tromo/docint/frontend && pnpm install`
Expected: `pnpm-lock.yaml` is created, no errors.

- [ ] **Step 12: Verify the dev server boots**

Run: `cd /Users/himarc/dev/nos-tromo/docint/frontend && pnpm dev` (then Ctrl-C after seeing output)
Expected: Vite reports `Local: http://localhost:5173/`. Visit it in a browser; you see "DocInt".

- [ ] **Step 13: Commit**

```bash
cd /Users/himarc/dev/nos-tromo/docint
git add frontend/
git commit -m "Scaffold Vite + React + TypeScript frontend"
```

---

### Task 0.2: Add Tailwind with dark theme baseline

**Files:**
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.js`
- Create: `frontend/src/styles/globals.css`
- Create: `frontend/src/lib/cn.ts`

- [ ] **Step 1: Write `frontend/tailwind.config.ts`**

```ts
import type { Config } from 'tailwindcss'

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: 'rgb(9 9 11)',
        foreground: 'rgb(244 244 245)',
        muted: 'rgb(39 39 42)',
        'muted-foreground': 'rgb(161 161 170)',
        border: 'rgb(39 39 42)',
        accent: 'rgb(82 82 91)',
        primary: 'rgb(244 244 245)',
        'primary-foreground': 'rgb(9 9 11)'
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['ui-monospace', 'SFMono-Regular']
      }
    }
  },
  plugins: []
} satisfies Config
```

- [ ] **Step 2: Write `frontend/postcss.config.js`**

```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {}
  }
}
```

- [ ] **Step 3: Write `frontend/src/styles/globals.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  color-scheme: dark;
}

html, body, #root {
  height: 100%;
}

body {
  @apply bg-background text-foreground antialiased;
}

* {
  @apply border-border;
}

::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}
::-webkit-scrollbar-track { @apply bg-zinc-900; }
::-webkit-scrollbar-thumb { @apply bg-zinc-700 rounded-full; }
::-webkit-scrollbar-thumb:hover { @apply bg-zinc-600; }
```

- [ ] **Step 4: Write `frontend/src/lib/cn.ts`**

```ts
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

- [ ] **Step 5: Verify Tailwind compiles**

Run: `cd frontend && pnpm dev` (Ctrl-C after page loads)
Expected: page background is dark zinc; "DocInt" text is light. No PostCSS errors in terminal.

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "Add Tailwind with dark theme baseline"
```

---

### Task 0.3: Set up Vitest + Testing Library

**Files:**
- Create: `frontend/src/test/setup.ts`
- Create: `frontend/src/lib/cn.test.ts`

- [ ] **Step 1: Write `frontend/src/test/setup.ts`**

```ts
import '@testing-library/jest-dom/vitest'
```

- [ ] **Step 2: Write a failing test in `frontend/src/lib/cn.test.ts`**

```ts
import { describe, it, expect } from 'vitest'
import { cn } from './cn'

describe('cn', () => {
  it('merges class names', () => {
    expect(cn('a', 'b')).toBe('a b')
  })

  it('deduplicates conflicting tailwind classes', () => {
    expect(cn('p-2', 'p-4')).toBe('p-4')
  })

  it('drops falsy values', () => {
    expect(cn('a', false && 'b', null, undefined, 'c')).toBe('a c')
  })
})
```

- [ ] **Step 3: Run the test suite**

Run: `cd frontend && pnpm test`
Expected: all three tests pass.

- [ ] **Step 4: Commit**

```bash
git add frontend/
git commit -m "Wire Vitest + Testing Library setup"
```

---

## Phase 1 — API client and streaming primitives

### Task 1.1: Hand-written API types

**Files:**
- Create: `frontend/src/api/types.ts`

- [ ] **Step 1: Write `frontend/src/api/types.ts`**

```ts
export interface Source {
  id: string
  file_hash: string
  filename: string
  page_label?: string | null
  row_label?: string | null
  score: number
  text?: string
  reference_metadata?: Record<string, unknown>
  ner?: { entities?: Entity[]; relations?: Relation[] }
}

export interface Entity {
  text: string
  type: string
  count?: number
  variants?: string[]
}

export interface Relation {
  subject: string
  predicate: string
  object: string
  count?: number
}

export interface ValidationFields {
  validation_status?: 'ok' | 'warning' | 'failed' | string
  validation_message?: string | null
  validation_details?: Record<string, unknown> | null
}

export interface ChatFinalEvent extends ValidationFields {
  status?: 'answer' | 'clarification'
  answer?: string
  message?: string
  sources: Source[]
  session_id: string
  intent?: string
  confidence?: number
  tool_used?: string
  reason?: string
  graph_debug?: unknown
  retrieval_query?: string
  coverage_unit?: string
  entity_match_candidates?: unknown[]
  entity_match_groups?: unknown[]
}

export interface MetadataFilter {
  field: string
  operator: string
  value?: unknown
  values?: unknown[]
}

export type QueryMode = 'answer' | 'entity_occurrence' | 'entity_occurrence_multi'
export type RetrievalMode = 'stateless' | 'session'

export interface ChatRequest {
  question: string
  session_id?: string
  metadata_filters?: MetadataFilter[]
  retrieval_mode?: RetrievalMode
  query_mode?: QueryMode
}

export interface SessionSummary {
  session_id: string
  title?: string | null
  created_at: string
  updated_at: string
  message_count: number
}

export interface SessionMessage {
  role: 'user' | 'assistant'
  content: string
  citations?: Source[]
  created_at: string
}

export interface DocumentRecord {
  filename: string
  file_hash: string
  mimetype?: string
  page_count?: number
  row_count?: number
  node_count?: number
  entity_types?: string[]
}

export interface NerStats {
  totals: { entities: number; relations: number; documents: number }
  top_entities: Array<{ text: string; type: string; count: number }>
  entity_types: Array<{ type: string; count: number }>
  top_relations: Array<{ subject: string; predicate: string; object: string; count: number }>
  documents: Array<{ filename: string; entity_count: number }>
}

export interface IngestEvent {
  event:
    | 'start'
    | 'upload_progress'
    | 'file_saved'
    | 'ingestion_started'
    | 'ingestion_progress'
    | 'ingestion_complete'
    | 'error'
  data: Record<string, unknown>
}

export interface SummaryDiagnostics {
  total_documents: number
  covered_documents: number
  coverage_ratio: number
  uncovered_documents: string[]
  coverage_target: number
  candidate_count: number
  deduped_count: number
  sampled_count: number
}

export interface SummaryResponse extends ValidationFields {
  summary: string
  sources: Source[]
  summary_diagnostics?: SummaryDiagnostics
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/api/types.ts
git commit -m "Add hand-written API types for frontend"
```

---

### Task 1.2: Base fetch client

**Files:**
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/client.test.ts`

- [ ] **Step 1: Write the failing test `frontend/src/api/client.test.ts`**

```ts
import { describe, it, expect, vi, afterEach } from 'vitest'
import { apiGet, apiPost, apiDelete, ApiError } from './client'

afterEach(() => {
  vi.restoreAllMocks()
})

function mockFetch(body: unknown, init: { status?: number; ok?: boolean } = {}) {
  const status = init.status ?? 200
  const ok = init.ok ?? status < 400
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  }))
}

describe('client', () => {
  it('apiGet returns parsed JSON', async () => {
    mockFetch({ hello: 'world' })
    expect(await apiGet<{ hello: string }>('/x')).toEqual({ hello: 'world' })
  })

  it('apiPost sends JSON body', async () => {
    mockFetch({ ok: true })
    await apiPost('/x', { a: 1 })
    const call = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(call[1].method).toBe('POST')
    expect(call[1].headers['Content-Type']).toBe('application/json')
    expect(call[1].body).toBe('{"a":1}')
  })

  it('apiDelete uses DELETE method', async () => {
    mockFetch({ ok: true })
    await apiDelete('/x')
    const call = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(call[1].method).toBe('DELETE')
  })

  it('throws ApiError on non-2xx', async () => {
    mockFetch({ detail: 'bad' }, { status: 400, ok: false })
    await expect(apiGet('/x')).rejects.toBeInstanceOf(ApiError)
  })
})
```

- [ ] **Step 2: Run; expect failure**

Run: `cd frontend && pnpm test src/api/client.test.ts`
Expected: FAIL with `Cannot find module './client'`.

- [ ] **Step 3: Write `frontend/src/api/client.ts`**

```ts
const BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(`API error ${status}: ${JSON.stringify(detail)}`)
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: unknown
    try {
      detail = await res.json()
    } catch {
      detail = await res.text()
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export function url(path: string) {
  return `${BASE}${path}`
}

export async function apiGet<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const qs = params
    ? '?' +
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join('&')
    : ''
  return handle<T>(await fetch(url(path) + qs))
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return handle<T>(
    await fetch(url(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body)
    })
  )
}

export async function apiDelete<T>(path: string): Promise<T> {
  return handle<T>(await fetch(url(path), { method: 'DELETE' }))
}
```

- [ ] **Step 4: Run; expect pass**

Run: `cd frontend && pnpm test src/api/client.test.ts`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/client.ts frontend/src/api/client.test.ts
git commit -m "Add base fetch client with ApiError"
```

---

### Task 1.3: SSE stream helper

**Files:**
- Create: `frontend/src/api/sse.ts`
- Create: `frontend/src/api/sse.test.ts`

- [ ] **Step 1: Write the failing test `frontend/src/api/sse.test.ts`**

```ts
import { describe, it, expect, vi, afterEach } from 'vitest'
import { streamSse } from './sse'

afterEach(() => vi.restoreAllMocks())

function bodyFromString(s: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(c) {
      c.enqueue(enc.encode(s))
      c.close()
    }
  })
}

describe('streamSse', () => {
  it('parses event/data frames into objects', async () => {
    const frames =
      'event: token\n' +
      'data: {"token":"hello"}\n\n' +
      'event: token\n' +
      'data: {"token":" world"}\n\n' +
      'event: done\n' +
      'data: {"answer":"hello world","sources":[]}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, body: bodyFromString(frames) })
    )

    const events: Array<{ event: string; data: unknown }> = []
    for await (const ev of streamSse('/x', { foo: 'bar' })) events.push(ev)

    expect(events).toEqual([
      { event: 'token', data: { token: 'hello' } },
      { event: 'token', data: { token: ' world' } },
      { event: 'done', data: { answer: 'hello world', sources: [] } }
    ])
  })

  it('handles frames split across chunks', async () => {
    const enc = new TextEncoder()
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(enc.encode('event: token\nda'))
        c.enqueue(enc.encode('ta: {"token":"hi"}\n\n'))
        c.close()
      }
    })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: stream }))

    const events: Array<{ event: string; data: unknown }> = []
    for await (const ev of streamSse('/x')) events.push(ev)
    expect(events).toEqual([{ event: 'token', data: { token: 'hi' } }])
  })
})
```

- [ ] **Step 2: Run; expect failure**

Run: `cd frontend && pnpm test src/api/sse.test.ts`
Expected: FAIL with `Cannot find module './sse'`.

- [ ] **Step 3: Write `frontend/src/api/sse.ts`**

```ts
import { url } from './client'

export interface SseEvent {
  event: string
  data: unknown
}

export async function* streamSse(
  path: string,
  body?: unknown,
  signal?: AbortSignal
): AsyncGenerator<SseEvent, void, unknown> {
  const res = await fetch(url(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal
  })
  if (!res.ok || !res.body) {
    throw new Error(`SSE request failed: ${res.status}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const event = parseFrame(frame)
      if (event) yield event
    }
  }
}

function parseFrame(frame: string): SseEvent | null {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return null
  const raw = dataLines.join('\n')
  try {
    return { event, data: JSON.parse(raw) }
  } catch {
    return { event, data: raw }
  }
}
```

- [ ] **Step 4: Run; expect pass**

Run: `cd frontend && pnpm test src/api/sse.test.ts`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/sse.ts frontend/src/api/sse.test.ts
git commit -m "Add streamSse helper for chat/summarize endpoints"
```

---

### Task 1.4: Multipart upload streamer

**Files:**
- Create: `frontend/src/api/upload.ts`
- Create: `frontend/src/api/upload.test.ts`

- [ ] **Step 1: Write the failing test `frontend/src/api/upload.test.ts`**

```ts
import { describe, it, expect, vi, afterEach } from 'vitest'
import { streamUpload } from './upload'

afterEach(() => vi.restoreAllMocks())

function bodyFromString(s: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(c) {
      c.enqueue(enc.encode(s))
      c.close()
    }
  })
}

describe('streamUpload', () => {
  it('posts FormData and yields parsed SSE frames', async () => {
    const frames =
      'event: start\ndata: {"collection":"c1"}\n\n' +
      'event: ingestion_complete\ndata: {"collection":"c1","data_dir":"/x"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, body: bodyFromString(frames) })
    )

    const fd = new FormData()
    fd.append('collection', 'c1')

    const events: Array<{ event: string }> = []
    for await (const ev of streamUpload('/ingest/upload', fd)) {
      events.push({ event: ev.event })
    }

    expect(events).toEqual([{ event: 'start' }, { event: 'ingestion_complete' }])
    const call = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(call[1].method).toBe('POST')
    expect(call[1].body).toBe(fd)
  })
})
```

- [ ] **Step 2: Run; expect failure**

Run: `cd frontend && pnpm test src/api/upload.test.ts`
Expected: FAIL with `Cannot find module './upload'`.

- [ ] **Step 3: Write `frontend/src/api/upload.ts`**

```ts
import { url } from './client'
import type { SseEvent } from './sse'

export async function* streamUpload(
  path: string,
  formData: FormData,
  signal?: AbortSignal
): AsyncGenerator<SseEvent, void, unknown> {
  const res = await fetch(url(path), {
    method: 'POST',
    body: formData,
    signal
  })
  if (!res.ok || !res.body) {
    throw new Error(`Upload failed: ${res.status}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = 'message'
      const dataLines: string[] = []
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (dataLines.length === 0) continue
      const raw = dataLines.join('\n')
      try {
        yield { event, data: JSON.parse(raw) }
      } catch {
        yield { event, data: raw }
      }
    }
  }
}
```

- [ ] **Step 4: Run; expect pass**

Run: `cd frontend && pnpm test src/api/upload.test.ts`
Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/upload.ts frontend/src/api/upload.test.ts
git commit -m "Add multipart streamUpload helper for /ingest/upload"
```

---

### Task 1.5: Resource API modules

**Files:**
- Create: `frontend/src/api/collections.ts`
- Create: `frontend/src/api/sessions.ts`
- Create: `frontend/src/api/analysis.ts`
- Create: `frontend/src/api/chat.ts`
- Create: `frontend/src/api/ingest.ts`

- [ ] **Step 1: Write `frontend/src/api/collections.ts`**

```ts
import { apiDelete, apiGet, apiPost } from './client'
import type { DocumentRecord, NerStats } from './types'

export const listCollections = () => apiGet<string[]>('/collections/list')

export const selectCollection = (name: string) =>
  apiPost<{ ok: boolean; name: string }>('/collections/select', { name })

export const deleteCollection = (name: string) =>
  apiDelete<{ ok: boolean }>(`/collections/${encodeURIComponent(name)}`)

export const listDocuments = () =>
  apiGet<{ documents: DocumentRecord[] }>('/collections/documents')

export const getNerStats = (params: {
  top_k?: number
  min_mentions?: number
  entity_type?: string
  include_relations?: boolean
  entity_merge_mode?: 'orthographic' | 'exact'
}) => apiGet<NerStats>('/collections/ner/stats', params)

export const getNer = (refresh?: boolean) =>
  apiGet<{ entities: unknown[]; relations: unknown[] }>('/collections/ner', { refresh })

export const getHateSpeech = () =>
  apiGet<{ results: unknown[] }>('/collections/hate-speech')

export const getIeStats = (collection: string) =>
  apiGet<unknown>(`/collections/${encodeURIComponent(collection)}/ie-stats`)
```

- [ ] **Step 2: Write `frontend/src/api/sessions.ts`**

```ts
import { apiDelete, apiGet } from './client'
import type { SessionMessage, SessionSummary } from './types'

export const listSessions = () =>
  apiGet<{ sessions: SessionSummary[] }>('/sessions/list')

export const getSessionHistory = (id: string) =>
  apiGet<{ messages: SessionMessage[] }>(`/sessions/${encodeURIComponent(id)}/history`)

export const deleteSession = (id: string) =>
  apiDelete<{ ok: boolean }>(`/sessions/${encodeURIComponent(id)}`)
```

- [ ] **Step 3: Write `frontend/src/api/analysis.ts`**

```ts
import { apiPost } from './client'
import { streamSse } from './sse'
import type { SummaryResponse } from './types'

export const summarize = (refresh?: boolean) =>
  apiPost<SummaryResponse>('/summarize' + (refresh ? '?refresh=true' : ''))

export const streamSummary = (refresh?: boolean) =>
  streamSse('/summarize/stream' + (refresh ? '?refresh=true' : ''))
```

- [ ] **Step 4: Write `frontend/src/api/chat.ts`**

```ts
import { streamSse } from './sse'
import type { ChatRequest } from './types'

export const streamAgentChat = (req: ChatRequest, signal?: AbortSignal) =>
  streamSse('/agent/chat/stream', req, signal)

export const streamQuery = (req: ChatRequest, signal?: AbortSignal) =>
  streamSse('/stream_query', req, signal)
```

- [ ] **Step 5: Write `frontend/src/api/ingest.ts`**

```ts
import { streamUpload } from './upload'

export const streamIngestUpload = (
  collection: string,
  files: File[],
  hybrid: boolean,
  signal?: AbortSignal
) => {
  const fd = new FormData()
  fd.append('collection', collection)
  fd.append('hybrid', hybrid ? 'true' : 'false')
  for (const f of files) fd.append('files', f, f.name)
  return streamUpload('/ingest/upload', fd, signal)
}

export const sourcePreviewUrl = (collection: string, file_hash: string) =>
  `/sources/preview?collection=${encodeURIComponent(collection)}&file_hash=${encodeURIComponent(file_hash)}`
```

- [ ] **Step 6: Typecheck**

Run: `cd frontend && pnpm typecheck`
Expected: No errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api/
git commit -m "Add resource API modules for collections/sessions/chat/ingest/analysis"
```

---

## Phase 2 — State stores and query hooks

### Task 2.1: UI state store with localStorage persistence

**Files:**
- Create: `frontend/src/stores/ui.ts`
- Create: `frontend/src/stores/ui.test.ts`

- [ ] **Step 1: Write the failing test `frontend/src/stores/ui.test.ts`**

```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { useUiStore } from './ui'

beforeEach(() => {
  localStorage.clear()
  useUiStore.setState({ selectedCollection: null, currentSessionId: null, previewModal: null })
})

describe('useUiStore', () => {
  it('updates selected collection', () => {
    useUiStore.getState().setSelectedCollection('c1')
    expect(useUiStore.getState().selectedCollection).toBe('c1')
  })

  it('persists collection to localStorage', () => {
    useUiStore.getState().setSelectedCollection('c1')
    expect(JSON.parse(localStorage.getItem('docint-ui')!).state.selectedCollection).toBe('c1')
  })

  it('clears current session', () => {
    useUiStore.getState().setCurrentSessionId('s1')
    useUiStore.getState().setCurrentSessionId(null)
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })
})
```

- [ ] **Step 2: Run; expect failure**

Run: `cd frontend && pnpm test src/stores/ui.test.ts`
Expected: FAIL with `Cannot find module './ui'`.

- [ ] **Step 3: Write `frontend/src/stores/ui.ts`**

```ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface PreviewModal {
  collection: string
  file_hash: string
  filename: string
}

interface UiState {
  selectedCollection: string | null
  currentSessionId: string | null
  previewModal: PreviewModal | null
  setSelectedCollection: (name: string | null) => void
  setCurrentSessionId: (id: string | null) => void
  openPreview: (modal: PreviewModal) => void
  closePreview: () => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      selectedCollection: null,
      currentSessionId: null,
      previewModal: null,
      setSelectedCollection: (name) => set({ selectedCollection: name }),
      setCurrentSessionId: (id) => set({ currentSessionId: id }),
      openPreview: (modal) => set({ previewModal: modal }),
      closePreview: () => set({ previewModal: null })
    }),
    {
      name: 'docint-ui',
      partialize: (s) => ({
        selectedCollection: s.selectedCollection,
        currentSessionId: s.currentSessionId
      })
    }
  )
)
```

- [ ] **Step 4: Run; expect pass**

Run: `cd frontend && pnpm test src/stores/ui.test.ts`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/stores/
git commit -m "Add Zustand UI store with localStorage persistence"
```

---

### Task 2.2: Chat filters store

**Files:**
- Create: `frontend/src/stores/chatFilters.ts`

- [ ] **Step 1: Write `frontend/src/stores/chatFilters.ts`**

```ts
import { create } from 'zustand'
import type { MetadataFilter, QueryMode, RetrievalMode } from '@/api/types'

export interface CustomRule {
  id: string
  field: string
  operator: string
  value: string
}

interface ChatFiltersState {
  queryMode: QueryMode
  retrievalMode: RetrievalMode
  filterEnabled: boolean
  mimePattern: string
  dateFrom: string
  dateTo: string
  hateSpeechOnly: boolean
  customRules: CustomRule[]
  setQueryMode: (m: QueryMode) => void
  setRetrievalMode: (m: RetrievalMode) => void
  setFilterEnabled: (b: boolean) => void
  setMimePattern: (s: string) => void
  setDateFrom: (s: string) => void
  setDateTo: (s: string) => void
  setHateSpeechOnly: (b: boolean) => void
  addRule: () => void
  updateRule: (id: string, patch: Partial<CustomRule>) => void
  removeRule: (id: string) => void
  reset: () => void
  buildPayload: () => MetadataFilter[]
}

const initial = {
  queryMode: 'answer' as QueryMode,
  retrievalMode: 'session' as RetrievalMode,
  filterEnabled: false,
  mimePattern: '',
  dateFrom: '',
  dateTo: '',
  hateSpeechOnly: false,
  customRules: [] as CustomRule[]
}

export const useChatFiltersStore = create<ChatFiltersState>((set, get) => ({
  ...initial,
  setQueryMode: (queryMode) => set({ queryMode }),
  setRetrievalMode: (retrievalMode) => set({ retrievalMode }),
  setFilterEnabled: (filterEnabled) => set({ filterEnabled }),
  setMimePattern: (mimePattern) => set({ mimePattern }),
  setDateFrom: (dateFrom) => set({ dateFrom }),
  setDateTo: (dateTo) => set({ dateTo }),
  setHateSpeechOnly: (hateSpeechOnly) => set({ hateSpeechOnly }),
  addRule: () =>
    set((s) => ({
      customRules: [
        ...s.customRules,
        { id: crypto.randomUUID(), field: '', operator: 'equals', value: '' }
      ]
    })),
  updateRule: (id, patch) =>
    set((s) => ({
      customRules: s.customRules.map((r) => (r.id === id ? { ...r, ...patch } : r))
    })),
  removeRule: (id) =>
    set((s) => ({ customRules: s.customRules.filter((r) => r.id !== id) })),
  reset: () => set(initial),
  buildPayload: () => {
    const s = get()
    if (!s.filterEnabled) return []
    const out: MetadataFilter[] = []
    if (s.mimePattern) out.push({ field: 'mimetype', operator: 'matches', value: s.mimePattern })
    if (s.dateFrom) out.push({ field: 'date', operator: 'gte', value: s.dateFrom })
    if (s.dateTo) out.push({ field: 'date', operator: 'lte', value: s.dateTo })
    if (s.hateSpeechOnly) out.push({ field: 'hate_speech_flagged', operator: 'eq', value: true })
    for (const r of s.customRules) {
      if (r.field && r.operator) out.push({ field: r.field, operator: r.operator, value: r.value })
    }
    return out
  }
}))
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && pnpm typecheck`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/stores/chatFilters.ts
git commit -m "Add chat filters Zustand store"
```

---

### Task 2.3: Query client + base hooks

**Files:**
- Create: `frontend/src/api/queryClient.ts`
- Create: `frontend/src/hooks/useCollections.ts`
- Create: `frontend/src/hooks/useSessions.ts`
- Create: `frontend/src/hooks/useDocuments.ts`
- Create: `frontend/src/hooks/useNer.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write `frontend/src/api/queryClient.ts`**

```ts
import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1, refetchOnWindowFocus: false }
  }
})
```

- [ ] **Step 2: Write `frontend/src/hooks/useCollections.ts`**

```ts
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteCollection, listCollections, selectCollection } from '@/api/collections'

export const collectionsKey = ['collections'] as const

export function useCollections() {
  return useQuery({ queryKey: collectionsKey, queryFn: listCollections })
}

export function useSelectCollection() {
  return useMutation({ mutationFn: (name: string) => selectCollection(name) })
}

export function useDeleteCollection() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => deleteCollection(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: collectionsKey })
  })
}
```

- [ ] **Step 3: Write `frontend/src/hooks/useSessions.ts`**

```ts
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteSession, getSessionHistory, listSessions } from '@/api/sessions'

export const sessionsKey = ['sessions'] as const
export const sessionHistoryKey = (id: string) => ['sessions', id, 'history'] as const

export function useSessions() {
  return useQuery({ queryKey: sessionsKey, queryFn: listSessions })
}

export function useSessionHistory(id: string | null) {
  return useQuery({
    queryKey: id ? sessionHistoryKey(id) : ['sessions', 'none'],
    queryFn: () => getSessionHistory(id!),
    enabled: !!id
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => deleteSession(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: sessionsKey })
  })
}
```

- [ ] **Step 4: Write `frontend/src/hooks/useDocuments.ts`**

```ts
import { useQuery } from '@tanstack/react-query'
import { listDocuments } from '@/api/collections'
import { useUiStore } from '@/stores/ui'

export function useDocuments() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['documents', collection],
    queryFn: listDocuments,
    enabled: !!collection
  })
}
```

- [ ] **Step 5: Write `frontend/src/hooks/useNer.ts`**

```ts
import { useQuery } from '@tanstack/react-query'
import { getHateSpeech, getIeStats, getNer, getNerStats } from '@/api/collections'
import { useUiStore } from '@/stores/ui'

export function useNerStats(params: Parameters<typeof getNerStats>[0]) {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['ner-stats', collection, params],
    queryFn: () => getNerStats(params),
    enabled: !!collection
  })
}

export function useNer(refresh?: boolean) {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['ner', collection, refresh ?? false],
    queryFn: () => getNer(refresh),
    enabled: !!collection
  })
}

export function useHateSpeech() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['hate-speech', collection],
    queryFn: getHateSpeech,
    enabled: !!collection
  })
}

export function useIeStats() {
  const collection = useUiStore((s) => s.selectedCollection)
  return useQuery({
    queryKey: ['ie-stats', collection],
    queryFn: () => getIeStats(collection!),
    enabled: !!collection
  })
}
```

- [ ] **Step 6: Update `frontend/src/App.tsx`**

```tsx
import { QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { queryClient } from './api/queryClient'
import { Router } from './routes/Router'

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Router />
      </BrowserRouter>
    </QueryClientProvider>
  )
}
```

- [ ] **Step 7: Create stub `frontend/src/routes/Router.tsx`** (real routes added in later phases)

```tsx
export function Router() {
  return <div className="p-8">Routes coming online…</div>
}
```

- [ ] **Step 8: Typecheck and run tests**

Run: `cd frontend && pnpm typecheck && pnpm test`
Expected: All green.

- [ ] **Step 9: Commit**

```bash
git add frontend/
git commit -m "Wire QueryClient and base TanStack Query hooks"
```

---

## Phase 3 — Backend integration changes

### Task 3.1: Update CORS allowlist default to include Vite dev port

**Files:**
- Modify: `docint/utils/env_cfg.py:466` (the `default_cors_origins=` argument)
- Modify: `tests/test_env_cfg.py` (whichever test asserts the default)

- [ ] **Step 1: Read the relevant region**

Run: `grep -n "8501\|cors_allowed_origins\|default_cors_origins" /Users/himarc/dev/nos-tromo/docint/docint/utils/env_cfg.py`

- [ ] **Step 2: Update the default in `docint/utils/env_cfg.py`**

Change the default value passed to `default_cors_origins` so it includes `http://localhost:5173,http://127.0.0.1:5173` in addition to the existing `8501` entries:

```python
default_cors_origins: str = (
    "http://localhost:8501,http://127.0.0.1:8501,"
    "http://localhost:5173,http://127.0.0.1:5173"
),
```

- [ ] **Step 3: Find any test that pins the old default**

Run: `grep -n "8501" /Users/himarc/dev/nos-tromo/docint/tests/test_env_cfg.py`

If a test asserts the exact default string, update it to match. If the test reads via `os.environ` overrides only, no change needed.

- [ ] **Step 4: Run the suite**

Run: `cd /Users/himarc/dev/nos-tromo/docint && uv run pytest tests/test_env_cfg.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docint/utils/env_cfg.py tests/test_env_cfg.py
git commit -m "Allow Vite dev origin (:5173) in default CORS allowlist"
```

---

### Task 3.2: Mount frontend/dist as static files

**Files:**
- Modify: `docint/core/api.py` (append a static mount near the bottom of the module)
- Create: `tests/test_api_static_mount.py`

- [ ] **Step 1: Locate the FastAPI app construction**

Run: `grep -n "^app = FastAPI\|@app\." /Users/himarc/dev/nos-tromo/docint/docint/core/api.py | head -20`

- [ ] **Step 2: Write the failing test `tests/test_api_static_mount.py`**

```python
"""Verifies the SPA static mount behavior."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_static_mount_present_only_when_dist_exists(repo_root: Path) -> None:
    """The /static_index sentinel route always responds; / serves SPA only when built."""
    from docint.core.api import app

    client = TestClient(app)

    dist = repo_root / "frontend" / "dist" / "index.html"
    if dist.is_file():
        res = client.get("/")
        assert res.status_code == 200
        assert "<!doctype html" in res.text.lower() or "<html" in res.text.lower()
    else:
        res = client.get("/")
        assert res.status_code in (404, 405)
```

- [ ] **Step 3: Run; expect failure**

Run: `cd /Users/himarc/dev/nos-tromo/docint && uv run pytest tests/test_api_static_mount.py -v`
Expected: FAIL (the route doesn't exist yet, but the test imports `app` and may pass on the 404 branch — see step 4).

- [ ] **Step 4: Add the static mount at the bottom of `docint/core/api.py`**

Append after all other route registrations:

```python
# --- Frontend SPA ---
from pathlib import Path as _Path  # noqa: E402  (intentional late import)
from fastapi.staticfiles import StaticFiles  # noqa: E402

_FRONTEND_DIST = _Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=_FRONTEND_DIST, html=True),
        name="frontend",
    )
```

- [ ] **Step 5: Run again**

Run: `cd /Users/himarc/dev/nos-tromo/docint && uv run pytest tests/test_api_static_mount.py -v`
Expected: PASS.

- [ ] **Step 6: Confirm existing API tests still pass**

Run: `cd /Users/himarc/dev/nos-tromo/docint && uv run pytest tests/test_api.py -v`
Expected: PASS (the mount is at `/` last, so it should not shadow named routes).

- [ ] **Step 7: Commit**

```bash
git add docint/core/api.py tests/test_api_static_mount.py
git commit -m "Mount frontend/dist as SPA static files when present"
```

---

## Phase 4 — Layout shell + sidebar

### Task 4.1: Layout shell with sidebar slot

**Files:**
- Create: `frontend/src/layout/Shell.tsx`
- Create: `frontend/src/layout/Shell.test.tsx`

- [ ] **Step 1: Write the failing test `frontend/src/layout/Shell.test.tsx`**

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Shell } from './Shell'

describe('Shell', () => {
  it('renders sidebar and main slot', () => {
    render(
      <MemoryRouter>
        <Shell>
          <p>main content</p>
        </Shell>
      </MemoryRouter>
    )
    expect(screen.getByText(/docint/i)).toBeInTheDocument()
    expect(screen.getByText('main content')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run; expect failure**

Run: `cd frontend && pnpm test src/layout/Shell.test.tsx`
Expected: FAIL with `Cannot find module './Shell'`.

- [ ] **Step 3: Write `frontend/src/layout/Shell.tsx`**

```tsx
import type { ReactNode } from 'react'
import { Sidebar } from './Sidebar'

export function Shell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  )
}
```

- [ ] **Step 4: Create a stub `frontend/src/layout/Sidebar.tsx`** (real content in next task)

```tsx
export function Sidebar() {
  return (
    <aside className="w-64 border-r border-border p-4">
      <h2 className="text-lg font-semibold">DocInt</h2>
    </aside>
  )
}
```

- [ ] **Step 5: Run; expect pass**

Run: `cd frontend && pnpm test src/layout/Shell.test.tsx`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/layout/
git commit -m "Add layout shell with sidebar slot"
```

---

### Task 4.2: Sidebar with navigation, collection select, sessions

**Files:**
- Modify: `frontend/src/layout/Sidebar.tsx`

- [ ] **Step 1: Write `frontend/src/layout/Sidebar.tsx`**

```tsx
import { NavLink, useNavigate } from 'react-router-dom'
import { useCollections, useDeleteCollection, useSelectCollection } from '@/hooks/useCollections'
import { useDeleteSession, useSessions } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { cn } from '@/lib/cn'

const NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/chat', label: 'Chat' },
  { to: '/ingest', label: 'Ingest' },
  { to: '/analysis', label: 'Analysis' },
  { to: '/inspector', label: 'Inspector' }
]

export function Sidebar() {
  const navigate = useNavigate()
  const { data: collections } = useCollections()
  const selectMutation = useSelectCollection()
  const deleteCollectionMutation = useDeleteCollection()
  const { data: sessionsData } = useSessions()
  const deleteSessionMutation = useDeleteSession()
  const selected = useUiStore((s) => s.selectedCollection)
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)

  const onSelectCollection = async (name: string) => {
    await selectMutation.mutateAsync(name)
    setSelected(name)
  }

  const onDeleteCollection = (name: string) => {
    if (!confirm(`Delete collection "${name}"? This cannot be undone.`)) return
    deleteCollectionMutation.mutate(name, {
      onSuccess: () => {
        if (selected === name) setSelected(null)
      }
    })
  }

  const onNewChat = () => {
    setCurrentSessionId(null)
    navigate('/chat')
  }

  const onPickSession = (id: string) => {
    setCurrentSessionId(id)
    navigate(`/chat/${id}`)
  }

  const onDeleteSession = (id: string) => {
    if (!confirm('Delete this chat?')) return
    deleteSessionMutation.mutate(id, {
      onSuccess: () => {
        if (currentSessionId === id) setCurrentSessionId(null)
      }
    })
  }

  return (
    <aside className="w-72 border-r border-border p-4 flex flex-col gap-4 bg-zinc-950">
      <h2 className="text-lg font-semibold tracking-tight">DocInt</h2>

      <nav className="flex flex-col gap-1">
        {NAV.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'rounded-md px-3 py-2 text-sm hover:bg-zinc-800',
                isActive && 'bg-zinc-800 text-foreground'
              )
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>

      <section>
        <label className="text-xs uppercase text-muted-foreground">Collection</label>
        <select
          className="mt-1 w-full bg-zinc-900 border border-border rounded-md px-2 py-1 text-sm"
          value={selected ?? ''}
          onChange={(e) => onSelectCollection(e.target.value)}
        >
          <option value="" disabled>
            — choose —
          </option>
          {collections?.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        {selected && (
          <button
            type="button"
            onClick={() => onDeleteCollection(selected)}
            className="mt-2 text-xs text-red-400 hover:text-red-300"
          >
            Delete this collection
          </button>
        )}
      </section>

      <section className="flex-1 min-h-0 flex flex-col">
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase text-muted-foreground">Sessions</label>
          <button
            type="button"
            onClick={onNewChat}
            className="text-xs px-2 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700"
          >
            + New
          </button>
        </div>
        <ul className="mt-2 flex-1 overflow-auto space-y-1">
          {sessionsData?.sessions.map((s) => {
            const active = currentSessionId === s.session_id
            return (
              <li key={s.session_id} className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => onPickSession(s.session_id)}
                  className={cn(
                    'flex-1 text-left text-sm px-2 py-1 rounded-md truncate',
                    active ? 'bg-zinc-800' : 'hover:bg-zinc-900'
                  )}
                  title={s.title ?? s.session_id}
                >
                  {s.title?.trim() || `Session ${s.session_id.slice(0, 8)}`}
                </button>
                <button
                  type="button"
                  onClick={() => onDeleteSession(s.session_id)}
                  className="text-xs text-zinc-500 hover:text-red-400 px-1"
                  aria-label="Delete session"
                >
                  ×
                </button>
              </li>
            )
          })}
        </ul>
      </section>
    </aside>
  )
}
```

- [ ] **Step 2: Update Shell test to wrap with QueryClientProvider**

The new Sidebar uses TanStack Query hooks, so the test now needs a `QueryClientProvider`. Rewrite `frontend/src/layout/Shell.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Shell } from './Shell'

function renderShell() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Shell>
          <p>main content</p>
        </Shell>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('Shell', () => {
  it('renders sidebar and main slot', () => {
    renderShell()
    expect(screen.getByText(/docint/i)).toBeInTheDocument()
    expect(screen.getByText('main content')).toBeInTheDocument()
  })
})
```

Run: `cd frontend && pnpm test src/layout/Shell.test.tsx`
Expected: PASS.

- [ ] **Step 3: Typecheck**

Run: `cd frontend && pnpm typecheck`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/layout/
git commit -m "Implement sidebar with collection select, sessions, navigation"
```

---

## Phase 5 — Routes scaffolding

### Task 5.1: Wire all five top-level routes with placeholder pages

**Files:**
- Modify: `frontend/src/routes/Router.tsx`
- Create: `frontend/src/routes/Dashboard.tsx`
- Create: `frontend/src/routes/Chat.tsx`
- Create: `frontend/src/routes/Ingest.tsx`
- Create: `frontend/src/routes/Analysis.tsx`
- Create: `frontend/src/routes/Inspector.tsx`

- [ ] **Step 1: Write `frontend/src/routes/Router.tsx`**

```tsx
import { Route, Routes } from 'react-router-dom'
import { Shell } from '@/layout/Shell'
import { Dashboard } from './Dashboard'
import { Chat } from './Chat'
import { Ingest } from './Ingest'
import { Analysis } from './Analysis'
import { Inspector } from './Inspector'

export function Router() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/chat/:sessionId" element={<Chat />} />
        <Route path="/ingest" element={<Ingest />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/inspector" element={<Inspector />} />
      </Routes>
    </Shell>
  )
}
```

- [ ] **Step 2: Create placeholder route files**

Each is a single-line page that will be filled in by later tasks. Write each file explicitly:

`frontend/src/routes/Dashboard.tsx`:
```tsx
export function Dashboard() {
  return <div className="p-8">Dashboard — coming up.</div>
}
```

`frontend/src/routes/Chat.tsx`:
```tsx
export function Chat() {
  return <div className="p-8">Chat — coming up.</div>
}
```

`frontend/src/routes/Ingest.tsx`:
```tsx
export function Ingest() {
  return <div className="p-8">Ingest — coming up.</div>
}
```

`frontend/src/routes/Analysis.tsx`:
```tsx
export function Analysis() {
  return <div className="p-8">Analysis — coming up.</div>
}
```

`frontend/src/routes/Inspector.tsx`:
```tsx
export function Inspector() {
  return <div className="p-8">Inspector — coming up.</div>
}
```

- [ ] **Step 3: Verify in the browser**

Run: `cd frontend && pnpm dev`
Click each sidebar link; the corresponding placeholder text shows. Confirm the URL changes and the layout chrome stays put.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/routes/
git commit -m "Wire all top-level routes with placeholder pages"
```

---

## Phase 6 — Dashboard

### Task 6.1: KPI card and dashboard data

**Files:**
- Create: `frontend/src/components/common/KpiCard.tsx`
- Create: `frontend/src/components/common/KpiCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { KpiCard } from './KpiCard'

describe('KpiCard', () => {
  it('renders label and value', () => {
    render(<KpiCard label="Documents" value={42} />)
    expect(screen.getByText('Documents')).toBeInTheDocument()
    expect(screen.getByText('42')).toBeInTheDocument()
  })

  it('renders dash for missing value', () => {
    render(<KpiCard label="X" value={null} />)
    expect(screen.getByText('—')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run; expect failure**

Run: `cd frontend && pnpm test src/components/common/KpiCard.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Write `frontend/src/components/common/KpiCard.tsx`**

```tsx
export function KpiCard({
  label,
  value,
  hint
}: {
  label: string
  value: number | string | null
  hint?: string
}) {
  return (
    <div className="rounded-lg border border-border bg-zinc-900 p-4">
      <div className="text-xs uppercase text-muted-foreground">{label}</div>
      <div className="mt-2 text-2xl font-semibold tracking-tight">{value ?? '—'}</div>
      {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
    </div>
  )
}
```

- [ ] **Step 4: Run; expect pass**

Run: `cd frontend && pnpm test src/components/common/KpiCard.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/common/
git commit -m "Add KpiCard component"
```

---

### Task 6.2: Top entities chart

**Files:**
- Create: `frontend/src/components/dashboard/TopEntitiesChart.tsx`

- [ ] **Step 1: Write `frontend/src/components/dashboard/TopEntitiesChart.tsx`**

```tsx
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export interface EntityRow {
  text: string
  type: string
  count: number
}

export function TopEntitiesChart({ data }: { data: EntityRow[] }) {
  if (!data.length) {
    return <div className="text-sm text-muted-foreground">No entities yet.</div>
  }
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} layout="vertical" margin={{ left: 24, right: 16 }}>
        <XAxis type="number" stroke="rgb(161,161,170)" />
        <YAxis type="category" dataKey="text" stroke="rgb(161,161,170)" width={140} />
        <Tooltip
          contentStyle={{
            background: 'rgb(24 24 27)',
            border: '1px solid rgb(39 39 42)',
            borderRadius: 6
          }}
        />
        <Bar dataKey="count" fill="rgb(244 244 245)" />
      </BarChart>
    </ResponsiveContainer>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && pnpm typecheck`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/dashboard/
git commit -m "Add TopEntitiesChart Recharts component"
```

---

### Task 6.3: Wire Dashboard route

**Files:**
- Modify: `frontend/src/routes/Dashboard.tsx`

- [ ] **Step 1: Replace `frontend/src/routes/Dashboard.tsx`**

```tsx
import { useState } from 'react'
import { useCollections } from '@/hooks/useCollections'
import { useDocuments } from '@/hooks/useDocuments'
import { useNerStats } from '@/hooks/useNer'
import { useSessions } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { KpiCard } from '@/components/common/KpiCard'
import { TopEntitiesChart } from '@/components/dashboard/TopEntitiesChart'

export function Dashboard() {
  const collection = useUiStore((s) => s.selectedCollection)
  const { data: collections, isError } = useCollections()
  const { data: documents } = useDocuments()
  const { data: sessionsData } = useSessions()
  const [topK, setTopK] = useState(15)
  const [minMentions, setMinMentions] = useState(2)
  const stats = useNerStats({ top_k: topK, min_mentions: minMentions, include_relations: false })

  return (
    <div className="p-8 space-y-6">
      <h1 className="text-2xl font-semibold">Dashboard</h1>

      <div className="grid grid-cols-4 gap-4">
        <KpiCard label="Backend" value={isError ? 'offline' : 'online'} />
        <KpiCard label="Collections" value={collections?.length ?? null} />
        <KpiCard
          label="Documents"
          value={collection ? documents?.documents.length ?? null : '—'}
          hint={collection ? `in ${collection}` : 'select a collection'}
        />
        <KpiCard label="Sessions" value={sessionsData?.sessions.length ?? null} />
      </div>

      <section className="rounded-lg border border-border bg-zinc-900 p-4">
        <header className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium">Top entities</h2>
          <div className="flex gap-3 text-sm">
            <label className="flex items-center gap-2">
              top-k
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-16 bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
            <label className="flex items-center gap-2">
              min mentions
              <input
                type="number"
                min={1}
                value={minMentions}
                onChange={(e) => setMinMentions(Number(e.target.value))}
                className="w-16 bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
          </div>
        </header>
        {!collection ? (
          <div className="text-sm text-muted-foreground">Select a collection to see entities.</div>
        ) : (
          <TopEntitiesChart data={stats.data?.top_entities ?? []} />
        )}
      </section>

      <section className="rounded-lg border border-border bg-zinc-900 p-4">
        <h2 className="text-lg font-medium mb-3">Recent sessions</h2>
        <ul className="space-y-1 text-sm">
          {sessionsData?.sessions.slice(0, 10).map((s) => (
            <li key={s.session_id} className="flex justify-between">
              <span>{s.title?.trim() || s.session_id.slice(0, 8)}</span>
              <span className="text-muted-foreground">{s.message_count} msgs</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
```

- [ ] **Step 2: Verify visually**

Run: `cd frontend && pnpm dev` (with backend running). Navigate to `/`. Confirm KPI cards render, top-entities chart renders if a collection is selected.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/routes/Dashboard.tsx
git commit -m "Implement Dashboard route with KPIs, entities chart, recent sessions"
```

---

## Phase 7 — Chat

### Task 7.1: Citation component

**Files:**
- Create: `frontend/src/components/chat/Citation.tsx`
- Create: `frontend/src/lib/sourceLabel.ts`
- Create: `frontend/src/lib/sourceLabel.test.ts`

- [ ] **Step 1: Write the failing test for the helper**

`frontend/src/lib/sourceLabel.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { sourceLabel } from './sourceLabel'

describe('sourceLabel', () => {
  it('uses filename + page_label when present', () => {
    expect(sourceLabel({ filename: 'a.pdf', page_label: '12' } as any)).toBe('a.pdf · p. 12')
  })
  it('uses filename + row_label when no page', () => {
    expect(sourceLabel({ filename: 'a.csv', row_label: 'r3' } as any)).toBe('a.csv · row r3')
  })
  it('falls back to filename', () => {
    expect(sourceLabel({ filename: 'x.txt' } as any)).toBe('x.txt')
  })
})
```

- [ ] **Step 2: Run; expect failure**

Run: `cd frontend && pnpm test src/lib/sourceLabel.test.ts`
Expected: FAIL.

- [ ] **Step 3: Write `frontend/src/lib/sourceLabel.ts`**

```ts
import type { Source } from '@/api/types'

export function sourceLabel(s: Source): string {
  if (s.page_label) return `${s.filename} · p. ${s.page_label}`
  if (s.row_label) return `${s.filename} · row ${s.row_label}`
  return s.filename
}

export function formatScore(n: number): string {
  return n.toFixed(3)
}
```

- [ ] **Step 4: Run; expect pass**

Run: `cd frontend && pnpm test src/lib/sourceLabel.test.ts`
Expected: PASS.

- [ ] **Step 5: Write `frontend/src/components/chat/Citation.tsx`**

```tsx
import { useState } from 'react'
import type { Source } from '@/api/types'
import { sourcePreviewUrl } from '@/api/ingest'
import { useUiStore } from '@/stores/ui'
import { formatScore, sourceLabel } from '@/lib/sourceLabel'

export function Citation({ source }: { source: Source }) {
  const [open, setOpen] = useState(false)
  const collection = useUiStore((s) => s.selectedCollection)
  return (
    <div className="rounded-md border border-border bg-zinc-900 px-3 py-2 text-sm">
      <button
        type="button"
        className="flex items-center justify-between w-full gap-2"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="truncate">{sourceLabel(source)}</span>
        <span className="text-xs text-muted-foreground">{formatScore(source.score)}</span>
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {source.text && (
            <pre className="whitespace-pre-wrap text-xs bg-zinc-950 p-2 rounded">
              {source.text}
            </pre>
          )}
          {collection && (
            <a
              href={sourcePreviewUrl(collection, source.file_hash)}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              Open original ↗
            </a>
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/chat/ frontend/src/lib/sourceLabel.ts frontend/src/lib/sourceLabel.test.ts
git commit -m "Add Citation component and sourceLabel helper"
```

---

### Task 7.2: Validation banner + GraphRAG debug panel

**Files:**
- Create: `frontend/src/components/chat/ValidationBanner.tsx`
- Create: `frontend/src/components/chat/GraphDebugPanel.tsx`

- [ ] **Step 1: Write `frontend/src/components/chat/ValidationBanner.tsx`**

```tsx
import type { ValidationFields } from '@/api/types'
import { cn } from '@/lib/cn'

export function ValidationBanner({ v }: { v: ValidationFields }) {
  if (!v.validation_status) return null
  const tone =
    v.validation_status === 'ok'
      ? 'border-emerald-700 bg-emerald-950 text-emerald-200'
      : v.validation_status === 'warning'
        ? 'border-amber-700 bg-amber-950 text-amber-200'
        : 'border-red-700 bg-red-950 text-red-200'
  return (
    <div className={cn('rounded-md border px-3 py-2 text-xs', tone)}>
      <div className="font-medium uppercase tracking-wide">{v.validation_status}</div>
      {v.validation_message && <div className="mt-1">{v.validation_message}</div>}
    </div>
  )
}
```

- [ ] **Step 2: Write `frontend/src/components/chat/GraphDebugPanel.tsx`**

```tsx
import { useState } from 'react'

export function GraphDebugPanel({ data }: { data: unknown }) {
  const [open, setOpen] = useState(false)
  if (!data) return null
  return (
    <div className="rounded-md border border-border bg-zinc-900">
      <button
        type="button"
        className="w-full text-left px-3 py-2 text-xs uppercase text-muted-foreground"
        onClick={() => setOpen((v) => !v)}
      >
        Graph debug {open ? '▾' : '▸'}
      </button>
      {open && (
        <pre className="text-xs p-3 overflow-auto max-h-80 bg-zinc-950">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/chat/
git commit -m "Add ValidationBanner and GraphDebugPanel for chat"
```

---

### Task 7.3: Filter builder

**Files:**
- Create: `frontend/src/components/chat/FilterBuilder.tsx`

- [ ] **Step 1: Write `frontend/src/components/chat/FilterBuilder.tsx`**

```tsx
import { useChatFiltersStore } from '@/stores/chatFilters'

const OPERATORS = ['equals', 'not_equals', 'contains', 'gte', 'lte']

export function FilterBuilder() {
  const s = useChatFiltersStore()
  return (
    <div className="rounded-md border border-border bg-zinc-900 p-3 space-y-3 text-sm">
      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={s.filterEnabled}
          onChange={(e) => s.setFilterEnabled(e.target.checked)}
        />
        Enable metadata filters
      </label>

      {s.filterEnabled && (
        <>
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">MIME pattern</span>
              <input
                value={s.mimePattern}
                onChange={(e) => s.setMimePattern(e.target.value)}
                placeholder="application/pdf"
                className="bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Hate-speech only</span>
              <input
                type="checkbox"
                checked={s.hateSpeechOnly}
                onChange={(e) => s.setHateSpeechOnly(e.target.checked)}
                className="self-start"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Date from</span>
              <input
                type="date"
                value={s.dateFrom}
                onChange={(e) => s.setDateFrom(e.target.value)}
                className="bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">Date to</span>
              <input
                type="date"
                value={s.dateTo}
                onChange={(e) => s.setDateTo(e.target.value)}
                className="bg-zinc-950 border border-border rounded-md px-2 py-1"
              />
            </label>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-muted-foreground">Custom rules</span>
              <button
                type="button"
                onClick={() => s.addRule()}
                className="text-xs px-2 py-1 rounded-md bg-zinc-800 hover:bg-zinc-700"
              >
                + Rule
              </button>
            </div>
            <ul className="space-y-2">
              {s.customRules.map((r) => (
                <li key={r.id} className="grid grid-cols-[1fr_auto_1fr_auto] gap-2 items-center">
                  <input
                    value={r.field}
                    onChange={(e) => s.updateRule(r.id, { field: e.target.value })}
                    placeholder="field"
                    className="bg-zinc-950 border border-border rounded-md px-2 py-1"
                  />
                  <select
                    value={r.operator}
                    onChange={(e) => s.updateRule(r.id, { operator: e.target.value })}
                    className="bg-zinc-950 border border-border rounded-md px-2 py-1"
                  >
                    {OPERATORS.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                  <input
                    value={r.value}
                    onChange={(e) => s.updateRule(r.id, { value: e.target.value })}
                    placeholder="value"
                    className="bg-zinc-950 border border-border rounded-md px-2 py-1"
                  />
                  <button
                    type="button"
                    onClick={() => s.removeRule(r.id)}
                    className="text-xs text-red-400 hover:text-red-300"
                    aria-label="Remove rule"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/chat/
git commit -m "Add FilterBuilder for chat metadata filters"
```

---

### Task 7.4: Chat route with streaming

**Files:**
- Modify: `frontend/src/routes/Chat.tsx`
- Create: `frontend/src/components/chat/ChatTurn.tsx`

- [ ] **Step 1: Write `frontend/src/components/chat/ChatTurn.tsx`**

```tsx
import type { ChatFinalEvent, Source } from '@/api/types'
import { Citation } from './Citation'
import { ValidationBanner } from './ValidationBanner'
import { GraphDebugPanel } from './GraphDebugPanel'

export interface ChatTurnData {
  user: string
  assistant: string
  done: boolean
  meta: ChatFinalEvent | null
}

export function ChatTurn({ turn }: { turn: ChatTurnData }) {
  const sources: Source[] = turn.meta?.sources ?? []
  return (
    <article className="space-y-3">
      <div className="rounded-md bg-zinc-900 px-4 py-2 self-end max-w-2xl ml-auto">
        <div className="text-xs text-muted-foreground mb-1">You</div>
        <div className="whitespace-pre-wrap">{turn.user}</div>
      </div>
      <div className="rounded-md bg-zinc-950 border border-border px-4 py-3">
        <div className="text-xs text-muted-foreground mb-1">Assistant</div>
        <div className="whitespace-pre-wrap">
          {turn.assistant || (turn.done ? '(no answer)' : '…')}
        </div>
        {turn.meta && <ValidationBanner v={turn.meta} />}
        {sources.length > 0 && (
          <div className="mt-3 space-y-2">
            <div className="text-xs uppercase text-muted-foreground">Sources</div>
            {sources.map((s) => (
              <Citation key={s.id} source={s} />
            ))}
          </div>
        )}
        {turn.meta?.graph_debug && <GraphDebugPanel data={turn.meta.graph_debug} />}
      </div>
    </article>
  )
}
```

- [ ] **Step 2: Write `frontend/src/routes/Chat.tsx`**

```tsx
import { useEffect, useReducer, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { streamAgentChat } from '@/api/chat'
import type { ChatFinalEvent } from '@/api/types'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSessionHistory } from '@/hooks/useSessions'
import { useUiStore } from '@/stores/ui'
import { useQueryClient } from '@tanstack/react-query'
import { sessionsKey } from '@/hooks/useSessions'
import { ChatTurn, type ChatTurnData } from '@/components/chat/ChatTurn'
import { FilterBuilder } from '@/components/chat/FilterBuilder'

interface State {
  turns: ChatTurnData[]
  inflight: boolean
  draft: string
}
type Action =
  | { type: 'set_turns'; turns: ChatTurnData[] }
  | { type: 'set_draft'; value: string }
  | { type: 'start'; user: string }
  | { type: 'token'; token: string }
  | { type: 'finalize'; meta: ChatFinalEvent }
  | { type: 'fail' }

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'set_turns':
      return { ...s, turns: a.turns }
    case 'set_draft':
      return { ...s, draft: a.value }
    case 'start':
      return {
        ...s,
        draft: '',
        inflight: true,
        turns: [...s.turns, { user: a.user, assistant: '', done: false, meta: null }]
      }
    case 'token': {
      const last = s.turns[s.turns.length - 1]
      const updated = { ...last, assistant: last.assistant + a.token }
      return { ...s, turns: [...s.turns.slice(0, -1), updated] }
    }
    case 'finalize': {
      const last = s.turns[s.turns.length - 1]
      const finalText = a.meta.answer ?? a.meta.message ?? last.assistant
      const updated = { ...last, assistant: finalText, done: true, meta: a.meta }
      return { ...s, inflight: false, turns: [...s.turns.slice(0, -1), updated] }
    }
    case 'fail': {
      const last = s.turns[s.turns.length - 1]
      const updated = { ...last, done: true }
      return { ...s, inflight: false, turns: [...s.turns.slice(0, -1), updated] }
    }
  }
}

export function Chat() {
  const params = useParams()
  const sessionIdParam = params.sessionId ?? null
  const setCurrentSessionId = useUiStore((s) => s.setCurrentSessionId)
  const currentSessionId = useUiStore((s) => s.currentSessionId)
  const filters = useChatFiltersStore()
  const qc = useQueryClient()
  const history = useSessionHistory(sessionIdParam)
  const [state, dispatch] = useReducer(reducer, { turns: [], inflight: false, draft: '' })
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    setCurrentSessionId(sessionIdParam)
  }, [sessionIdParam, setCurrentSessionId])

  useEffect(() => {
    if (!history.data) return
    const turns: ChatTurnData[] = []
    let pendingUser: string | null = null
    for (const m of history.data.messages) {
      if (m.role === 'user') pendingUser = m.content
      else {
        turns.push({
          user: pendingUser ?? '',
          assistant: m.content,
          done: true,
          meta: m.citations ? ({ sources: m.citations, session_id: sessionIdParam ?? '' } as ChatFinalEvent) : null
        })
        pendingUser = null
      }
    }
    dispatch({ type: 'set_turns', turns })
  }, [history.data, sessionIdParam])

  const send = async () => {
    const message = state.draft.trim()
    if (!message || state.inflight) return
    dispatch({ type: 'start', user: message })

    const ac = new AbortController()
    abortRef.current = ac
    try {
      for await (const ev of streamAgentChat(
        {
          question: message,
          session_id: currentSessionId ?? undefined,
          metadata_filters: filters.buildPayload(),
          query_mode: filters.queryMode,
          retrieval_mode: filters.retrievalMode
        },
        ac.signal
      )) {
        if (ev.event === 'token') {
          const tok = (ev.data as { token?: string }).token ?? ''
          dispatch({ type: 'token', token: tok })
        } else if (ev.event === 'done') {
          const final = ev.data as ChatFinalEvent
          dispatch({ type: 'finalize', meta: final })
          if (!currentSessionId && final.session_id) {
            setCurrentSessionId(final.session_id)
          }
          qc.invalidateQueries({ queryKey: sessionsKey })
        }
      }
    } catch {
      dispatch({ type: 'fail' })
    } finally {
      abortRef.current = null
    }
  }

  return (
    <div className="p-8 grid grid-cols-[1fr_22rem] gap-6 h-full">
      <section className="flex flex-col h-full">
        <h1 className="text-2xl font-semibold mb-4">Chat</h1>
        <div className="flex-1 overflow-auto space-y-6 pr-2">
          {state.turns.map((t, i) => (
            <ChatTurn key={i} turn={t} />
          ))}
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            void send()
          }}
          className="mt-4 flex gap-2"
        >
          <textarea
            value={state.draft}
            onChange={(e) => dispatch({ type: 'set_draft', value: e.target.value })}
            placeholder="Ask something…"
            rows={2}
            className="flex-1 bg-zinc-900 border border-border rounded-md px-3 py-2"
          />
          <button
            type="submit"
            disabled={state.inflight || !state.draft.trim()}
            className="px-4 py-2 rounded-md bg-zinc-100 text-zinc-900 disabled:opacity-50"
          >
            {state.inflight ? '…' : 'Send'}
          </button>
        </form>
      </section>

      <aside className="space-y-4">
        <div className="flex flex-col gap-2 text-sm">
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase text-muted-foreground">Query mode</span>
            <select
              value={filters.queryMode}
              onChange={(e) => filters.setQueryMode(e.target.value as typeof filters.queryMode)}
              className="bg-zinc-900 border border-border rounded-md px-2 py-1"
            >
              <option value="answer">Answer</option>
              <option value="entity_occurrence">Entity occurrence</option>
              <option value="entity_occurrence_multi">Entity occurrence (multi)</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase text-muted-foreground">Retrieval</span>
            <select
              value={filters.retrievalMode}
              onChange={(e) =>
                filters.setRetrievalMode(e.target.value as typeof filters.retrievalMode)
              }
              className="bg-zinc-900 border border-border rounded-md px-2 py-1"
            >
              <option value="session">Session</option>
              <option value="stateless">Stateless</option>
            </select>
          </label>
        </div>
        <FilterBuilder />
      </aside>
    </div>
  )
}
```

- [ ] **Step 3: Verify in browser**

Run: `cd frontend && pnpm dev`. Pick a collection in the sidebar, navigate to `/chat`, send a message. Confirm tokens stream into the assistant bubble, sources appear after `done`, validation banner renders if returned, and the sidebar's session list updates.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/routes/Chat.tsx frontend/src/components/chat/
git commit -m "Implement Chat route with streaming, filters, citations"
```

---

## Phase 8 — Ingest

### Task 8.1: Dropzone

**Files:**
- Create: `frontend/src/components/ingest/Dropzone.tsx`

- [ ] **Step 1: Write `frontend/src/components/ingest/Dropzone.tsx`**

```tsx
import { useRef, useState, type DragEvent } from 'react'
import { cn } from '@/lib/cn'

export function Dropzone({
  onFiles,
  disabled
}: {
  onFiles: (files: File[]) => void
  disabled?: boolean
}) {
  const [hover, setHover] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handle = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setHover(false)
    if (disabled) return
    const list = Array.from(e.dataTransfer.files)
    if (list.length) onFiles(list)
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setHover(true)
      }}
      onDragLeave={() => setHover(false)}
      onDrop={handle}
      onClick={() => inputRef.current?.click()}
      className={cn(
        'rounded-lg border-2 border-dashed p-10 text-center cursor-pointer',
        hover ? 'border-foreground bg-zinc-900' : 'border-border',
        disabled && 'opacity-50 pointer-events-none'
      )}
    >
      <p>Drop files here or click to choose.</p>
      <input
        ref={inputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          const list = Array.from(e.target.files ?? [])
          if (list.length) onFiles(list)
          e.target.value = ''
        }}
      />
    </div>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/ingest/
git commit -m "Add Dropzone component"
```

---

### Task 8.2: Event timeline + Ingest route

**Files:**
- Create: `frontend/src/components/ingest/EventTimeline.tsx`
- Modify: `frontend/src/routes/Ingest.tsx`

- [ ] **Step 1: Write `frontend/src/components/ingest/EventTimeline.tsx`**

```tsx
import type { IngestEvent } from '@/api/types'

const ICON: Record<string, string> = {
  start: '▶',
  upload_progress: '↑',
  file_saved: '✓',
  ingestion_started: '⚙',
  ingestion_progress: '·',
  ingestion_complete: '✓',
  error: '!'
}

export function EventTimeline({ events }: { events: IngestEvent[] }) {
  return (
    <ol className="text-sm font-mono space-y-1">
      {events.map((e, i) => (
        <li key={i} className="flex gap-2">
          <span className="text-muted-foreground w-4">{ICON[e.event] ?? '•'}</span>
          <span className="text-muted-foreground w-44 shrink-0">{e.event}</span>
          <span className="truncate">{describe(e)}</span>
        </li>
      ))}
    </ol>
  )
}

function describe(e: IngestEvent): string {
  const d = e.data as Record<string, unknown>
  if (e.event === 'upload_progress') return `${d.filename} (${d.bytes_written} bytes)`
  if (e.event === 'file_saved') return `${d.filename} → ${d.file_hash}`
  if (e.event === 'ingestion_progress') return String(d.message ?? '')
  if (e.event === 'ingestion_complete') return `done · ${d.collection}`
  if (e.event === 'error') return String(d.message ?? d)
  return Object.entries(d).map(([k, v]) => `${k}=${String(v)}`).join(' ')
}
```

- [ ] **Step 2: Write `frontend/src/routes/Ingest.tsx`**

```tsx
import { useReducer, useState } from 'react'
import { streamIngestUpload } from '@/api/ingest'
import { useSelectCollection, useCollections, collectionsKey } from '@/hooks/useCollections'
import { useQueryClient } from '@tanstack/react-query'
import { useUiStore } from '@/stores/ui'
import type { IngestEvent } from '@/api/types'
import { Dropzone } from '@/components/ingest/Dropzone'
import { EventTimeline } from '@/components/ingest/EventTimeline'

interface State {
  collection: string
  hybrid: boolean
  files: File[]
  events: IngestEvent[]
  busy: boolean
}
type Action =
  | { type: 'set_collection'; v: string }
  | { type: 'set_hybrid'; v: boolean }
  | { type: 'add_files'; v: File[] }
  | { type: 'reset_files' }
  | { type: 'start' }
  | { type: 'event'; v: IngestEvent }
  | { type: 'done' }

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'set_collection':
      return { ...s, collection: a.v }
    case 'set_hybrid':
      return { ...s, hybrid: a.v }
    case 'add_files':
      return { ...s, files: [...s.files, ...a.v] }
    case 'reset_files':
      return { ...s, files: [] }
    case 'start':
      return { ...s, busy: true, events: [] }
    case 'event':
      return { ...s, events: [...s.events, a.v] }
    case 'done':
      return { ...s, busy: false, files: [] }
  }
}

export function Ingest() {
  const [state, dispatch] = useReducer(reducer, {
    collection: '',
    hybrid: true,
    files: [],
    events: [],
    busy: false
  })
  const [error, setError] = useState<string | null>(null)
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  const selectMutation = useSelectCollection()
  const qc = useQueryClient()
  const { data: collections } = useCollections()

  const submit = async () => {
    if (!state.collection || state.files.length === 0) return
    setError(null)
    dispatch({ type: 'start' })
    try {
      for await (const ev of streamIngestUpload(state.collection, state.files, state.hybrid)) {
        dispatch({ type: 'event', v: ev as IngestEvent })
        if (ev.event === 'ingestion_complete') {
          await selectMutation.mutateAsync(state.collection)
          setSelected(state.collection)
          await qc.invalidateQueries({ queryKey: collectionsKey })
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      dispatch({ type: 'done' })
    }
  }

  return (
    <div className="p-8 max-w-3xl space-y-4">
      <h1 className="text-2xl font-semibold">Ingest</h1>

      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs uppercase text-muted-foreground">Collection</span>
          <input
            list="existing-collections"
            value={state.collection}
            onChange={(e) => dispatch({ type: 'set_collection', v: e.target.value })}
            placeholder="my-collection"
            className="bg-zinc-900 border border-border rounded-md px-2 py-1"
          />
          <datalist id="existing-collections">
            {collections?.map((c) => (
              <option key={c} value={c} />
            ))}
          </datalist>
        </label>
        <label className="flex items-center gap-2 text-sm mt-5">
          <input
            type="checkbox"
            checked={state.hybrid}
            onChange={(e) => dispatch({ type: 'set_hybrid', v: e.target.checked })}
          />
          Hybrid search
        </label>
      </div>

      <Dropzone disabled={state.busy} onFiles={(v) => dispatch({ type: 'add_files', v })} />

      {state.files.length > 0 && (
        <ul className="text-sm space-y-1">
          {state.files.map((f) => (
            <li key={f.name}>
              {f.name} <span className="text-muted-foreground">({f.size} bytes)</span>
            </li>
          ))}
        </ul>
      )}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={submit}
          disabled={state.busy || !state.collection || state.files.length === 0}
          className="px-4 py-2 rounded-md bg-zinc-100 text-zinc-900 disabled:opacity-50"
        >
          {state.busy ? 'Ingesting…' : 'Ingest'}
        </button>
        {state.files.length > 0 && (
          <button
            type="button"
            onClick={() => dispatch({ type: 'reset_files' })}
            className="px-4 py-2 rounded-md border border-border"
          >
            Clear files
          </button>
        )}
      </div>

      {error && <div className="text-red-400 text-sm">{error}</div>}
      {state.events.length > 0 && (
        <div className="rounded-lg border border-border bg-zinc-900 p-4">
          <EventTimeline events={state.events} />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Verify in browser**

Run: `cd frontend && pnpm dev`. Pick a collection name, drop a small PDF or text file, click Ingest. Confirm the event timeline streams in real time and the new collection becomes the selected one.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/routes/Ingest.tsx frontend/src/components/ingest/
git commit -m "Implement Ingest route with dropzone and event timeline"
```

---

## Phase 9 — Analysis

### Task 9.1: CSV export helper

**Files:**
- Create: `frontend/src/lib/csv.ts`
- Create: `frontend/src/lib/csv.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, it, expect } from 'vitest'
import { toCsv } from './csv'

describe('toCsv', () => {
  it('emits header and rows', () => {
    const csv = toCsv([{ a: 1, b: 'x' }, { a: 2, b: 'y, z' }], ['a', 'b'])
    expect(csv).toBe('a,b\n1,x\n2,"y, z"')
  })

  it('escapes quotes', () => {
    const csv = toCsv([{ a: 'he said "hi"' }], ['a'])
    expect(csv).toBe('a\n"he said ""hi"""')
  })
})
```

- [ ] **Step 2: Run; expect failure**

Run: `cd frontend && pnpm test src/lib/csv.test.ts`
Expected: FAIL.

- [ ] **Step 3: Write `frontend/src/lib/csv.ts`**

```ts
function escape(value: unknown): string {
  if (value === null || value === undefined) return ''
  const s = String(value)
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

export function toCsv<T extends Record<string, unknown>>(rows: T[], columns: (keyof T)[]): string {
  const header = columns.map((c) => escape(c)).join(',')
  const body = rows.map((r) => columns.map((c) => escape(r[c])).join(',')).join('\n')
  return `${header}\n${body}`
}

export function downloadCsv(filename: string, csv: string): void {
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}
```

- [ ] **Step 4: Run; expect pass**

Run: `cd frontend && pnpm test src/lib/csv.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/csv.ts frontend/src/lib/csv.test.ts
git commit -m "Add CSV export helper"
```

---

### Task 9.2: Analysis route — NER tab

**Files:**
- Modify: `frontend/src/routes/Analysis.tsx`
- Create: `frontend/src/components/analysis/NerTable.tsx`

- [ ] **Step 1: Write `frontend/src/components/analysis/NerTable.tsx`**

```tsx
import { useMemo, useState } from 'react'
import { downloadCsv, toCsv } from '@/lib/csv'

export interface EntityRow {
  text: string
  type: string
  count: number
}

export function NerTable({ rows }: { rows: EntityRow[] }) {
  const [filter, setFilter] = useState('')
  const [type, setType] = useState('')
  const types = useMemo(() => Array.from(new Set(rows.map((r) => r.type))).sort(), [rows])
  const filtered = rows.filter(
    (r) => (!type || r.type === type) && (!filter || r.text.toLowerCase().includes(filter.toLowerCase()))
  )

  return (
    <div className="space-y-3">
      <div className="flex gap-2 text-sm">
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="filter…"
          className="bg-zinc-900 border border-border rounded-md px-2 py-1 flex-1"
        />
        <select
          value={type}
          onChange={(e) => setType(e.target.value)}
          className="bg-zinc-900 border border-border rounded-md px-2 py-1"
        >
          <option value="">All types</option>
          {types.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => downloadCsv('entities.csv', toCsv(filtered, ['text', 'type', 'count']))}
          className="px-3 py-1 rounded-md border border-border"
        >
          CSV
        </button>
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-muted-foreground">
          <tr>
            <th className="py-2">Text</th>
            <th>Type</th>
            <th className="text-right">Count</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((r, i) => (
            <tr key={i} className="border-t border-border">
              <td className="py-1">{r.text}</td>
              <td>{r.type}</td>
              <td className="text-right">{r.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Write the route shell `frontend/src/routes/Analysis.tsx`** (only NER tab so far; other tabs added in next tasks)

```tsx
import { useState } from 'react'
import { useNerStats } from '@/hooks/useNer'
import { NerTable } from '@/components/analysis/NerTable'
import { cn } from '@/lib/cn'

const TABS = ['NER', 'Hate speech', 'Summary'] as const
type Tab = (typeof TABS)[number]

export function Analysis() {
  const [tab, setTab] = useState<Tab>('NER')
  const stats = useNerStats({ top_k: 100, min_mentions: 1, include_relations: true })

  return (
    <div className="p-8 space-y-4">
      <h1 className="text-2xl font-semibold">Analysis</h1>
      <nav className="flex gap-2 border-b border-border">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={cn(
              'px-3 py-2 text-sm -mb-px border-b-2',
              tab === t ? 'border-foreground' : 'border-transparent text-muted-foreground'
            )}
          >
            {t}
          </button>
        ))}
      </nav>
      {tab === 'NER' && <NerTable rows={stats.data?.top_entities ?? []} />}
      {tab !== 'NER' && <div className="text-sm text-muted-foreground">{tab} — wired in next task.</div>}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/routes/Analysis.tsx frontend/src/components/analysis/
git commit -m "Implement Analysis NER tab with CSV export"
```

---

### Task 9.3: Analysis hate-speech tab + streaming summary

**Files:**
- Create: `frontend/src/components/analysis/HateSpeechTable.tsx`
- Create: `frontend/src/components/analysis/SummaryPanel.tsx`
- Modify: `frontend/src/routes/Analysis.tsx`

- [ ] **Step 1: Write `frontend/src/components/analysis/HateSpeechTable.tsx`**

```tsx
import { downloadCsv, toCsv } from '@/lib/csv'

export interface HateSpeechRow {
  filename: string
  page_label?: string | null
  text: string
  score?: number
}

export function HateSpeechTable({ rows }: { rows: HateSpeechRow[] }) {
  if (!rows.length) return <div className="text-sm text-muted-foreground">No flagged content.</div>
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() =>
            downloadCsv(
              'hate-speech.csv',
              toCsv(rows, ['filename', 'page_label', 'score', 'text'])
            )
          }
          className="px-3 py-1 rounded-md border border-border text-sm"
        >
          CSV
        </button>
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-muted-foreground">
          <tr>
            <th className="py-2">File</th>
            <th>Page</th>
            <th>Score</th>
            <th>Text</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-border align-top">
              <td className="py-1">{r.filename}</td>
              <td>{r.page_label ?? ''}</td>
              <td>{r.score?.toFixed(3) ?? ''}</td>
              <td className="whitespace-pre-wrap">{r.text}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Write `frontend/src/components/analysis/SummaryPanel.tsx`**

```tsx
import { useReducer } from 'react'
import { streamSummary } from '@/api/analysis'
import { Citation } from '@/components/chat/Citation'
import { ValidationBanner } from '@/components/chat/ValidationBanner'
import type { SummaryResponse } from '@/api/types'

interface State {
  text: string
  done: boolean
  busy: boolean
  meta: SummaryResponse | null
  error: string | null
}
type Action =
  | { type: 'start' }
  | { type: 'token'; v: string }
  | { type: 'done'; meta: SummaryResponse }
  | { type: 'fail'; error: string }

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'start':
      return { text: '', done: false, busy: true, meta: null, error: null }
    case 'token':
      return { ...s, text: s.text + a.v }
    case 'done':
      return { ...s, busy: false, done: true, meta: a.meta, text: a.meta.summary || s.text }
    case 'fail':
      return { ...s, busy: false, done: true, error: a.error }
  }
}

export function SummaryPanel() {
  const [state, dispatch] = useReducer(reducer, {
    text: '',
    done: false,
    busy: false,
    meta: null,
    error: null
  })

  const generate = async (refresh: boolean) => {
    dispatch({ type: 'start' })
    try {
      for await (const ev of streamSummary(refresh)) {
        if (ev.event === 'token') dispatch({ type: 'token', v: (ev.data as { token: string }).token })
        else if (ev.event === 'done') dispatch({ type: 'done', meta: ev.data as SummaryResponse })
      }
    } catch (e) {
      dispatch({ type: 'fail', error: e instanceof Error ? e.message : String(e) })
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => generate(false)}
          disabled={state.busy}
          className="px-3 py-1 rounded-md bg-zinc-100 text-zinc-900 disabled:opacity-50"
        >
          {state.busy ? 'Generating…' : 'Generate'}
        </button>
        <button
          type="button"
          onClick={() => generate(true)}
          disabled={state.busy}
          className="px-3 py-1 rounded-md border border-border"
        >
          Refresh
        </button>
      </div>

      {state.error && <div className="text-red-400 text-sm">{state.error}</div>}
      {state.text && (
        <div className="rounded-md border border-border bg-zinc-900 p-4 whitespace-pre-wrap text-sm">
          {state.text}
        </div>
      )}
      {state.meta && <ValidationBanner v={state.meta} />}
      {state.meta?.summary_diagnostics && (
        <div className="text-xs text-muted-foreground">
          Coverage {state.meta.summary_diagnostics.covered_documents}/
          {state.meta.summary_diagnostics.total_documents} · ratio{' '}
          {(state.meta.summary_diagnostics.coverage_ratio * 100).toFixed(0)}%
        </div>
      )}
      {state.meta?.sources && state.meta.sources.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs uppercase text-muted-foreground">Sources</div>
          {state.meta.sources.map((s) => (
            <Citation key={s.id} source={s} />
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Rewrite `frontend/src/routes/Analysis.tsx`** in full

```tsx
import { useState } from 'react'
import { useHateSpeech, useNerStats } from '@/hooks/useNer'
import { NerTable } from '@/components/analysis/NerTable'
import { HateSpeechTable, type HateSpeechRow } from '@/components/analysis/HateSpeechTable'
import { SummaryPanel } from '@/components/analysis/SummaryPanel'
import { cn } from '@/lib/cn'

const TABS = ['NER', 'Hate speech', 'Summary'] as const
type Tab = (typeof TABS)[number]

export function Analysis() {
  const [tab, setTab] = useState<Tab>('NER')
  const stats = useNerStats({ top_k: 100, min_mentions: 1, include_relations: true })
  const hate = useHateSpeech()

  return (
    <div className="p-8 space-y-4">
      <h1 className="text-2xl font-semibold">Analysis</h1>
      <nav className="flex gap-2 border-b border-border">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={cn(
              'px-3 py-2 text-sm -mb-px border-b-2',
              tab === t ? 'border-foreground' : 'border-transparent text-muted-foreground'
            )}
          >
            {t}
          </button>
        ))}
      </nav>
      {tab === 'NER' && <NerTable rows={stats.data?.top_entities ?? []} />}
      {tab === 'Hate speech' && (
        <HateSpeechTable rows={(hate.data?.results ?? []) as HateSpeechRow[]} />
      )}
      {tab === 'Summary' && <SummaryPanel />}
    </div>
  )
}
```

- [ ] **Step 4: Verify in browser**

Run: `cd frontend && pnpm dev`. Click each tab; confirm NER table renders, hate-speech tab shows results or empty state, Summary tab streams tokens and shows sources + diagnostics.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/routes/Analysis.tsx frontend/src/components/analysis/
git commit -m "Implement Analysis hate-speech and streaming Summary tabs"
```

---

## Phase 10 — Inspector

### Task 10.1: Document table

**Files:**
- Create: `frontend/src/components/inspector/DocumentTable.tsx`
- Modify: `frontend/src/routes/Inspector.tsx`

- [ ] **Step 1: Write `frontend/src/components/inspector/DocumentTable.tsx`**

```tsx
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
  type ColumnDef
} from '@tanstack/react-table'
import { useMemo, useState } from 'react'
import type { DocumentRecord } from '@/api/types'
import { downloadCsv, toCsv } from '@/lib/csv'

const COLUMNS: ColumnDef<DocumentRecord>[] = [
  { accessorKey: 'filename', header: 'Filename' },
  { accessorKey: 'mimetype', header: 'MIME' },
  {
    accessorFn: (r) => r.page_count ?? r.row_count ?? 0,
    id: 'units',
    header: 'Pages/Rows'
  },
  { accessorKey: 'node_count', header: 'Nodes' },
  {
    accessorFn: (r) => (r.entity_types ?? []).join(', '),
    id: 'entity_types',
    header: 'Entity types'
  },
  { accessorKey: 'file_hash', header: 'Hash' }
]

export function DocumentTable({ docs }: { docs: DocumentRecord[] }) {
  const [sorting, setSorting] = useState<SortingState>([])
  const data = useMemo(() => docs, [docs])
  const table = useReactTable({
    data,
    columns: COLUMNS,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel()
  })
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => downloadCsv('documents.csv', toCsv(docs, ['filename', 'mimetype', 'page_count', 'row_count', 'node_count', 'file_hash']))}
          className="px-3 py-1 rounded-md border border-border text-sm"
        >
          CSV
        </button>
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-muted-foreground">
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => (
                <th
                  key={h.id}
                  onClick={h.column.getToggleSortingHandler()}
                  className="cursor-pointer py-2 select-none"
                >
                  {flexRender(h.column.columnDef.header, h.getContext())}
                  {h.column.getIsSorted() === 'asc' && ' ↑'}
                  {h.column.getIsSorted() === 'desc' && ' ↓'}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-t border-border">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="py-1 align-top">
                  {flexRender(cell.column.columnDef.cell ?? cell.column.columnDef.header, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Replace `frontend/src/routes/Inspector.tsx`**

```tsx
import { useDocuments } from '@/hooks/useDocuments'
import { DocumentTable } from '@/components/inspector/DocumentTable'

export function Inspector() {
  const { data, isLoading } = useDocuments()
  return (
    <div className="p-8 space-y-4">
      <h1 className="text-2xl font-semibold">Inspector</h1>
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading…</div>
      ) : (
        <DocumentTable docs={data?.documents ?? []} />
      )}
    </div>
  )
}
```

- [ ] **Step 3: Verify**

Run: `cd frontend && pnpm dev`, navigate to `/inspector`, click column headers to sort, click CSV. Confirm sorting works and CSV downloads.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/routes/Inspector.tsx frontend/src/components/inspector/
git commit -m "Implement Inspector with sortable TanStack Table and CSV export"
```

---

### Task 10.2: Session ZIP archive

**Files:**
- Create: `frontend/src/components/inspector/SessionZipButton.tsx`
- Modify: `frontend/src/routes/Inspector.tsx`

- [ ] **Step 1: Write `frontend/src/components/inspector/SessionZipButton.tsx`**

```tsx
import { useState } from 'react'
import JSZip from 'jszip'
import { useUiStore } from '@/stores/ui'
import { useSessionHistory } from '@/hooks/useSessions'
import { sourcePreviewUrl } from '@/api/ingest'

export function SessionZipButton() {
  const sessionId = useUiStore((s) => s.currentSessionId)
  const collection = useUiStore((s) => s.selectedCollection)
  const history = useSessionHistory(sessionId)
  const [busy, setBusy] = useState(false)

  const build = async () => {
    if (!sessionId || !collection || !history.data) return
    setBusy(true)
    try {
      const zip = new JSZip()
      const seen = new Set<string>()
      for (const m of history.data.messages) {
        for (const c of m.citations ?? []) {
          if (seen.has(c.file_hash)) continue
          seen.add(c.file_hash)
          const res = await fetch(sourcePreviewUrl(collection, c.file_hash))
          if (!res.ok) continue
          zip.file(c.filename, await res.blob())
        }
      }
      const blob = await zip.generateAsync({ type: 'blob' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `session-${sessionId}-sources.zip`
      a.click()
      URL.revokeObjectURL(a.href)
    } finally {
      setBusy(false)
    }
  }

  if (!sessionId || !collection) return null
  return (
    <button
      type="button"
      onClick={build}
      disabled={busy}
      className="px-3 py-1 rounded-md border border-border text-sm disabled:opacity-50"
    >
      {busy ? 'Building…' : 'Download session sources (ZIP)'}
    </button>
  )
}
```

- [ ] **Step 2: Add the button to `frontend/src/routes/Inspector.tsx`**

```tsx
import { SessionZipButton } from '@/components/inspector/SessionZipButton'
// ...
<div className="flex justify-between items-center">
  <h1 className="text-2xl font-semibold">Inspector</h1>
  <SessionZipButton />
</div>
```

- [ ] **Step 3: Verify**

Run: `cd frontend && pnpm dev`, run a chat that produces citations, navigate to `/inspector`, click "Download session sources (ZIP)". Confirm a ZIP downloads with the source files inside.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/inspector/ frontend/src/routes/Inspector.tsx
git commit -m "Add session source ZIP archive in Inspector"
```

---

## Phase 11 — Production build pipeline

### Task 11.1: Repoint the docint console script to uvicorn

**Files:**
- Create: `docint/cli/serve.py`
- Modify: `pyproject.toml` (the `[project.scripts]` table)

- [ ] **Step 1: Write `docint/cli/serve.py`**

```python
"""Console entry point that runs the FastAPI app via uvicorn.

Replaces the old Streamlit ``docint.app:run`` entry point.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the FastAPI app on host 0.0.0.0 with the configured port."""
    host = os.getenv("DOCINT_HOST", "0.0.0.0")
    port = int(os.getenv("DOCINT_PORT", "8000"))
    uvicorn.run("docint.core.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update `pyproject.toml`**

Find:

```toml
docint = "docint.app:run"
```

Replace with:

```toml
docint = "docint.cli.serve:main"
```

- [ ] **Step 3: Reinstall the project**

Run: `cd /Users/himarc/dev/nos-tromo/docint && uv sync --extra cpu`
Expected: no errors.

- [ ] **Step 4: Smoke test the new entry point**

Run: `uv run docint &` then `curl -s http://localhost:8000/collections/list` then `kill %1`
Expected: a JSON array (possibly empty) is returned.

- [ ] **Step 5: Commit**

```bash
git add docint/cli/serve.py pyproject.toml
git commit -m "Repoint docint console script to uvicorn entry point"
```

---

### Task 11.2: Build the SPA inside the backend Docker images

**Files:**
- Modify: `Dockerfile.backend.cpu` (add a frontend-builder stage and a COPY of the dist)
- Modify: `Dockerfile.backend.cuda` (same)
- Create: `frontend/.dockerignore`

- [ ] **Step 1: Write `frontend/.dockerignore`**

```
node_modules
dist
.vite
coverage
```

- [ ] **Step 2: Read `Dockerfile.backend.cpu` to find insertion points**

Run: `grep -n "FROM\|COPY\|CMD\|EXPOSE" /Users/himarc/dev/nos-tromo/docint/Dockerfile.backend.cpu`

- [ ] **Step 3: Add a frontend-builder stage at the top of `Dockerfile.backend.cpu`**

Insert before the existing first `FROM`:

```dockerfile
FROM node:20-alpine AS frontend-builder
WORKDIR /build
RUN corepack enable && corepack prepare pnpm@9.12.0 --activate
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ .
RUN pnpm build
```

Then in the runtime stage, add a `COPY` after the application source is copied:

```dockerfile
COPY --from=frontend-builder /build/dist /app/frontend/dist
```

- [ ] **Step 4: Repeat the same edits in `Dockerfile.backend.cuda`**

- [ ] **Step 5: Build the CPU image and confirm the SPA is reachable**

Run: `cd /Users/himarc/dev/nos-tromo/docint && docker build -f Dockerfile.backend.cpu -t docint-backend-cpu:test .`
Expected: build succeeds.

Then start it briefly: `docker run --rm -p 8000:8000 docint-backend-cpu:test &`
After ~10 s: `curl -s http://localhost:8000/ | head -1`
Expected: HTML beginning with `<!doctype html>`.

`docker kill <container-id>` to stop.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile.backend.cpu Dockerfile.backend.cuda frontend/.dockerignore
git commit -m "Build SPA inside backend Docker images"
```

---

### Task 11.3: Remove Streamlit from compose, Dockerfile, and dependency list

**Files:**
- Modify: `docker-compose.yml` (delete the `frontend-cpu`, `frontend-cuda` service blocks and their YAML anchors)
- Delete: `Dockerfile.frontend`
- Modify: `pyproject.toml` (remove `streamlit`, the `frontend` dependency-group, and `watchdog` if only Streamlit needed it — verify with grep first)
- Delete: `docint/ui/`
- Delete: `docint/app.py`
- Modify: `tests/test_ingest_ui.py` and `tests/test_app_ner.py` (delete or rewrite — they import from `docint.ui` and `docint.app`)

- [ ] **Step 1: Confirm what depends on Streamlit and `docint.ui`**

Run: `grep -rn "import streamlit\|from streamlit\|docint\.ui\|docint\.app" /Users/himarc/dev/nos-tromo/docint --include="*.py"`

Expected: hits in `docint/app.py`, `docint/ui/*`, `tests/test_ingest_ui.py`, `tests/test_app_ner.py`. No production code outside `docint/ui/` should import streamlit.

- [ ] **Step 2: Delete the Streamlit code and Dockerfile**

```bash
rm -r docint/ui/
rm docint/app.py
rm Dockerfile.frontend
```

- [ ] **Step 3: Delete or rewrite the affected tests**

If `test_ingest_ui.py` and `test_app_ner.py` only verify Streamlit-side behavior, delete them. If they test data-shaping helpers (such as `aggregate_ner`), move those helpers to a new module `docint/utils/ner_aggregate.py` and re-point the test imports. Use `grep` to decide:

Run: `grep -n "aggregate_ner\|format_score\|normalize_entities\|normalize_relations\|source_label" /Users/himarc/dev/nos-tromo/docint/tests/test_app_ner.py /Users/himarc/dev/nos-tromo/docint/tests/test_ingest_ui.py`

For helpers that have non-UI value, port them to `docint/utils/ner_aggregate.py` (one function per port, copy the source verbatim) and update imports. Otherwise delete the test files:

```bash
git rm tests/test_app_ner.py tests/test_ingest_ui.py
```

- [ ] **Step 4: Edit `pyproject.toml`**

Remove these lines from the `[project] dependencies` array:

```toml
"streamlit>=1.56.0",
"watchdog>=6.0.0",
```

(`watchdog` is only used by Streamlit; `grep -rn "watchdog" docint/` should confirm.)

Remove the `[dependency-groups] frontend = [...]` block entirely.

- [ ] **Step 5: Edit `docker-compose.yml`**

Delete the `frontend-cpu` and `frontend-cuda` service blocks (lines around 80-130) and their YAML anchors `&frontend-build`, `&frontend-ports`, `&frontend-volumes`, `&frontend-networks`. Verify nothing else references those anchors:

Run: `grep -n "frontend-" /Users/himarc/dev/nos-tromo/docint/docker-compose.yml`
Expected after edit: no matches.

If `${DOCINT_HOST_PORT:-8501}` is referenced elsewhere, leave it — it just controls the (now backend-only) host-port mapping if anyone still sets it.

- [ ] **Step 6: Reinstall and run the suite**

Run: `cd /Users/himarc/dev/nos-tromo/docint && uv sync --extra cpu && uv run pytest`
Expected: all remaining tests pass.

- [ ] **Step 7: Verify compose still validates**

Run: `cd /Users/himarc/dev/nos-tromo/docint && docker compose --profile cpu config > /dev/null`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "Remove Streamlit code, Dockerfile, dependency, and compose services"
```

---

### Task 11.4: Update README and CLAUDE.md

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Edit `README.md`**

- Change "a Streamlit UI" to "a React SPA served by the FastAPI backend" in the bullet list at the top.
- Replace the "Open the app — UI: <http://localhost:8501>, API: <http://localhost:8000>" lines with a single "App: <http://localhost:8000>" entry.
- Replace the local-dev "Start the UI in another terminal" block (`uv run docint`) with:

  ```bash
  cd frontend
  pnpm install
  pnpm dev      # → http://localhost:5173 (proxies /api to :8000)
  ```

- Update the "Repository Shape" entry: change `docint/ui: Streamlit pages and components` to `frontend/: React SPA (Vite + TypeScript)`.

- [ ] **Step 2: Edit `CLAUDE.md`**

In the "Commands" block:

- Remove `# Start Streamlit UI` and `uv run docint`.
- Add a new "Start frontend (Vite dev server)" block:

  ```bash
  cd frontend && pnpm install
  cd frontend && pnpm dev
  cd frontend && pnpm test
  cd frontend && pnpm build
  ```

In the "Architecture" block, change the request flow to:

```
React SPA (frontend/) → FastAPI (docint/core/api.py) → AgentOrchestrator (docint/agents/)
    → understanding → clarification → retrieval → generation
    → RAG engine (docint/core/rag.py) ↔ Qdrant vector store
```

In "Key Conventions", remove the bullet about `Streamlit UI pages live in docint/ui/` and replace it with: `Frontend lives in frontend/. Keep business logic in the API/agents layer. Frontend dev: pnpm; tests: Vitest.`

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Document React SPA frontend in README and CLAUDE.md"
```

---

### Task 11.5: Final smoke test

- [ ] **Step 1: Build a fresh image end-to-end**

Run: `cd /Users/himarc/dev/nos-tromo/docint && docker compose --profile cpu build`
Expected: build succeeds.

- [ ] **Step 2: Bring the stack up**

Run: `docker compose --profile cpu up -d`
Wait ~30 s for Qdrant + backend health.

- [ ] **Step 3: Hit the SPA**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/`
Expected: `200`.

- [ ] **Step 4: Hit a JSON route**

Run: `curl -s http://localhost:8000/collections/list`
Expected: a JSON array.

- [ ] **Step 5: Open the browser**

Visit `http://localhost:8000/`. Walk every route (Dashboard, Chat, Ingest, Analysis, Inspector). Confirm dark theme, no console errors, no Streamlit chrome anywhere.

- [ ] **Step 6: Tear down**

Run: `docker compose down`

- [ ] **Step 7: Final commit (if any housekeeping changes were made)**

```bash
git status
# only commit if there are uncommitted housekeeping changes
```

---

## Self-review notes

- Every spec section has at least one task: scaffold (0.1-0.3), backend changes (3.1-3.2), state stores (2.1-2.2), query hooks (2.3), all five routes (6, 7, 8, 9, 10), Docker pipeline (11.2), Streamlit removal (11.3), docs (11.4).
- Every API endpoint listed in the spec is consumed: `/collections/list`, `/collections/select`, `/collections/{name}` DELETE, `/collections/documents`, `/collections/ner`, `/collections/ner/stats`, `/collections/hate-speech`, `/collections/{c}/ie-stats`, `/sessions/list`, `/sessions/{id}/history`, `/sessions/{id}` DELETE, `/agent/chat/stream`, `/stream_query`, `/summarize/stream`, `/ingest/upload`, `/sources/preview`.
- Naming consistent across tasks: `useUiStore`, `useChatFiltersStore`, `streamSse`, `streamUpload`, `sessionsKey`, `collectionsKey`, `Source`, `ChatFinalEvent`.
- No "TBD" / "TODO" / "implement later" anywhere in the plan.
- Each task ends with a commit; commits are small and revertable.
