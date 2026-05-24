import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// @tanstack/react-virtual relies on ResizeObserver + measurable layout that
// happy-dom doesn't provide, so it renders zero virtual items in test mode.
// Mock the hook to return every item — tests assert behavior, not virtualization.
vi.mock('@tanstack/react-virtual', async () => {
  const actual = await vi.importActual<typeof import('@tanstack/react-virtual')>(
    '@tanstack/react-virtual'
  )
  type Opts = { count: number; estimateSize: (i: number) => number }
  return {
    ...actual,
    useVirtualizer: <O extends Opts>(opts: O) => {
      const sizes = Array.from({ length: opts.count }, (_, i) => opts.estimateSize(i))
      let offset = 0
      const items = sizes.map((size, index) => {
        const start = offset
        offset += size
        return { index, key: index, start, end: start + size, size, lane: 0 }
      })
      return {
        getVirtualItems: () => items,
        getTotalSize: () => offset,
        measureElement: () => undefined,
        scrollToIndex: () => undefined,
        scrollToOffset: () => undefined
      }
    }
  }
})

// Node.js v22+ defines localStorage as undefined globally, which prevents
// happy-dom from injecting its implementation via populateGlobal.
// This guard ensures the happy-dom localStorage is available in tests.
if (typeof localStorage === 'undefined') {
  const store: Record<string, string> = {}
  Object.defineProperty(globalThis, 'localStorage', {
    value: {
      getItem: (k: string) => store[k] ?? null,
      setItem: (k: string, v: string) => { store[k] = String(v) },
      removeItem: (k: string) => { delete store[k] },
      clear: () => { Object.keys(store).forEach(k => delete store[k]) },
      get length() { return Object.keys(store).length },
      key: (i: number) => Object.keys(store)[i] ?? null
    },
    writable: true,
    configurable: true
  })
}
