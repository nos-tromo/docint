import '@testing-library/jest-dom/vitest'

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
