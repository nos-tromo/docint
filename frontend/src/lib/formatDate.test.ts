import { afterEach, describe, expect, it } from 'vitest'
import { formatDate } from './formatDate'

const originalTz = process.env.TZ

afterEach(() => {
  process.env.TZ = originalTz
})

describe('formatDate', () => {
  it('writes a date the way the active language does', () => {
    expect(formatDate('2026-10-21T06:45:00Z', 'de')).toBe('21.10.2026')
    expect(formatDate('2026-10-21T06:45:00Z', 'en')).toBe('Oct 21, 2026')
  })

  it('reads the date in UTC, the day the server acts on, wherever the browser is', () => {
    // Late evening UTC is already the next day in the browser's zone here.
    process.env.TZ = 'Pacific/Kiritimati'
    expect(formatDate('2026-10-20T23:30:00Z', 'de')).toBe('20.10.2026')
  })
})
