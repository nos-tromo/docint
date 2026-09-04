import { describe, it, expect } from 'vitest'
import { parseClockSeconds, formatClockSeconds } from './clock'

describe('parseClockSeconds', () => {
  it('reads a bare number as seconds', () => {
    expect(parseClockSeconds('90')).toBe(90)
  })

  it('reads minutes and seconds', () => {
    expect(parseClockSeconds('1:30')).toBe(90)
  })

  it('reads hours, minutes and seconds', () => {
    expect(parseClockSeconds('01:02:03')).toBe(3723)
  })

  it('tolerates surrounding whitespace', () => {
    expect(parseClockSeconds('  2:00  ')).toBe(120)
  })

  it('refuses an empty field, which is no bound rather than zero', () => {
    expect(parseClockSeconds('')).toBeNull()
    expect(parseClockSeconds('   ')).toBeNull()
  })

  it('refuses prose', () => {
    expect(parseClockSeconds('half past')).toBeNull()
  })

  it('refuses a minute or second past 59, which reads as a typo', () => {
    expect(parseClockSeconds('1:75')).toBeNull()
  })

  it('refuses more parts than a clock has', () => {
    expect(parseClockSeconds('1:2:3:4')).toBeNull()
  })
})

describe('formatClockSeconds', () => {
  it('writes under an hour as m:ss', () => {
    expect(formatClockSeconds(90)).toBe('1:30')
  })

  it('writes an hour and over as h:mm:ss', () => {
    expect(formatClockSeconds(3723)).toBe('1:02:03')
  })

  it('never writes a negative position', () => {
    expect(formatClockSeconds(-5)).toBe('0:00')
  })
})
