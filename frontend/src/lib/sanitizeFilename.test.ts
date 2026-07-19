import { describe, expect, it } from 'vitest'
import { sanitizeExportFilename } from './sanitizeFilename'

describe('sanitizeExportFilename', () => {
  it('lowercases the input', () => {
    expect(sanitizeExportFilename('My Reports')).toBe('my-reports')
  })

  it('replaces spaces with dashes', () => {
    expect(sanitizeExportFilename('My Reports')).toBe('my-reports')
  })

  it('replaces slashes with dashes', () => {
    expect(sanitizeExportFilename('My Reports/2026')).toBe('my-reports-2026')
  })

  it('replaces special characters with dashes', () => {
    expect(sanitizeExportFilename('My@Reports#2026')).toBe('my-reports-2026')
  })

  it('collapses consecutive dashes', () => {
    expect(sanitizeExportFilename('My--Reports')).toBe('my-reports')
    expect(sanitizeExportFilename('My//Reports')).toBe('my-reports')
    expect(sanitizeExportFilename('My--//Reports')).toBe('my-reports')
  })

  it('trims leading and trailing dashes', () => {
    expect(sanitizeExportFilename('-My Reports-')).toBe('my-reports')
    expect(sanitizeExportFilename('---My Reports---')).toBe('my-reports')
  })

  it('preserves alphanumeric characters and allowed punctuation', () => {
    expect(sanitizeExportFilename('my_reports.2026')).toBe('my_reports.2026')
  })

  it('returns fallback when input is empty', () => {
    expect(sanitizeExportFilename('')).toBe('docint')
  })

  it('returns fallback when input becomes empty after sanitization', () => {
    expect(sanitizeExportFilename('---')).toBe('docint')
    expect(sanitizeExportFilename('!!!')).toBe('docint')
  })

  it('handles complex real-world examples', () => {
    expect(sanitizeExportFilename('My Reports/2026')).toBe('my-reports-2026')
    expect(sanitizeExportFilename('Q4 2026 - Final Report')).toBe('q4-2026-final-report')
    expect(sanitizeExportFilename('Analysis (Draft v2)')).toBe('analysis-draft-v2')
  })
})
