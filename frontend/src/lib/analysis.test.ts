import { describe, it, expect } from 'vitest'
import { isPngDataUri } from './analysis'

describe('isPngDataUri', () => {
  it('accepts a base64 PNG data URI only', () => {
    expect(isPngDataUri('data:image/png;base64,iVBORw0KGgo=')).toBe(true)
    for (const other of [
      'data:image/svg+xml;base64,PHN2Zy8+',
      'data:image/jpeg;base64,/9j/4AAQ',
      'data:image/png,raw',
      'data:text/html;base64,PGI+',
      'https://example.test/chart.png',
      'javascript:alert(1)',
      ' data:image/png;base64,iVBORw0KGgo=',
      '',
      null,
      7
    ]) {
      expect(isPngDataUri(other)).toBe(false)
    }
  })
})
