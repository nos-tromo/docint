import { describe, expect, it } from 'vitest'
import { catalogs, type Strings } from '@/i18n'
import { hateCategoryLabel, hateConfidenceLabel } from './hateCategoryLabel'

const t = (lang: 'en' | 'de') => (key: keyof Strings) => catalogs[lang][key]

describe('hateCategoryLabel', () => {
  it('labels a fixed category in the active language', () => {
    expect(hateCategoryLabel('sexual_orientation', t('de'))).toBe('Sexuelle Orientierung')
  })

  it('shows a category outside the enum as stored', () => {
    expect(hateCategoryLabel('slur', t('de'))).toBe('slur')
  })
})

describe('hateConfidenceLabel', () => {
  it('labels the fixed confidence values in the active language', () => {
    expect(hateConfidenceLabel('high', t('de'))).toBe('hoch')
    expect(hateConfidenceLabel('MEDIUM', t('de'))).toBe('mittel')
    expect(hateConfidenceLabel('low', t('en'))).toBe('low')
  })

  it('shows an unrecognized value as stored', () => {
    expect(hateConfidenceLabel('certain', t('de'))).toBe('certain')
  })
})
