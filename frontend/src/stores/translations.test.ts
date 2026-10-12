import { beforeEach, describe, expect, it } from 'vitest'
import { storedTranslation, useTranslationsStore } from './translations'

describe('translations store', () => {
  beforeEach(() => {
    useTranslationsStore.setState({ byText: {}, sameLanguage: {} })
  })

  it('stores a payload retrievable by its raw text', () => {
    const payload = { text: 'the translated line', target_lang: 'de', model: 'test-model' }

    useTranslationsStore.getState().put('the original line', payload)

    expect(storedTranslation('the original line')).toEqual(payload)
  })

  it('returns undefined for text nobody translated', () => {
    expect(storedTranslation('never translated')).toBeUndefined()
  })

  it('overwrites the payload when the same text is translated again', () => {
    const put = useTranslationsStore.getState().put
    put('same text', { text: 'first', target_lang: 'de', model: 'test-model' })

    put('same text', { text: 'second', target_lang: 'de', model: 'other-model' })

    expect(storedTranslation('same text')?.text).toBe('second')
  })

  it('ignores an empty key so untranslatable rows never share one entry', () => {
    useTranslationsStore.getState().put('', { text: 'nothing', target_lang: 'de', model: 'test-model' })

    expect(storedTranslation('')).toBeUndefined()
  })

  it('remembers text already in the target language apart from translations, ignoring an empty key', () => {
    const { markSameLanguage } = useTranslationsStore.getState()

    markSameLanguage('Schon deutsch.')
    markSameLanguage('')

    expect(useTranslationsStore.getState().sameLanguage).toEqual({ 'Schon deutsch.': true })
    // Never a translation: the report snapshots and "Add all" read only those.
    expect(storedTranslation('Schon deutsch.')).toBeUndefined()
  })
})
