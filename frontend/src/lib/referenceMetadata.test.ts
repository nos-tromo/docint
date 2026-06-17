import { describe, expect, it } from 'vitest'
import { REFERENCE_METADATA_FIELDS, referenceMetadataItems } from './referenceMetadata'

describe('referenceMetadata detected_language', () => {
  it('registers a Detected Language field in the map', () => {
    expect(
      REFERENCE_METADATA_FIELDS.some((f) => f.key === 'detected_language' && f.label === 'Detected Language')
    ).toBe(true)
  })

  it('renders the detected language as a labeled item, after Language', () => {
    const items = referenceMetadataItems({ language: 'en', detected_language: 'de' })
    expect(items).toContainEqual({ label: 'Detected Language', value: 'de' })
    const labels = items.map((i) => i.label)
    expect(labels.indexOf('Detected Language')).toBeGreaterThan(labels.indexOf('Language'))
  })
})
