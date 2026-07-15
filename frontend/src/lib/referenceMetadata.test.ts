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

describe('referenceMetadata posting reference fields', () => {
  it('mirrors the Python registry: link ids + posting fields are registered in order', () => {
    const keys = REFERENCE_METADATA_FIELDS.map((f) => f.key as string)
    for (const key of [
      'posting_uuid',
      'posting_id',
      'media_id',
      'url',
      'posting_network',
      'posting_author',
      'posting_author_id',
      'posting_vanity',
      'posting_timestamp',
      'posting_url',
      'posting_text'
    ]) {
      expect(keys).toContain(key)
    }
    // Posting context is grouped right after the link ids, mirroring Python.
    expect(keys.indexOf('url')).toBe(keys.indexOf('media_id') + 1)
  })

  it('renders posting fields additively next to the artifact identity', () => {
    const items = referenceMetadataItems({
      network: 'nextext',
      type: 'transcript_segment',
      posting_network: 'Facebook',
      posting_author: 'Jane Poster',
      posting_url: 'https://fb.example/p1',
      posting_text: 'Original post body'
    })
    expect(items).toContainEqual({ label: 'Network', value: 'nextext' })
    expect(items).toContainEqual({ label: 'Posting Network', value: 'Facebook' })
    expect(items).toContainEqual({ label: 'Posting Author', value: 'Jane Poster' })
    expect(items).toContainEqual({ label: 'Posting URL', value: 'https://fb.example/p1' })
    expect(items).toContainEqual({ label: 'Posting Text', value: 'Original post body' })
  })
})
