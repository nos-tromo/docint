import { describe, expect, it } from 'vitest'
import { REFERENCE_METADATA_FIELDS, referenceMetadataItems, referenceMetadataPills } from './referenceMetadata'

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

describe('referenceMetadataPills', () => {
  it('excludes opaque IDs and body-text fields', () => {
    const pills = referenceMetadataPills({
      network: 'Instagram',
      uuid: 'u-1',
      posting_uuid: 'pu-1',
      posting_id: 'p-1',
      media_id: 'm-1',
      author_id: 'a-1',
      posting_author_id: 'pa-1',
      text_id: 't-1',
      text: 'body',
      parent_text: 'parent',
      anchor_text: 'anchor'
    })
    expect(pills).toEqual([{ key: 'network', value: 'Instagram' }])
  })

  it('trims fractional seconds off timestamps', () => {
    const pills = referenceMetadataPills({
      posting_timestamp: '2025-09-17 15:15:30.000000',
      timestamp: '2026-02-14'
    })
    expect(pills).toContainEqual({ key: 'posting_timestamp', value: '2025-09-17 15:15:30' })
    expect(pills).toContainEqual({ key: 'timestamp', value: '2026-02-14' })
  })

  it('renders URLs as link pills with the open-link label', () => {
    const pills = referenceMetadataPills({ posting_url: 'https://fb.example/p1' })
    expect(pills).toEqual([
      { key: 'posting_url', value: 'Open link ↗', href: 'https://fb.example/p1' }
    ])
  })

  it('labels person-ish fields and leaves self-explanatory ones bare', () => {
    const pills = referenceMetadataPills({
      network: 'Instagram',
      type: 'posting',
      posting_author: 'Jane Poster',
      speaker: 'SPEAKER_00',
      language: 'de',
      source_file: 'clip.mp4'
    })
    expect(pills).toContainEqual({ key: 'posting_author', label: 'Posting Author', value: 'Jane Poster' })
    expect(pills).toContainEqual({ key: 'speaker', label: 'Speaker', value: 'SPEAKER_00' })
    expect(pills).toContainEqual({ key: 'network', value: 'Instagram' })
    expect(pills).toContainEqual({ key: 'type', value: 'posting' })
    expect(pills).toContainEqual({ key: 'language', value: 'de' })
    expect(pills).toContainEqual({ key: 'source_file', value: 'clip.mp4' })
  })

  it('preserves registry order and skips empty values', () => {
    const pills = referenceMetadataPills({
      url: 'https://ig.example/x',
      network: 'Instagram',
      posting_author: '   ',
      type: 'posting'
    })
    expect(pills.map((p) => p.key)).toEqual(['network', 'type', 'url'])
  })

  it('returns [] for undefined metadata', () => {
    expect(referenceMetadataPills(undefined)).toEqual([])
  })

  it('excludes posting_text like the other body-text fields', () => {
    const pills = referenceMetadataPills({
      network: 'Instagram',
      posting_text: 'A full posting body that should never render as a bare pill.'
    })
    expect(pills).toEqual([{ key: 'network', value: 'Instagram' }])
  })

  it('only emits an href for http(s) URLs, falling back to a plain value pill otherwise', () => {
    const pills = referenceMetadataPills({
      url: 'javascript:alert(1)',
      posting_url: 'https://fb.example/p1'
    })
    expect(pills).toContainEqual({ key: 'url', value: 'javascript:alert(1)' })
    expect(pills).toContainEqual({
      key: 'posting_url',
      value: 'Open link ↗',
      href: 'https://fb.example/p1'
    })
  })

  it('falls back to a plain value pill for a malformed URL', () => {
    const pills = referenceMetadataPills({ url: 'not a url' })
    expect(pills).toEqual([{ key: 'url', value: 'not a url' }])
  })
})
