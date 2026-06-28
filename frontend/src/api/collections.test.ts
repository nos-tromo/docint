import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  csvExportHref,
  getDocumentsCount,
  getDocumentsPage,
  getHateSpeech,
  getHateSpeechPage,
  getNerGraph,
  getNerSourcesPage,
  getNerStats,
  listDocuments,
  warmCollectionNer
} from './collections'

afterEach(() => {
  vi.restoreAllMocks()
})

function mockFetch(body: unknown = {}) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body)
    })
  )
}

function lastUrl() {
  return String((fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0])
}

function lastInit() {
  return (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][1]
}

describe('collections api carries the selected collection', () => {
  it('listDocuments sends collection as a query param', async () => {
    mockFetch({ documents: [] })
    await listDocuments('docs')
    expect(lastUrl()).toContain('/collections/documents')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('getDocumentsPage sends collection as a query param', async () => {
    mockFetch({ items: [], next_cursor: null })
    await getDocumentsPage({ limit: 50, collection: 'docs' })
    expect(lastUrl()).toContain('collection=docs')
  })

  it('getDocumentsCount sends collection as a query param', async () => {
    mockFetch({ count: 0 })
    await getDocumentsCount('docs')
    expect(lastUrl()).toContain('/collections/documents/count')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('getNerStats sends collection as a query param', async () => {
    mockFetch({})
    await getNerStats({ top_k: 10, collection: 'docs' })
    expect(lastUrl()).toContain('/collections/ner/stats')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('getNerSourcesPage sends collection as a query param', async () => {
    mockFetch({ items: [], next_cursor: null })
    await getNerSourcesPage({ limit: 50, entity_key: 'A::PER', collection: 'docs' })
    expect(lastUrl()).toContain('/collections/ner/sources')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('getNerGraph sends collection as a query param', async () => {
    mockFetch({ nodes: [], edges: [], meta: { node_count: 0, edge_count: 0 } })
    await getNerGraph({ top_k_nodes: 80, collection: 'docs' })
    expect(lastUrl()).toContain('/collections/ner/graph')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('getHateSpeech sends collection as a query param', async () => {
    mockFetch({ results: [] })
    await getHateSpeech('docs')
    expect(lastUrl()).toContain('/collections/hate-speech')
    expect(lastUrl()).toContain('collection=docs')
  })

  it('getHateSpeechPage sends collection as a query param', async () => {
    mockFetch({ items: [], next_cursor: null })
    await getHateSpeechPage({ limit: 50, collection: 'docs' })
    expect(lastUrl()).toContain('collection=docs')
  })

  it('warmCollectionNer POSTs with the collection query param', async () => {
    mockFetch({ ok: true })
    await warmCollectionNer('docs')
    expect(lastUrl()).toContain('/collections/ner/warm')
    expect(lastUrl()).toContain('collection=docs')
    expect(lastInit().method).toBe('POST')
  })

  it('csvExportHref encodes the collection into the export path', () => {
    expect(csvExportHref('my docs', 'documents')).toContain(
      '/collections/my%20docs/export/documents.csv'
    )
    expect(csvExportHref('docs', 'hate-speech')).toContain(
      '/collections/docs/export/hate-speech.csv'
    )
  })
})
