import { describe, it, expect, beforeEach } from 'vitest'
import { useUiStore } from './ui'

beforeEach(() => {
  localStorage.clear()
  useUiStore.setState({
    selectedCollection: null,
    currentSessionId: null,
    previewModal: null,
    entityMergeMode: 'resolved',
    graphTopK: null
  })
})

describe('useUiStore', () => {
  it('updates selected collection', () => {
    useUiStore.getState().setSelectedCollection('c1')
    expect(useUiStore.getState().selectedCollection).toBe('c1')
  })

  it('persists the active collection across reloads', () => {
    useUiStore.getState().setSelectedCollection('c1')
    const persisted = JSON.parse(localStorage.getItem('docint-ui')!).state
    expect(persisted.selectedCollection).toBe('c1')
  })

  it('clears current session', () => {
    useUiStore.getState().setCurrentSessionId('s1')
    useUiStore.getState().setCurrentSessionId(null)
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })

  it('defaults the entity merge mode to resolved', () => {
    expect(useUiStore.getState().entityMergeMode).toBe('resolved')
  })

  it('updates the entity merge mode', () => {
    useUiStore.getState().setEntityMergeMode('orthographic')
    expect(useUiStore.getState().entityMergeMode).toBe('orthographic')
  })

  it('persists the entity merge mode across reloads', () => {
    useUiStore.getState().setEntityMergeMode('exact')
    const persisted = JSON.parse(localStorage.getItem('docint-ui')!).state
    expect(persisted.entityMergeMode).toBe('exact')
  })

  it('defaults graphTopK to null', () => {
    expect(useUiStore.getState().graphTopK).toBeNull()
  })

  it('updates graphTopK', () => {
    useUiStore.getState().setGraphTopK(200)
    expect(useUiStore.getState().graphTopK).toBe(200)
  })

  it('persists graphTopK across reloads', () => {
    useUiStore.getState().setGraphTopK(150)
    const persisted = JSON.parse(localStorage.getItem('docint-ui')!).state
    expect(persisted.graphTopK).toBe(150)
  })

  it('clears the current session when the selected collection changes', () => {
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 's1' })
    useUiStore.getState().setSelectedCollection('beta')
    expect(useUiStore.getState().selectedCollection).toBe('beta')
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })

  it('clears the current session when the collection is cleared to null', () => {
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 's1' })
    useUiStore.getState().setSelectedCollection(null)
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })

  it('keeps the current session when re-selecting the same collection', () => {
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 's1' })
    useUiStore.getState().setSelectedCollection('alpha')
    expect(useUiStore.getState().currentSessionId).toBe('s1')
  })
})
