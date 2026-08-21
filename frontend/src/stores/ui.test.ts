import { describe, it, expect, beforeEach } from 'vitest'
import { useUiStore } from './ui'
import { useReportStore } from './report'

beforeEach(() => {
  localStorage.clear()
  useUiStore.setState({
    selectedCollection: null,
    selectedOwner: null,
    currentSessionId: null,
    previewModal: null,
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

  it('drops the active report when the collection changes', () => {
    // A report holds one collection's evidence. Keeping the old one active
    // across a switch is how findings from one collection ended up in a report
    // scoped to another, with their images looked up in the wrong companion.
    useUiStore.getState().setSelectedCollection('c1')
    useReportStore.setState({ activeReportId: 7 })

    useUiStore.getState().setSelectedCollection('c2')

    expect(useReportStore.getState().activeReportId).toBeNull()
  })

  it('keeps the active report when the same collection is re-selected', () => {
    useUiStore.getState().setSelectedCollection('c1')
    useReportStore.setState({ activeReportId: 7 })

    useUiStore.getState().setSelectedCollection('c1')

    expect(useReportStore.getState().activeReportId).toBe(7)
  })

  it('drops the active report when only the owner changes', () => {
    // A foreign collection with the same name is a different collection.
    useUiStore.getState().setSelectedCollection('shared', 'alice')
    useReportStore.setState({ activeReportId: 7 })

    useUiStore.getState().setSelectedCollection('shared', 'bob')

    expect(useReportStore.getState().activeReportId).toBeNull()
  })

  it('clears current session', () => {
    useUiStore.getState().setCurrentSessionId('s1')
    useUiStore.getState().setCurrentSessionId(null)
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })

  it('no longer carries an entity merge mode', () => {
    const state = useUiStore.getState() as unknown as Record<string, unknown>
    expect('entityMergeMode' in state).toBe(false)
    expect('setEntityMergeMode' in state).toBe(false)
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

  it('tracks the selected owner and clears it on own-collection selection', () => {
    useUiStore.getState().setSelectedCollection('theirs', 'jane.doe')
    expect(useUiStore.getState().selectedOwner).toBe('jane.doe')

    useUiStore.getState().setSelectedCollection('mine')
    expect(useUiStore.getState().selectedOwner).toBeNull()
  })

  it('changing owner of the same-named collection drops the open session', () => {
    useUiStore.setState({ selectedCollection: 'alpha', selectedOwner: null, currentSessionId: 's1' })
    useUiStore.getState().setSelectedCollection('alpha', 'jane.doe')
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })
})
