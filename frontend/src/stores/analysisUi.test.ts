import { beforeEach, describe, expect, it } from 'vitest'
import { selectEntityKeyFor, useAnalysisUiStore } from './analysisUi'

beforeEach(() => useAnalysisUiStore.setState({ tab: 'ner', nerView: 'table', entity: null }))

describe('useAnalysisUiStore', () => {
  it('keeps tab and view as global preferences', () => {
    useAnalysisUiStore.getState().setTab('hate')
    useAnalysisUiStore.getState().setNerView('graph')

    expect(useAnalysisUiStore.getState().tab).toBe('hate')
    expect(useAnalysisUiStore.getState().nerView).toBe('graph')
  })

  it('restores the entity selection for its own collection', () => {
    useAnalysisUiStore.getState().setEntity('Acme::ORG', 'mydocs')
    expect(selectEntityKeyFor('mydocs')(useAnalysisUiStore.getState())).toBe('Acme::ORG')
  })

  it('does not restore an entity selected under another collection', () => {
    useAnalysisUiStore.getState().setEntity('Acme::ORG', 'mydocs')
    expect(selectEntityKeyFor('otherdocs')(useAnalysisUiStore.getState())).toBeNull()
  })

  it('clears the selection when passed null', () => {
    useAnalysisUiStore.getState().setEntity('Acme::ORG', 'mydocs')
    useAnalysisUiStore.getState().setEntity(null, null)
    expect(useAnalysisUiStore.getState().entity).toBeNull()
  })

  it('persists tab, view, and entity', () => {
    useAnalysisUiStore.getState().setTab('summary')
    useAnalysisUiStore.getState().setEntity('Acme::ORG', 'mydocs')

    const persisted = JSON.parse(localStorage.getItem('docint-analysis-ui') ?? '{}')
    expect(persisted.state.tab).toBe('summary')
    expect(persisted.state.entity).toEqual({ key: 'Acme::ORG', collection: 'mydocs' })
  })
})
