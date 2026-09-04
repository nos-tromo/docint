import { describe, it, expect, beforeEach } from 'vitest'
import { useChatFiltersStore } from './chatFilters'

beforeEach(() => {
  localStorage.clear()
  useChatFiltersStore.setState({
    retrievalMode: 'session',
    retrievalTarget: 'all',
    visualSourceType: 'any',
    visualClipFile: '',
    visualTimeFrom: '',
    visualTimeTo: '',
    filterEnabled: false,
    mimePattern: '',
    dateFrom: '',
    dateTo: '',
    hateSpeechOnly: false,
    customRules: []
  })
})

describe('useChatFiltersStore', () => {
  it('persists the retrieval mode', () => {
    useChatFiltersStore.getState().setRetrievalMode('stateless')

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.retrievalMode).toBe('stateless')
  })

  it('persists the reasoning toggle, off by default', () => {
    expect(useChatFiltersStore.getState().reasoning).toBe(false)

    useChatFiltersStore.getState().setReasoning(true)

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.reasoning).toBe(true)
  })

  it('persists custom filter rules', () => {
    useChatFiltersStore.getState().setFilterEnabled(true)
    useChatFiltersStore.getState().addRule()

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.filterEnabled).toBe(true)
    expect(persisted.state.customRules).toHaveLength(1)
  })

  it('emits the API date operators, not gte/lte aliases', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setDateFrom('2026-01-01')
    s.setDateTo('2026-02-01')

    const operators = useChatFiltersStore.getState().buildPayload().map((r) => r.operator)

    expect(operators).toContain('date_on_or_after')
    expect(operators).toContain('date_on_or_before')
    expect(operators).not.toContain('date_gte')
    expect(operators).not.toContain('date_lte')
  })

  it('applies date bounds to both timestamp keys', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setDateFrom('2026-01-01')

    const [rule] = useChatFiltersStore.getState().buildPayload()

    expect(rule.fields).toEqual([
      'reference_metadata.timestamp',
      'reference_metadata.posting_timestamp'
    ])
    expect(rule.field).toBeUndefined()
  })

  it('filters hate speech on the real nested payload key', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setHateSpeechOnly(true)

    const [rule] = useChatFiltersStore.getState().buildPayload()

    expect(rule.field).toBe('hate_speech.hate_speech')
    expect(rule.operator).toBe('eq')
    expect(rule.value).toBe(true)
  })

  it('emits nothing while filters are disabled', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(false)
    s.setMimePattern('application/pdf')
    s.setDateFrom('2026-01-01')

    expect(useChatFiltersStore.getState().buildPayload()).toEqual([])
  })

  it('does not persist action functions', () => {
    useChatFiltersStore.getState().setRetrievalMode('stateless')

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.setRetrievalMode).toBeUndefined()
    expect(persisted.state.buildPayload).toBeUndefined()
  })

  it('answers from everything until told otherwise', () => {
    expect(useChatFiltersStore.getState().retrievalTarget).toBe('all')
  })

  it('persists the retrieval target', () => {
    useChatFiltersStore.getState().setRetrievalTarget('visual')

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.retrievalTarget).toBe('visual')
  })

  it('keeps the visual presets out of a document turn, where they match nothing', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setRetrievalTarget('documents')
    s.setVisualSourceType('video')
    s.setVisualTimeFrom('1:00')

    expect(useChatFiltersStore.getState().buildPayload()).toEqual([])
  })

  it('narrows a visual turn to one kind of imagery', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setRetrievalTarget('visual')
    s.setVisualSourceType('video')

    expect(useChatFiltersStore.getState().buildPayload()).toContainEqual({
      field: 'source_type',
      operator: 'eq',
      value: 'video_keyframe'
    })
  })

  it('treats pictures as everything that is not a keyframe', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setRetrievalTarget('visual')
    s.setVisualSourceType('image')

    expect(useChatFiltersStore.getState().buildPayload()).toContainEqual({
      field: 'source_type',
      operator: 'in',
      values: ['social_media', 'standalone', 'document']
    })
  })

  it('reads a time range the way a player writes it', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setRetrievalTarget('visual')
    s.setVisualTimeFrom('1:30')
    s.setVisualTimeTo('2:00')

    const payload = useChatFiltersStore.getState().buildPayload()
    expect(payload).toContainEqual({ field: 'keyframe_time_sec', operator: 'gte', value: 90 })
    expect(payload).toContainEqual({ field: 'keyframe_time_sec', operator: 'lte', value: 120 })
  })

  it('drops an unreadable time bound rather than sending it as zero', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setRetrievalTarget('visual')
    s.setVisualTimeFrom('half past')

    expect(useChatFiltersStore.getState().buildPayload()).toEqual([])
  })

  it('pins a visual turn to one clip', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setRetrievalTarget('visual')
    s.setVisualClipFile('clip.mp4')

    expect(useChatFiltersStore.getState().buildPayload()).toContainEqual({
      field: 'source_file',
      operator: 'eq',
      value: 'clip.mp4'
    })
  })

  it('gives a state stored before the target existed the default', () => {
    localStorage.setItem(
      'docint-chat-filters',
      JSON.stringify({ state: { retrievalMode: 'session', filterEnabled: false }, version: 1 })
    )

    useChatFiltersStore.persist.rehydrate()

    expect(useChatFiltersStore.getState().retrievalTarget).toBe('all')
    expect(useChatFiltersStore.getState().visualSourceType).toBe('any')
  })
})
