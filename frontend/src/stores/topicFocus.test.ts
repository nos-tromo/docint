import { beforeEach, describe, expect, it } from 'vitest'
import { topicFocusRule, useTopicFocusStore, type TopicFocus } from './topicFocus'

const RUN = '20260102-030405-a1b2c3'
const EARLIER_RUN = '20251201-101010-0f0f0f'

const FOCUS: TopicFocus = { collection: 'projekt-alpha', runId: RUN, topicId: 3, label: 'Harbour cranes' }

beforeEach(() => useTopicFocusStore.setState({ focus: null }))

describe('topicFocusRule', () => {
  it('builds the rule only for the matching collection and run', () => {
    expect(topicFocusRule(FOCUS, 'projekt-alpha', RUN)).toEqual({
      field: 'docint_topic',
      operator: 'eq',
      value: '20260102-030405-a1b2c3:3'
    })

    expect(topicFocusRule(null, 'projekt-alpha', RUN)).toBeNull()
    expect(topicFocusRule(FOCUS, 'projekt-beta', RUN)).toBeNull()
    expect(topicFocusRule(FOCUS, null, RUN)).toBeNull()
    // A rebuild gives the collection a new run, whose values the old topic's never match.
    expect(topicFocusRule(FOCUS, 'projekt-alpha', EARLIER_RUN)).toBeNull()
    // No run known yet, or none at all.
    expect(topicFocusRule(FOCUS, 'projekt-alpha', null)).toBeNull()
  })
})

describe('useTopicFocusStore', () => {
  it('clears the focus', () => {
    useTopicFocusStore.getState().setFocus(FOCUS)
    expect(useTopicFocusStore.getState().focus).toEqual(FOCUS)

    useTopicFocusStore.getState().clear()

    expect(useTopicFocusStore.getState().focus).toBeNull()
  })
})
