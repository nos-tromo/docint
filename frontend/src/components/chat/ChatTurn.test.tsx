import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ChatTurn, type ChatTurnData } from './ChatTurn'
import type { AgentTrail, ChatFinalEvent } from '@/api/types'
import { LanguageContext } from '@/i18n/LanguageContext'
import { useUiStore } from '@/stores/ui'

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'docs', previewModal: null })
})

function renderTurn(
  turn: ChatTurnData,
  lang: 'en' | 'de' = 'en',
  props: Omit<Parameters<typeof ChatTurn>[0], 'turn'> = {}
) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const ui = (current: ChatTurnData) => (
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <LanguageContext value={lang}>
          <ChatTurn turn={current} {...props} />
        </LanguageContext>
      </MemoryRouter>
    </QueryClientProvider>
  )
  const view = render(ui(turn))
  return { ...view, update: (next: ChatTurnData) => view.rerender(ui(next)) }
}

describe('ChatTurn', () => {
  it('surfaces the entities behind the answer above its sources', () => {
    const meta = {
      session_id: 's',
      sources: [
        {
          filename: 'handbook.pdf',
          page: 26,
          text: 'Alice Weber met the Acme Corp board.',
          entities: [
            { text: 'Alice Weber', type: 'PER' },
            { text: 'Acme Corp', type: 'ORG' }
          ]
        }
      ]
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice Weber.', done: true, meta })

    const pills = screen.getAllByTestId('answer-entity')
    expect(pills.map((p) => p.textContent)).toEqual(
      expect.arrayContaining([expect.stringContaining('Alice Weber')])
    )

    // Order matters: the entities summarize the answer, so they belong
    // between it and the evidence, not after the citation list.
    const entities = screen.getByTestId('answer-entities')
    const sources = screen.getByText(/handbook\.pdf/)
    expect(entities.compareDocumentPosition(sources)).toBe(Node.DOCUMENT_POSITION_FOLLOWING)
  })

  it('renders the sources as pills matching the entity presentation', () => {
    const meta = {
      session_id: 's',
      sources: [
        { id: 's1', filename: 'handbook.pdf', page: 26, text: 'first', citation_index: 1 },
        { id: 's2', filename: 'notes.md', text: 'second', citation_index: 2 }
      ]
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice Weber.', done: true, meta })

    const pills = screen.getAllByTestId('source-pill')
    expect(pills).toHaveLength(2)
    expect(pills[0]).toHaveTextContent('handbook.pdf · page 26')
    // Compact by default: no detail card until a pill is clicked.
    expect(screen.queryByText('first')).not.toBeInTheDocument()
  })

  it('renders no entity row for an answer whose sources carry none', () => {
    const meta = {
      session_id: 's',
      sources: [{ filename: 'handbook.pdf', page: 26, text: 'no entities here' }]
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Nobody.', done: true, meta })

    expect(screen.queryByTestId('answer-entities')).not.toBeInTheDocument()
  })

  it('stays quiet when the server confirms it answered from the selection', () => {
    const meta = {
      session_id: 's',
      sources: [],
      retrieval_mode: 'scoped',
      scoped_chunk_count: 2
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta, scopeRequested: 2 })

    expect(screen.queryByTestId('scope-not-applied')).not.toBeInTheDocument()
  })

  it('flags a turn that asked for a scope and got an unscoped answer', () => {
    // Without this the two are indistinguishable in the transcript, which is
    // how a dropped scope came to be presented as hand-picked evidence.
    const meta = {
      session_id: 's',
      sources: [],
      retrieval_mode: 'rewrite_compact'
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Something else.', done: true, meta, scopeRequested: 2 })

    expect(screen.getByTestId('scope-not-applied')).toBeInTheDocument()
  })

  it('flags a turn whose sources were not re-ranked', () => {
    // Measured on a live stack: the reranker had been down for a day and every
    // answer shipped its top-5 by raw fusion order, indistinguishable from a
    // healthy turn. The final frame now says so; the transcript must show it.
    const meta = {
      session_id: 's',
      sources: [{ id: 1 }],
      retrieval_mode: 'rewrite_compact',
      rerank: { applied: false, error: 'upstream down' }
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta })

    expect(screen.getByTestId('rerank-not-applied')).toBeInTheDocument()
  })

  it('says so when a visual answer never got to see the pictures', () => {
    // The point of the visual target is that the model looks at the imagery.
    // A turn where the thumbnails could not be fetched answered from captions
    // alone, and must not read as an ordinary visual answer.
    const meta = {
      session_id: 's',
      sources: [{ id: 1 }],
      retrieval_target: 'visual',
      visual: { images_attached: 0 }
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'what is at the gate?', assistant: 'A barrier.', done: true, meta })

    expect(screen.getByTestId('visual-pixels-missing')).toBeInTheDocument()
  })

  it('stays quiet when a visual answer did see the pictures', () => {
    const meta = {
      session_id: 's',
      sources: [{ id: 1 }],
      retrieval_target: 'visual',
      visual: { images_attached: 3 }
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'what is at the gate?', assistant: 'A barrier.', done: true, meta })

    expect(screen.queryByTestId('visual-pixels-missing')).not.toBeInTheDocument()
  })

  it('stays quiet on a text turn, which was never going to show pictures', () => {
    const meta = { session_id: 's', sources: [{ id: 1 }], retrieval_target: 'all' } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta })

    expect(screen.queryByTestId('visual-pixels-missing')).not.toBeInTheDocument()
  })

  it('stays quiet when the sources were re-ranked, or no reranker was in the loop', () => {
    const healthy = {
      session_id: 's',
      sources: [{ id: 1 }],
      retrieval_mode: 'rewrite_compact',
      rerank: { applied: true, error: null }
    } as unknown as ChatFinalEvent
    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta: healthy })
    expect(screen.queryByTestId('rerank-not-applied')).not.toBeInTheDocument()

    const absent = { session_id: 's', sources: [], retrieval_mode: 'scoped' } as unknown as ChatFinalEvent
    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta: absent })
    expect(screen.queryByTestId('rerank-not-applied')).not.toBeInTheDocument()
  })

  it('names the corrective retry a reloaded session replays', () => {
    // The live retry frame is long gone by then; the provenance has to come
    // off the persisted turn or a reloaded transcript would present a second
    // attempt as though it were the original answer.
    const meta = {
      session_id: 's',
      sources: [],
      retried: true,
      retry_query: 'Security Council resolutions'
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'what did the UN say?', assistant: 'Three resolutions.', done: true, meta })

    expect(screen.getByTestId('retry-notice')).toHaveTextContent(/Security Council resolutions/)
  })

  it('gives a stateless answer its own report identity, never a shared placeholder', () => {
    // A stateless turn is saved in no session. Keyed by the placeholder id the
    // server used to hand back, every such answer shared `chat:stateless:<n>`:
    // a fresh one read as already in the report, and the toggle then removed a
    // different answer.
    const meta = { session_id: null, sources: [] } as unknown as ChatFinalEvent

    renderTurn(
      { user: 'who?', assistant: 'Alice.', done: true, meta, unsavedId: 'k1' },
      'en',
      { turnIdx: 0, reportDedupeKeys: new Set(['chat:stateless:0']) }
    )

    expect(screen.getByRole('button', { name: 'Add to report' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('offers no Add to report for a turn that failed', () => {
    // A failed turn is never saved, so after a reload the next saved answer takes
    // its place in the list and, with it, the report key this one would claim.
    renderTurn(
      { user: 'who?', assistant: 'Alice', done: true, meta: null, error: 'The stream ended.' },
      'en',
      { sessionId: 's', turnIdx: 0, reportDedupeKeys: new Set<string>() }
    )

    expect(screen.queryByRole('button', { name: 'Add to report' })).not.toBeInTheDocument()
  })

  it('shows no retry notice on an ordinary answer', () => {
    const meta = { session_id: 's', sources: [] } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta })

    expect(screen.queryByTestId('retry-notice')).not.toBeInTheDocument()
  })

  it('renders a Markdown answer as structure, its citation markers as text', () => {
    const answer = [
      '### Board members',
      '',
      '- **Alice Weber** chairs the board [1]',
      '- Bob Stein keeps the minutes [2][3]',
      '',
      '| Name | Role |',
      '| --- | --- |',
      '| Alice Weber | Chair |'
    ].join('\n')

    const { container } = renderTurn({ user: 'who sits on the board?', assistant: answer, done: true, meta: null })

    expect(screen.getByRole('heading', { level: 3, name: 'Board members' })).toBeInTheDocument()
    expect(container.querySelectorAll('li')).toHaveLength(2)
    expect(screen.getByText('Alice Weber', { selector: 'strong' })).toBeInTheDocument()
    expect(container.querySelector('li')).toHaveTextContent('Alice Weber chairs the board [1]')
    expect(screen.getByText(/keeps the minutes \[2\]\[3\]/)).toBeInTheDocument()
    // A wide table scrolls inside its own box instead of stretching the panel.
    expect(screen.getByTestId('markdown-table')).toContainElement(screen.getByRole('table'))
    expect(screen.getByRole('cell', { name: 'Chair' })).toBeInTheDocument()
  })

  it('never renders raw HTML from an answer as markup', () => {
    const { container } = renderTurn({
      user: 'q',
      assistant: 'Before <img src="x" onerror="alert(1)"> <b>bold</b> after',
      done: true,
      meta: null
    })

    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('b')).toBeNull()
    expect(container).toHaveTextContent('<b>bold</b> after')
  })

  const TRAIL: AgentTrail = {
    steps: 1,
    actions: [
      { action: 'seed', detail: '', found: 2 },
      { action: 'search', detail: 'evaluation datasets', found: 1 }
    ],
    reason: null,
    answer: 'split',
    shortened: 0,
    evidence_chars: 9
  }

  function assessmentMeta(trail: AgentTrail = TRAIL): ChatFinalEvent {
    return { session_id: 's', sources: [], answer_mode: 'assessment', agent: trail } as ChatFinalEvent
  }

  it('shows the live research trail while an assessment turn runs', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 0, action: 'seed', detail: '', found: 2 },
        { step: 1, action: 'search', detail: 'evaluation datasets', found: 1 }
      ]
    })
    const trail = screen.getByTestId('agent-trail-live')
    expect(trail).toHaveTextContent('Initial search: 2 passages')
    expect(trail).toHaveTextContent('evaluation datasets')
  })

  it('marks a finished assessment and lists its searches behind a disclosure', async () => {
    renderTurn({ user: 'q', assistant: 'Fact [1].', done: true, meta: assessmentMeta() })

    expect(screen.getByTestId('assessment-notice')).toHaveTextContent(/model’s own assessment/i)
    expect(screen.queryByTestId('agent-trail-live')).not.toBeInTheDocument()
    const toggle = screen.getByRole('button', { name: /show the searches/i })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await userEvent.click(toggle)
    expect(screen.getByRole('button', { name: /hide the searches/i })).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/evaluation datasets/)).toBeInTheDocument()
  })

  it('labels a keyword check by its count and the field it matched', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 1, action: 'keyword', detail: 'Kontrollgruppe', found: 0, field: 'text', matches: 0 },
        { step: 2, action: 'keyword', detail: 'Krieger', found: 2, field: 'author', matches: 1 },
        { step: 3, action: 'keyword', detail: 'Nachtschicht', found: 1, field: 'text' }
      ]
    })
    const trail = screen.getByTestId('agent-trail-live')
    expect(trail).toHaveTextContent('Keyword check “Kontrollgruppe”: 0 matches, 0 new')
    expect(trail).toHaveTextContent('Keyword check “Krieger” in Author: 1 match, 2 new')
    expect(trail).toHaveTextContent('Keyword check “Nachtschicht”: count unavailable, 1 new')
  })

  it('lists keyword checks behind the disclosure, even when research only checked words', async () => {
    renderTurn({
      user: 'q',
      assistant: 'Fact [1].',
      done: true,
      meta: assessmentMeta({
        ...TRAIL,
        actions: [
          { action: 'seed', detail: '', found: 2 },
          { action: 'keyword', detail: 'Kontrollgruppe', found: 0, field: 'text', matches: 0 },
          { action: 'keyword', detail: 'Nachtschicht', found: 1, field: 'text', matches: null }
        ]
      })
    })
    await userEvent.click(screen.getByRole('button', { name: /show the searches/i }))
    expect(screen.getByText('Keyword check “Kontrollgruppe”: 0 matches, 0 new')).toBeInTheDocument()
    expect(screen.getByText('Keyword check “Nachtschicht”: count unavailable, 1 new')).toBeInTheDocument()
  })

  it('shows an amber notice for a degraded research step', () => {
    renderTurn({
      user: 'q',
      assistant: 'Fact [1].',
      done: true,
      meta: assessmentMeta({ ...TRAIL, reason: 'tools_unavailable' })
    })
    expect(screen.getByTestId('assessment-research-degraded')).toHaveTextContent(/could not run its own searches/i)
  })

  it.each([
    ['en', 'The model could not run its own searches, so this assessment rests on the first search’s passages only.'],
    [
      'de',
      'Das Modell konnte keine eigenen Suchen ausführen; diese Einschätzung beruht nur auf den Passagen der ersten Suche.'
    ]
  ] as const)('names no single cause for unavailable tools in the %s notice', (lang, copy) => {
    // The token covers a rejected tool call and a prose reply alike.
    renderTurn(
      {
        user: 'q',
        assistant: 'Fact [1].',
        done: true,
        meta: assessmentMeta({ ...TRAIL, reason: 'tools_unavailable' })
      },
      lang
    )

    expect(screen.getByTestId('assessment-research-degraded')).toHaveTextContent(copy)
  })

  it.each([
    ['en', 'The model’s context window was full, so research stopped early or part of the evidence was left out.'],
    [
      'de',
      'Das Kontextfenster des Modells war voll; daher endete die Recherche vorzeitig oder ein Teil der Belege wurde ausgelassen.'
    ]
  ] as const)('says in the %s notice that a full context window can also leave evidence out', (lang, copy) => {
    // The token covers research the room stopped and evidence left out whole or cut below its minimum.
    renderTurn(
      {
        user: 'q',
        assistant: 'Fact [1].',
        done: true,
        meta: assessmentMeta({ ...TRAIL, reason: 'context_budget' })
      },
      lang
    )

    expect(screen.getByTestId('assessment-research-degraded')).toHaveTextContent(copy)
  })

  it('says when the model did not separate its judgement', () => {
    renderTurn({
      user: 'q',
      assistant: 'View.',
      done: true,
      meta: assessmentMeta({ ...TRAIL, answer: 'unsplit', evidence_chars: 0 })
    })
    expect(screen.getByTestId('assessment-answer-degraded')).toHaveTextContent(/did not separate/i)
  })

  it('renders no assessment notice on a lookup turn', () => {
    renderTurn({
      user: 'q',
      assistant: 'Alice.',
      done: true,
      meta: { session_id: 's', sources: [] } as unknown as ChatFinalEvent
    })
    expect(screen.queryByTestId('assessment-notice')).not.toBeInTheDocument()
  })

  it('names the answer step in the live trail', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [{ step: 2, action: 'answer', detail: '', found: 0 }]
    })
    expect(screen.getByTestId('agent-trail-live')).toHaveTextContent('Writing the assessment…')
  })

  it('does not mention shortening when nothing was shortened', () => {
    renderTurn({ user: 'q', assistant: 'Fact [1].', done: true, meta: assessmentMeta() })
    expect(screen.getByTestId('assessment-notice')).not.toHaveTextContent(/shortened/i)
  })

  it('offers no search disclosure when the assessment ran none of its own', () => {
    renderTurn({
      user: 'q',
      assistant: 'Fact [1].',
      done: true,
      meta: assessmentMeta({ ...TRAIL, actions: [{ action: 'seed', detail: '', found: 2 }] })
    })
    expect(screen.getByTestId('assessment-notice')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /searches/i })).not.toBeInTheDocument()
  })

  it.each([
    ['brand_new_reason', 'brand_new_answer'],
    ['constructor', 'toString']
  ])('renders the unknown trail tokens %s and %s without crashing or a notice', (reason, answer) => {
    // A token this build does not know (version skew, a persisted trail) must not take the SPA down.
    const unknown = { ...TRAIL, reason, answer } as unknown as AgentTrail
    renderTurn({ user: 'q', assistant: 'Fact [1].', done: true, meta: assessmentMeta(unknown) })

    // An unknown answer token cannot say whether the text carries the label.
    expect(screen.queryByTestId('assessment-notice')).not.toBeInTheDocument()
    expect(screen.queryByTestId('assessment-research-degraded')).not.toBeInTheDocument()
    expect(screen.queryByTestId('assessment-answer-degraded')).not.toBeInTheDocument()
  })

  it.each([
    ['empty', null, false],
    ['split', null, false],
    ['unsplit', 0, true],
    ['split', 9, true]
  ] as const)(
    'shows the assessment notice for a %s answer with evidence_chars %s only when the label is in the text: %s',
    (answer, evidenceChars, shown) => {
      renderTurn({
        user: 'q',
        assistant: 'Fact [1].',
        done: true,
        meta: assessmentMeta({ ...TRAIL, answer, evidence_chars: evidenceChars })
      })

      expect(screen.queryByTestId('assessment-notice') !== null).toBe(shown)
    }
  )

  it.each([
    ['en', /whole answer is marked as its assessment\.$/, /not validated/i],
    ['de', /als seine Einschätzung markiert\.$/, /nicht validiert/i]
  ] as const)('leaves "not validated" to the banner in the %s unsplit notice', (lang, sentence, banned) => {
    renderTurn(
      {
        user: 'q',
        assistant: 'View.',
        done: true,
        meta: assessmentMeta({ ...TRAIL, answer: 'unsplit', evidence_chars: 0 })
      },
      lang
    )

    const notice = screen.getByTestId('assessment-answer-degraded')
    expect(notice).toHaveTextContent(sentence)
    expect(notice).not.toHaveTextContent(banned)
  })

  it('marks only the searches in the live trail with a magnifier', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 0, action: 'seed', detail: '', found: 2 },
        { step: 1, action: 'search', detail: 'evaluation datasets', found: 1 },
        { step: 2, action: 'answer', detail: '', found: 3 }
      ]
    })

    const rows = within(screen.getByTestId('agent-trail-live')).getAllByRole('listitem')
    expect(rows.map((row) => row.querySelector('svg') !== null)).toEqual([true, true, false])
    // A spacer of the icon's size keeps the answer row's text aligned with the searches.
    expect(rows[2].firstElementChild).toHaveAttribute('aria-hidden', 'true')
    expect(rows[2].firstElementChild).toHaveClass('h-3.5', 'w-3.5', 'shrink-0')
  })

  it('uses the singular for a single passage in the live trail', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 0, action: 'seed', detail: '', found: 1 },
        { step: 1, action: 'search', detail: 'evaluation datasets', found: 1 }
      ]
    })
    expect(screen.getByText('Initial search: 1 passage')).toBeInTheDocument()
    expect(screen.getByText('Searched “evaluation datasets”: 1 new passage')).toBeInTheDocument()
  })

  it('keeps the plural for no passages', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 0, action: 'seed', detail: '', found: 0 },
        { step: 1, action: 'search', detail: 'evaluation datasets', found: 0 }
      ]
    })
    expect(screen.getByText('Initial search: 0 passages')).toBeInTheDocument()
    expect(screen.getByText('Searched “evaluation datasets”: 0 new passages')).toBeInTheDocument()
  })

  it.each([
    [1, /1 passage was shortened/i],
    [3, /3 passages were shortened/i]
  ])('words %i shortened passage(s) with the right number', (shortened, expected) => {
    renderTurn({
      user: 'q',
      assistant: 'Fact [1].',
      done: true,
      meta: assessmentMeta({ ...TRAIL, shortened })
    })
    expect(screen.getByTestId('assessment-notice')).toHaveTextContent(expected)
  })

  it('uses the German singular and plural for passage counts in the live trail', () => {
    renderTurn(
      {
        user: 'q',
        assistant: '',
        done: false,
        meta: null,
        agentSteps: [
          { step: 0, action: 'seed', detail: '', found: 1 },
          { step: 1, action: 'search', detail: 'Bewertungsdaten', found: 1 },
          { step: 2, action: 'search', detail: 'Messreihen', found: 3 },
          { step: 3, action: 'answer', detail: '', found: 0 }
        ]
      },
      'de'
    )
    expect(screen.getByText('Erste Suche: 1 Passage')).toBeInTheDocument()
    expect(screen.getByText('Gesucht nach „Bewertungsdaten“: 1 neue Passage')).toBeInTheDocument()
    expect(screen.getByText('Gesucht nach „Messreihen“: 3 neue Passagen')).toBeInTheDocument()
    expect(screen.getByText('Einschätzung wird geschrieben…')).toBeInTheDocument()
  })

  it.each([
    [1, /1 Passage wurde gekürzt/],
    [4, /4 Passagen wurden gekürzt/]
  ])('uses the German wording for %i shortened passage(s)', (shortened, expected) => {
    renderTurn(
      { user: 'q', assistant: 'Fakt [1].', done: true, meta: assessmentMeta({ ...TRAIL, shortened }) },
      'de'
    )
    expect(screen.getByTestId('assessment-notice')).toHaveTextContent(expected)
  })

  it('has a status region from the first moment a turn runs, and none once it is done', () => {
    const running = renderTurn({ user: 'q', assistant: '', done: false, meta: null })
    expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite')
    // `status` implies atomic; each new step must be announced alone, not re-read with the whole list.
    expect(screen.getByRole('status')).toHaveAttribute('aria-atomic', 'false')
    expect(screen.queryByTestId('agent-trail-live')).not.toBeInTheDocument()
    running.unmount()

    renderTurn({ user: 'q', assistant: 'Alice.', done: true, meta: assessmentMeta() })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('puts the first research step inside the status region that was already mounted', () => {
    const view = renderTurn({ user: 'q', assistant: '', done: false, meta: null })
    const region = screen.getByRole('status')

    view.update({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [{ step: 0, action: 'seed', detail: '', found: 2 }]
    })

    expect(screen.getByRole('status')).toBe(region)
    expect(region).toContainElement(screen.getByTestId('agent-trail-live'))
  })

  it.each([
    ['no actions key', undefined],
    ['a null actions value', null],
    ['an actions value that is not a list', {}]
  ])('still renders the notice, with no search list, for a stored trail with %s', (_name, actions) => {
    // The backend only checks that a stored trail is an object, so its shape is not guaranteed.
    const stored: Record<string, unknown> = { ...TRAIL, actions }
    if (actions === undefined) delete stored.actions
    renderTurn({ user: 'q', assistant: 'Fact [1].', done: true, meta: assessmentMeta(stored as unknown as AgentTrail) })

    expect(screen.getByTestId('assessment-notice')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /searches/i })).not.toBeInTheDocument()
  })
})
