import type { ReactElement } from 'react'
import { describe, it, expect, beforeEach, onTestFinished, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DocumentsIcon, ImageIcon, LayersIcon, SearchIcon } from '@infra/ui'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ChatTurn, type ChatTurnData } from './ChatTurn'
import type { AgentReason, AgentStep, AgentTrail, ChatFinalEvent } from '@/api/types'
import { en } from '@/i18n/en'
import { LanguageContext } from '@/i18n/LanguageContext'
import { useUiStore } from '@/stores/ui'

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'docs', previewModal: null })
})

/** The markup inside an icon's `<svg>`, so two drawings can be compared. */
function innerMarkup(icon: ReactElement): string | undefined {
  const { container, unmount } = render(icon)
  const markup = container.querySelector('svg')?.innerHTML
  unmount()
  return markup
}

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

  it('shows the target chip on documents and visual turns only', () => {
    // A turn narrowed to one kind of evidence must say so: it is the only way
    // to tell a text-only answer from one that simply found no pictures. The
    // default says nothing, and a turn with no recorded target says nothing
    // either, rather than being read as `all`.
    const cases = [
      ['documents', 'Answered from documents only', DocumentsIcon],
      ['visual', 'Answered from images only', ImageIcon],
      ['all', null, null],
      [undefined, null, null]
    ] as const

    for (const [target, text, Icon] of cases) {
      const meta = { session_id: 's', sources: [], retrieval_target: target } as unknown as ChatFinalEvent
      const { unmount } = renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta })

      if (text && Icon) {
        const chip = screen.getByTestId('retrieval-target-used')
        expect(chip).toHaveTextContent(text)
        // The drawing the picker shows for the same target.
        expect(chip.querySelector('svg')?.innerHTML).toBe(innerMarkup(<Icon />))
        // A narrow chat column truncates the text rather than wrapping the
        // chip onto two lines, so the tooltip carries the whole sentence.
        expect(chip).toHaveAttribute('title', text)
        expect(chip.querySelector('.truncate')).toHaveTextContent(text)
      } else {
        expect(screen.queryByTestId('retrieval-target-used')).not.toBeInTheDocument()
      }
      unmount()
    }
  })

  it('sits in the assistant header row, between its label and the answer', () => {
    const meta = { session_id: 's', sources: [], retrieval_target: 'documents' } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta })

    const label = screen.getByText('Assistant')
    const chip = screen.getByTestId('retrieval-target-used')
    const answer = screen.getByText('Alice.')
    expect(label.compareDocumentPosition(chip) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(chip.compareDocumentPosition(answer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('no target chip on a scoped turn', () => {
    // The final frame still reports the requested target, but a pinned
    // selection outranks it: the answer came from the hand-picked chunks, and
    // "documents only" would describe a restriction that never applied.
    const meta = {
      session_id: 's',
      sources: [],
      retrieval_mode: 'scoped',
      scoped_chunk_count: 2,
      retrieval_target: 'documents'
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'who?', assistant: 'Alice.', done: true, meta, scopeRequested: 2 })

    expect(screen.queryByTestId('retrieval-target-used')).not.toBeInTheDocument()
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
    expect(trail).toHaveTextContent('Started from 2 passages')
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

  it('shows an amber notice for a degraded research step', () => {
    renderTurn({
      user: 'q',
      assistant: 'Fact [1].',
      done: true,
      meta: assessmentMeta({ ...TRAIL, reason: 'tools_unavailable' })
    })
    expect(screen.getByTestId('agent-reason-notice')).toHaveTextContent(/could not run its own searches/i)
  })

  it.each([
    ['en', 'The model could not run its own searches, so this answer rests on a standard search instead.'],
    ['de', 'Das Modell konnte keine eigenen Suchen ausführen; diese Antwort beruht daher auf einer Standardsuche.']
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

    expect(screen.getByTestId('agent-reason-notice')).toHaveTextContent(copy)
  })

  it.each([
    ['en', 'Stopped searching: the room for more evidence was used up.'],
    ['de', 'Suche beendet: Der Platz für weitere Belege war aufgebraucht.']
  ] as const)('says in the %s note that the room for evidence was used up', (lang, copy) => {
    // At the smallest window a turn often stops here by design, pictures above all, so it reads as information.
    renderTurn(
      {
        user: 'q',
        assistant: 'Fact [1].',
        done: true,
        meta: assessmentMeta({ ...TRAIL, reason: 'context_budget' })
      },
      lang
    )

    const note = screen.getByTestId('agent-reason-notice')
    expect(note).toHaveTextContent(copy)
    expect(note).toHaveAttribute('role', 'note')
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
    expect(screen.queryByTestId('agent-reason-notice')).not.toBeInTheDocument()
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
    expect(screen.getByText('Started from 1 passage')).toBeInTheDocument()
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
    expect(screen.getByText('Started from 0 passages')).toBeInTheDocument()
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
    expect(screen.getByText('Ausgangspunkt: 1 Passage')).toBeInTheDocument()
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

  it('shows a spinner while the answer is pending, announced once through the live region', () => {
    const view = renderTurn({ user: 'q', assistant: '', done: false, meta: null })
    expect(screen.getByTestId('answer-pending')).toHaveAttribute('aria-hidden', 'true')
    expect(screen.queryByText('…')).not.toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('Working on an answer…')

    view.update({ user: 'q', assistant: 'Alice', done: false, meta: null })
    expect(screen.queryByTestId('answer-pending')).not.toBeInTheDocument()
    expect(screen.getByRole('status')).not.toHaveTextContent('Working on an answer…')

    view.update({ user: 'q', assistant: '', done: true, meta: null })
    expect(screen.queryByTestId('answer-pending')).not.toBeInTheDocument()
    expect(screen.getByText('(no answer)')).toBeInTheDocument()
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

  const LOOKUP_TRAIL: AgentTrail = {
    steps: 1,
    actions: [
      { action: 'note', detail: 'Ich suche zuerst nach dem Hafen.', found: 0 },
      { action: 'search', detail: 'Hafen Regenjacken', found: 3 },
      { action: 'search_images', detail: 'zwei Personen in Regenjacken am Hafen', found: 2 }
    ],
    reason: null,
    answer: 'plain'
  }

  function lookupMeta(trail: AgentTrail = LOOKUP_TRAIL): ChatFinalEvent {
    return { session_id: 's', sources: [], retrieval_mode: 'agent', answer_mode: 'lookup', agent: trail } as ChatFinalEvent
  }

  function trailRows(testId: string): (string | null)[] {
    return within(screen.getByTestId(testId))
      .getAllByRole('listitem')
      .map((row) => row.textContent)
  }

  /** One step of every kind the agent loop added, as a live trail streams them. */
  const NEW_STEPS: AgentStep[] = [
    { step: 1, action: 'note', detail: 'Ich sehe mir zuerst die Bilder an.', found: 0 },
    { step: 1, action: 'search_images', detail: 'Regenjacken, bedeckter Himmel', found: 2 },
    { step: 2, action: 'search_images', detail: 'Person A am Hafen', found: 1 },
    { step: 2, action: 'keyframes', detail: 'harbour_walk.webm', found: 4 },
    { step: 3, action: 'keyframes', detail: 'harbour_walk.webm #3', found: 1 },
    { step: 3, action: 'read', detail: '2', found: 0 },
    { step: 4, action: 'keyword', detail: 'Regenjacke', found: 1, field: 'text', matches: 3, unchecked: 2 }
  ]

  it('a researched lookup shows its steps but no assessment notice', async () => {
    renderTurn({ user: 'Wer steht am Hafen?', assistant: 'Person A steht am Hafen [1].', done: true, meta: lookupMeta() })

    expect(screen.queryByTestId('assessment-notice')).not.toBeInTheDocument()
    expect(screen.queryByTestId('agent-reason-notice')).not.toBeInTheDocument()
    const toggle = screen.getByRole('button', { name: 'Show the searches behind this answer' })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await userEvent.click(toggle)

    expect(screen.getByRole('button', { name: 'Hide the searches behind this answer' })).toHaveAttribute(
      'aria-expanded',
      'true'
    )
    expect(trailRows('agent-trail')).toEqual([
      'Note: Ich suche zuerst nach dem Hafen.',
      'Searched “Hafen Regenjacken”: 3 new passages',
      'Searched the images for “zwei Personen in Regenjacken am Hafen”: 2 new images'
    ])
  })

  it.each([
    ['a lookup', 'lookup'],
    ['a plain answer an agent turn handed over', undefined]
  ] as const)('marks no judgement on %s, whatever its trail token says', (_name, answerMode) => {
    const meta = { ...lookupMeta({ ...LOOKUP_TRAIL, answer: 'unsplit' }), answer_mode: answerMode } as ChatFinalEvent

    renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta })

    expect(screen.queryByTestId('assessment-notice')).not.toBeInTheDocument()
    expect(screen.queryByTestId('assessment-answer-degraded')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Show the searches behind this answer' })).toBeInTheDocument()
  })

  it('image-search, keyframe, read, keyword and note steps have labels', () => {
    renderTurn({ user: 'Was zeigt der Clip?', assistant: '', done: false, meta: null, agentSteps: NEW_STEPS })

    expect(trailRows('agent-trail-live')).toEqual([
      'Note: Ich sehe mir zuerst die Bilder an.',
      'Searched the images for “Regenjacken, bedeckter Himmel”: 2 new images',
      'Searched the images for “Person A am Hafen”: 1 new image',
      'Viewed the keyframes of “harbour_walk.webm”: 4 new keyframes',
      'Viewed keyframe #3 of “harbour_walk.webm”',
      'Read passage [2]',
      'Keyword check “Regenjacke”: 3 matches (2 chunks not checked), 1 new'
    ])
  })

  it('labels the steps the agent loop added in German', () => {
    renderTurn({ user: 'Was zeigt der Clip?', assistant: '', done: false, meta: null, agentSteps: NEW_STEPS }, 'de')

    expect(trailRows('agent-trail-live')).toEqual([
      'Notiz: Ich sehe mir zuerst die Bilder an.',
      'Bilder durchsucht nach „Regenjacken, bedeckter Himmel“: 2 neue Bilder',
      'Bilder durchsucht nach „Person A am Hafen“: 1 neues Bild',
      'Keyframes von „harbour_walk.webm“ angesehen: 4 neue Keyframes',
      'Keyframe #3 von „harbour_walk.webm“ angesehen',
      'Passage [2] gelesen',
      'Wortprüfung „Regenjacke“: 3 Treffer (2 Abschnitte nicht geprüft), 1 neu'
    ])
  })

  it.each([
    [
      'en',
      ['Listed the topics', 'Read topic T12: 1 passage', 'Read topic T3: 4 passages', 'Read topic T7: 0 passages']
    ],
    [
      'de',
      ['Themen aufgelistet', 'Thema T12 gelesen: 1 Passage', 'Thema T3 gelesen: 4 Passagen', 'Thema T7 gelesen: 0 Passagen']
    ]
  ] as const)('labels a topics listing and a topic read in %s', (lang, labels) => {
    renderTurn(
      {
        user: 'q',
        assistant: '',
        done: false,
        meta: null,
        agentSteps: [
          { step: 1, action: 'topics', detail: '', found: 0 },
          { step: 2, action: 'topics', detail: 'T12', found: 1 },
          { step: 2, action: 'topics', detail: 'T3', found: 4 },
          { step: 3, action: 'topics', detail: 'T7', found: 0 }
        ]
      },
      lang
    )

    expect(trailRows('agent-trail-live')).toEqual(labels)
  })

  it('draws a topics step with the layers icon, not a magnifier', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 1, action: 'topics', detail: '', found: 0 },
        { step: 2, action: 'topics', detail: 'T12', found: 3 }
      ]
    })

    const layers = innerMarkup(<LayersIcon />)
    expect(layers).not.toBe(innerMarkup(<SearchIcon />))
    const rows = within(screen.getByTestId('agent-trail-live')).getAllByRole('listitem')
    expect(rows.map((row) => row.querySelector('svg')?.innerHTML)).toEqual([layers, layers])
  })

  it('keeps a topics step in a finished trail', async () => {
    const stored: AgentTrail = {
      ...LOOKUP_TRAIL,
      actions: [
        { action: 'topics', detail: '', found: 0 },
        { action: 'topics', detail: 'T12', found: 2 }
      ]
    }

    renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta: lookupMeta(stored) })

    await userEvent.click(screen.getByRole('button', { name: 'Show the searches behind this answer' }))
    expect(trailRows('agent-trail')).toEqual(['Listed the topics', 'Read topic T12: 2 passages'])
  })

  it('names a passage the model asked for as it cites it, by its number alone', () => {
    // `read` takes the number as the model wrote it, `[2]` included.
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [{ step: 1, action: 'read', detail: '[2]', found: 0 }]
    })

    expect(trailRows('agent-trail-live')).toEqual(['Read passage [2]'])
  })

  it.each([
    [
      'en',
      [
        'Keyword check “Kontrollgruppe”: 0 matches, 0 new',
        'Keyword check “Alice Weber” in Author: 1 match, 2 new',
        'Keyword check “Studie”: 14 matches, 3 new'
      ]
    ],
    [
      'de',
      [
        'Wortprüfung „Kontrollgruppe“: 0 Treffer, 0 neu',
        'Wortprüfung „Alice Weber“ im Feld Autor: 1 Treffer, 2 neu',
        'Wortprüfung „Studie“: 14 Treffer, 3 neu'
      ]
    ]
  ] as const)('labels a keyword check by its count and the field it matched in %s', (lang, labels) => {
    renderTurn(
      {
        user: 'q',
        assistant: '',
        done: false,
        meta: null,
        agentSteps: [
          { step: 1, action: 'keyword', detail: 'Kontrollgruppe', found: 0, field: 'text', matches: 0 },
          { step: 1, action: 'keyword', detail: 'Alice Weber', found: 2, field: 'author', matches: 1 },
          { step: 2, action: 'keyword', detail: 'Studie', found: 3, field: 'text', matches: 14 }
        ]
      },
      lang
    )

    expect(trailRows('agent-trail-live')).toEqual(labels)
  })

  it.each([
    ['en', 'author', 'Alice Weber', 'Keyword check “Alice Weber” in Author: 1 match, 1 new'],
    ['en', 'network', 'Forum A', 'Keyword check “Forum A” in Network: 1 match, 1 new'],
    ['en', 'uuid', 'a1b2c3', 'Keyword check “a1b2c3” in UUID: 1 match, 1 new'],
    ['de', 'author', 'Alice Weber', 'Wortprüfung „Alice Weber“ im Feld Autor: 1 Treffer, 1 neu'],
    ['de', 'network', 'Forum A', 'Wortprüfung „Forum A“ im Feld Netzwerk: 1 Treffer, 1 neu'],
    ['de', 'uuid', 'a1b2c3', 'Wortprüfung „a1b2c3“ im Feld UUID: 1 Treffer, 1 neu']
  ] as const)('names the %s keyword field %s by the Search panel’s label', (lang, field, detail, label) => {
    renderTurn(
      {
        user: 'q',
        assistant: '',
        done: false,
        meta: null,
        agentSteps: [{ step: 1, action: 'keyword', detail, found: 1, field, matches: 1 }]
      },
      lang
    )

    expect(trailRows('agent-trail-live')).toEqual([label])
  })

  it.each(['constructor', 'brand_new_field', undefined])(
    'names no field for a keyword check on the field %j',
    (field) => {
      // The text needs no name, and a token this build does not know is not copy: it must not crash or print.
      renderTurn({
        user: 'q',
        assistant: '',
        done: false,
        meta: null,
        agentSteps: [{ step: 1, action: 'keyword', detail: 'Studie', found: 1, field, matches: 2 }]
      })

      expect(trailRows('agent-trail-live')).toEqual(['Keyword check “Studie”: 2 matches, 1 new'])
    }
  )

  it.each([
    [
      'en',
      [
        'Keyword check “Kontrollgruppe”: 0 matches (12 chunks not checked), 0 new',
        'Keyword check “Studie”: 1 match (1 chunk not checked), 1 new'
      ]
    ],
    [
      'de',
      [
        'Wortprüfung „Kontrollgruppe“: 0 Treffer (12 Abschnitte nicht geprüft), 0 neu',
        'Wortprüfung „Studie“: 1 Treffer (1 Abschnitt nicht geprüft), 1 neu'
      ]
    ]
  ] as const)('names the chunks a keyword check could not reach in %s', (lang, labels) => {
    // A zero over a partly indexed collection is no proof of absence, so the count says what it left out.
    renderTurn(
      {
        user: 'q',
        assistant: '',
        done: false,
        meta: null,
        agentSteps: [
          { step: 1, action: 'keyword', detail: 'Kontrollgruppe', found: 0, field: 'text', matches: 0, unchecked: 12 },
          { step: 2, action: 'keyword', detail: 'Studie', found: 1, field: 'text', matches: 1, unchecked: 1 }
        ]
      },
      lang
    )

    expect(trailRows('agent-trail-live')).toEqual(labels)
  })

  it('keeps the note on unchecked chunks once the turn is done, and adds none when nothing was missed', async () => {
    const { update } = renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 1, action: 'keyword', detail: 'Kontrollgruppe', found: 0, field: 'text', matches: 0, unchecked: 12 }
      ]
    })
    expect(trailRows('agent-trail-live')).toEqual([
      'Keyword check “Kontrollgruppe”: 0 matches (12 chunks not checked), 0 new'
    ])

    update({
      user: 'q',
      assistant: 'Person A [1].',
      done: true,
      meta: lookupMeta({
        ...LOOKUP_TRAIL,
        actions: [
          { action: 'keyword', detail: 'Kontrollgruppe', found: 0, field: 'text', matches: 0, unchecked: 12 },
          { action: 'keyword', detail: 'Studie', found: 1, field: 'text', matches: 2, unchecked: 0 }
        ]
      })
    })
    await userEvent.click(screen.getByRole('button', { name: 'Show the searches behind this answer' }))

    expect(trailRows('agent-trail')).toEqual([
      'Keyword check “Kontrollgruppe”: 0 matches (12 chunks not checked), 0 new',
      'Keyword check “Studie”: 2 matches, 1 new'
    ])
  })

  it.each([
    ['en', 'Keyword check “Nachtschicht”: count unavailable, 1 new'],
    ['de', 'Wortprüfung „Nachtschicht“: Anzahl nicht verfügbar, 1 neu']
  ] as const)('words a failed keyword count as unavailable in %s, with no note on unchecked chunks', (lang, label) => {
    // A failed count is no zero, and it leaves no number for the note on unchecked chunks to qualify.
    renderTurn(
      {
        user: 'q',
        assistant: '',
        done: false,
        meta: null,
        agentSteps: [{ step: 1, action: 'keyword', detail: 'Nachtschicht', found: 1, field: 'text', unchecked: 4 }]
      },
      lang
    )

    expect(trailRows('agent-trail-live')).toEqual([label])
  })

  it('words a failed keyword count alike whether the frame omits it or the stored trail keeps null', async () => {
    const live = renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [{ step: 1, action: 'keyword', detail: 'Nachtschicht', found: 1, field: 'text', unchecked: 4 }]
    })
    const streamed = trailRows('agent-trail-live')
    live.unmount()

    renderTurn({
      user: 'q',
      assistant: 'Person A [1].',
      done: true,
      meta: lookupMeta({
        ...LOOKUP_TRAIL,
        actions: [{ action: 'keyword', detail: 'Nachtschicht', found: 1, field: 'text', matches: null, unchecked: 4 }]
      })
    })
    await userEvent.click(screen.getByRole('button', { name: 'Show the searches behind this answer' }))

    expect(streamed).toEqual(['Keyword check “Nachtschicht”: count unavailable, 1 new'])
    expect(trailRows('agent-trail')).toEqual(streamed)
  })

  /** A stored trail whose own research was only keyword checks: one count known, one failed. */
  const KEYWORD_ONLY_ACTIONS: AgentTrail['actions'] = [
    { action: 'seed', detail: '', found: 2 },
    { action: 'keyword', detail: 'Kontrollgruppe', found: 0, field: 'text', matches: 0 },
    { action: 'keyword', detail: 'Nachtschicht', found: 1, field: 'text', matches: null }
  ]

  it.each([
    ['a lookup', () => lookupMeta({ ...LOOKUP_TRAIL, actions: KEYWORD_ONLY_ACTIONS })],
    ['an assessment', () => assessmentMeta({ ...TRAIL, actions: KEYWORD_ONLY_ACTIONS })]
  ] as const)('lists the keyword checks of %s behind the disclosure', async (_name, meta) => {
    renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta: meta() })

    await userEvent.click(screen.getByRole('button', { name: 'Show the searches behind this answer' }))

    expect(trailRows('agent-trail')).toEqual([
      'Keyword check “Kontrollgruppe”: 0 matches, 0 new',
      'Keyword check “Nachtschicht”: count unavailable, 1 new'
    ])
  })

  it('draws a keyword check as a search, with the magnifier', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [{ step: 1, action: 'keyword', detail: 'Studie', found: 1, field: 'text', matches: 2 }]
    })

    const row = within(screen.getByTestId('agent-trail-live')).getByRole('listitem')
    expect(row.querySelector('svg')?.innerHTML).toBe(innerMarkup(<SearchIcon />))
  })

  it('renders a step this build does not know by its detail, without crashing', () => {
    const unknown = { step: 1, action: 'brand_new_tool', detail: 'Hafen', found: 1 } as unknown as AgentStep

    renderTurn({ user: 'q', assistant: '', done: false, meta: null, agentSteps: [unknown] })

    expect(screen.getByTestId('agent-trail-live')).toHaveTextContent('Hafen')
  })

  it('draws a seed as passages, as a read is, never as a search', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 0, action: 'seed', detail: '', found: 2 },
        { step: 1, action: 'search', detail: 'Hafen', found: 1 },
        { step: 1, action: 'read', detail: '1', found: 0 }
      ]
    })

    const [seed, search, read] = within(screen.getByTestId('agent-trail-live'))
      .getAllByRole('listitem')
      .map((row) => row.querySelector('svg')?.innerHTML)
    expect(seed).toBe(read)
    expect(seed).not.toBe(search)
  })

  it('renders a carried step in the live and the finished trail', async () => {
    const live = renderTurn({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [
        { step: 0, action: 'carried', detail: '', found: 2 },
        { step: 1, action: 'search', detail: 'Hafen', found: 1 }
      ]
    })
    expect(trailRows('agent-trail-live')).toEqual([
      'Carried over 2 passages from earlier answers',
      'Searched “Hafen”: 1 new passage'
    ])
    // Drawn as passages, as a seed and a read are, never as a search.
    const [carried] = within(screen.getByTestId('agent-trail-live')).getAllByRole('listitem')
    expect(carried.querySelector('svg')?.innerHTML).toBe(innerMarkup(<DocumentsIcon />))
    live.unmount()

    // Stored, a follow-up's trail opens with the step, and unlike a pinned seed it stays behind the disclosure.
    const stored: AgentTrail = {
      ...LOOKUP_TRAIL,
      actions: [
        { action: 'carried', detail: '', found: 2 },
        { action: 'search', detail: 'Hafen', found: 1 }
      ]
    }
    renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta: lookupMeta(stored) })
    await userEvent.click(screen.getByRole('button', { name: 'Show the searches behind this answer' }))
    expect(trailRows('agent-trail')).toEqual([
      'Carried over 2 passages from earlier answers',
      'Searched “Hafen”: 1 new passage'
    ])
  })

  it('shows a follow-up that only carried passages as one line, with no search disclosure', () => {
    const stored: AgentTrail = { ...LOOKUP_TRAIL, actions: [{ action: 'carried', detail: '', found: 2 }] }

    renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta: lookupMeta(stored) })

    // It searched nothing, so there is no title or toggle naming searches to put the row behind.
    expect(screen.queryByRole('button', { name: /searches/i })).not.toBeInTheDocument()
    expect(screen.queryByText('Searches behind this answer')).not.toBeInTheDocument()
    expect(trailRows('agent-trail')).toEqual(['Carried over 2 passages from earlier answers'])
    // The row is the trail's own: the passages icon, in the small muted text.
    const trail = screen.getByTestId('agent-trail')
    expect(trail).toHaveClass('text-xs', 'text-muted-foreground')
    expect(within(trail).getByRole('listitem').querySelector('svg')?.innerHTML).toBe(innerMarkup(<DocumentsIcon />))
  })

  it('folds the carried row behind the disclosure when the follow-up also searched', async () => {
    const stored: AgentTrail = {
      ...LOOKUP_TRAIL,
      actions: [
        { action: 'carried', detail: '', found: 2 },
        { action: 'search', detail: 'Hafen', found: 1 }
      ]
    }

    renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta: lookupMeta(stored) })

    // Shut, the disclosure holds both rows: the carried one is not also a line under the answer.
    expect(screen.queryByText(/Carried over/)).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Show the searches behind this answer' }))
    expect(trailRows('agent-trail')).toEqual([
      'Carried over 2 passages from earlier answers',
      'Searched “Hafen”: 1 new passage'
    ])
  })

  it('keeps the carried line of a follow-up whose stream failed before it searched', () => {
    renderTurn({
      user: 'q',
      assistant: '',
      done: true,
      meta: null,
      error: 'Internal server error',
      agentSteps: [{ step: 0, action: 'carried', detail: '', found: 2 }]
    })

    expect(screen.queryByRole('button', { name: /searches/i })).not.toBeInTheDocument()
    expect(trailRows('agent-trail')).toEqual(['Carried over 2 passages from earlier answers'])
  })

  it.each([
    ['en', 1, 'Carried over 1 passage from earlier answers'],
    ['en', 3, 'Carried over 3 passages from earlier answers'],
    ['de', 1, '1 Passage aus früheren Antworten übernommen'],
    ['de', 3, '3 Passagen aus früheren Antworten übernommen']
  ] as const)('words a carried step in %s for %i passage(s)', (lang, found, label) => {
    renderTurn(
      {
        user: 'q',
        assistant: '',
        done: false,
        meta: null,
        agentSteps: [{ step: 0, action: 'carried', detail: '', found }]
      },
      lang
    )

    expect(trailRows('agent-trail-live')).toEqual([label])
  })

  it('marks the steps that read the collection with an icon, and a note with none', () => {
    renderTurn({ user: 'Was zeigt der Clip?', assistant: '', done: false, meta: null, agentSteps: NEW_STEPS })

    const rows = within(screen.getByTestId('agent-trail-live')).getAllByRole('listitem')
    expect(rows.map((row) => row.querySelector('svg') !== null)).toEqual([false, true, true, true, true, true, true])
    expect(rows[0].firstElementChild).toHaveAttribute('aria-hidden', 'true')
  })

  it('a legacy trail with seed and answer steps still renders', async () => {
    // Live, as #648 streamed an assessment: its seed, its searches, then the answer step.
    const live = renderTurn({
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
    expect(trailRows('agent-trail-live')).toEqual([
      'Started from 2 passages',
      'Searched “evaluation datasets”: 1 new passage',
      'Writing the assessment…'
    ])
    live.unmount()

    // Stored, with `shortened`. Seed and answer are no research of the model's own, so the list holds the search.
    const stored: AgentTrail = {
      ...TRAIL,
      actions: [...TRAIL.actions, { action: 'answer', detail: '', found: 3 }],
      shortened: 2
    }
    renderTurn({ user: 'q', assistant: 'Fact [1].', done: true, meta: assessmentMeta(stored) })
    expect(screen.getByTestId('assessment-notice')).toHaveTextContent('2 passages were shortened')
    await userEvent.click(screen.getByRole('button', { name: 'Show the searches behind this answer' }))
    expect(trailRows('agent-trail')).toEqual(['Searched “evaluation datasets”: 1 new passage'])
  })

  it.each([
    'tools_unavailable',
    'round_budget',
    'time_budget',
    'context_budget',
    'idle',
    'research_failed',
    'no_evidence',
    'answer_cut'
  ] as const)('explains the %s reason on an agent lookup', (reason: AgentReason) => {
    renderTurn({
      user: 'q',
      assistant: 'Person A [1].',
      done: true,
      meta: lookupMeta({ ...LOOKUP_TRAIL, reason })
    })

    const copy: Record<string, string> = en
    expect(screen.getByTestId('agent-reason-notice')).toHaveTextContent(copy[`chat.agent_reason.${reason}`])
  })

  it.each(['round_budget', 'time_budget', 'context_budget', 'idle'] as const)(
    'shows the %s limit as a muted note, not an alert',
    (reason: AgentReason) => {
      renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta: lookupMeta({ ...LOOKUP_TRAIL, reason }) })

      const note = screen.getByTestId('agent-reason-notice')
      expect(note).toHaveAttribute('role', 'note')
      expect(note.className).not.toMatch(/amber/)
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    }
  )

  it.each(['research_failed', 'tools_unavailable', 'no_evidence', 'answer_cut'] as const)(
    'keeps the amber alert for %s',
    (reason: AgentReason) => {
      renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta: lookupMeta({ ...LOOKUP_TRAIL, reason }) })

      const alert = screen.getByRole('alert')
      expect(alert).toHaveAttribute('data-testid', 'agent-reason-notice')
      expect(alert.className).toMatch(/amber/)
    }
  )

  it.each([
    ['en', 'length limit'],
    ['de', 'Längengrenze']
  ] as const)('says in %s that an answer cut at the length limit may be incomplete', (lang, limit) => {
    renderTurn(
      {
        user: 'q',
        assistant: 'Person A steht am Hafen [1], und',
        done: true,
        meta: lookupMeta({ ...LOOKUP_TRAIL, reason: 'answer_cut' })
      },
      lang
    )

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent(limit)
    expect(alert).toHaveTextContent(/incomplete|unvollständig/)
  })

  it.each([
    ['en', 'Some evidence may be missing'],
    ['de', 'Möglicherweise fehlen einige Belege']
  ] as const)('says in %s that evidence may be missing beside a complete answer, not that it failed', (lang, copy) => {
    renderTurn(
      {
        user: 'q',
        assistant: 'Person A steht am Hafen [1].',
        done: true,
        meta: lookupMeta({ ...LOOKUP_TRAIL, reason: 'research_failed' })
      },
      lang
    )

    const notice = screen.getByTestId('agent-reason-notice')
    expect(notice).toHaveTextContent(copy)
    expect(notice).not.toHaveTextContent(/fail|fehlgeschlagen|abgebrochen/i)
  })

  it('explains a plain answer an agent turn handed over when its tools were refused', () => {
    // The fallback's envelope: a plain lookup with no answer mode and a trail of no steps.
    const meta = {
      session_id: 's',
      sources: [],
      retrieval_mode: 'rewrite_compact',
      agent: { steps: 0, actions: [], reason: 'tools_unavailable', answer: 'plain' }
    } as unknown as ChatFinalEvent

    renderTurn({ user: 'q', assistant: 'Person A [1].', done: true, meta })

    expect(screen.getByTestId('agent-reason-notice')).toHaveTextContent(/could not run its own searches/i)
    expect(screen.queryByTestId('assessment-notice')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /searches/i })).not.toBeInTheDocument()
  })

  it('keeps every call of a round in the live trail as more steps arrive', () => {
    // One round's calls share its step number.
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {})
    onTestFinished(() => errors.mockRestore())
    const round: AgentStep[] = [
      { step: 1, action: 'search', detail: 'Hafen', found: 2 },
      { step: 1, action: 'search', detail: 'Regenjacken', found: 1 }
    ]
    const view = renderTurn({ user: 'q', assistant: '', done: false, meta: null, agentSteps: round })
    view.update({
      user: 'q',
      assistant: '',
      done: false,
      meta: null,
      agentSteps: [...round, { step: 2, action: 'search', detail: 'bedeckter Himmel', found: 0 }]
    })

    expect(trailRows('agent-trail-live')).toEqual([
      'Searched “Hafen”: 2 new passages',
      'Searched “Regenjacken”: 1 new passage',
      'Searched “bedeckter Himmel”: 0 new passages'
    ])
    expect(errors.mock.calls.flat().join(' ')).not.toMatch(/same key/i)
  })

  it('flags an agent turn on the visual target that saw no pictures, as any visual turn', () => {
    const meta = { ...lookupMeta(), retrieval_target: 'visual', visual: { images_attached: 0 } } as ChatFinalEvent

    renderTurn({ user: 'Was ist am Hafen zu sehen?', assistant: 'Zwei Personen [1].', done: true, meta })

    expect(screen.getByTestId('visual-pixels-missing')).toBeInTheDocument()
  })
})
