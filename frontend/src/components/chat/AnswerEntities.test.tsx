import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import type { Source } from '@/api/types'
import { AnswerEntities } from './AnswerEntities'
import { useUiStore } from '@/stores/ui'
import { useAnalysisUiStore } from '@/stores/analysisUi'

const navigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return { ...actual, useNavigate: () => navigate }
})

beforeEach(() => {
  navigate.mockClear()
  useUiStore.setState({ selectedCollection: 'docs' })
  useAnalysisUiStore.setState({ tab: 'hate', entity: null })
})

function renderPills(sources: Partial<Source>[]) {
  return render(
    <MemoryRouter>
      <AnswerEntities sources={sources as Source[]} />
    </MemoryRouter>
  )
}

describe('AnswerEntities', () => {
  it('renders nothing when no source carries entities', () => {
    const { container } = renderPills([{ filename: 'a.pdf' }])
    expect(container).toBeEmptyDOMElement()
  })

  it('merges the same entity across sources and sums its mentions', () => {
    renderPills([
      { filename: 'a.pdf', entities: [{ text: 'Acme Corp', type: 'ORG', count: 2 }] },
      { filename: 'b.pdf', entities: [{ text: 'acme corp', type: 'ORG', count: 3 }] }
    ])

    const pills = screen.getAllByTestId('answer-entity')
    expect(pills).toHaveLength(1)
    expect(pills[0]).toHaveTextContent('Acme Corp')
    expect(pills[0]).toHaveTextContent('5')
  })

  it('ranks by mention count', () => {
    renderPills([
      {
        filename: 'a.pdf',
        entities: [
          { text: 'Rare', type: 'ORG', count: 1 },
          { text: 'Common', type: 'ORG', count: 9 }
        ]
      }
    ])

    const pills = screen.getAllByTestId('answer-entity')
    expect(pills[0]).toHaveTextContent('Common')
    expect(pills[1]).toHaveTextContent('Rare')
  })

  it('caps the visible pills and reveals the rest on demand', async () => {
    const entities = Array.from({ length: 15 }, (_, i) => ({
      text: `Entity ${i}`,
      type: 'ORG',
      count: 15 - i
    }))
    renderPills([{ filename: 'a.pdf', entities }])

    expect(screen.getAllByTestId('answer-entity')).toHaveLength(12)

    await userEvent.click(screen.getByRole('button', { name: '+3 more' }))

    expect(screen.getAllByTestId('answer-entity')).toHaveLength(15)
  })

  it('opens the entity in the Analysis tab when clicked', async () => {
    renderPills([{ filename: 'a.pdf', entities: [{ text: 'Alice Weber', type: 'PER' }] }])

    await userEvent.click(screen.getByRole('button', { name: /Alice Weber/ }))

    expect(useAnalysisUiStore.getState().tab).toBe('ner')
    expect(useAnalysisUiStore.getState().entity).toEqual({
      key: 'Alice Weber::PER',
      collection: 'docs'
    })
    expect(navigate).toHaveBeenCalledWith('/analysis')
  })

  it('is not clickable without an active collection to resolve against', () => {
    useUiStore.setState({ selectedCollection: null })
    renderPills([{ filename: 'a.pdf', entities: [{ text: 'Alice Weber', type: 'PER' }] }])

    expect(screen.queryByRole('button', { name: /Alice Weber/ })).not.toBeInTheDocument()
    expect(screen.getByTestId('answer-entity')).toHaveTextContent('Alice Weber')
  })
})
