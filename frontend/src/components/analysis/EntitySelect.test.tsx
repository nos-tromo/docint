import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntitySelect } from './EntitySelect'
import type { NerEntityRow } from '@/api/types'

const entities: NerEntityRow[] = [
  { text: 'Berlin', type: 'LOC', mentions: 4 },
  { text: 'Paris', type: 'LOC', mentions: 3 },
  { text: 'Alice', type: 'PER', mentions: 2 }
]

const keyOf = (e: NerEntityRow) => `${e.text}::${e.type}`

describe('EntitySelect', () => {
  it('lists every entity option labelled with type and mentions by default', async () => {
    render(
      <EntitySelect entities={entities} selectedKey="Berlin::LOC" onSelectEntity={() => {}} keyOf={keyOf} />
    )
    await userEvent.click(screen.getByRole('combobox', { name: /^entity$/i }))
    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual([
      'Berlin [LOC] · 4',
      'Paris [LOC] · 3',
      'Alice [PER] · 2'
    ])
  })

  it('filters the entity list by the chosen category and pre-selects its top entity', async () => {
    const onSelectEntity = vi.fn()
    render(
      <EntitySelect entities={entities} selectedKey="Berlin::LOC" onSelectEntity={onSelectEntity} keyOf={keyOf} />
    )
    // Selecting the PER category must (a) re-filter the entity dropdown and
    // (b) pre-select that category's top entity — the previous ref-based
    // selector did neither.
    await userEvent.click(screen.getByRole('combobox', { name: /entity category/i }))
    await userEvent.click(screen.getByRole('option', { name: 'PER' }))
    expect(onSelectEntity).toHaveBeenCalledWith('Alice::PER')

    await userEvent.click(screen.getByRole('combobox', { name: /^entity$/i }))
    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['Alice [PER] · 2'])
  })

  it('emits the new key when the user picks a different entity', async () => {
    const onSelectEntity = vi.fn()
    render(
      <EntitySelect entities={entities} selectedKey="Berlin::LOC" onSelectEntity={onSelectEntity} keyOf={keyOf} />
    )
    await userEvent.click(screen.getByRole('combobox', { name: /^entity$/i }))
    await userEvent.click(screen.getByRole('option', { name: 'Alice [PER] · 2' }))
    expect(onSelectEntity).toHaveBeenCalledWith('Alice::PER')
  })

  it('keeps the captions readable beside the pickers', () => {
    render(
      <EntitySelect entities={entities} selectedKey="Berlin::LOC" onSelectEntity={() => {}} keyOf={keyOf} />
    )
    // The trigger shows a value, so the caption has to stay on screen as well
    // as in the accessible name.
    expect(screen.getByText(/entity category/i)).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: /^entity$/i })).toHaveTextContent('Berlin [LOC] · 4')
  })

  it('falls back to a helpful message when there are no entities', () => {
    render(<EntitySelect entities={[]} selectedKey={null} onSelectEntity={() => {}} keyOf={keyOf} />)
    expect(screen.getByText(/no entities found/i)).toBeInTheDocument()
  })
})
