import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntityGraph } from './EntityGraph'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'

const nodes: NerGraphNode[] = [
  { id: 'acme::org', text: 'Acme', type: 'ORG', mentions: 9 },
  { id: 'rivertown::loc', text: 'Rivertown', type: 'LOC', mentions: 4 },
  { id: 'widget::product', text: 'Widget', type: 'PRODUCT', mentions: 2 }
]

const edges: NerGraphEdge[] = [
  { source: 'acme::org', target: 'rivertown::loc', label: 'located_in', kind: 'relation', weight: 3 },
  { source: 'acme::org', target: 'widget::product', label: 'makes', kind: 'relation', weight: 1 }
]

const keyForNode = (n: NerGraphNode) => `${n.text}::${n.type}`

describe('EntityGraph', () => {
  it('renders a node per entity with its label', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByText('Acme')).toBeInTheDocument()
    expect(screen.getByText('Rivertown')).toBeInTheDocument()
    expect(screen.getByText('Widget')).toBeInTheDocument()
  })

  it('selects an entity (by text::type key) when its node is clicked', async () => {
    const onSelectEntity = vi.fn()
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={onSelectEntity}
        keyForNode={keyForNode}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG, 9 mentions\)/ }))
    expect(onSelectEntity).toHaveBeenCalledWith('Acme::ORG')
  })

  it('marks the selected node pressed for assistive tech', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Acme::ORG"
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    const selected = screen.getByRole('button', { name: /Acme \(ORG/ })
    expect(selected).toHaveAttribute('aria-pressed', 'true')
  })

  it('renders a type legend', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByText('Types')).toBeInTheDocument()
    expect(screen.getByText('ORG')).toBeInTheDocument()
    expect(screen.getByText('PRODUCT')).toBeInTheDocument()
  })

  it('shows an empty state when there are no nodes', () => {
    render(
      <EntityGraph
        nodes={[]}
        edges={[]}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByText(/no entity relationships to graph/i)).toBeInTheDocument()
  })
})
