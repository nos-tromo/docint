import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntityGraph } from './EntityGraph'
import { colorForType } from '@/lib/entityGraphElements'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'

async function selectAcme() {
  await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG\)/ }))
}

/** A minimal controlled wrapper, mirroring how a real parent owns `selectedKey`. */
function ControlledEntityGraph(props: {
  nodes: NerGraphNode[]
  edges: NerGraphEdge[]
  initialSelectedKey?: string | null
  onSelectEntity?: (key: string | null) => void
}) {
  const [selectedKey, setSelectedKey] = useState<string | null>(props.initialSelectedKey ?? null)
  return (
    <EntityGraph
      nodes={props.nodes}
      edges={props.edges}
      selectedKey={selectedKey}
      onSelectEntity={(key) => {
        setSelectedKey(key)
        props.onSelectEntity?.(key)
      }}
      keyForNode={keyForNode}
    />
  )
}

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

  it('renders an edge (svg line) per graph edge', () => {
    const { container } = render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(container.querySelectorAll('line')).toHaveLength(edges.length)
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
    await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG\)/ }))
    expect(onSelectEntity).toHaveBeenCalledWith('Acme::ORG')
  })

  it('clears the selection when the background is clicked', () => {
    const onSelectEntity = vi.fn()
    const { container } = render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Acme::ORG"
        onSelectEntity={onSelectEntity}
        keyForNode={keyForNode}
      />
    )
    const backgroundRect = container.querySelector('svg > rect')
    expect(backgroundRect).toBeTruthy()
    fireEvent.pointerDown(backgroundRect!, { clientX: 10, clientY: 10 })
    fireEvent.pointerUp(backgroundRect!, { clientX: 10, clientY: 10 })
    expect(onSelectEntity).toHaveBeenCalledWith(null)
  })

  it('still selects an entity by key when a node is clicked (regression)', async () => {
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
    await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG\)/ }))
    expect(onSelectEntity).toHaveBeenCalledWith('Acme::ORG')
  })

  it('marks the selected node pressed for assistive tech', async () => {
    // Mounting with a pre-existing `selectedKey` no longer marks it pressed
    // (see the "no dimmed nodes on mount" tests below) — selecting via an
    // in-graph click still does.
    render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    const selected = screen.getByRole('button', { name: /Acme \(ORG\)/ })
    expect(selected).toHaveAttribute('aria-pressed', 'true')
  })

  it('renders a type legend for the top entity types', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByText('ORG')).toBeInTheDocument()
    expect(screen.getByText('LOC')).toBeInTheDocument()
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

  it('does not show the empty state while loading, even with zero nodes', () => {
    render(
      <EntityGraph
        nodes={[]}
        edges={[]}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        isLoading
      />
    )
    expect(screen.queryByText(/no entity relationships to graph/i)).not.toBeInTheDocument()
  })

  it('renders the node-count control inline with the other graph controls', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        nodeCount={80}
        nodeCountMax={500}
        onNodeCountChange={() => {}}
        onResetNodeCount={() => {}}
      />
    )
    expect(screen.getByLabelText('Graph node count')).toHaveValue(80)
  })

  it('hides the node-count control when no handler is wired', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.queryByLabelText('Graph node count')).not.toBeInTheDocument()
  })

  it('fires onNodeCountChange when the node-count control commits a new value', async () => {
    const onNodeCountChange = vi.fn()
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        nodeCount={80}
        nodeCountMax={500}
        onNodeCountChange={onNodeCountChange}
        onResetNodeCount={() => {}}
      />
    )
    const input = screen.getByLabelText('Graph node count')
    await userEvent.clear(input)
    await userEvent.type(input, '120')
    await userEvent.tab()
    expect(onNodeCountChange).toHaveBeenCalledWith(120)
  })

  it('resets the node count to the deploy default via its own reset control', async () => {
    const onResetNodeCount = vi.fn()
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        nodeCount={250}
        nodeCountMax={500}
        onNodeCountChange={() => {}}
        onResetNodeCount={onResetNodeCount}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /reset node count/i }))
    expect(onResetNodeCount).toHaveBeenCalledTimes(1)
  })
})

describe('EntityGraph node removal (view-only)', () => {
  it('removes a selected node on "Remove node", keeping other nodes', async () => {
    render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))

    expect(screen.queryByText('Acme')).not.toBeInTheDocument()
    expect(screen.getByText('Rivertown')).toBeInTheDocument()
    expect(screen.getByText('Widget')).toBeInTheDocument()
  })

  it('removing a node also removes its incident edges', async () => {
    const { container } = render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))

    // Both edges touched Acme, so both should be gone.
    expect(container.querySelectorAll('line')).toHaveLength(0)
  })

  it('deselects via onSelectEntity(null) when the removed node was the selected entity', async () => {
    const onSelectEntity = vi.fn()
    render(<ControlledEntityGraph nodes={nodes} edges={edges} onSelectEntity={onSelectEntity} />)
    await selectAcme()
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))
    expect(onSelectEntity).toHaveBeenCalledWith(null)
  })

  it('removes the selected node on Backspace', async () => {
    render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    fireEvent.keyDown(document, { key: 'Backspace' })

    expect(screen.queryByText('Acme')).not.toBeInTheDocument()
    expect(screen.getByText('Rivertown')).toBeInTheDocument()
  })

  it('restores removed nodes when the nodes prop identity changes (fresh fetch)', async () => {
    let selectedKey: string | null = null
    const onSelectEntity = (key: string | null) => {
      selectedKey = key
    }
    const { rerender, container } = render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={selectedKey}
        onSelectEntity={onSelectEntity}
        keyForNode={keyForNode}
      />
    )
    await selectAcme()
    rerender(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={selectedKey}
        onSelectEntity={onSelectEntity}
        keyForNode={keyForNode}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))
    expect(screen.queryByText('Acme')).not.toBeInTheDocument()

    const freshNodes = nodes.map((n) => ({ ...n }))
    const freshEdges = edges.map((e) => ({ ...e }))
    rerender(
      <EntityGraph
        nodes={freshNodes}
        edges={freshEdges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )

    expect(screen.getByText('Acme')).toBeInTheDocument()
    expect(container.querySelectorAll('line')).toHaveLength(edges.length)
  })

  it('reflects the filtered node/edge set in the type legend after removal', async () => {
    render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))
    expect(screen.queryByText('ORG')).not.toBeInTheDocument()
    expect(screen.getByText('LOC')).toBeInTheDocument()
    expect(screen.getByText('PRODUCT')).toBeInTheDocument()
  })
})

describe('EntityGraph multi-select (marquee/shift)', () => {
  it('keeps all shift-selected nodes on the canvas and reports the last one to the panel', async () => {
    const onSelectEntity = vi.fn()
    render(<ControlledEntityGraph nodes={nodes} edges={edges} onSelectEntity={onSelectEntity} />)

    await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG\)/ }))
    fireEvent.click(screen.getByRole('button', { name: /Rivertown \(LOC\)/ }), { shiftKey: true })
    fireEvent.click(screen.getByRole('button', { name: /Widget \(PRODUCT\)/ }), { shiftKey: true })

    const pressed = screen.getAllByRole('button', { pressed: true })
    expect(pressed).toHaveLength(3)
    expect(onSelectEntity).toHaveBeenLastCalledWith('Widget::PRODUCT')
  })

  it('mounting with a pre-existing selectedKey renders no dimmed nodes', () => {
    const { container } = render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Acme::ORG"
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.queryAllByRole('button', { pressed: true })).toHaveLength(0)
    const nodeGroups = container.querySelectorAll('svg g[role="button"]')
    expect(nodeGroups).toHaveLength(3)
    for (const g of nodeGroups) {
      expect(g.getAttribute('opacity')).toBe('1')
    }
  })

  it('a selectedKey change after mount selects exactly that node', () => {
    const { rerender, container } = render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Acme::ORG"
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    // No selection yet (mount doesn't adopt the pre-existing key).
    expect(container.querySelectorAll('svg g[aria-pressed="true"]')).toHaveLength(0)

    rerender(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Rivertown::LOC"
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    const selected = screen.getByRole('button', { name: /Rivertown \(LOC\)/ })
    expect(selected).toHaveAttribute('aria-pressed', 'true')
    expect(screen.queryAllByRole('button', { pressed: true })).toHaveLength(1)
  })

  it('removing a shift-selected set of two removes both and their edges, clearing the panel if it pointed at one', async () => {
    const onSelectEntity = vi.fn()
    const { container } = render(
      <ControlledEntityGraph
        nodes={nodes}
        edges={edges}
        initialSelectedKey="Acme::ORG"
        onSelectEntity={onSelectEntity}
      />
    )
    // Select Acme (already the panel's entity) + Rivertown, leave Widget alone.
    await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG\)/ }))
    fireEvent.click(screen.getByRole('button', { name: /Rivertown \(LOC\)/ }), { shiftKey: true })
    expect(screen.getAllByRole('button', { pressed: true })).toHaveLength(2)

    await userEvent.click(screen.getByRole('button', { name: /remove 2 nodes/i }))

    expect(screen.queryByText('Acme')).not.toBeInTheDocument()
    expect(screen.queryByText('Rivertown')).not.toBeInTheDocument()
    expect(screen.getByText('Widget')).toBeInTheDocument()
    // Both incident edges touched Acme or Rivertown, so both should be gone.
    expect(container.querySelectorAll('line')).toHaveLength(0)
    expect(onSelectEntity).toHaveBeenCalledWith(null)
  })
})

describe('EntityGraph export (JSON/GraphML/HTML)', () => {
  let capturedBlob: Blob | null = null
  let capturedFilename: string | null = null

  beforeEach(() => {
    capturedBlob = null
    capturedFilename = null
    vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => {
      capturedBlob = blob as Blob
      return 'blob:mock-url'
    })
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    // Mock HTMLAnchorElement.click to capture the download attribute
    HTMLAnchorElement.prototype.click = vi.fn(function (this: HTMLAnchorElement) {
      capturedFilename = this.download
    })
  })

  it('hides the export buttons when there is no graph to export', () => {
    render(
      <EntityGraph
        nodes={[]}
        edges={[]}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.queryByRole('button', { name: 'Export JSON' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Export GraphML' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Export HTML' })).not.toBeInTheDocument()
  })

  it('shows the export buttons once there is graph data', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByRole('button', { name: 'Export JSON' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export GraphML' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export HTML' })).toBeInTheDocument()
  })

  it('triggers a JSON download via the Blob/createObjectURL mock on click', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1)
    expect(capturedBlob).toBeTruthy()
    const text = await capturedBlob!.text()
    expect(JSON.parse(text).nodes).toHaveLength(nodes.length)
  })

  it('exports only the post-removal (visible) node/edge set', async () => {
    render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))

    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    const text = await capturedBlob!.text()
    const parsed = JSON.parse(text)
    expect(parsed.nodes.map((n: { id: string }) => n.id)).not.toContain('acme::org')
    expect(parsed.nodes).toHaveLength(nodes.length - 1)
  })

  it('exports with default filename when no exportName is provided', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    expect(capturedFilename).toBe('docint_entity_graph.json')
  })

  it('exports with collection-named filename when exportName is provided', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="My Reports"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    expect(capturedFilename).toBe('my-reports_entity_graph.json')
  })

  it('sanitizes exportName: converts spaces to dashes', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="My Reports 2026"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    expect(capturedFilename).toBe('my-reports-2026_entity_graph.json')
  })

  it('sanitizes exportName: converts slashes and special chars to dashes', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="My Reports/2026"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    expect(capturedFilename).toBe('my-reports-2026_entity_graph.json')
  })

  it('sanitizes exportName: collapses consecutive dashes', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="My--Reports//2026"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    expect(capturedFilename).toBe('my-reports-2026_entity_graph.json')
  })

  it('sanitizes exportName: trims leading/trailing dashes', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="-My Reports-"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    expect(capturedFilename).toBe('my-reports_entity_graph.json')
  })

  it('sanitizes exportName: falls back to "docint" when exportName becomes empty after sanitization', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="---"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export JSON' }))
    expect(capturedFilename).toBe('docint_entity_graph.json')
  })

  it('uses sanitized exportName in HTML export title', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="My Reports 2026"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export HTML' }))
    const text = await capturedBlob!.text()
    expect(text).toContain('<title>My Reports 2026 — entity graph</title>')
  })

  it('uses "Entity graph" title in HTML export when no exportName provided', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export HTML' }))
    const text = await capturedBlob!.text()
    expect(text).toContain('<title>Entity graph</title>')
  })

  it('exports GraphML with collection-named filename', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="My Reports"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export GraphML' }))
    expect(capturedFilename).toBe('my-reports_entity_graph.graphml')
  })

  it('exports HTML with collection-named filename', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        exportName="My Reports"
      />
    )
    await userEvent.click(screen.getByRole('button', { name: 'Export HTML' }))
    expect(capturedFilename).toBe('my-reports_entity_graph.html')
  })
})

describe('colorForType', () => {
  it('is stable: the same type always resolves to the same color', () => {
    expect(colorForType('ORG')).toBe(colorForType('ORG'))
    expect(colorForType('PERSON')).toBe(colorForType('PERSON'))
  })

  it('differentiates at least some distinct types', () => {
    const colors = new Set(['ORG', 'PERSON', 'LOC', 'PRODUCT', 'EVENT'].map(colorForType))
    expect(colors.size).toBeGreaterThan(1)
  })
})
