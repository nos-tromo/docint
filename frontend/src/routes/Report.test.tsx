import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { act, render, screen, waitFor, fireEvent, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Report } from './Report'
import { resetReportPdfJobs } from '@/hooks/useReportPdf'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'

const overview = {
  collection: 'docs',
  captured_at: '2026-07-01T09:00:00Z',
  document_count: 1,
  node_count: 6,
  file_types: [{ label: 'PDF', count: 1 }],
  entity_types: ['ORG'],
  documents: [
    { filename: 'c.pdf', type_label: 'PDF', page_count: 4, row_count: null, node_count: 6, file_hash: '0123456789abcdef' }
  ]
}

const reportDetail = {
  id: 1,
  title: 'Case Alpha',
  collection_name: 'docs',
  show_toc: true,
  show_collection_overview: true,
  session_id: null,
  created_at: null,
  updated_at: null,
  item_count: 2,
  items: [
    {
      id: 10,
      artifact_type: 'entity_finding',
      dedupe_key: 'entity:c1',
      position: 0,
      note: null,
      snapshot: { entity_label: 'Acme [ORG]', chunk_text: 'Acme text', filename: 'a.pdf', page: 1 },
      created_at: null
    },
    {
      id: 11,
      artifact_type: 'hate_speech_finding',
      dedupe_key: 'hate:c2',
      position: 1,
      note: null,
      snapshot: { category: 'slur', confidence: 'high', reason: 'bad', filename: 'b.json' },
      created_at: null
    }
  ],
  collection_overview: overview
}

const NO_PDF = { job: null, pdf: null }

function mockFetch(
  detail: Record<string, unknown> = reportDetail,
  pdfStatus: Record<string, unknown> = NO_PDF
) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (u: string, init?: RequestInit) => {
      const url = String(u)
      // Ahead of the `/reports/1` branch, which would answer them with the report.
      if (url.includes('/reports/1/pdf/status')) {
        return { ok: true, status: 200, json: async () => pdfStatus }
      }
      if (url.includes('/reports/1/pdf') && init?.method === 'POST') {
        return { ok: true, status: 202, json: async () => ({ job_id: 'f00d01' }) }
      }
      if (url.includes('/reports/1')) {
        return { ok: true, status: 200, json: async () => detail }
      }
      if (url.endsWith('/reports')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            reports: [{ ...detail, items: undefined }]
          })
        }
      }
      return { ok: true, status: 200, json: async () => ({}) }
    })
  )
}

function renderReport() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <Report />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  localStorage.clear()
  useReportStore.setState({ activeReportId: 1 })
  mockFetch()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Report view', () => {
  it('shows the active report title, grouped items and export links', async () => {
    renderReport()
    expect(await screen.findByDisplayValue('Case Alpha')).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText('Acme [ORG]')).toBeInTheDocument()
    })
    expect(screen.getByText(/Entity findings/i)).toBeInTheDocument()
    expect(screen.getByText(/Hate-speech findings/i)).toBeInTheDocument()
    // The exports re-sort hate-speech findings, which the hand order must not hide.
    expect(
      screen.getByText('The PDF, HTML and Markdown exports list these by posting time, newest first.')
    ).toBeInTheDocument()

    // The formats live behind the download button now, and it opens on a
    // click rather than on hover — which is what made them reachable by touch
    // and by keyboard at all.
    expect(screen.queryByRole('menuitem', { name: 'PDF' })).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /download/i }))

    // No stored render yet, so the PDF entry starts one rather than linking.
    const pdf = screen.getByRole('menuitem', { name: 'PDF' })
    expect(pdf.tagName).toBe('BUTTON')
    expect(pdf).not.toHaveAttribute('href')
    const md = screen.getByRole('menuitem', { name: 'Markdown' })
    expect(md).toHaveAttribute('href', expect.stringContaining('/reports/1/export.md'))
    expect(md).toHaveAttribute('download')
    const html = screen.getByRole('menuitem', { name: 'HTML' })
    expect(html).toHaveAttribute('target', '_blank')
    expect(html).toHaveAttribute('rel', 'noreferrer')
  })

  it('prompts to select a report when none is active', async () => {
    useReportStore.setState({ activeReportId: null })
    renderReport()
    expect(await screen.findByText(/select a report/i)).toBeInTheDocument()
  })

  it('reflects show_toc and toggles it via PATCH', async () => {
    renderReport()
    const toggle = await screen.findByRole('checkbox', { name: /contents/i })
    expect(toggle).toBeChecked()
    fireEvent.click(toggle)
    await waitFor(() => {
      const patch = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find(
        (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(patch).toBeTruthy()
      expect(JSON.parse(String((patch![1] as RequestInit).body))).toMatchObject({ show_toc: false })
    })
  })
})

describe('Report view — PDF export', () => {
  const RENDER = {
    report_id: 1,
    filename: 'report-1-Case Alpha.pdf',
    size: 52_000,
    created_at: '2026-03-01T10:05:00Z',
    report_updated_at: '2026-03-01T10:00:00Z',
    pages: 12
  }

  function pdfJob(status: string) {
    return {
      job_id: 'f00d01',
      collection: 'docs',
      status,
      message: null,
      error: null,
      empty: false,
      resolution: null,
      kind: 'report_pdf',
      target: '1',
      artifact: null,
      created_at: '2026-03-01T10:01:00Z',
      run_started_at: '2026-03-01T10:01:00Z',
      started_at: null,
      finished_at: null,
      duration_ms: null
    }
  }

  const STATUS_LINES = [
    'Queued — waiting for another export to finish',
    'Rendering the PDF…',
    'Finishing the PDF…',
    'The PDF could not be created.'
  ]

  /** The menu's PDF entry, re-read each time: it is a different element as a link than as a button. */
  const pdfEntry = () => screen.getByRole('menuitem', { name: 'PDF' })

  async function openExportMenu() {
    await screen.findByText('Acme [ORG]')
    await userEvent.click(screen.getByRole('button', { name: /download/i }))
    return pdfEntry()
  }

  beforeEach(() => {
    useIngestJobsStore.getState().clear()
    resetReportPdfJobs()
  })

  afterEach(() => {
    useIngestJobsStore.getState().clear()
    vi.unstubAllGlobals()
  })

  it('links a PDF that matches the report straight to its download', async () => {
    mockFetch(reportDetail, { job: pdfJob('completed'), pdf: { ...RENDER, current: true } })
    renderReport()
    await openExportMenu()
    await waitFor(() => expect(pdfEntry().tagName).toBe('A'))
    expect(pdfEntry().getAttribute('href')).toMatch(/\/reports\/1\/pdf$/)
    expect(pdfEntry()).toHaveAttribute('download', 'report-1-Case Alpha.pdf')
  })

  it('renders again instead of linking a PDF the report has outgrown', async () => {
    mockFetch(reportDetail, { job: pdfJob('completed'), pdf: { ...RENDER, current: false } })
    renderReport()
    await openExportMenu()
    await waitFor(() => expect(pdfEntry()).toHaveAttribute('title', expect.stringMatching(/changed since the last PDF/)))
    expect(pdfEntry().tagName).toBe('BUTTON')
    expect(pdfEntry()).not.toHaveAttribute('aria-disabled')
  })

  it('starts a render from the PDF entry, then keeps it from starting another', async () => {
    let pdfStatus: Record<string, unknown> = NO_PDF
    const posts: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/reports/1/pdf/status')) return { ok: true, status: 200, json: async () => pdfStatus }
        if (url.includes('/reports/1/pdf') && init?.method === 'POST') {
          posts.push(url)
          pdfStatus = { job: pdfJob('queued'), pdf: null }
          return { ok: true, status: 202, json: async () => ({ job_id: 'f00d01' }) }
        }
        if (url.includes('/reports/1')) return { ok: true, status: 200, json: async () => reportDetail }
        if (url.endsWith('/reports')) {
          return { ok: true, status: 200, json: async () => ({ reports: [{ ...reportDetail, items: undefined }] }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    renderReport()
    await openExportMenu()
    await waitFor(() => expect(pdfEntry()).toHaveAttribute('title', expect.stringMatching(/downloads when it is ready/)))
    await userEvent.click(pdfEntry())

    await waitFor(() => expect(posts).toHaveLength(1))
    expect(await screen.findByText('Queued — waiting for another export to finish')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /download/i }))
    expect(pdfEntry()).toHaveAttribute('aria-disabled', 'true')
    await userEvent.click(pdfEntry())
    expect(posts).toHaveLength(1)
  })

  it('says a queued render is waiting', async () => {
    mockFetch(reportDetail, { job: pdfJob('queued'), pdf: null })
    renderReport()
    expect(await screen.findByText('Queued — waiting for another export to finish')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Queued — waiting for another export to finish' })).toBeInTheDocument()
    expect(await openExportMenu()).toHaveAttribute('aria-disabled', 'true')
  })

  it('follows a running render page by page', async () => {
    mockFetch(reportDetail, { job: pdfJob('running'), pdf: null })
    const store = useIngestJobsStore.getState()
    store.appendEvent('f00d01', { event: 'report_pdf_started', data: { job_id: 'f00d01' } })
    store.appendEvent('f00d01', {
      event: 'report_pdf_progress',
      data: { job_id: 'f00d01', message: 'Laying out page 4', stage: 'layout', page: 4 }
    })
    renderReport()
    expect(await screen.findByText('Rendering the PDF… page 4')).toBeInTheDocument()
    expect(screen.getByRole('status', { name: 'Rendering the PDF… page 4' })).toBeInTheDocument()
    expect(await openExportMenu()).toHaveAttribute('aria-disabled', 'true')

    act(() => {
      store.appendEvent('f00d01', {
        event: 'report_pdf_progress',
        data: { job_id: 'f00d01', message: 'Finishing the PDF', stage: 'finishing', page: null }
      })
    })
    expect(await screen.findByText('Finishing the PDF…')).toBeInTheDocument()
  })

  it('reports a failed render, and offers to try again', async () => {
    mockFetch(reportDetail, { job: pdfJob('failed'), pdf: null })
    renderReport()
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The PDF could not be created.')
    expect(within(alert).getByRole('img', { name: 'The PDF could not be created.' })).toBeInTheDocument()
    expect(await openExportMenu()).not.toHaveAttribute('aria-disabled')
  })

  it('says nothing beside the menu while a current PDF is ready', async () => {
    mockFetch(reportDetail, { job: pdfJob('completed'), pdf: { ...RENDER, current: true } })
    renderReport()
    await openExportMenu()
    await waitFor(() => expect(pdfEntry().tagName).toBe('A'))
    for (const line of STATUS_LINES) expect(screen.queryByText(line)).not.toBeInTheDocument()
  })
})

describe('Report view — document overview', () => {
  it('shows both the items and the overview preview when a report has both', async () => {
    renderReport()
    await screen.findByText('Acme [ORG]')
    // The manifest is collapsed by default, so its rows are behind the
    // heading — what matters here is that the overview section is present.
    await userEvent.click(await screen.findByRole('button', { name: /document overview/i }))
    expect(screen.getByText('c.pdf')).toBeInTheDocument()
    expect(screen.getByText('0123456789ab')).toBeInTheDocument()
    expect(screen.queryByText(/this report is empty/i)).not.toBeInTheDocument()
  })

  it('shows items without a preview when the report has no overview snapshot yet', async () => {
    mockFetch({ ...reportDetail, collection_overview: null })
    renderReport()
    await screen.findByText('Acme [ORG]')
    expect(screen.queryByText('c.pdf')).not.toBeInTheDocument()
    expect(screen.queryByText(/this report is empty/i)).not.toBeInTheDocument()
    expect(await screen.findByRole('button', { name: /capture overview/i })).toBeInTheDocument()
  })

  it('shows the overview preview instead of the empty message when there are no items', async () => {
    mockFetch({ ...reportDetail, items: [], item_count: 0 })
    renderReport()
    await userEvent.click(await screen.findByRole('button', { name: /document overview/i }))
    expect(await screen.findByText('c.pdf')).toBeInTheDocument()
    expect(screen.queryByText(/this report is empty/i)).not.toBeInTheDocument()
  })

  it('shows the empty message when there are no items and no overview', async () => {
    mockFetch({ ...reportDetail, items: [], item_count: 0, collection_overview: null })
    renderReport()
    expect(await screen.findByText(/this report is empty/i)).toBeInTheDocument()
  })

  it('shows the empty message (not a blank area) when items are empty and the overview is toggled off', async () => {
    mockFetch({ ...reportDetail, items: [], item_count: 0, show_collection_overview: false })
    renderReport()
    expect(await screen.findByText(/this report is empty/i)).toBeInTheDocument()
    expect(screen.queryByText('c.pdf')).not.toBeInTheDocument()
  })

  it('shows the empty message when items are empty and the overview snapshot has no documents', async () => {
    mockFetch({
      ...reportDetail,
      items: [],
      item_count: 0,
      collection_overview: { ...overview, documents: [], document_count: 0 }
    })
    renderReport()
    expect(await screen.findByText(/this report is empty/i)).toBeInTheDocument()
    expect(screen.queryByText('c.pdf')).not.toBeInTheDocument()
  })

  it('reflects show_collection_overview and toggles it via PATCH', async () => {
    renderReport()
    const toggle = await screen.findByRole('checkbox', { name: 'Document overview' })
    expect(toggle).toBeChecked()
    fireEvent.click(toggle)
    await waitFor(() => {
      const patch = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find(
        (c) => (c[1] as RequestInit | undefined)?.method === 'PATCH'
      )
      expect(patch).toBeTruthy()
      expect(JSON.parse(String((patch![1] as RequestInit).body))).toMatchObject({ show_collection_overview: false })
    })
  })

  it('shows the captured date and refreshes the overview on click', async () => {
    renderReport()
    const button = await screen.findByRole('button', { name: /refresh overview \(captured 2026-07-01\)/i })
    fireEvent.click(button)
    await waitFor(() => {
      const post = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.find((c) =>
        String(c[0]).includes('/reports/1/collection-overview/refresh')
      )
      expect(post).toBeTruthy()
    })
  })

  it('persists a manual operator edit without re-applying the prefill', async () => {
    const patches: { url: string; body: unknown }[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/whoami')) {
          return { ok: true, status: 200, json: async () => ({ username: 'jane.doe', display_name: 'Jane Doe' }) }
        }
        if (url.includes('/reports/1') && init?.method === 'PATCH') {
          patches.push({ url, body: JSON.parse(String(init.body)) })
          return { ok: true, status: 200, json: async () => reportDetail }
        }
        if (url.includes('/reports/1')) return { ok: true, status: 200, json: async () => reportDetail }
        if (url.endsWith('/reports')) {
          return { ok: true, status: 200, json: async () => ({ reports: [{ ...reportDetail, items: undefined }] }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    renderReport()
    const input = await screen.findByPlaceholderText(/operator|sachbearbeit/i)
    fireEvent.change(input, { target: { value: 'Someone Else' } })
    fireEvent.blur(input)
    await waitFor(() => expect(patches).toHaveLength(1))
    expect((patches[0].body as { operator?: string }).operator).toBe('Someone Else')
  })

  it('creates a report with the signed-in display name as operator', async () => {
    const calls: { url: string; body: unknown }[] = []
    let whoamiCallCount = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/whoami')) {
          whoamiCallCount++
          return { ok: true, status: 200, json: async () => ({ username: 'jane.doe', display_name: 'Jane Doe' }) }
        }
        if (url.endsWith('/reports') && init?.method === 'POST') {
          calls.push({ url, body: JSON.parse(String(init.body)) })
          return { ok: true, status: 200, json: async () => ({ ...reportDetail, id: 2 }) }
        }
        if (url.includes('/reports/')) return { ok: true, status: 200, json: async () => reportDetail }
        if (url.endsWith('/reports')) {
          return { ok: true, status: 200, json: async () => ({ reports: [{ ...reportDetail, items: undefined }] }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    renderReport()
    // Wait for whoami to be fetched
    await waitFor(() => expect(whoamiCallCount).toBeGreaterThan(0))
    fireEvent.click(await screen.findByRole('button', { name: /New/i }))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect((calls[0].body as { operator?: string }).operator).toBe('Jane Doe')
  })

  it('omits operator when no identity is available', async () => {
    const calls: { body: unknown }[] = []
    let whoamiCallCount = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        if (url.includes('/whoami')) {
          whoamiCallCount++
          return { ok: false, status: 401, json: async () => ({}) }
        }
        if (url.endsWith('/reports') && init?.method === 'POST') {
          calls.push({ body: JSON.parse(String(init.body)) })
          return { ok: true, status: 200, json: async () => ({ ...reportDetail, id: 3 }) }
        }
        if (url.includes('/reports/')) return { ok: true, status: 200, json: async () => reportDetail }
        if (url.endsWith('/reports')) {
          return { ok: true, status: 200, json: async () => ({ reports: [{ ...reportDetail, items: undefined }] }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    renderReport()
    // Wait for whoami to fail
    await waitFor(() => expect(whoamiCallCount).toBeGreaterThan(0))
    fireEvent.click(await screen.findByRole('button', { name: /New/i }))
    await waitFor(() => expect(calls).toHaveLength(1))
    expect('operator' in (calls[0].body as Record<string, unknown>)).toBe(false)
  })
})

describe('Report view — the header selector', () => {
  /** Two reports in the list, so the selector has something to switch between. */
  function mockTwoReports() {
    const calls: Array<{ url: string; method: string; body?: string }> = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        calls.push({ url, method: init?.method ?? 'GET', body: init?.body as string | undefined })
        if (url.includes('/reports/1')) {
          return { ok: true, status: 200, json: async () => reportDetail }
        }
        if (url.includes('/reports/2')) {
          return { ok: true, status: 200, json: async () => ({ ...reportDetail, id: 2, title: 'Case Beta', item_count: 0, items: [] }) }
        }
        if (url.endsWith('/reports')) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              reports: [
                { ...reportDetail, items: undefined },
                { ...reportDetail, items: undefined, id: 2, title: 'Case Beta', item_count: 0 }
              ]
            })
          }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    return calls
  }

  it('switches the active report from the header selector', async () => {
    mockTwoReports()
    renderReport()

    // The trigger keeps role="combobox", so this query survived the move off a
    // native <select>. What changed is that a closed menu has no options in
    // the DOM at all, so the list has to be opened before one can be picked.
    const trigger = await screen.findByRole('combobox', { name: /select report/i })
    await waitFor(() => expect(trigger).toHaveTextContent('Case Alpha (2)'))
    await userEvent.click(trigger)
    await userEvent.click(await screen.findByRole('option', { name: 'Case Beta (0)' }))

    await waitFor(() => {
      expect(useReportStore.getState().activeReportId).toBe(2)
    })
  })

  it('renames the active report from the title field', async () => {
    const calls = mockTwoReports()
    renderReport()

    const title = await screen.findByDisplayValue('Case Alpha')
    fireEvent.change(title, { target: { value: 'Case Gamma' } })
    fireEvent.blur(title)

    await waitFor(() => {
      const patch = calls.find((c) => c.method === 'PATCH')
      expect(patch?.body).toContain('Case Gamma')
    })
  })

  it('deletes the active report from the header and clears the selection', async () => {
    const calls = mockTwoReports()
    vi.stubGlobal('confirm', vi.fn(() => true))
    renderReport()

    fireEvent.click(await screen.findByRole('button', { name: /delete report/i }))

    await waitFor(() => {
      expect(calls.some((c) => c.method === 'DELETE' && c.url.includes('/reports/1'))).toBe(true)
    })
    await waitFor(() => {
      expect(useReportStore.getState().activeReportId).toBeNull()
    })
  })

  it('offers only the create action when there are no reports', async () => {
    useReportStore.setState({ activeReportId: null })
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string) => {
        if (String(u).endsWith('/reports')) {
          return { ok: true, status: 200, json: async () => ({ reports: [] }) }
        }
        return { ok: true, status: 200, json: async () => ({}) }
      })
    )
    renderReport()

    // With nothing to pick, the placeholder says so rather than inviting a
    // choice; delete and export need a report and must not sit there dead.
    // The empty message lives on the closed trigger, exactly where the native
    // select's disabled placeholder option used to render it — not hidden
    // behind a click.
    const emptyTrigger = await screen.findByRole('combobox', { name: /select report/i })
    expect(emptyTrigger).toHaveTextContent(/no reports yet/i)
    expect(emptyTrigger).toHaveAttribute('aria-disabled', 'true')
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /delete report/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /New/i })).toBeInTheDocument()
  })
})

describe('Report view — frozen thumbnails', () => {
  const DATA_URI = 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='

  beforeEach(() => {
    useReportStore.setState({ activeReportId: 1 })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    useReportStore.setState({ activeReportId: null })
  })

  it('labels a keyframe as a keyframe, not as an image', async () => {
    const detail = {
      ...reportDetail,
      items: [
        {
          ...reportDetail.items[0],
          snapshot: { ...reportDetail.items[0].snapshot, thumbnail: { data_uri: DATA_URI, kind: 'video_keyframe' } }
        }
      ]
    }
    mockFetch(detail)
    renderReport()

    const img = await screen.findByRole('img', { name: /video keyframe/i })
    expect(img).toHaveAttribute('src', DATA_URI)
    expect(screen.queryByRole('img', { name: /image evidence/i })).not.toBeInTheDocument()
  })

  it('renders one image per chat source that carries a thumbnail', async () => {
    const detail = {
      ...reportDetail,
      items: [
        {
          id: 12,
          artifact_type: 'chat_answer',
          dedupe_key: 'chat:s1:0',
          position: 0,
          note: null,
          snapshot: {
            user_text: 'q',
            model_response: 'a',
            sources: [
              { filename: 'fig.png', text: 'caption', thumbnail: { data_uri: DATA_URI, kind: 'image' } },
              { filename: 'a.pdf', text: 'prose' }
            ]
          },
          created_at: null
        }
      ]
    }
    mockFetch(detail)
    renderReport()

    const imgs = await screen.findAllByRole('img', { name: /image evidence/i })
    expect(imgs).toHaveLength(1)
    expect(imgs[0]).toHaveAttribute('src', DATA_URI)
  })

  it('captions each chat figure with the number the answer cites', async () => {
    const detail = {
      ...reportDetail,
      items: [
        {
          id: 13,
          artifact_type: 'chat_answer',
          dedupe_key: 'chat:s1:0',
          position: 0,
          note: null,
          snapshot: {
            user_text: 'q',
            model_response: 'a',
            sources: [
              {
                filename: 'chart.png',
                citation_index: 1,
                thumbnail: { data_uri: DATA_URI, kind: 'image' }
              },
              {
                filename: 'photo.jpg',
                citation_index: 2,
                thumbnail: { data_uri: DATA_URI, kind: 'image' }
              }
            ]
          },
          created_at: null
        }
      ]
    }
    mockFetch(detail)
    renderReport()

    expect(await screen.findByText('[1] chart.png')).toBeInTheDocument()
    expect(screen.getByText('[2] photo.jpg')).toBeInTheDocument()
  })

  it('never renders a non-image data URI from a snapshot', async () => {
    const detail = {
      ...reportDetail,
      items: [
        {
          ...reportDetail.items[0],
          snapshot: { ...reportDetail.items[0].snapshot, thumbnail: { data_uri: 'javascript:alert(1)' } }
        }
      ]
    }
    mockFetch(detail)
    renderReport()

    await screen.findByText('Acme [ORG]')
    expect(screen.queryByRole('img', { name: /image evidence/i })).not.toBeInTheDocument()
  })
})

describe('Report view — enlarging frozen evidence', () => {
  const DATA_URI = 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='

  beforeEach(() => {
    useReportStore.setState({ activeReportId: 1 })
    useUiStore.setState({ previewModal: null })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    useReportStore.setState({ activeReportId: null })
  })

  it('enlarges a chat figure from the frozen bytes, not from the source store', async () => {
    const detail = {
      ...reportDetail,
      items: [
        {
          id: 14,
          artifact_type: 'chat_answer',
          dedupe_key: 'chat:s1:0',
          position: 0,
          note: null,
          snapshot: {
            user_text: 'q',
            model_response: 'a',
            sources: [
              { filename: 'chart.png', citation_index: 1, thumbnail: { data_uri: DATA_URI, kind: 'image' } }
            ]
          },
          created_at: null
        }
      ]
    }
    mockFetch(detail)
    renderReport()

    await userEvent.click(await screen.findByRole('img', { name: /image evidence/i }))

    expect(useUiStore.getState().previewModal).toEqual({ filename: 'chart.png', data_uri: DATA_URI })
  })

  it('enlarges a finding figure by its own filename', async () => {
    const detail = {
      ...reportDetail,
      items: [
        {
          ...reportDetail.items[0],
          snapshot: {
            ...reportDetail.items[0].snapshot,
            filename: 'folie.jpg',
            thumbnail: { data_uri: DATA_URI, kind: 'video_keyframe' }
          }
        }
      ]
    }
    mockFetch(detail)
    renderReport()

    await userEvent.click(await screen.findByRole('img', { name: /video keyframe/i }))

    expect(useUiStore.getState().previewModal).toEqual({ filename: 'folie.jpg', data_uri: DATA_URI })
  })
})

describe('Report view — image findings', () => {
  const parts = { ocr_text: 'PRINTED SLOGAN', image_description: 'A poster in a town square.', image_tags: ['poster'] }

  it('labels the printed words, description and tags of an entity finding instead of one block', async () => {
    const detail = {
      ...reportDetail,
      items: [
        {
          ...reportDetail.items[0],
          snapshot: { ...reportDetail.items[0].snapshot, chunk_text: 'PRINTED SLOGAN\n\nA poster in a town square.', ...parts }
        }
      ]
    }
    mockFetch(detail)
    renderReport()

    expect((await screen.findByText('Text in the image')).nextElementSibling).toHaveTextContent('PRINTED SLOGAN')
    expect(screen.getByText('Image description').nextElementSibling).toHaveTextContent('A poster in a town square.')
    expect(screen.getByText('Tags').nextElementSibling).toHaveTextContent('poster')
  })

  it('keeps a hate-speech finding’s reason as its body, and labels the parts only without one', async () => {
    const hate = reportDetail.items[1]
    const detail = {
      ...reportDetail,
      items: [
        { ...hate, snapshot: { ...hate.snapshot, ...parts } },
        { ...hate, id: 12, dedupe_key: 'hate:c3', position: 2, snapshot: { ...hate.snapshot, reason: '', ...parts } }
      ]
    }
    mockFetch(detail)
    renderReport()

    expect(await screen.findByText('bad')).toBeInTheDocument()
    expect(screen.getAllByText('Text in the image')).toHaveLength(1)
  })
})
