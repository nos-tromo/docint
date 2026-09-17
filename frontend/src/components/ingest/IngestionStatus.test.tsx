import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { IngestionStatus } from './IngestionStatus'
import type { IngestStatus } from '@/lib/ingestStatus'
import { LanguageContext } from '@/i18n/LanguageContext'

function errorStatus(overrides: Partial<IngestStatus> = {}): IngestStatus {
  return {
    phase: 'error',
    totalFiles: 1,
    filesSaved: 0,
    tasks: [],
    indexed: 0,
    totalChunks: 0,
    ...overrides,
  } as IngestStatus
}

describe('IngestionStatus error body', () => {
  it('renders the client-composed error message when present', () => {
    render(<IngestionStatus status={errorStatus({ errorMessage: 'Upload failed: every batch was rejected.' })} />)
    expect(screen.getByText('Upload failed: every batch was rejected.')).toBeInTheDocument()
  })

  it('falls back to the generic catalog copy without a message', () => {
    render(<IngestionStatus status={errorStatus()} />)
    expect(screen.getByText('Ingestion failed.')).toBeInTheDocument()
  })

  it('tags a backend job error with its validated code token', () => {
    // A backend-composed error carries errorCode, never errorMessage; the
    // catalog copy is rendered with the code appended for support triage
    // (streamErrorText), matching the chat/summary stream-error behavior.
    render(<IngestionStatus status={errorStatus({ errorCode: 'ingestion_failed' })} />)
    expect(screen.getByText('Ingestion failed. (ingestion_failed)')).toBeInTheDocument()
  })

  it('does not append a token that fails code validation', () => {
    // The code is protocol, regex-validated before display; anything not
    // matching the closed-enum token shape renders the fallback copy alone.
    render(<IngestionStatus status={errorStatus({ errorCode: 'Not A Valid Token!' })} />)
    expect(screen.getByText('Ingestion failed.')).toBeInTheDocument()
  })

  it('keeps the elapsed timer visible on a failed job', () => {
    render(
      <IngestionStatus
        status={errorStatus({ startedAt: 10_000, finishedAt: 75_000 })}
      />
    )
    expect(screen.getByText('01:05')).toBeInTheDocument()
  })
})

function completeStatus(overrides: Partial<IngestStatus> = {}): IngestStatus {
  return {
    phase: 'complete',
    totalFiles: 2,
    filesSaved: 2,
    tasks: [],
    indexed: 2,
    totalChunks: 10,
    ...overrides,
  } as IngestStatus
}

describe('IngestionStatus completion timing', () => {
  it('shows the frozen elapsed timer in the header', () => {
    render(
      <IngestionStatus
        status={completeStatus({ startedAt: 0, finishedAt: 3_725_000 })}
      />
    )
    expect(screen.getByText('1:02:05')).toBeInTheDocument()
  })

  it('renders the duration once, leaving it out of the completion summary', () => {
    // The header timer freezes in place where it was already ticking; a second
    // copy in the summary showed the same number twice in one card.
    render(
      <IngestionStatus
        status={completeStatus({ startedAt: 10_000, finishedAt: 75_000 })}
      />
    )
    expect(screen.getByText('01:05')).toBeInTheDocument()
    expect(screen.getByText('2 files indexed · 10 chunks')).toBeInTheDocument()
    expect(screen.queryByText(/Duration:/)).not.toBeInTheDocument()
  })

  it('omits the timer when nothing was measured at all', () => {
    render(<IngestionStatus status={completeStatus()} />)
    expect(screen.getByText('2 files indexed · 10 chunks')).toBeInTheDocument()
    expect(screen.queryByText('00:00')).not.toBeInTheDocument()
  })

  it('renders the server duration in preference to its own delta', () => {
    // The reported bug: the card floored its own start→finish (19.0 s) while
    // the backend log floored the run it measured (18.9 s), so one run showed
    // two durations a second apart. The server's number is the only one now.
    render(
      <IngestionStatus
        status={completeStatus({
          startedAt: 0,
          finishedAt: 19_004,
          durationMs: 18_942
        })}
      />
    )
    expect(screen.getByText('00:18')).toBeInTheDocument()
    expect(screen.queryByText('00:19')).not.toBeInTheDocument()
  })

  it('shows the timer for a reattached run that has only the server duration', () => {
    render(<IngestionStatus status={completeStatus({ durationMs: 65_000 })} />)
    expect(screen.getByText('01:05')).toBeInTheDocument()
  })
})

describe('IngestionStatus markers are drawn, never typed', () => {
  // A text character renders from whatever font the OS falls back to, and `⏱`
  // in particular carries emoji presentation on some platforms — it can arrive
  // full-colour beside otherwise monochrome chrome.
  it('marks a completed run with a drawn check and a drawn stopwatch', () => {
    const { container } = render(
      <IngestionStatus status={completeStatus({ startedAt: 0, finishedAt: 65_000 })} />
    )
    expect(container.querySelectorAll('svg').length).toBeGreaterThanOrEqual(2)
    expect(container.textContent).not.toMatch(/[✓✗⏱]/)
  })

  it('marks a failed run with a drawn cross', () => {
    const { container } = render(
      <IngestionStatus status={errorStatus({ startedAt: 0, finishedAt: 65_000 })} />
    )
    expect(container.querySelector('svg')).not.toBeNull()
    expect(container.textContent).not.toMatch(/[✓✗⏱]/)
  })
})

describe('IngestionStatus task bars', () => {
  // The label a counter carries joins to the catalog by string, and an
  // unmapped one falls back to the raw English silently. Asserting in German
  // is what separates a real mapping from that fallback — under `en` the two
  // read identically. The summary is the long tail of a run, so its bar is
  // the one an operator watches longest.
  it('renders the summary counter through the catalog, not its raw label', () => {
    render(
      <LanguageContext value="de">
        <IngestionStatus
          status={
            {
              phase: 'processing',
              totalFiles: 1,
              filesSaved: 1,
              indexed: 0,
              totalChunks: 0,
              tasks: [{ key: 'summary', label: 'Summarizing collection', current: 12, total: 412 }],
            } as IngestStatus
          }
        />
      </LanguageContext>
    )
    expect(screen.getByText('Zusammenfassung')).toBeInTheDocument()
    expect(screen.getByText('12/412')).toBeInTheDocument()
    expect(screen.queryByText('Summarizing collection')).not.toBeInTheDocument()
  })
})

describe('IngestionStatus preprocess bars', () => {
  function uploading(stages: IngestStatus['preprocess']): IngestStatus {
    return {
      phase: 'uploading',
      totalFiles: 40,
      filesSaved: 6,
      indexed: 0,
      totalChunks: 0,
      tasks: [],
      warnings: [],
      preprocess: stages,
    } as IngestStatus
  }

  // The whole point of the feature: the server starts reading a file the
  // moment it lands, and on a large batch that runs for the entire upload.
  it('shows what the server is already reading while the upload is still running', () => {
    render(
      <LanguageContext value="de">
        <IngestionStatus
          status={uploading([
            { stage: 'pdf', done: 1, total: 4, failed: 0 },
            { stage: 'ocr_pages', done: 12, total: 40, failed: 0 },
          ])}
        />
      </LanguageContext>
    )

    // Asserted in German: an unmapped label falls back to the raw stage id,
    // which under `en` would read the same as a real mapping.
    expect(screen.getByText('Gescannte Seiten werden erkannt')).toBeInTheDocument()
    expect(screen.getByText('12/40')).toBeInTheDocument()
    expect(screen.queryByText('ocr_pages')).not.toBeInTheDocument()
  })

  it('draws no bar for a stage the batch has no work for', () => {
    render(<IngestionStatus status={uploading([{ stage: 'media', done: 0, total: 0, failed: 0 }])} />)

    expect(screen.queryByText('Transcribing clips')).not.toBeInTheDocument()
  })

  it('names failures, since a stalled bar and a failing one read alike', () => {
    render(<IngestionStatus status={uploading([{ stage: 'image', done: 3, total: 10, failed: 2 }])} />)

    expect(screen.getByText(/2 failed/)).toBeInTheDocument()
  })

  it('keeps showing them once the run reaches the job', () => {
    // The pool keeps working through finalize — the job's own prefetch
    // re-submits the batch — so the bars must not vanish at the hand-over.
    render(
      <IngestionStatus
        status={
          {
            phase: 'processing',
            totalFiles: 40,
            filesSaved: 40,
            indexed: 2,
            totalChunks: 0,
            tasks: [{ key: 'embedding', label: 'Embedding', current: 1, total: 8 }],
            warnings: [],
            preprocess: [{ stage: 'media', done: 1, total: 3, failed: 0 }],
          } as IngestStatus
        }
      />
    )

    expect(screen.getByText('Transcribing clips')).toBeInTheDocument()
    expect(screen.getByText('Embedding and storing')).toBeInTheDocument()
  })
})
