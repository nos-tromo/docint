import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { IngestionStatus } from './IngestionStatus'
import type { IngestStatus } from '@/lib/ingestStatus'

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
