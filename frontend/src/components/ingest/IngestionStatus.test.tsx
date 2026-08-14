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

  it('includes the duration in the completion summary', () => {
    render(
      <IngestionStatus
        status={completeStatus({ startedAt: 10_000, finishedAt: 75_000 })}
      />
    )
    expect(
      screen.getByText(/2 files indexed · 10 chunks · Duration: 01:05/)
    ).toBeInTheDocument()
  })

  it('omits the summary duration when no start time is known', () => {
    render(<IngestionStatus status={completeStatus()} />)
    expect(screen.getByText('2 files indexed · 10 chunks')).toBeInTheDocument()
    expect(screen.queryByText(/Duration:/)).not.toBeInTheDocument()
  })
})
