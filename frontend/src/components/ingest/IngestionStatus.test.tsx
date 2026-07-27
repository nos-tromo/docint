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
})
