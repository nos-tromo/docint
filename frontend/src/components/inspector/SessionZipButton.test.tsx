import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { clearAllPurposes, setPurpose } from '@infra/ui'
import { PURPOSE_SCOPE, setOwnerParam } from '@/api/client'
import { useUiStore } from '@/stores/ui'
import { captureDownloads, fileResponse } from '@/test/downloads'
import { SessionZipButton } from './SessionZipButton'

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 's1' })
})

afterEach(() => {
  clearAllPurposes()
  sessionStorage.clear()
  setOwnerParam(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('SessionZipButton', () => {
  it('renders nothing without an open session', () => {
    useUiStore.setState({ currentSessionId: null })
    const { container } = render(<SessionZipButton />)
    expect(container).toBeEmptyDOMElement()
  })

  it("downloads the session's sources with the stated purpose, in the owner's namespace", async () => {
    const downloads = captureDownloads()
    const fetchMock = vi.fn(async (_input: unknown, _init?: { headers?: Record<string, string> }) =>
      fileResponse('PK', { type: 'application/zip', filename: 'session-s1-sources.zip' })
    )
    vi.stubGlobal('fetch', fetchMock)
    setPurpose(PURPOSE_SCOPE, { reason: 'Review' })
    // An admin working in another user's collection: the archive is theirs.
    setOwnerParam('jane.doe')
    render(<SessionZipButton />)

    await userEvent.click(screen.getByRole('button', { name: 'Download session sources (ZIP)' }))

    await waitFor(() => expect(downloads.saved()).toHaveLength(1))
    expect(String(fetchMock.mock.calls[0][0])).toBe('/sessions/s1/sources.zip?owner=jane.doe')
    expect(fetchMock.mock.calls[0][1]?.headers?.['X-Audit-Purpose']).toBe('reason=Review')
    expect(downloads.saved()[0].download).toBe('session-s1-sources.zip')
  })
})
