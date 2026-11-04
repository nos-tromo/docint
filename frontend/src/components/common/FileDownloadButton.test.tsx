import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ApiError } from '@/api/client'
import { FileDownloadButton } from './FileDownloadButton'

afterEach(() => vi.restoreAllMocks())

describe('FileDownloadButton', () => {
  it('runs its download on a click, busy and closed to a second click meanwhile', async () => {
    let finish!: () => void
    const download = vi.fn(() => new Promise<void>((resolve) => (finish = resolve)))
    render(<FileDownloadButton label="Export CSV" download={download} />)
    const button = screen.getByRole('button', { name: 'Export CSV' })

    await userEvent.click(button)

    expect(download).toHaveBeenCalledTimes(1)
    expect(button).toHaveAttribute('aria-busy', 'true')
    expect(button).toBeDisabled()
    finish()
    await waitFor(() => expect(button).not.toBeDisabled())
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })

  it('says so when the download fails, and can be tried again', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const download = vi.fn().mockRejectedValueOnce(new ApiError(500, 'boom')).mockResolvedValueOnce(undefined)
    render(<FileDownloadButton label="Export CSV" download={download} />)

    await userEvent.click(screen.getByRole('button', { name: 'Export CSV' }))

    expect(await screen.findByRole('status')).toHaveTextContent('The download failed. Try again.')
    // The name still says what the control is; the tooltip says what happened.
    const button = screen.getByRole('button', { name: 'Export CSV' })
    expect(button).toHaveAttribute('title', 'The download failed. Try again.')
    await userEvent.click(button)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())
    expect(download).toHaveBeenCalledTimes(2)
  })

  it('is no failure when the user declined to state a purpose', async () => {
    const download = vi.fn().mockRejectedValue(new ApiError(428, 'A stated purpose is required for this request.'))
    render(<FileDownloadButton label="Export CSV" download={download} />)

    await userEvent.click(screen.getByRole('button', { name: 'Export CSV' }))

    await waitFor(() => expect(screen.getByRole('button', { name: 'Export CSV' })).not.toBeDisabled())
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Export CSV' })).toHaveAttribute('title', 'Export CSV')
  })
})
