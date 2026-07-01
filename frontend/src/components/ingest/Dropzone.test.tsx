import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { Dropzone } from './Dropzone'

describe('Dropzone folder picker', () => {
  it('exposes a folder input with webkitdirectory and forwards picked files', () => {
    const onFiles = vi.fn()
    render(<Dropzone onFiles={onFiles} />)

    expect(screen.getByRole('button', { name: /choose a folder/i })).toBeInTheDocument()

    const folderInput = Array.from(
      document.querySelectorAll('input[type="file"]')
    ).find((el) => el.hasAttribute('webkitdirectory')) as HTMLInputElement
    expect(folderInput).toBeTruthy()

    const f = new File([new Uint8Array([1])], 'a.jpg', { type: 'image/jpeg' })
    Object.defineProperty(folderInput, 'files', { value: [f] })
    fireEvent.change(folderInput)
    expect(onFiles).toHaveBeenCalledWith([f])
  })
})
