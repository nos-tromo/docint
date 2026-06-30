import { useRef, useState, type DragEvent } from 'react'
import { cn } from '@/lib/cn'

export function Dropzone({
  onFiles,
  disabled
}: {
  onFiles: (files: File[]) => void
  disabled?: boolean
}) {
  const [hover, setHover] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)

  const handle = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setHover(false)
    if (disabled) return
    const list = Array.from(e.dataTransfer.files)
    if (list.length) onFiles(list)
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setHover(true)
      }}
      onDragLeave={() => setHover(false)}
      onDrop={handle}
      onClick={() => inputRef.current?.click()}
      className={cn(
        'rounded-lg border-2 border-dashed p-10 text-center cursor-pointer',
        hover ? 'border-foreground bg-zinc-900' : 'border-border',
        disabled && 'opacity-50 pointer-events-none'
      )}
    >
      <p>Drop files here or click to choose.</p>
      <button
        type="button"
        className="mt-3 underline"
        onClick={(e) => {
          e.stopPropagation()
          folderInputRef.current?.click()
        }}
      >
        Or choose a folder
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          const list = Array.from(e.target.files ?? [])
          if (list.length) onFiles(list)
          e.target.value = ''
        }}
      />
      <input
        ref={folderInputRef}
        type="file"
        multiple
        className="hidden"
        {...({ webkitdirectory: '', directory: '' } as Record<string, string>)}
        onChange={(e) => {
          const list = Array.from(e.target.files ?? [])
          if (list.length) onFiles(list)
          e.target.value = ''
        }}
      />
    </div>
  )
}
