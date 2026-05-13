import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface PreviewModal {
  collection: string
  file_hash: string
  filename: string
}

interface UiState {
  selectedCollection: string | null
  currentSessionId: string | null
  previewModal: PreviewModal | null
  setSelectedCollection: (name: string | null) => void
  setCurrentSessionId: (id: string | null) => void
  openPreview: (modal: PreviewModal) => void
  closePreview: () => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      selectedCollection: null,
      currentSessionId: null,
      previewModal: null,
      setSelectedCollection: (name) => set({ selectedCollection: name }),
      setCurrentSessionId: (id) => set({ currentSessionId: id }),
      openPreview: (modal) => set({ previewModal: modal }),
      closePreview: () => set({ previewModal: null })
    }),
    {
      name: 'docint-ui',
      partialize: (s) => ({
        selectedCollection: s.selectedCollection,
        currentSessionId: s.currentSessionId
      })
    }
  )
)
