import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { EntityMergeMode } from '@/api/types'

export interface PreviewModal {
  collection: string
  file_hash: string
  filename: string
}

interface UiState {
  selectedCollection: string | null
  currentSessionId: string | null
  previewModal: PreviewModal | null
  entityMergeMode: EntityMergeMode
  graphTopK: number | null
  setSelectedCollection: (name: string | null) => void
  setCurrentSessionId: (id: string | null) => void
  setEntityMergeMode: (mode: EntityMergeMode) => void
  setGraphTopK: (n: number | null) => void
  openPreview: (modal: PreviewModal) => void
  closePreview: () => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      selectedCollection: null,
      currentSessionId: null,
      previewModal: null,
      entityMergeMode: 'resolved',
      graphTopK: null,
      setSelectedCollection: (name) => set({ selectedCollection: name }),
      setCurrentSessionId: (id) => set({ currentSessionId: id }),
      setEntityMergeMode: (mode) => set({ entityMergeMode: mode }),
      setGraphTopK: (n) => set({ graphTopK: n }),
      openPreview: (modal) => set({ previewModal: modal }),
      closePreview: () => set({ previewModal: null })
    }),
    {
      name: 'docint-ui',
      // Active collection is intentionally not persisted: the backend
      // singleton resets to "no active collection" on every restart, so
      // carrying a stale UI selection across reloads only produces a
      // chain of 400s until the user re-picks. Each session starts fresh
      // and the user must deliberately lock in a collection.
      partialize: (s) => ({
        currentSessionId: s.currentSessionId,
        entityMergeMode: s.entityMergeMode,
        graphTopK: s.graphTopK
      }),
      // v0 builds saved `selectedCollection` to localStorage. Strip it on
      // rehydrate so existing users do not inherit a stale selection that
      // disagrees with the freshly-started backend singleton.
      version: 1,
      migrate: (persisted) => {
        const prior = (persisted ?? {}) as { currentSessionId?: string | null }
        return { currentSessionId: prior.currentSessionId ?? null }
      }
    }
  )
)
