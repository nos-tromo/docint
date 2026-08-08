import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * Return the draft key for a chat.
 *
 * Drafts are keyed per session so switching sessions does not drag a
 * half-typed question along; an unstarted chat has no id yet and uses `'new'`.
 *
 * @param sessionId - The open session id, or null for an unstarted chat.
 * @returns The key under which this chat's draft is stored.
 */
export const draftKey = (sessionId: string | null): string => sessionId ?? 'new'

/** Unsent chat input, keyed by session. Persisted so a reload keeps it. */
export interface ChatUiState {
  drafts: Record<string, string>
  /** Whether the search/filters column beside the transcript is expanded. */
  sidePanelOpen: boolean
  setDraft: (key: string, value: string) => void
  clearDraft: (key: string) => void
  setSidePanelOpen: (open: boolean) => void
  toggleSidePanel: () => void
}

export const useChatUiStore = create<ChatUiState>()(
  persist(
    (set) => ({
      drafts: {},
      sidePanelOpen: true,
      setDraft: (key, value) => set((s) => ({ drafts: { ...s.drafts, [key]: value } })),
      clearDraft: (key) =>
        set((s) => {
          if (!(key in s.drafts)) return s // stable reference -> no needless re-render
          const drafts = { ...s.drafts }
          delete drafts[key]
          return { drafts }
        }),
      setSidePanelOpen: (sidePanelOpen) => set({ sidePanelOpen }),
      toggleSidePanel: () => set((s) => ({ sidePanelOpen: !s.sidePanelOpen }))
    }),
    {
      name: 'docint-chat-ui',
      version: 2,
      // v1 had no panel flag; default it open so an upgrade does not silently
      // hide a column the user never chose to hide.
      migrate: (persisted) => {
        const prior = (persisted ?? {}) as Partial<ChatUiState>
        return { drafts: prior.drafts ?? {}, sidePanelOpen: prior.sidePanelOpen ?? true }
      }
    }
  )
)
