import type { Strings } from '@/i18n'

// The hate-speech `category` value is protocol (the fixed enum in
// docint/utils/prompts/{en,de}/hate_speech.txt): never translate the raw
// value itself, only its display label — an unrecognized value (e.g. a
// future category) falls back to showing the raw string as-is. Shared
// between HateSpeechTable (the live findings table) and Report.tsx (which
// renders the same frozen `category` snapshot value in an item's title), so
// the same category always displays the same label in both places.
const CATEGORY_LABEL_KEY: Partial<Record<string, keyof Strings>> = {
  race: 'hate.category_race',
  ethnicity: 'hate.category_ethnicity',
  religion: 'hate.category_religion',
  gender: 'hate.category_gender',
  sexual_orientation: 'hate.category_sexual_orientation',
  disability: 'hate.category_disability',
  nationality: 'hate.category_nationality',
  extremism: 'hate.category_extremism',
  other: 'hate.category_other',
  none: 'hate.category_none',
  unknown: 'hate.category_unknown'
}

export function hateCategoryLabel(raw: string, t: (key: keyof Strings) => string): string {
  const key = CATEGORY_LABEL_KEY[raw.toLowerCase()]
  return key ? t(key) : raw
}
