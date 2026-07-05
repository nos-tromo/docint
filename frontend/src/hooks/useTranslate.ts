import { useMutation } from '@tanstack/react-query'
import { translate } from '@/api/translate'

/** On-demand snippet translation (fail-soft: the endpoint returns ok:false, never throws server-side). */
export function useTranslate() {
  return useMutation({ mutationFn: (text: string) => translate(text) })
}
