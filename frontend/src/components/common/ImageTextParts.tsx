import type { ReactNode } from 'react'
import { ClampedText } from '@/components/common/ClampedText'
import type { ImageParts } from '@/lib/reportSnapshots'
import { useT } from '@/i18n/LanguageContext'

interface Props {
  parts: ImageParts
  /** The printed words' translation, shown in their place when set. */
  ocrTranslation?: string | null
  /** Renders one part's text, e.g. with entity mentions highlighted. */
  renderText?: (text: string) => ReactNode
}

const LABEL_CLASS = 'mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground'

/**
 * An image finding's text, one labelled part each: the words printed in the
 * picture, then docint's description of it, then its tags. Run together they
 * read as one passage, and nothing says which words the picture itself says.
 */
export function ImageTextParts({ parts, ocrTranslation, renderText = (text) => text }: Props) {
  const t = useT()
  const printed = (parts.ocr_text ?? '').trim()
  const description = (parts.image_description ?? '').trim()
  const tags = (parts.image_tags ?? [])
    .map((tag) => tag.trim())
    .filter(Boolean)
    .join(', ')
  const translated = ocrTranslation != null
  return (
    <dl className="space-y-2">
      {printed && (
        <div>
          <dt className={LABEL_CLASS}>
            {translated ? `${t('common.image_text')} · ${t('common.translation')}` : t('common.image_text')}
          </dt>
          <dd>
            <ClampedText length={(ocrTranslation ?? printed).length}>
              {translated ? ocrTranslation : renderText(printed)}
            </ClampedText>
          </dd>
        </div>
      )}
      {description && (
        <div>
          <dt className={LABEL_CLASS}>{t('common.image_description')}</dt>
          <dd>
            <ClampedText length={description.length}>{renderText(description)}</ClampedText>
          </dd>
        </div>
      )}
      {tags && (
        <div>
          <dt className={LABEL_CLASS}>{t('common.image_tags')}</dt>
          <dd className="text-xs text-muted-foreground break-words">{renderText(tags)}</dd>
        </div>
      )}
    </dl>
  )
}
