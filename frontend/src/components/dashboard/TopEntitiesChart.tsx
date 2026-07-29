import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { NerEntityRow } from '@/api/types'
import { useT } from '@/i18n/LanguageContext'

export function TopEntitiesChart({ data }: { data: NerEntityRow[] }) {
  const t = useT()
  if (!data.length) {
    return <div className="text-sm text-muted-foreground">{t('dashboard.chart_empty')}</div>
  }
  // Recharts' ResponsiveContainer measures the parent via ResizeObserver
  // and collapses to 0×0 inside flex columns until layout settles. The
  // canonical fix is to wrap it in a fixed-dimension <div> and let the
  // container fill it with 100% / 100%.
  const height = Math.max(240, data.length * 22 + 32)
  // Theme-reactive: these read the semantic tokens from @infra/ui's
  // theme.css (light default, dark override via [data-theme='dark'] /
  // prefers-color-scheme) so the chart matches the app in both themes.
  const axisStroke = 'var(--color-muted-foreground)'
  const tickStyle = { fill: 'var(--color-muted-foreground)', fontSize: 10 }
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 16, right: 16, top: 4, bottom: 4 }}>
          <XAxis type="number" stroke={axisStroke} tick={tickStyle} allowDecimals={false} />
          <YAxis
            type="category"
            dataKey="text"
            stroke={axisStroke}
            tick={tickStyle}
            width={140}
            interval={0}
          />
          <Tooltip
            cursor={{ fill: 'var(--color-muted)' }}
            contentStyle={{
              background: 'var(--color-background)',
              border: '1px solid var(--color-border)',
              borderRadius: 6,
              fontSize: 11
            }}
            labelStyle={{ color: 'var(--color-foreground)' }}
          />
          <Bar dataKey="mentions" name={t('dashboard.chart_mentions')} fill="var(--color-primary)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
