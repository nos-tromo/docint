import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { NerEntityRow } from '@/api/types'

export function TopEntitiesChart({ data }: { data: NerEntityRow[] }) {
  if (!data.length) {
    return <div className="text-sm text-muted-foreground">No entities yet.</div>
  }
  // Recharts' ResponsiveContainer measures the parent via ResizeObserver
  // and collapses to 0×0 inside flex columns until layout settles. The
  // canonical fix is to wrap it in a fixed-dimension <div> and let the
  // container fill it with 100% / 100%.
  const height = Math.max(240, data.length * 22 + 32)
  const tickStyle = { fill: 'rgb(161,161,170)', fontSize: 10 }
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 16, right: 16, top: 4, bottom: 4 }}>
          <XAxis type="number" stroke="rgb(161,161,170)" tick={tickStyle} allowDecimals={false} />
          <YAxis
            type="category"
            dataKey="text"
            stroke="rgb(161,161,170)"
            tick={tickStyle}
            width={140}
            interval={0}
          />
          <Tooltip
            cursor={{ fill: 'rgba(82, 82, 91, 0.25)' }}
            contentStyle={{
              background: 'rgb(24 24 27)',
              border: '1px solid rgb(39 39 42)',
              borderRadius: 6,
              fontSize: 11
            }}
            labelStyle={{ color: 'rgb(244 244 245)' }}
          />
          <Bar dataKey="mentions" name="Mentions" fill="rgb(244 244 245)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
