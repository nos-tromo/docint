import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export interface EntityRow {
  text: string
  type: string
  count: number
}

export function TopEntitiesChart({ data }: { data: EntityRow[] }) {
  if (!data.length) {
    return <div className="text-sm text-muted-foreground">No entities yet.</div>
  }
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} layout="vertical" margin={{ left: 24, right: 16 }}>
        <XAxis type="number" stroke="rgb(161,161,170)" />
        <YAxis type="category" dataKey="text" stroke="rgb(161,161,170)" width={140} />
        <Tooltip
          contentStyle={{
            background: 'rgb(24 24 27)',
            border: '1px solid rgb(39 39 42)',
            borderRadius: 6
          }}
        />
        <Bar dataKey="count" fill="rgb(244 244 245)" />
      </BarChart>
    </ResponsiveContainer>
  )
}
