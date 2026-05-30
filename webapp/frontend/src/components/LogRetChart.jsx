import {
  ResponsiveContainer, ComposedChart, Bar, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, Cell, ReferenceLine,
} from 'recharts'

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="glass rounded-lg p-3 text-xs min-w-[200px]">
      <div className="text-slate-400 mb-2 font-medium">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-4 py-0.5">
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
            <span className="text-slate-300">{p.name}</span>
          </span>
          <span className="font-mono font-semibold text-slate-100">
            {(p.value * 100).toFixed(3)}%
          </span>
        </div>
      ))}
    </div>
  )
}

export default function LogRetChart({ data, height = 280 }) {
  // Build series: actual log_ret as colored bars, predicted as a line.
  const chartData = data.map((p) => ({
    date: p.date,
    actual: p.actual_log_ret,
    predicted: p.predicted_log_ret,
  }))
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <ComposedChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 6" />
          <XAxis dataKey="date" tickMargin={8} />
          <YAxis
            tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
            tickMargin={6}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ paddingTop: 8, fontSize: 12 }} iconType="circle" />
          <ReferenceLine y={0} stroke="#475569" strokeDasharray="2 3" />
          <Bar dataKey="actual" name="Actual log return" radius={[2, 2, 0, 0]}>
            {chartData.map((p, i) => (
              <Cell key={i} fill={p.actual >= 0 ? '#10b981' : '#f43f5e'} fillOpacity={0.55} />
            ))}
          </Bar>
          <Line
            type="monotone"
            dataKey="predicted"
            name="Predicted log return"
            stroke="#a78bfa"
            strokeWidth={2.5}
            dot={{ r: 3, fill: '#a78bfa' }}
            activeDot={{ r: 5 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
