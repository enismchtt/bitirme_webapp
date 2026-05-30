import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts'

function fmtNum(v) {
  if (v == null || Number.isNaN(v)) return '—'
  if (v >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 2 })
  if (v >= 1) return v.toFixed(4)
  return v.toFixed(6)
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null
  return (
    <div className="glass rounded-lg p-3 text-xs min-w-[180px]">
      <div className="text-slate-400 mb-2 font-medium">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-4 py-0.5">
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
            <span className="text-slate-300">{p.name}</span>
          </span>
          <span className="font-mono font-semibold text-slate-100">
            {fmtNum(p.value)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function PriceChart({
  data,
  showActual = true,
  showAutoregressive = false,
  forecastStartDate = null,
  predictionColor = '#a78bfa',
  predictionLabel = '1-step prediction',
  arColor = '#fbbf24',
  height = 360,
}) {
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="predFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={predictionColor} stopOpacity={0.35} />
              <stop offset="100%" stopColor={predictionColor} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 6" />
          <XAxis dataKey="date" tickMargin={8} />
          <YAxis
            domain={['auto', 'auto']}
            tickFormatter={(v) =>
              v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v >= 1 ? v.toFixed(2) : v.toFixed(4)
            }
            tickMargin={6}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ paddingTop: 8, fontSize: 12 }} iconType="circle" />
          {forecastStartDate && (
            <ReferenceLine
              x={forecastStartDate}
              stroke="#475569"
              strokeDasharray="4 4"
              label={{ value: 'Forecast starts', fill: '#94a3b8', fontSize: 11, position: 'top' }}
            />
          )}
          {showActual && (
            <Area
              type="monotone"
              dataKey="actual_close"
              name="Actual"
              stroke="#22d3ee"
              strokeWidth={2}
              fill="url(#actualFill)"
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls
            />
          )}
          <Area
            type="monotone"
            dataKey="predicted_close"
            name={predictionLabel}
            stroke={predictionColor}
            strokeWidth={2}
            fill={showActual ? 'transparent' : 'url(#predFill)'}
            dot={showActual ? false : { r: 3, fill: predictionColor }}
            activeDot={{ r: 4 }}
            connectNulls
          />
          {showAutoregressive && (
            <Line
              type="monotone"
              dataKey="predicted_close_ar"
              name="Autoregressive"
              stroke={arColor}
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
