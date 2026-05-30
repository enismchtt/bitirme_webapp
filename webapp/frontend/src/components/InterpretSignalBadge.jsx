const SIGNAL_STYLES = {
  BUY: {
    label: 'Buy',
    ring: 'ring-emerald-500/40',
    bg: 'bg-emerald-500/15',
    border: 'border-emerald-400/50',
    text: 'text-emerald-300',
    glow: 'shadow-emerald-500/25',
  },
  SELL: {
    label: 'Sell',
    ring: 'ring-rose-500/40',
    bg: 'bg-rose-500/15',
    border: 'border-rose-400/50',
    text: 'text-rose-300',
    glow: 'shadow-rose-500/25',
  },
  HOLD: {
    label: 'Hold',
    ring: 'ring-amber-500/40',
    bg: 'bg-amber-500/15',
    border: 'border-amber-400/50',
    text: 'text-amber-200',
    glow: 'shadow-amber-500/20',
  },
}

export function normalizeSignal(raw) {
  const s = String(raw || '').toUpperCase().trim()
  if (s === 'BUY' || s === 'SELL' || s === 'HOLD') return s
  return 'HOLD'
}

export default function InterpretSignalBadge({ signal, horizonDays }) {
  const key = normalizeSignal(signal)
  const style = SIGNAL_STYLES[key]

  return (
    <div
      className={`flex flex-col items-center justify-center rounded-2xl border-2 px-8 py-6 mb-5 shadow-lg ${style.bg} ${style.border} ${style.ring} ring-2 ${style.glow}`}
      role="status"
      aria-label={`Recommendation: ${key}`}
    >
      <span className="text-xs uppercase tracking-[0.2em] text-slate-400 mb-2">
        Recommendation{horizonDays ? ` · ${horizonDays} day horizon` : ''}
      </span>
      <span className={`text-5xl md:text-6xl font-black tracking-tight ${style.text}`}>
        {key}
      </span>
      <span className={`text-sm font-medium mt-2 ${style.text} opacity-80`}>
        {style.label} — based on model forecasts
      </span>
    </div>
  )
}
