const MODEL_LABELS = {
  xg_boost: 'XGBoost',
  lstm: 'LSTM',
  cnn_lstm: 'CNN-LSTM',
  tcn: 'TCN',
}

const MODEL_COLORS = {
  xg_boost: 'text-violet-300 border-violet-400/60 bg-violet-500/10',
  lstm: 'text-amber-300 border-amber-400/60 bg-amber-500/10',
  cnn_lstm: 'text-emerald-300 border-emerald-400/60 bg-emerald-500/10',
  tcn: 'text-rose-300 border-rose-400/60 bg-rose-500/10',
}

const MODEL_COLORS_ACTIVE = {
  xg_boost: 'text-violet-100 border-violet-400 bg-violet-500/30',
  lstm: 'text-amber-100 border-amber-400 bg-amber-500/30',
  cnn_lstm: 'text-emerald-100 border-emerald-400 bg-emerald-500/30',
  tcn: 'text-rose-100 border-rose-400 bg-rose-500/30',
}

export const MODEL_STROKE = {
  xg_boost: '#a78bfa',
  lstm: '#fbbf24',
  cnn_lstm: '#34d399',
  tcn: '#fb7185',
}

export default function ModelFilter({ models, selected, onChange }) {
  if (!models || models.length === 0) return null
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-slate-400 mr-1">Model:</span>
      {models.map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={`px-3 py-1 rounded-full border text-xs font-medium transition-all ${
            selected === m ? MODEL_COLORS_ACTIVE[m] : MODEL_COLORS[m]
          }`}
        >
          {MODEL_LABELS[m] || m}
        </button>
      ))}
    </div>
  )
}

export { MODEL_LABELS }
