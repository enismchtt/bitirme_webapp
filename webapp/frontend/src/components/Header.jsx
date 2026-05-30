import { TrendingUp, History, Sparkles } from 'lucide-react'

export default function Header({ tab, onTabChange }) {
  const tabs = [
    { id: 'historical', label: 'Historical Analysis', icon: History },
    { id: 'forecast', label: 'Future Forecast', icon: Sparkles },
  ]
  return (
    <header className="sticky top-0 z-30 backdrop-blur-xl bg-bg-900/70 border-b border-white/5">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-400 to-cyan-glow flex items-center justify-center shadow-lg shadow-accent-500/30">
            <TrendingUp size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">Crypto Prediction</h1>
            <p className="text-xs text-slate-400 -mt-0.5">XGBoost · 7-day lags · live Binance data</p>
          </div>
        </div>
        <nav className="flex items-center gap-1 p-1 rounded-xl bg-bg-800/60 border border-white/5">
          {tabs.map((t) => {
            const Icon = t.icon
            const active = tab === t.id
            return (
              <button
                key={t.id}
                onClick={() => onTabChange(t.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  active
                    ? 'bg-gradient-to-r from-accent-500 to-cyan-glow text-white shadow-md shadow-accent-500/30'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Icon size={16} />
                <span>{t.label}</span>
              </button>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
