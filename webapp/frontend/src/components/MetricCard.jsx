export default function MetricCard({ label, value, sub, accent = 'accent', icon: Icon }) {
  const accents = {
    accent: 'from-accent-400 to-accent-600',
    cyan: 'from-cyan-glow to-accent-400',
    green: 'from-emerald-400 to-cyan-glow',
    red: 'from-rose-400 to-orange-400',
  }
  return (
    <div className="card relative overflow-hidden">
      <div
        className={`absolute -top-12 -right-12 w-32 h-32 rounded-full bg-gradient-to-br ${accents[accent]} opacity-10 blur-2xl`}
      />
      <div className="flex items-start justify-between">
        <span className="label">{label}</span>
        {Icon && <Icon size={16} className="text-slate-400" />}
      </div>
      <div className="mt-2 text-2xl font-bold tracking-tight">{value}</div>
      {sub && <div className="text-xs text-slate-400 mt-1">{sub}</div>}
    </div>
  )
}
