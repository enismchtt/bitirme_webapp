import { useEffect, useRef, useState } from 'react'
import { ChevronDown, Search } from 'lucide-react'

const COIN_COLORS = {
  BTC: '#f7931a', ETH: '#627eea', BNB: '#f3ba2f', XRP: '#23292f',
  ADA: '#0033ad', DOGE: '#c2a633', MATIC: '#8247e5', LTC: '#345d9d',
  LINK: '#2a5ada', ATOM: '#2e3148', XLM: '#7d00ff', ETC: '#3ab14a',
  XMR: '#ff6600', ALGO: '#000000', VET: '#15bdff', TRX: '#ff060a',
  EOS: '#000000', NEO: '#58bf00', IOTA: '#222', CHZ: '#cd1041',
}

export default function CoinSelector({ coins, value, onChange }) {
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const ref = useRef(null)

  useEffect(() => {
    const onClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  const filtered = coins.filter((c) => c.toLowerCase().includes(filter.toLowerCase()))
  const color = COIN_COLORS[value] || '#7c3aed'

  return (
    <div className={`relative ${open ? 'z-50' : 'z-0'}`} ref={ref}>
      <span className="label block mb-1.5">Coin</span>
      {/* button below */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="input w-full flex items-center justify-between gap-2"
      >
        <span className="flex items-center gap-2">
          <span
            className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
            style={{ background: color }}
          >
            {value?.slice(0, 1)}
          </span>
          <span className="font-semibold">{value}</span>
          <span className="text-slate-400 text-xs">/USDT</span>
        </span>
        <ChevronDown size={16} className={`text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="absolute z-50 mt-2 w-full glass rounded-xl overflow-hidden shadow-xl">
          <div className="p-2 border-b border-white/5 flex items-center gap-2">
            <Search size={14} className="text-slate-400" />
            <input
              autoFocus
              placeholder="Search coin..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="bg-transparent text-sm outline-none flex-1 text-slate-100 placeholder-slate-500"
            />
          </div>
          <div className="max-h-72 overflow-auto py-1">
            {filtered.map((c) => (
              <button
                key={c}
                onClick={() => {
                  onChange(c)
                  setOpen(false)
                  setFilter('')
                }}
                className={`w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-white/5 transition-colors ${
                  c === value ? 'bg-white/5 text-white' : 'text-slate-300'
                }`}
              >
                <span
                  className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
                  style={{ background: COIN_COLORS[c] || '#7c3aed' }}
                >
                  {c.slice(0, 1)}
                </span>
                <span className="font-medium">{c}</span>
                <span className="text-slate-500 text-xs">/USDT</span>
              </button>
            ))}
            {filtered.length === 0 && (
              <div className="px-3 py-4 text-center text-sm text-slate-500">No matches</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export { COIN_COLORS }
