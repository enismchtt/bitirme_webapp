import { useEffect, useMemo, useState } from 'react'
import { Activity, Target, Percent, TrendingUp, Play, AlertCircle } from 'lucide-react'
import CoinSelector from './CoinSelector.jsx'
import DateRangePicker from './DateRangePicker.jsx'
import MetricCard from './MetricCard.jsx'
import PriceChart from './PriceChart.jsx'
import LogRetChart from './LogRetChart.jsx'
import { getCoinInfo, getHistorical } from '../api.js'

function shiftDays(dateStr, n) {
  const d = new Date(dateStr)
  d.setDate(d.getDate() + n)
  return d.toISOString().slice(0, 10)
}

export default function HistoricalView({ coins, coin, onCoinChange }) {
  const [coinInfo, setCoinInfo] = useState(null)
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Load coin date bounds whenever coin changes.
  useEffect(() => {
    let cancelled = false
    setCoinInfo(null)
    setData(null)
    setError(null)
    getCoinInfo(coin)
      .then((info) => {
        if (cancelled) return
        setCoinInfo(info)
        const latest = info.latest_date
        const defaultStart = shiftDays(latest, -30)
        const defaultEnd = shiftDays(latest, -1)
        setStart(defaultStart)
        setEnd(defaultEnd)
      })
      .catch((e) => {
        if (cancelled) return
        setError(e?.response?.data?.detail || e.message)
      })
    return () => { cancelled = true }
  }, [coin])

  const minStartDate = useMemo(() => {
    if (!coinInfo) return undefined
    // Need >= 7+30 = 37 days of training history.
    return shiftDays(coinInfo.earliest_date, 40)
  }, [coinInfo])

  async function run() {
    if (!start || !end) return
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const res = await getHistorical({ coin, start, end })
      setData(res)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const days = data?.points?.length || 0

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="grid md:grid-cols-[1fr_2fr_auto] gap-4 items-end">
          <CoinSelector coins={coins} value={coin} onChange={onCoinChange} />
          <DateRangePicker
            start={start}
            end={end}
            onStartChange={setStart}
            onEndChange={setEnd}
            minDate={minStartDate}
            maxDate={coinInfo?.latest_date}
          />
          <button
            onClick={run}
            disabled={loading || !start || !end}
            className="btn btn-primary h-[42px] md:w-40"
          >
            <Play size={16} />
            {loading ? 'Running...' : 'Compare'}
          </button>
        </div>
        {coinInfo && (
          <div className="mt-4 flex flex-wrap gap-2 text-xs">
            <span className="chip">Data range: {coinInfo.earliest_date} → {coinInfo.latest_date}</span>
            <span className="chip">Latest close: ${coinInfo.latest_close.toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
            <span className="chip">Model: XGBoost · lags=7</span>
            {data?.features && (
              <span className="chip text-cyan-glow">
                Features: {data.features.join(' + ')}
              </span>
            )}
          </div>
        )}
        <p className="mt-3 text-xs text-slate-500 leading-relaxed">
          For each day in the selected window the model predicts using the{' '}
          <span className="text-slate-300 font-medium">previous 7 actual days</span> of log-returns.
          The model is trained strictly on data before your start date, so the window is fully
          unseen by the model.
        </p>
      </div>

      {error && (
        <div className="card flex items-start gap-3 border-rose-500/30 bg-rose-500/5">
          <AlertCircle size={18} className="text-rose-400 mt-0.5" />
          <div>
            <div className="font-medium text-rose-300">Error</div>
            <div className="text-sm text-slate-300">{error}</div>
          </div>
        </div>
      )}

      {data && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Days"
              value={days}
              accent="cyan"
              icon={Activity}
            />
            <MetricCard
              label="Direction accuracy"
              value={`${(data.direction_accuracy * 100).toFixed(1)}%`}
              sub={`Over ${days} days`}
              accent={data.direction_accuracy >= 0.5 ? 'green' : 'red'}
              icon={Target}
            />
            <MetricCard
              label="Price MAPE"
              value={Number.isFinite(data.mape) ? `${data.mape.toFixed(2)}%` : '—'}
              sub="Mean absolute % error"
              accent="accent"
              icon={Percent}
            />
            <MetricCard
              label="RMSE (log return)"
              value={Number.isFinite(data.rmse_log_ret) ? data.rmse_log_ret.toFixed(5) : '—'}
              sub={`Price RMSE: ${Number.isFinite(data.rmse_price) ? '$' + data.rmse_price.toFixed(2) : '—'}`}
              accent="cyan"
              icon={TrendingUp}
            />
          </div>

          <div className="card">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-semibold">Actual vs. Predicted · {coin}/USDT</h3>
              <div className="text-xs text-slate-400">
                {start} → {end}
              </div>
            </div>
            <p className="text-xs text-slate-500 mb-3">
              <span className="text-purple-300 font-medium">Purple (1-step-ahead prediction):</span>{' '}
              each day uses the previous <strong>7 actual days</strong> as input. Highly accurate,
              but because daily log-returns are small the line stays close to the actual price and
              can look "lagged" by one day.{' '}
              <span className="text-amber-300 font-medium">Yellow dashed (autoregressive forecast):</span>{' '}
              only sees actual data at the start, then feeds its own predictions back as input —
              looks like a "pure forecast" but can drift from reality over time.
            </p>
            <PriceChart data={data.points} showActual showAutoregressive />
          </div>

          <div className="card">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-semibold">Daily log return · Actual vs. Predicted</h3>
              <div className="text-xs text-slate-400">
                Direction accuracy: <span className="text-slate-200 font-semibold">{(data.direction_accuracy * 100).toFixed(1)}%</span>
              </div>
            </div>
            <p className="text-xs text-slate-500 mb-3">
              Green/red bars show the <em>actual</em> daily log return
              (green = up, red = down). The purple line is what the model predicted.
              When the bar and line share the same sign, the model called the direction correctly.
            </p>
            <LogRetChart data={data.points} />
          </div>

          <div className="card">
            <h3 className="font-semibold mb-3 text-sm">Daily detail</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-white/5">
                    <th className="py-2 pr-4">Date</th>
                    <th className="py-2 pr-4 text-right">Actual $</th>
                    <th className="py-2 pr-4 text-right">Predicted $</th>
                    <th className="py-2 pr-4 text-right">Diff</th>
                    <th className="py-2 pr-4 text-right">Actual logret</th>
                    <th className="py-2 pr-4 text-right">Pred logret</th>
                    <th className="py-2 text-center">Dir</th>
                  </tr>
                </thead>
                <tbody>
                  {data.points.map((p) => {
                    const diff = p.predicted_close - p.actual_close
                    const diffPct = p.actual_close ? (diff / p.actual_close) * 100 : 0
                    const sameDir = Math.sign(p.actual_log_ret) === Math.sign(p.predicted_log_ret)
                    return (
                      <tr key={p.date} className="border-b border-white/5 hover:bg-white/[0.02]">
                        <td className="py-1.5 pr-4 font-mono">{p.date}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{p.actual_close?.toFixed(2)}</td>
                        <td className="py-1.5 pr-4 text-right font-mono">{p.predicted_close?.toFixed(2)}</td>
                        <td className={`py-1.5 pr-4 text-right font-mono ${diff >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {diff >= 0 ? '+' : ''}{diff.toFixed(2)} ({diffPct.toFixed(2)}%)
                        </td>
                        <td className="py-1.5 pr-4 text-right font-mono text-slate-300">
                          {(p.actual_log_ret * 100).toFixed(3)}%
                        </td>
                        <td className="py-1.5 pr-4 text-right font-mono text-slate-300">
                          {(p.predicted_log_ret * 100).toFixed(3)}%
                        </td>
                        <td className="py-1.5 text-center">
                          <span className={sameDir ? 'text-emerald-400' : 'text-rose-400'}>
                            {sameDir ? '✓' : '✗'}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {!data && !error && !loading && coinInfo && (
        <div className="card text-center text-slate-400 py-12">
          Pick a date range and click <span className="text-slate-200 font-medium">Compare</span>.
        </div>
      )}
    </div>
  )
}
