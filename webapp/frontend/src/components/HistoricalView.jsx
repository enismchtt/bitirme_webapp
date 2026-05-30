import { useEffect, useMemo, useState } from 'react'
import { Activity, Target, Percent, TrendingUp, Play, AlertCircle } from 'lucide-react'
import CoinSelector from './CoinSelector.jsx'
import DateRangePicker from './DateRangePicker.jsx'
import MetricCard from './MetricCard.jsx'
import PriceChart from './PriceChart.jsx'
import LogRetChart from './LogRetChart.jsx'
import ModelFilter, { MODEL_STROKE, MODEL_LABELS } from './ModelFilter.jsx'
import { getCoinInfo, getHistorical } from '../api.js'

function shiftDays(dateStr, n) {
  const d = new Date(dateStr)
  d.setDate(d.getDate() + n)
  return d.toISOString().slice(0, 10)
}

const SUPPORTED_MODELS = ['xg_boost', 'lstm', 'cnn_lstm']

export default function HistoricalView({ coins, coin, onCoinChange }) {
  const [coinInfo, setCoinInfo] = useState(null)
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedModel, setSelectedModel] = useState('xg_boost')

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
        setStart(shiftDays(latest, -30))
        setEnd(shiftDays(latest, -1))
      })
      .catch((e) => {
        if (cancelled) return
        setError(e?.response?.data?.detail || e.message)
      })
    return () => { cancelled = true }
  }, [coin])

  // Require 30-day window + 26 indicator warmup before start.
  const minStartDate = useMemo(() => {
    if (!coinInfo) return undefined
    return shiftDays(coinInfo.earliest_date, 56)
  }, [coinInfo])

  async function run() {
    if (!start || !end) return
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const res = await getHistorical({ coin, start, end })
      setData(res)
      // Default to first model with results.
      const available = SUPPORTED_MODELS.filter((m) => res.models?.[m])
      if (available.length && !available.includes(selectedModel)) {
        setSelectedModel(available[0])
      }
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const modelResult = data?.models?.[selectedModel]
  const points = modelResult?.points ?? []
  const days = points.length
  const stroke = MODEL_STROKE[selectedModel] ?? '#a78bfa'
  const availableModels = data ? SUPPORTED_MODELS.filter((m) => data.models?.[m]) : []

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
            <span className="chip">Lookback: 30 days · Interval: 1d</span>
          </div>
        )}
        <p className="mt-3 text-xs text-slate-500 leading-relaxed">
          All three models run on click. Each uses its own{' '}
          <span className="text-slate-300 font-medium">30-day feature window</span>{' '}
          (ending the day before each eval date) to predict next-day log return.
          The autoregressive line feeds each day's prediction into the next window.
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
          {/* Model filter + training note */}
          <div className="card flex flex-wrap items-center gap-4 justify-between">
            <ModelFilter
              models={availableModels}
              selected={selectedModel}
              onChange={setSelectedModel}
            />
            {data.training_note && (
              <p className="text-xs text-slate-500 max-w-lg">{data.training_note}</p>
            )}
          </div>

          {/* Metrics for selected model */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Days" value={days} accent="cyan" icon={Activity} />
            <MetricCard
              label="Direction accuracy"
              value={modelResult ? `${(modelResult.direction_accuracy * 100).toFixed(1)}%` : '—'}
              sub={`Over ${days} days`}
              accent={modelResult?.direction_accuracy >= 0.5 ? 'green' : 'red'}
              icon={Target}
            />
            <MetricCard
              label="Price MAPE"
              value={modelResult && Number.isFinite(modelResult.mape) ? `${modelResult.mape.toFixed(2)}%` : '—'}
              sub="Mean absolute % error"
              accent="accent"
              icon={Percent}
            />
            <MetricCard
              label="RMSE (log return)"
              value={modelResult && Number.isFinite(modelResult.rmse_log_ret) ? modelResult.rmse_log_ret.toFixed(5) : '—'}
              sub={`Price RMSE: ${modelResult && Number.isFinite(modelResult.rmse_price) ? '$' + modelResult.rmse_price.toFixed(2) : '—'}`}
              accent="cyan"
              icon={TrendingUp}
            />
          </div>

          {/* All-model metric comparison */}
          <div className="card">
            <h3 className="font-semibold text-sm mb-3">Model comparison · direction accuracy</h3>
            <div className="grid grid-cols-3 gap-3">
              {availableModels.map((m) => {
                const mr = data.models[m]
                const acc = mr?.direction_accuracy ?? 0
                return (
                  <button
                    key={m}
                    onClick={() => setSelectedModel(m)}
                    className={`rounded-lg p-3 border text-left transition-all ${
                      m === selectedModel ? 'border-white/20 bg-white/5' : 'border-white/5 bg-white/[0.02] hover:bg-white/5'
                    }`}
                  >
                    <div className="text-xs text-slate-400 mb-1">{MODEL_LABELS[m]}</div>
                    <div className="text-lg font-bold" style={{ color: MODEL_STROKE[m] }}>
                      {(acc * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      RMSE log: {mr?.rmse_log_ret?.toFixed(5) ?? '—'}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Price chart */}
          <div className="card">
            <div className="flex items-center justify-between mb-1 flex-wrap gap-3">
              <h3 className="font-semibold">Actual vs. Predicted · {coin}/USDT · {MODEL_LABELS[selectedModel]}</h3>
              <div className="text-xs text-slate-400">{start} → {end}</div>
            </div>
            <p className="text-xs text-slate-500 mb-3">
              <span style={{ color: stroke }} className="font-medium">Solid line (1-step):</span>{' '}
              each day uses the previous 30 actual days as input — most accurate.{' '}
              <span className="text-amber-300 font-medium">Dashed line (autoregressive):</span>{' '}
              seeds from actual data at {start}, then feeds its own predictions forward.
            </p>
            <PriceChart
              data={points}
              showActual
              showAutoregressive
              predictionColor={stroke}
              predictionLabel={`${MODEL_LABELS[selectedModel]} 1-step`}
            />
          </div>

          {/* Log return chart */}
          <div className="card">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-semibold">Daily log return · Actual vs. {MODEL_LABELS[selectedModel]}</h3>
              <div className="text-xs text-slate-400">
                Direction accuracy:{' '}
                <span className="text-slate-200 font-semibold">
                  {modelResult ? `${(modelResult.direction_accuracy * 100).toFixed(1)}%` : '—'}
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-500 mb-3">
              Green/red bars = actual daily return. Colored line = model prediction. Matching signs = correct direction call.
            </p>
            <LogRetChart data={points} predictionColor={stroke} />
          </div>

          {/* Daily detail table */}
          <div className="card">
            <h3 className="font-semibold mb-3 text-sm">Daily detail · {MODEL_LABELS[selectedModel]}</h3>
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
                  {points.map((p) => {
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
          All models run simultaneously.
        </div>
      )}
    </div>
  )
}
