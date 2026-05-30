import { useEffect, useMemo, useState } from 'react'
import { Sparkles, Play, AlertCircle, BrainCircuit, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import CoinSelector from './CoinSelector.jsx'
import PriceChart from './PriceChart.jsx'
import MetricCard from './MetricCard.jsx'
import { getCoinInfo, getForecast, getInterpretation, getRecent } from '../api.js'

export default function ForecastView({ coins, coin, onCoinChange }) {
  const [coinInfo, setCoinInfo] = useState(null)
  const [days, setDays] = useState(7)
  const [forecast, setForecast] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [aiLoading, setAiLoading] = useState(false)
  const [aiText, setAiText] = useState(null)
  const [aiProvider, setAiProvider] = useState(null)
  const [aiError, setAiError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setCoinInfo(null)
    setForecast(null)
    setAiText(null)
    setError(null)
    getCoinInfo(coin)
      .then((info) => !cancelled && setCoinInfo(info))
      .catch((e) => !cancelled && setError(e?.response?.data?.detail || e.message))
    return () => { cancelled = true }
  }, [coin])

  async function run() {
    setLoading(true)
    setError(null)
    setForecast(null)
    setAiText(null)
    try {
      const res = await getForecast({ coin, days })
      setForecast(res)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  async function runAi() {
    if (!forecast) return
    setAiLoading(true)
    setAiError(null)
    setAiText(null)
    try {
      const recent = recentActual.length
        ? recentActual
        : await getRecent({ coin, days: 14 })
      const res = await getInterpretation({
        coin,
        recent,
        forecast: forecast.points,
      })
      setAiText(res.interpretation)
      setAiProvider(res.provider)
    } catch (e) {
      setAiError(e?.response?.data?.detail || e.message)
    } finally {
      setAiLoading(false)
    }
  }

  // Recent actual candles for chart context (cheap, no training).
  const [recentActual, setRecentActual] = useState([])
  useEffect(() => {
    if (!forecast) {
      setRecentActual([])
      return
    }
    getRecent({ coin, days: 14 })
      .then(setRecentActual)
      .catch(() => setRecentActual([]))
  }, [coin, forecast])

  const chartData = useMemo(() => {
    if (!forecast) return []
    const left = recentActual.map((r) => ({
      date: r.date,
      actual_close: r.close,
      predicted_close: null,
    }))
    const lastActual = recentActual.length
      ? { date: forecast.last_known_date, actual_close: forecast.last_known_close, predicted_close: forecast.last_known_close }
      : null
    const right = forecast.points.map((p) => ({
      date: p.date,
      actual_close: null,
      predicted_close: p.predicted_close,
    }))
    return [...left, ...(lastActual ? [lastActual] : []), ...right]
  }, [forecast, recentActual])

  const summary = useMemo(() => {
    if (!forecast || forecast.points.length === 0) return null
    const last = forecast.last_known_close
    const final = forecast.points[forecast.points.length - 1].predicted_close
    const pct = ((final - last) / last) * 100
    const peak = Math.max(...forecast.points.map((p) => p.predicted_close))
    const trough = Math.min(...forecast.points.map((p) => p.predicted_close))
    const totalRet = forecast.points.reduce((acc, p) => acc + p.predicted_log_ret, 0)
    return { last, final, pct, peak, trough, totalRet }
  }, [forecast])

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="grid md:grid-cols-[1fr_auto_auto] gap-4 items-end">
          <CoinSelector coins={coins} value={coin} onChange={onCoinChange} />
          <div>
            <span className="label block mb-1.5">Days</span>
            <input
              type="number"
              min="1"
              max="30"
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value, 10) || 7)}
              className="input w-24 text-center"
            />
          </div>
          <button onClick={run} disabled={loading} className="btn btn-primary h-[42px] md:w-44">
            <Play size={16} />
            {loading ? 'Forecasting...' : 'Forecast Future'}
          </button>
        </div>
        {coinInfo && (
          <div className="mt-4 flex flex-wrap gap-2 text-xs">
            <span className="chip">Latest data: {coinInfo.latest_date}</span>
            <span className="chip">Latest close: ${coinInfo.latest_close.toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
            <span className="chip">Model: XGBoost · autoregressive</span>
            {forecast?.features && (
              <span className="chip text-cyan-glow">
                Features: {forecast.features.join(' + ')}
              </span>
            )}
          </div>
        )}
        <p className="mt-3 text-xs text-slate-500 leading-relaxed">
          Starting from the last <span className="text-slate-300 font-medium">7 days</span> of log-returns,
          the model feeds its own predictions back as input for each subsequent day (autoregressive).
          Uncertainty grows as the horizon extends.
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

      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Last actual close"
            value={`$${summary.last.toLocaleString('en-US', { maximumFractionDigits: 4 })}`}
            sub={forecast.last_known_date}
            accent="cyan"
          />
          <MetricCard
            label={`Prediction in ${forecast.points.length} days`}
            value={`$${summary.final.toLocaleString('en-US', { maximumFractionDigits: 4 })}`}
            sub={forecast.points[forecast.points.length - 1].date}
            accent="accent"
          />
          <MetricCard
            label="Total expected change"
            value={`${summary.pct >= 0 ? '+' : ''}${summary.pct.toFixed(2)}%`}
            accent={summary.pct >= 0 ? 'green' : 'red'}
          />
          <MetricCard
            label="Forecast range"
            value={`$${summary.trough.toFixed(2)} – $${summary.peak.toFixed(2)}`}
            sub="Predicted min – max"
            accent="cyan"
          />
        </div>
      )}

      {forecast && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold flex items-center gap-2">
              <Sparkles size={16} className="text-accent-400" />
              {coin}/USDT · Next {forecast.points.length} Days
            </h3>
          </div>
          <PriceChart
            data={chartData}
            showActual
            forecastStartDate={forecast.last_known_date}
            height={380}
          />
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-slate-400 border-b border-white/5">
                  <th className="py-2 pr-4">Date</th>
                  <th className="py-2 pr-4 text-right">Predicted $</th>
                  <th className="py-2 pr-4 text-right">Daily change</th>
                  <th className="py-2 pr-4 text-right">Cumulative change</th>
                </tr>
              </thead>
              <tbody>
                {forecast.points.map((p, i) => {
                  const prev = i === 0 ? forecast.last_known_close : forecast.points[i - 1].predicted_close
                  const daily = ((p.predicted_close - prev) / prev) * 100
                  const total = ((p.predicted_close - forecast.last_known_close) / forecast.last_known_close) * 100
                  return (
                    <tr key={p.date} className="border-b border-white/5">
                      <td className="py-1.5 pr-4 font-mono">{p.date}</td>
                      <td className="py-1.5 pr-4 text-right font-mono">{p.predicted_close.toFixed(4)}</td>
                      <td className={`py-1.5 pr-4 text-right font-mono ${daily >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {daily >= 0 ? '+' : ''}{daily.toFixed(3)}%
                      </td>
                      <td className={`py-1.5 pr-4 text-right font-mono ${total >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {total >= 0 ? '+' : ''}{total.toFixed(3)}%
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {forecast && (
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold flex items-center gap-2">
              <BrainCircuit size={16} className="text-cyan-glow" />
              AI Interpretation
            </h3>
            <button onClick={runAi} disabled={aiLoading} className="btn btn-primary">
              {aiLoading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
              {aiLoading ? 'Generating...' : aiText ? 'Regenerate' : 'Interpret with AI'}
            </button>
          </div>
          {aiError && (
            <div className="flex items-start gap-3 p-3 rounded-lg bg-rose-500/5 border border-rose-500/30 text-sm text-rose-300">
              <AlertCircle size={16} className="mt-0.5" /> {aiError}
            </div>
          )}
          {!aiText && !aiError && !aiLoading && (
            <p className="text-sm text-slate-400">
              Sends the model's forecast and the recent actual data to Gemini (or a rule-based
              fallback engine) and returns a natural-language summary with caveats.
            </p>
          )}
          {aiText && (
            <>
              <div className="prose-ai">
                <ReactMarkdown>{aiText}</ReactMarkdown>
              </div>
              <div className="mt-3 text-[10px] uppercase tracking-wider text-slate-500">
                Provider: {aiProvider === 'gemini' ? 'Google Gemini' : 'Rule-based (fallback)'}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
