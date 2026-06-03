import { useEffect, useMemo, useState } from 'react'
import { Sparkles, Play, AlertCircle, BrainCircuit, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import CoinSelector from './CoinSelector.jsx'
import PriceChart from './PriceChart.jsx'
import MetricCard from './MetricCard.jsx'
import ModelFilter, { MODEL_STROKE, MODEL_LABELS } from './ModelFilter.jsx'
import InterpretSignalBadge from './InterpretSignalBadge.jsx'
import { getCoinInfo, getForecast, getInterpretation, getRecent } from '../api.js'

const SUPPORTED_MODELS = ['xg_boost', 'lstm', 'cnn_lstm', 'tcn']

export default function ForecastView({ coins, coin, onCoinChange }) {
  const [coinInfo, setCoinInfo] = useState(null)
  const [days, setDays] = useState(7)
  const [forecast, setForecast] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedModel, setSelectedModel] = useState('xg_boost')

  const [aiLoading, setAiLoading] = useState(false)
  const [aiText, setAiText] = useState(null)
  const [aiSignal, setAiSignal] = useState(null)
  const [aiProvider, setAiProvider] = useState(null)
  const [aiError, setAiError] = useState(null)

  const [recentActual, setRecentActual] = useState([])

  useEffect(() => {
    let cancelled = false
    setCoinInfo(null)
    setForecast(null)
    setAiText(null)
    setAiSignal(null)
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
    setAiSignal(null)
    try {
      const res = await getForecast({ coin, days })
      setForecast(res)
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

  useEffect(() => {
    if (!forecast) { setRecentActual([]); return }
    getRecent({ coin, days: 14 })
      .then(setRecentActual)
      .catch(() => setRecentActual([]))
  }, [coin, forecast])

  async function runAi() {
    if (!forecast) return
    setAiLoading(true)
    setAiError(null)
    setAiText(null)
    setAiSignal(null)
    try {
      const recent = recentActual.length
        ? recentActual
        : await getRecent({ coin, days: 14 })
      // Send all models so the LLM can compare them.
      const allModels = {}
      for (const m of SUPPORTED_MODELS) {
        if (forecast.models?.[m]?.points) allModels[m] = forecast.models[m].points
      }
      const res = await getInterpretation({
        coin,
        recent,
        models: allModels,
        last_known_close: forecast.last_known_close,
      })
      setAiText(res.interpretation)
      setAiSignal(res.signal ?? null)
      setAiProvider(res.provider)
    } catch (e) {
      setAiError(e?.response?.data?.detail || e.message)
    } finally {
      setAiLoading(false)
    }
  }

  const modelResult = forecast?.models?.[selectedModel]
  const modelPoints = modelResult?.points ?? []
  const stroke = MODEL_STROKE[selectedModel] ?? '#a78bfa'
  const availableModels = forecast ? SUPPORTED_MODELS.filter((m) => forecast.models?.[m]) : []

  const chartData = useMemo(() => {
    if (!forecast || modelPoints.length === 0) return []
    const left = recentActual.map((r) => ({
      date: r.date,
      actual_close: r.close,
      predicted_close: null,
    }))
    const bridge = recentActual.length
      ? [{ date: forecast.last_known_date, actual_close: forecast.last_known_close, predicted_close: forecast.last_known_close }]
      : []
    const right = modelPoints.map((p) => ({
      date: p.date,
      actual_close: null,
      predicted_close: p.predicted_close,
    }))
    return [...left, ...bridge, ...right]
  }, [forecast, modelPoints, recentActual])

  const summary = useMemo(() => {
    if (!forecast || modelPoints.length === 0) return null
    const last = forecast.last_known_close
    const final = modelPoints[modelPoints.length - 1].predicted_close
    const pct = ((final - last) / last) * 100
    const peak = Math.max(...modelPoints.map((p) => p.predicted_close))
    const trough = Math.min(...modelPoints.map((p) => p.predicted_close))
    const totalRet = modelPoints.reduce((acc, p) => acc + p.predicted_log_ret, 0)
    return { last, final, pct, peak, trough, totalRet }
  }, [forecast, modelPoints])

  return (
    <div className="space-y-6">
      <div className="card relative z-20 overflow-visible">
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
          <button onClick={run} disabled={loading} className="btn btn-action h-[42px] md:w-44">
            <Play size={16} />
            {loading ? 'Forecasting...' : 'Forecast Future'}
          </button>
        </div>
        {coinInfo && (
          <div className="mt-4 flex flex-wrap gap-2 text-xs">
            <span className="chip">Latest data: {coinInfo.latest_date}</span>
            <span className="chip">Latest close: ${coinInfo.latest_close.toLocaleString('en-US', { maximumFractionDigits: 2 })}</span>
            <span className="chip">Lookback: 30 days · Interval: 1d · Autoregressive</span>
          </div>
        )}
        <p className="mt-3 text-xs text-slate-500 leading-relaxed">
          All three models run at once. Each looks back{' '}
          <span className="text-slate-300 font-medium">30 days</span> from today and then
          rolls forward autoregressively for the selected number of days. Select a model below to inspect its forecast.
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

      {forecast && (
        <>
          {/* Model filter */}
          <div className="card flex flex-wrap items-center gap-4 justify-between">
            <ModelFilter
              models={availableModels}
              selected={selectedModel}
              onChange={setSelectedModel}
            />
            {forecast.training_note && (
              <p className="text-xs text-slate-500 max-w-lg">{forecast.training_note}</p>
            )}
          </div>

          {/* Summary metrics */}
          {summary && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard
                label="Last actual close"
                value={`$${summary.last.toLocaleString('en-US', { maximumFractionDigits: 4 })}`}
                sub={forecast.last_known_date}
                accent="cyan"
              />
              <MetricCard
                label={`${MODEL_LABELS[selectedModel]} in ${modelPoints.length} days`}
                value={`$${summary.final.toLocaleString('en-US', { maximumFractionDigits: 4 })}`}
                sub={modelPoints[modelPoints.length - 1]?.date}
                accent="accent"
              />
              <MetricCard
                label="Expected change"
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

          {/* All-model final-price comparison */}
          <div className="card">
            <h3 className="font-semibold text-sm mb-3">All-model comparison · predicted close in {days} days</h3>
            <div className="grid grid-cols-3 gap-3">
              {availableModels.map((m) => {
                const pts = forecast.models[m]?.points ?? []
                const finalClose = pts.length ? pts[pts.length - 1].predicted_close : null
                const chg = finalClose != null ? ((finalClose - forecast.last_known_close) / forecast.last_known_close) * 100 : null
                return (
                  <button
                    key={m}
                    onClick={() => setSelectedModel(m)}
                    className={`rounded-lg p-3 border text-left transition-all ${m === selectedModel ? 'border-white/20 bg-white/5' : 'border-white/5 bg-white/[0.02] hover:bg-white/5'
                      }`}
                  >
                    <div className="text-xs text-slate-400 mb-1">{MODEL_LABELS[m]}</div>
                    <div className="text-lg font-bold font-mono" style={{ color: MODEL_STROKE[m] }}>
                      {finalClose != null ? `$${finalClose.toLocaleString('en-US', { maximumFractionDigits: 2 })}` : '—'}
                    </div>
                    {chg != null && (
                      <div className={`text-xs mt-0.5 ${chg >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {chg >= 0 ? '+' : ''}{chg.toFixed(2)}%
                      </div>
                    )}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Forecast chart */}
          <div className="card">
            <div className="flex items-center justify-between mb-3 flex-wrap gap-3">
              <h3 className="font-semibold flex items-center gap-2">
                <Sparkles size={16} className="text-accent-400" />
                {coin}/USDT · Next {modelPoints.length} Days · {MODEL_LABELS[selectedModel]}
              </h3>
            </div>
            <PriceChart
              data={chartData}
              showActual
              forecastStartDate={forecast.last_known_date}
              predictionColor={stroke}
              predictionLabel={`${MODEL_LABELS[selectedModel]} forecast`}
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
                  {modelPoints.map((p, i) => {
                    const prev = i === 0 ? forecast.last_known_close : modelPoints[i - 1].predicted_close
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

          {/* AI Interpretation */}
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold flex items-center gap-2">
                <BrainCircuit size={16} className="text-cyan-glow" />
                AI Interpretation · all models
              </h3>
              <button onClick={runAi} disabled={aiLoading} className="btn btn-action">
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
                Sends all three models' forecasts to a local <span className="text-slate-200 font-medium">Ollama</span> instance
                for a concise buy/sell/hold commentary. If Ollama is not running, a rule-based consensus summary is shown instead.
              </p>
            )}
            {aiText && (
              <>
                {aiSignal && (
                  <InterpretSignalBadge
                    signal={aiSignal}
                    horizonDays={forecast?.days ?? days}
                  />
                )}
                <div className="prose-ai">
                  <ReactMarkdown>{aiText}</ReactMarkdown>
                </div>
                <div className={`mt-3 text-[10px] uppercase tracking-wider ${aiProvider?.includes('unreachable') || aiProvider?.startsWith('rule') ? 'text-amber-500' : 'text-slate-500'}`}>
                  Provider: {aiProvider}
                </div>
              </>
            )}
          </div>
        </>
      )}

      {!forecast && !error && !loading && coinInfo && (
        <div className="card text-center text-slate-400 py-12">
          Set the horizon and click <span className="text-slate-200 font-medium">Forecast Future</span>.
          All three models run simultaneously.
        </div>
      )}
    </div>
  )
}
