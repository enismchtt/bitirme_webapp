import { useEffect, useState } from 'react'
import Header from './components/Header.jsx'
import HistoricalView from './components/HistoricalView.jsx'
import ForecastView from './components/ForecastView.jsx'
import { getCoins } from './api.js'

export default function App() {
  const [tab, setTab] = useState('historical')
  const [coins, setCoins] = useState([])
  const [coin, setCoin] = useState('BTC')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getCoins()
      .then((list) => {
        setCoins(list)
        if (!list.includes(coin) && list.length) setCoin(list[0])
      })
      .catch((e) => setError(e?.response?.data?.detail || e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="min-h-screen">
      <Header tab={tab} onTabChange={setTab} />
      <main className="max-w-7xl mx-auto px-6 py-8">
        {error && (
          <div className="card text-rose-300 border-rose-500/30 bg-rose-500/5">
            Could not reach the backend: {error}
            <div className="mt-2 text-xs text-slate-400">
              Make sure the backend is running: <code className="font-mono">uvicorn app:app --reload</code>
            </div>
          </div>
        )}
        {loading && !error && (
          <div className="card text-center text-slate-400 py-12">Loading...</div>
        )}
        {!loading && !error && coins.length > 0 && (
          <>
            {tab === 'historical' && (
              <HistoricalView coins={coins} coin={coin} onCoinChange={setCoin} />
            )}
            {tab === 'forecast' && (
              <ForecastView coins={coins} coin={coin} onCoinChange={setCoin} />
            )}
          </>
        )}
      </main>
      <footer className="text-center text-xs text-slate-600 py-6">
        Hacettepe BBM479 design project
      </footer>
    </div>
  )
}
