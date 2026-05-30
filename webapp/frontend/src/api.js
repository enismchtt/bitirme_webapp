import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 180000,
})

export async function getCoins() {
  const { data } = await api.get('/coins')
  return data.coins
}

export async function getCoinInfo(symbol) {
  const { data } = await api.get(`/coins/${symbol}`)
  return data
}

export async function getHistorical({ coin, start, end }) {
  const { data } = await api.get('/historical', { params: { coin, start, end } })
  return data
}

export async function getForecast({ coin, days = 7 }) {
  const { data } = await api.get('/forecast', { params: { coin, days } })
  return data
}

export async function getRecent({ coin, days = 14 }) {
  const { data } = await api.get('/recent', { params: { coin, days } })
  return data.points
}

export async function getInterpretation({ coin, recent, models, last_known_close }) {
  const { data } = await api.post('/interpret', { coin, recent, models, last_known_close })
  return data
}
