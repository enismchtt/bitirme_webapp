# Crypto Prediction Dashboard

Two-tab React + FastAPI app that runs **three pre-trained models** (XGBoost, LSTM, CNN-LSTM) on **20 cryptocurrencies**, with a **30-day lookback** and training-aligned features. Candle data comes from Binance (cached locally). **Ollama** (local LLM) can comment on all models’ forecasts with a short buy/sell/hold-style summary.

```
webapp/
├── backend/                       FastAPI + PyTorch + XGBoost + Ollama client
│   ├── app.py                     REST endpoints
│   ├── config.py                  Coins, MODELS_DIR, Ollama, BEST_FEATURE_SET
│   ├── services/
│   │   ├── features.py            RSI/MACD/volatility (training parity)
│   │   ├── binance_fetcher.py     Cached Binance OHLCV
│   │   ├── model_registry.py      Load outputs/models/*.pt
│   │   ├── inference.py           1-step + autoregressive roll-forward
│   │   ├── forecaster.py          Historical + future (all models)
│   │   └── interpreter.py         Ollama interpretation + rule-based fallback
│   ├── requirements.txt
│   └── .env.example
├── start_backend.sh
├── start_frontend.sh
└── frontend/                      Vite + React + Tailwind + Recharts
```

Checkpoints (train first if missing):

```
outputs/models/
├── xg_boost.pt
├── lstm.pt
└── cnn_lstm.pt
```

From the **repository root**:

```bash
python train_models/train.py
```

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| **Python 3.10+** | Backend |
| **Node.js 18+** | Frontend (`npm`) |
| **`outputs/models/*.pt`** | Inference (see above) |
| **Ollama** *(optional)* | AI interpretation on Future Forecast tab |
| **~4 GB+ RAM** for Ollama | Use a **small** local model (see below) |

Backend pip packages: `fastapi`, `uvicorn`, `pandas`, `numpy`, `xgboost`, `torch`, `scikit-learn`, `requests`. No Gemini API key and no `python-binance` / `pandas_ta`.

---

## Run the whole app (recommended order)

Use **three terminals** (four if Ollama is not already running as a service).

### Terminal 1 — Ollama (local LLM)

Install Ollama: [https://ollama.com](https://ollama.com)

```bash
# Start the Ollama server (leave this running)
ollama serve
```

In **another** shell, pull a small model that fits **~4 GB RAM**:

```bash
ollama pull llama3.2:1b
# alternatives: ollama pull qwen2.5:1.5b   or   ollama pull phi3:mini
```

Match the model name in `webapp/backend/.env` (copy from `.env.example`):

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_TIMEOUT=60
```

Quick check:

```bash
curl http://localhost:11434/api/tags
```

If Ollama is **not** running, the app still works: forecasting and charts work; **Interpret with AI** shows a **rule-based** consensus summary and labels the provider as `rule-based (ollama unreachable)`.

---

### Terminal 2 — Backend (port 8000)

From the **repository root**, create a venv once if needed:

```bash
python -m venv .venv
.venv/bin/pip install -r webapp/backend/requirements.txt    # Linux/macOS
# Windows:  .venv\Scripts\pip install -r webapp\backend\requirements.txt
```

**Linux / macOS (script):**

```bash
./webapp/start_backend.sh
```

**Windows (PowerShell), from repo root:**

```powershell
cd webapp\backend
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
..\..\..venv\Scripts\pip install -r requirements.txt
..\..\..venv\Scripts\python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API: [http://localhost:8000](http://localhost:8000) · Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

First request per coin may fetch ~5 years of 1d candles into `webapp/backend/cache/`.

---

### Terminal 3 — Frontend (port 5173)

**Linux / macOS:**

```bash
./webapp/start_frontend.sh
```

**Windows:**

```powershell
cd webapp\frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Vite proxies `/api/*` to the backend.

---

### Startup checklist

| Step | Command / URL | OK when |
|------|----------------|---------|
| 1 | `ollama serve` + `ollama pull llama3.2:1b` | `curl localhost:11434/api/tags` lists your model |
| 2 | Backend on :8000 | `GET http://localhost:8000/api/health` returns `"ok": true` |
| 3 | Frontend on :5173 | Dashboard loads, coin list appears |
| 4 | Train checkpoints | `outputs/models/*.pt` exist (or historical/forecast returns 503) |

---

## What the app does

### Historical Analysis

- Pick coin + date range → **Compare** runs **all three models** in one request.
- Per model: **1-step** predictions (30 actual days before each day) + **autoregressive** overlay in range.
- UI: model filter (XGBoost / LSTM / CNN-LSTM), metrics, price + log-return charts.

Earliest start date needs ~**56 days** of history before `start` (30-day window + indicator warmup).

### Future Forecast

- Pick coin + **1–30 days** → **Forecast Future** runs all models from the last candle.
- **Interpret with AI** sends **all models’** forecast points + recent actuals to Ollama for a short commentary (buy/sell/hold style). Uses selected model only for the chart; interpretation always compares all three.

---

## Models

| Model | Checkpoint | Features |
|-------|------------|----------|
| `xg_boost` | `outputs/models/xg_boost.pt` | `rsi`, `macd`, `log_ret_close` |
| `lstm` | `outputs/models/lstm.pt` | `rsi`, `log_ret_close` |
| `cnn_lstm` | `outputs/models/cnn_lstm.pt` | `volatility`, `log_ret_close` |

Weights were trained on **BTC + ETH**. All 20 UI coins are supported; API may include a `training_note` that other symbols use the same checkpoints.

---

## Environment variables (`webapp/backend/.env`)

Copy `webapp/backend/.env.example` → `.env`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2:1b` | Model name (`ollama list`) |
| `OLLAMA_TIMEOUT` | `60` | Seconds before rule-based fallback |
| `SIGNAL_CONSENSUS_PCT_THRESHOLD` | `0.05` | Min \|% change\| for a model to vote up/down in consensus (fallback signal) |
| `CACHE_DIR` | `cache` | Binance CSV cache |
| `HISTORY_DAYS` | `1825` | Days fetched on first coin load |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | CORS |

---

## REST API (summary)

| Method | Path | Params |
|--------|------|--------|
| `GET` | `/api/health` | Ollama URL/model in response |
| `GET` | `/api/coins` | List of 20 coins |
| `GET` | `/api/coins/{symbol}` | Date bounds + latest close |
| `GET` | `/api/recent` | `coin`, `days` |
| `GET` | `/api/historical` | `coin`, `start`, `end` → `models` dict |
| `GET` | `/api/forecast` | `coin`, `days` (1–30) → `models` dict |
| `POST` | `/api/interpret` | `{ coin, recent, models, last_known_close }` |

No `model` query param on historical/forecast — one click runs all three.

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| **503 / missing `.pt`** | Run `python train_models/train.py` from repo root |
| **Ollama unreachable** | Start `ollama serve`, check `OLLAMA_URL`, pull model matching `OLLAMA_MODEL` |
| **Ollama very slow / OOM** | Use a smaller model (`llama3.2:1b`, `phi3:mini`); close other apps |
| **Could not load candles** | Network / Binance; see backend logs |
| **Not enough history before start** | Move start date later (need ~56 valid rows before range) |
| **`npm` not found** | Install Node.js 18+ |
| **Port 8000 in use** | Stop the other process or change uvicorn port |

---

## License / disclaimer

Educational project. Forecasts and AI text are **not financial advice**.
