# Crypto Prediction Dashboard

Two-tab React + FastAPI web app that visualises the team's XGBoost (lags=7)
log-return forecasts for **20 cryptocurrencies**, fetching candle data from
Binance on demand and offering an optional **Google Gemini** "interpret this
chart" button.

```
webapp/
├── backend/                       FastAPI + xgboost + Gemini
│   ├── app.py                     REST endpoints
│   ├── config.py                  Env, coin list, XGB hyperparams, feature set
│   ├── services/
│   │   ├── binance_fetcher.py     Cached Binance candle fetcher (REST, no SDK)
│   │   ├── forecaster.py          Historical + autoregressive forecasting
│   │   └── interpreter.py         Gemini LLM + rule-based fallback
│   ├── requirements.txt           Backend pip dependencies
│   └── .env.example               Copy to .env and add GEMINI_API_KEY
└── frontend/                      Vite + React + Tailwind + Recharts
    ├── src/
    │   ├── App.jsx
    │   ├── api.js
    │   └── components/
    └── package.json
```

## What the app does

### Historical Analysis tab

1. Pick a coin (20 supported) and a past date range.
2. Backend trains XGBoost (`lags=7`, same hyperparameters as `src/forecast.py`)
   on data strictly *before* the start date.
3. For every day in the chosen window the model produces a one-step-ahead
   prediction using the **previous 7 actual days** — i.e. *"predict today by
   looking at the past 7 days"*.
4. Predicted log-returns are converted to USD prices via
   `predicted_close = prev_actual_close · exp(predicted_log_ret)`.
5. The UI shows:
   * Actual vs predicted price line/area chart, with a dashed
     **autoregressive forecast** overlay (starts from the actual at day 0,
     then feeds its own predictions back — does not "lag" the actual line).
   * Daily log-return bar chart (green/red bars = actual direction, purple
     line = model prediction).
   * A 4-card metrics row: days, direction accuracy, MAPE, RMSE.
   * A day-by-day detail table.

### Future Forecast tab

1. Pick a coin and a number of days (1–30).
2. Backend trains XGBoost on **all** available candles.
3. The model is rolled forward autoregressively for N days — each step's
   prediction becomes the next step's input.
4. The frontend shows the last 14 actual days followed by the forecast, with
   a dashed marker at the forecast boundary.
5. **"Interpret with AI"** sends the recent history + forecast to Google
   Gemini (`gemini-1.5-flash` by default) and renders the markdown reply.
   If no key is set, a rule-based summary fires automatically.

## Prerequisites

| Tool        | Why                       | How to install (macOS)            |
| ----------- | ------------------------- | --------------------------------- |
| **Python 3.10+** | Backend runtime      | Already on macOS, or `brew install python` |
| **Node.js 18+ (with npm)** | Frontend dev server | <https://nodejs.org/> or `brew install node` |
| **Git**     | Cloning the repo          | Comes with Xcode tools or `brew install git` |
| **Google Gemini API key** *(optional)* | Real LLM interpretation | <https://aistudio.google.com/app/apikey> (free tier) |

The project already ships a `.venv` at the repo root with `darts`, `xgboost`,
`pandas`, `pandas_ta_classic`, and `python-binance` pre-installed. The
backend layers a few extra packages on top.

## Quick start (recommended)

Open **two terminals** in the repository root.

```bash
# Terminal 1 — Backend (port 8000)
./webapp/start_backend.sh

# Terminal 2 — Frontend (port 5173)
./webapp/start_frontend.sh
```

Then open <http://localhost:5173>.

The first request for a coin fetches ~5 years of 1d candles from Binance and
caches them in `backend/cache/{COIN}USDT_1d.csv`. Subsequent requests are
near-instant (XGBoost train + predict ≈ 50–200 ms).

### What `start_backend.sh` does

1. Locates the repo's `.venv` and verifies Python is available.
2. Copies `backend/.env.example` → `backend/.env` if missing.
3. Installs the extra pip packages from `backend/requirements.txt`
   (`fastapi`, `uvicorn`, `python-dotenv`, `pydantic`, `google-generativeai`).
4. Starts `uvicorn app:app --reload`.

### What `start_frontend.sh` does

1. Checks `npm` is on `PATH`.
2. Runs `npm install` if `node_modules/` is missing.
3. Starts `npm run dev` (Vite).

## Manual setup (without scripts)

```bash
# Backend
cd webapp/backend
../../.venv/bin/pip install -r requirements.txt
cp .env.example .env          # edit and add GEMINI_API_KEY=...
../../.venv/bin/uvicorn app:app --reload --port 8000

# Frontend (separate terminal)
cd webapp/frontend
npm install
npm run dev
```

Vite proxies `/api/*` to `localhost:8000` automatically (see
`vite.config.js`).

## Environment variables (`backend/.env`)

| Variable           | Default                 | Purpose                                            |
| ------------------ | ----------------------- | -------------------------------------------------- |
| `GEMINI_API_KEY`   | _(empty)_               | Free key from <https://aistudio.google.com/app/apikey>. If empty, the rule-based summary runs instead. |
| `GEMINI_MODEL`     | `gemini-1.5-flash`      | Gemini model name.                                 |
| `CACHE_DIR`        | `cache`                 | Where to keep fetched candles.                     |
| `HISTORY_DAYS`     | `1825`                  | How many days back to fetch on first request.      |
| `FRONTEND_ORIGIN`  | `http://localhost:5173` | CORS origin.                                       |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | _(empty)_ | Optional — public market data does NOT require them. |

## REST endpoints

| Method | Path                          | Description |
| ------ | ----------------------------- | ----------- |
| `GET`  | `/api/health`                 | Liveness + whether Gemini is configured. |
| `GET`  | `/api/coins`                  | List of the 20 supported coins. |
| `GET`  | `/api/coins/{symbol}`         | Date bounds + latest close (fetches/caches on demand). |
| `GET`  | `/api/recent?coin&days`       | Raw last N candles (no training). |
| `GET`  | `/api/historical?coin&start&end` | Actual + 1-step-ahead + autoregressive predictions in `[start, end]`. |
| `GET`  | `/api/forecast?coin&days`     | Autoregressive N-day forecast from the latest candle. |
| `POST` | `/api/interpret`              | `{coin, recent, forecast}` → Markdown analysis. |

Interactive API docs: <http://localhost:8000/docs>.

## How the predictions work

Both modes use the team's hyperparameters (see `config.XGB_KWARGS`, mirrored
from `src/forecast.py`):

```python
lags=7,
output_chunk_length=1,
n_estimators=300, learning_rate=0.05,
max_leaves=10, min_child_weight=7,
subsample=0.89, colsample_bytree=0.93,
gamma=0.005, reg_alpha=0.0, reg_lambda=1.0,
random_state=42, tree_method="hist"
```

The default feature set for 1d is **`['rsi', 'macd', 'log_ret_close']`**
(target + two past covariates, each with its own 7-step lag block). You can
change it in `backend/config.py → BEST_FEATURE_SET`.

Target column is `log_ret_close = log(close[t]) − log(close[t-1])`. Prices
are reconstructed via `predicted_close[t] = prev_close · exp(pred_log_ret)`.

## Troubleshooting

* **"Could not load candles for X 1d."** — Binance request failed. Check the
  backend terminal for the underlying exception (network blocked, region
  ban, missing TLS). The fetcher uses `https://api.binance.com/api/v3/klines`
  directly via `requests`, so no auth or VPN is required for most coins.

* **The "Actual vs Predicted" line looks like the actual price shifted by
  one day.** That is mathematically expected: log returns are small, so
  `prev_close · exp(small)` ≈ `prev_close`. Look at the dashed yellow
  *autoregressive* line and the daily log-return bar chart for a clearer
  picture of what the model is actually doing.

* **"npm: command not found"** — Install Node.js 18+ from
  <https://nodejs.org/> or `brew install node`.

* **`Address already in use` on port 8000** — Find the zombie process:
  `lsof -iTCP:8000 -sTCP:LISTEN` then `kill <pid>`.

* **Model output looks "too smooth" / flat** — Daily log returns are
  inherently small. Try the autoregressive overlay or extend the forecast
  horizon to amplify visible drift.

## License / disclaimer

Educational project. The forecasts are **not financial advice**.
