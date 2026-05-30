"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]  # webapp/backend → webapp → repo root

CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", "cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = REPO_ROOT / "outputs" / "models"

HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "1825"))  # 5 years

# ── Ollama local LLM ──────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Pick a model that fits in 4 GB RAM: llama3.2:1b, qwen2.5:1.5b, phi3:mini, etc.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # seconds

# Threshold for BUY/SELL consensus signal (% units).
# The sum of all models' predicted % changes is compared to this:
#   sum >  threshold  → BUY
#   sum < -threshold  → SELL
#   otherwise         → HOLD
SIGNAL_CONSENSUS_PCT_THRESHOLD = float(
    os.getenv("SIGNAL_CONSENSUS_PCT_THRESHOLD", "0.05")
)

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# ── Forecasting constants (mirror train_models/config.py) ─────────────────
SEQUENCE_LENGTH = 30
TARGET_COL = "target_log_ret_close_next_1d"

# Indicator warmup rows needed before the first valid feature row.
# MACD slow EMA requires min_periods=26; RSI requires 14. 26 is the binding constraint.
INDICATOR_WARMUP = 26

# Per-model feature sets (must stay in sync with train_models/config.py).
BEST_FEATURE_SET: dict[str, list[str]] = {
    "xg_boost": ["rsi", "macd", "log_ret_close"],
    "lstm": ["rsi", "log_ret_close"],
    "cnn_lstm": ["volatility", "log_ret_close"],
}

SUPPORTED_MODELS: list[str] = ["xg_boost", "lstm", "cnn_lstm"]

FORECAST_DAYS_MIN = 1
FORECAST_DAYS_MAX = 30

TRAINING_NOTE = (
    "Checkpoints were trained on BTC and ETH data combined. "
    "Predictions for other coins use the same global weights and may be less accurate."
)

# The 20 coins supported by the UI. Models were trained on BTC + ETH only.
SUPPORTED_COINS: list[str] = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "MATIC", "LTC",
    "LINK", "ATOM", "XLM", "ETC", "XMR", "ALGO", "VET", "TRX",
    "EOS", "NEO", "IOTA", "CHZ",
]

# Cache freshness: if the on-disk CSV's last candle is older than this many
# days, we refetch from Binance.
CACHE_REFRESH_DAYS = 1
