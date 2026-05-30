"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = BASE_DIR / os.getenv("CACHE_DIR", "cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "1825"))  # 5 years — matches src/forecast.py

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# The 20 coins the team trained models for. Listed in display order.
SUPPORTED_COINS: list[str] = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "MATIC", "LTC",
    "LINK", "ATOM", "XLM", "ETC", "XMR", "ALGO", "VET", "TRX",
    "EOS", "NEO", "IOTA", "CHZ",
]

# Forecasting hyperparameters mirror the ones the team used during research
# (see src/forecast.py and src/config.py).
XGB_LAGS = 7
XGB_OUTPUT_CHUNK = 1

# Feature combination used for predictions on 1d.
# Target is always log_ret_close; the rest are past covariates (each gets
# its own 7-step lag block, just like src/forecast.py with darts).
BEST_FEATURE_SET: dict[str, list[str]] = {
    "xg_boost": ["rsi", "macd", "log_ret_close"],
    "lstm": ["rsi", "log_ret_close"],
    "cnn_lstm": ["volatility", "log_ret_close"]
    }



XGB_KWARGS = {
    "lags": XGB_LAGS,
    "output_chunk_length": XGB_OUTPUT_CHUNK,
    "random_state": 42,
    "n_jobs": 1,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_leaves": 10,
    "min_child_weight": 7,
    "subsample": 0.89,
    "colsample_bytree": 0.93,
    "gamma": 0.005,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

# Cache freshness: if the on-disk CSV's last candle is older than this many
# days, we refetch from Binance.
CACHE_REFRESH_DAYS = 1
