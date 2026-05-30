"""Fetch OHLCV candles from Binance public REST API with a local CSV cache.

Uses plain `requests` (no python-binance Client) so there is no initialization
ping that could fail on startup.  For each (symbol, timeframe) we keep one CSV
under ``CACHE_DIR``.  If the cached file's last candle is older than
``CACHE_REFRESH_DAYS``, we pull the tail and merge it in.
"""
from __future__ import annotations

import logging
import threading
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

_BINANCE_URL = "https://api.binance.com/api/v3/klines"

# Per-symbol locks so concurrent requests for the same coin don't both hit
# Binance at the same time.
_locks: dict[str, threading.Lock] = {}


def _lock_for(symbol: str, timeframe: str) -> threading.Lock:
    key = f"{symbol}_{timeframe}"
    if key not in _locks:
        _locks[key] = threading.Lock()
    return _locks[key]


def _cache_path(symbol: str, timeframe: str) -> Path:
    return config.CACHE_DIR / f"{symbol}USDT_{timeframe}.csv"


def _read_cache(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def _write_cache(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    path = _cache_path(symbol, timeframe)
    df.sort_index().to_csv(path, index_label="date")


def _fetch_from_binance(
    symbol: str,
    timeframe: str,
    start_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Pull klines from Binance REST API (no SDK, no ping)."""
    full_symbol = f"{symbol}USDT"
    all_rows: list[list] = []

    # Binance returns max 1000 candles per request; paginate until caught up.
    current_start = start_ms
    while True:
        params = {
            "symbol": full_symbol,
            "interval": timeframe,
            "startTime": current_start,
            "limit": 1000,
        }
        resp = requests.get(_BINANCE_URL, params=params, timeout=20)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < 1000:
            break
        # Next page starts right after the last candle's open time.
        current_start = rows[-1][0] + 1

    if not all_rows:
        raise ValueError(f"No candles returned from Binance for {full_symbol} {timeframe}")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["date"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


import numpy as np
import pandas_ta_classic as ta  # type: ignore


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same indicators the team's training pipeline used
    (see src/data/binance_data.py).
    """
    out = df.copy().sort_index()
    out["log_ret_close"] = np.log(out["close"]).diff()
    out["log_ret_vol"] = np.log(out["volume"].replace(0, np.nan)).diff()
    out["volatility"] = out["volume"] / out["volume"].shift(1).rolling(window=7).mean()
    out["rsi"] = out.ta.rsi(length=14)
    macd_full = out.ta.macd(fast=12, slow=26, signal=9)
    out["macd"] = macd_full["MACDh_12_26_9"]
    bb_full = out.ta.bbands(length=20, std=2)
    out["bollinger_bands"] = bb_full["BBP_20_2.0"]
    return out


def get_candles(
    symbol: str,
    timeframe: str = "1d",
    history_days: Optional[int] = None,
) -> pd.DataFrame:
    """Return cached candles for ``symbol``, refreshing from Binance if stale.

    The returned DataFrame is indexed by ``date`` (tz-naive) and contains:
    ``open, high, low, close, volume, log_ret_close``.
    """
    if symbol not in config.SUPPORTED_COINS:
        raise ValueError(f"Unsupported coin: {symbol}")

    history_days = history_days or config.HISTORY_DAYS

    with _lock_for(symbol, timeframe):
        cached = _read_cache(symbol, timeframe)

        today = pd.Timestamp.utcnow().normalize()
        if today.tzinfo is not None:
            today = today.tz_localize(None)

        needs_refresh = True
        merged = cached
        if cached is not None and not cached.empty:
            last_date = cached.index.max()
            age_days = (today - last_date).days
            needs_refresh = age_days >= config.CACHE_REFRESH_DAYS

        fetch_error: Exception | None = None
        if needs_refresh:
            if cached is not None and not cached.empty:
                fetch_from = cached.index.max() - timedelta(days=2)
            else:
                fetch_from = today - timedelta(days=history_days)

            start_ms = int(fetch_from.timestamp() * 1000)
            try:
                logger.info(
                    "Fetching %sUSDT %s from Binance (start_ms=%s)",
                    symbol, timeframe, start_ms,
                )
                fresh = _fetch_from_binance(symbol, timeframe, start_ms)
            except Exception as exc:
                logger.exception("Binance fetch failed for %s %s", symbol, timeframe)
                fetch_error = exc
                fresh = None

            if fresh is not None and not fresh.empty:
                if cached is None:
                    merged = fresh
                else:
                    merged = pd.concat([cached, fresh])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                _write_cache(symbol, timeframe, merged)

        if merged is None or merged.empty:
            reason = f": {fetch_error}" if fetch_error else ""
            raise RuntimeError(
                f"Could not load candles for {symbol} {timeframe}{reason}"
            )

        return _enrich(merged)


def date_bounds(symbol: str, timeframe: str = "1d") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (earliest, latest) date currently available for a coin."""
    df = get_candles(symbol, timeframe)
    return df.index.min(), df.index.max()
