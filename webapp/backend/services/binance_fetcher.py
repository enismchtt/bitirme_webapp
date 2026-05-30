"""Fetch OHLCV candles from Binance public REST API with a local CSV cache.

Uses plain ``requests`` (no python-binance Client).  Incremental refresh:
if the cached CSV's last candle is older than ``CACHE_REFRESH_DAYS``, the tail
is re-fetched and merged in.  Indicator enrichment uses ``features.add_features``
so values are identical to those in the training CSVs.
"""
from __future__ import annotations

import logging
import threading
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from services.features import add_features, fetch_binance_klines, normalize_symbol, to_utc_ms

logger = logging.getLogger(__name__)

# Per-symbol locks so concurrent requests for the same coin don't both hit
# Binance at the same time.
_locks: dict[str, threading.Lock] = {}


def _lock_for(symbol: str) -> threading.Lock:
    if symbol not in _locks:
        _locks[symbol] = threading.Lock()
    return _locks[symbol]


def _cache_path(symbol: str) -> Path:
    return config.CACHE_DIR / f"{symbol}USDT_1d.csv"


def _read_cache(symbol: str) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def _write_cache(symbol: str, df: pd.DataFrame) -> None:
    path = _cache_path(symbol)
    df.sort_index().to_csv(path, index_label="date")


def _fetch_raw(symbol: str, start_ms: int, end_ms: Optional[int] = None) -> pd.DataFrame:
    """Pull raw OHLCV from Binance and return as DatetimeIndex DataFrame."""
    full_symbol = normalize_symbol(symbol)
    start_date = pd.Timestamp(start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    end_date = (
        pd.Timestamp(end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        if end_ms
        else pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    )
    raw = fetch_binance_klines(full_symbol, start_date, end_date)
    if raw.empty:
        return pd.DataFrame()

    # Convert string date → DatetimeIndex (tz-naive UTC calendar day).
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_convert(None).dt.normalize()
    raw = raw.set_index("date").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    return raw[["open", "high", "low", "close", "volume"]]


def get_candles(
    symbol: str,
    history_days: Optional[int] = None,
) -> pd.DataFrame:
    """Return cached + enriched candles for *symbol*.

    Refreshes from Binance if stale.  The returned DataFrame is indexed by
    ``date`` (tz-naive UTC calendar days) and includes all columns from
    ``features.add_features``:
        open, high, low, close, volume,
        log_ret_close, log_ret_vol, volatility, rsi, macd,
        target_log_ret_close_next_1d
    """
    if symbol not in config.SUPPORTED_COINS:
        raise ValueError(f"Unsupported coin: {symbol}")

    history_days = history_days or config.HISTORY_DAYS

    with _lock_for(symbol):
        cached = _read_cache(symbol)

        today = pd.Timestamp.utcnow().normalize().tz_localize(None)

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
                logger.info("Fetching %sUSDT 1d from Binance (from %s)", symbol, fetch_from.date())
                fresh = _fetch_raw(symbol, start_ms)
            except Exception as exc:
                logger.exception("Binance fetch failed for %s", symbol)
                fetch_error = exc
                fresh = None

            if fresh is not None and not fresh.empty:
                merged = fresh if cached is None else pd.concat([cached[["open", "high", "low", "close", "volume"]], fresh])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                _write_cache(symbol, merged)

        if merged is None or merged.empty:
            reason = f": {fetch_error}" if fetch_error else ""
            raise RuntimeError(f"Could not load candles for {symbol}{reason}")

        # Enrich with training-parity indicators.
        ohlcv = merged[["open", "high", "low", "close", "volume"]].copy()
        return add_features(ohlcv)
