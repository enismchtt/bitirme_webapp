"""Feature engineering — exact replica of train_dataset/derive_train_dataset.py.

All formulas are copied verbatim so that the indicator values produced here
are identical to those in the training CSV files.  Do NOT change RSI/MACD/
volatility formulas or the log-return definition without also retraining.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_INTERVAL = "1d"
_REQUEST_LIMIT = 1000


# ── Utilities ──────────────────────────────────────────────────────────────

def normalize_symbol(coin: str, quote: str = "USDT") -> str:
    coin = coin.upper().strip()
    quote = quote.upper().strip()
    return coin if coin.endswith(quote) else f"{coin}{quote}"


def to_utc_ms(date_value: str | pd.Timestamp, end_of_day: bool = False) -> int:
    """Convert a date string or Timestamp to UTC milliseconds.

    When *end_of_day* is True and the input is a plain ``YYYY-MM-DD`` string,
    the returned timestamp points to the last millisecond of that calendar day
    (inclusive end-date boundary for Binance queries).
    """
    ts = pd.Timestamp(date_value) if not isinstance(date_value, pd.Timestamp) else date_value
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    date_str = str(date_value).strip()
    if end_of_day and len(date_str) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

    return int(ts.timestamp() * 1000)


# ── Binance REST fetch ─────────────────────────────────────────────────────

def fetch_binance_klines(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily klines from Binance between two inclusive dates.

    Paginates with ``startTime`` / ``endTime`` exactly as derive_train_dataset.py
    does: ``current_start = last_close_time + 1``, 0.2 s sleep between pages.

    Returns a DataFrame with columns:
        date (string YYYY-MM-DD, UTC), symbol, open, high, low, close, volume
    Sorted ascending by date, no duplicates.
    """
    start_ms = to_utc_ms(start_date)
    end_ms = to_utc_ms(end_date, end_of_day=True)

    all_rows: list[list] = []
    current_start = start_ms

    while current_start <= end_ms:
        params = {
            "symbol": symbol,
            "interval": _INTERVAL,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": _REQUEST_LIMIT,
        }
        resp = requests.get(_BINANCE_KLINES_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        last_close_time = int(batch[-1][6])
        current_start = last_close_time + 1
        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError(f"No data from Binance for {symbol} {start_date}..{end_date}")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=columns)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df["symbol"] = symbol
    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["date", "symbol"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Technical indicators ───────────────────────────────────────────────────

def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-style RSI using exponential moving average (adjust=False)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def calculate_macd_histogram(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram = MACD line − signal line."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - signal_line


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features in-place, matching the training CSV schema.

    Input: DataFrame with at least ``close`` and ``volume`` columns.
    Output: Same DataFrame plus:
        log_ret_close, log_ret_vol, volatility, rsi, macd,
        target_log_ret_close_next_1d
    """
    out = df.copy()
    out["log_ret_close"] = np.log(out["close"] / out["close"].shift(1))
    out["log_ret_vol"] = np.log(out["volume"] / out["volume"].shift(1))
    out["volatility"] = out["volume"] / out["volume"].shift(1).rolling(window=7).mean()
    out["rsi"] = calculate_rsi(out["close"], length=14)
    out["macd"] = calculate_macd_histogram(out["close"], fast=12, slow=26, signal=9)
    out["target_log_ret_close_next_1d"] = out["log_ret_close"].shift(-1)
    return out
