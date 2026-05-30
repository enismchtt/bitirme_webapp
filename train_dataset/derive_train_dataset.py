#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1d"
REQUEST_LIMIT = 1000


def normalize_symbol(coin: str, quote_asset: str = "USDT") -> str:
    value = coin.upper().strip()
    quote_asset = quote_asset.upper().strip()

    if value.endswith(quote_asset):
        return value

    return f"{value}{quote_asset}"


def to_utc_ms(date_value: str, end_of_day: bool = False) -> int:
    ts = pd.Timestamp(date_value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    if end_of_day and len(date_value.strip()) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

    return int(ts.timestamp() * 1000)


def fetch_binance_klines(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    start_ms = to_utc_ms(start_date)
    end_ms = to_utc_ms(end_date, end_of_day=True)

    all_rows = []
    current_start = start_ms

    while current_start <= end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": REQUEST_LIMIT,
        }

        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        all_rows.extend(batch)

        last_close_time = int(batch[-1][6])
        current_start = last_close_time + 1

        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError(f"No data returned for {symbol} between {start_date} and {end_date}")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(all_rows, columns=columns)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df["symbol"] = symbol

    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["date", "symbol"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
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
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()

    macd_hist = macd_line - signal_line

    return macd_hist


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["log_ret_close"] = np.log(out["close"] / out["close"].shift(1))
    out["log_ret_vol"] = np.log(out["volume"] / out["volume"].shift(1))

    out["volatility"] = (
        out["volume"] / out["volume"].shift(1).rolling(window=7).mean()
    )

    out["rsi"] = calculate_rsi(out["close"], length=14)

    # Same idea as pandas_ta:
    # macd_full = out.ta.macd(fast=12, slow=26, signal=9)
    # out["macd"] = macd_full["MACDh_12_26_9"]
    out["macd"] = calculate_macd_histogram(out["close"], fast=12, slow=26, signal=9)

    # Useful supervised-learning target:
    # row t predicts next day's log_ret_close
    out["target_log_ret_close_next_1d"] = out["log_ret_close"].shift(-1)

    return out


def build_dataset(
    coin: str,
    start_date: str,
    end_date: str,
    quote_asset: str = "USDT",
    drop_na: bool = False,
) -> pd.DataFrame:
    symbol = normalize_symbol(coin, quote_asset)

    df = fetch_binance_klines(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    df = add_features(df)

    if drop_na:
        df = df.dropna().reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", required=True, help="Example: BTC, ETH, BTCUSDT")
    parser.add_argument("--start", default="2020-08-22", help="Example: 2020-08-22")
    parser.add_argument("--end", default="2023-04-15", help="Example: 2023-04-15")
    parser.add_argument("--quote", default="USDT")
    parser.add_argument("--output", default=None)
    parser.add_argument("--drop-na", action="store_true")

    args = parser.parse_args()

    symbol = normalize_symbol(args.coin, args.quote)

    output_path = args.output or f"{symbol}_1d_{args.start}_to_{args.end}.csv"

    df = build_dataset(
        coin=args.coin,
        start_date=args.start,
        end_date=args.end,
        quote_asset=args.quote,
        drop_na=args.drop_na,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()