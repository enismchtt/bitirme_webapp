"""Multi-model inference forecaster.

Every request runs all three models (xg_boost, lstm, cnn_lstm) and returns
per-model result series.  No online training — weights come from saved .pt
checkpoints.

Target: ``target_log_ret_close_next_1d`` = next calendar day's log_ret_close.

Window convention (mirrors training):
    For eval day T at DataFrame position t:
        X = df[feature_cols].iloc[t-30 : t]   (30 rows, ending at date T-1)
        y = log_ret_close at T = log(close[T] / close[T-1])
        predicted_close[T] = close[T-1] * exp(y_pred)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from services import binance_fetcher, inference

logger = logging.getLogger(__name__)

SEQ = config.SEQUENCE_LENGTH
MIN_HISTORY = SEQ + config.INDICATOR_WARMUP  # 56 rows before first eval day


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class HistoricalPoint:
    date: str
    actual_close: float
    actual_log_ret: float
    predicted_close: float        # 1-step prediction
    predicted_log_ret: float
    predicted_close_ar: float     # autoregressive chain
    predicted_log_ret_ar: float


@dataclass
class HistoricalModelResult:
    features: list[str]
    rmse_log_ret: float
    rmse_price: float
    direction_accuracy: float
    mape: float
    points: list[HistoricalPoint]


@dataclass
class HistoricalResult:
    coin: str
    timeframe: str
    start: str
    end: str
    training_note: str
    models: dict[str, HistoricalModelResult] = field(default_factory=dict)


@dataclass
class ForecastPoint:
    date: str
    predicted_close: float
    predicted_log_ret: float


@dataclass
class ForecastModelResult:
    features: list[str]
    points: list[ForecastPoint]


@dataclass
class ForecastResult:
    coin: str
    timeframe: str
    days: int
    last_known_date: str
    last_known_close: float
    training_note: str
    models: dict[str, ForecastModelResult] = field(default_factory=dict)


# ── Shared data loading ────────────────────────────────────────────────────

def _load_df(coin: str, feature_cols: list[str]) -> pd.DataFrame:
    """Fetch + enrich candles, drop NaN on required columns."""
    df = binance_fetcher.get_candles(coin)
    if df.empty:
        raise RuntimeError(f"No data returned for {coin}")
    df = df.replace([np.inf, -np.inf], np.nan)
    needed = list({*feature_cols, "close", "log_ret_close"})
    df = df.dropna(subset=needed).copy()
    return df


# ── Per-model historical inference ────────────────────────────────────────

def _historical_one_model(
    model_name: str,
    coin: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> HistoricalModelResult:
    feature_cols = config.BEST_FEATURE_SET[model_name]
    df = _load_df(coin, feature_cols)

    # Locate eval rows: dates strictly in [start, end].
    mask = (df.index >= start) & (df.index <= end)
    eval_dates = df.index[mask]
    if len(eval_dates) == 0:
        raise ValueError(
            f"No candles in {start.date()}..{end.date()} after dropna. "
            f"Available: {df.index.min().date()}..{df.index.max().date()}"
        )

    # Positional indices (integer positions in df, not DatetimeIndex labels).
    eval_positions = [df.index.get_loc(d) for d in eval_dates]
    t_0 = eval_positions[0]

    if t_0 < MIN_HISTORY:
        raise ValueError(
            f"Not enough history before {start.date()} for model '{model_name}'. "
            f"Need at least {MIN_HISTORY} valid rows before start "
            f"(got {t_0}). Move start date later."
        )

    # ── 1-step predictions ─────────────────────────────────────────────────
    preds_1step: list[float] = []
    for t in eval_positions:
        window = df[feature_cols].iloc[t - SEQ: t].to_numpy(dtype=np.float32)
        preds_1step.append(inference.predict_one_step(model_name, window))

    # ── Autoregressive chain ───────────────────────────────────────────────
    preds_ar = inference.roll_forward_historical_ar(
        model_name, df, feature_cols, eval_positions
    )

    # ── Assemble points ────────────────────────────────────────────────────
    # AR close is compounded from the close on the day before the first eval day.
    close_before_start = float(df["close"].iloc[t_0 - 1])
    running_ar_close = close_before_start

    points: list[HistoricalPoint] = []
    for k, t in enumerate(eval_positions):
        actual_close = float(df["close"].iloc[t])
        actual_log_ret = float(df["log_ret_close"].iloc[t])
        prev_close = float(df["close"].iloc[t - 1])

        pred_lr_1step = preds_1step[k]
        predicted_close_1step = prev_close * np.exp(pred_lr_1step)

        pred_lr_ar = preds_ar[k]
        running_ar_close *= np.exp(pred_lr_ar)

        points.append(HistoricalPoint(
            date=df.index[t].strftime("%Y-%m-%d"),
            actual_close=actual_close,
            actual_log_ret=actual_log_ret,
            predicted_close=float(predicted_close_1step),
            predicted_log_ret=float(pred_lr_1step),
            predicted_close_ar=float(running_ar_close),
            predicted_log_ret_ar=float(pred_lr_ar),
        ))

    # ── Metrics (1-step only) ──────────────────────────────────────────────
    preds_arr = np.array(preds_1step)
    actuals_arr = np.array([p.actual_log_ret for p in points])
    actual_close_arr = np.array([p.actual_close for p in points])
    pred_close_arr = np.array([p.predicted_close for p in points])

    rmse_log_ret = float(np.sqrt(np.mean((preds_arr - actuals_arr) ** 2)))
    direction_acc = float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr)))

    mask_price = (
        (actual_close_arr != 0)
        & ~np.isnan(actual_close_arr)
        & ~np.isnan(pred_close_arr)
    )
    if mask_price.any():
        rmse_price = float(np.sqrt(np.mean((actual_close_arr[mask_price] - pred_close_arr[mask_price]) ** 2)))
        mape = float(np.mean(np.abs((actual_close_arr[mask_price] - pred_close_arr[mask_price]) / actual_close_arr[mask_price])) * 100)
    else:
        rmse_price = float("nan")
        mape = float("nan")

    return HistoricalModelResult(
        features=feature_cols,
        rmse_log_ret=rmse_log_ret,
        rmse_price=rmse_price,
        direction_accuracy=direction_acc,
        mape=mape,
        points=points,
    )


# ── Per-model future inference ─────────────────────────────────────────────

def _forecast_one_model(
    model_name: str,
    coin: str,
    days: int,
    anchor_close: float,
    anchor_date: pd.Timestamp,
    df: pd.DataFrame,
) -> ForecastModelResult:
    feature_cols = config.BEST_FEATURE_SET[model_name]

    # Drop NaN on this model's features (shared df may need re-filtering).
    needed = list({*feature_cols, "close", "log_ret_close"})
    df_m = df.dropna(subset=needed).copy()

    # Compute anchor position in the (possibly filtered) per-model df.
    try:
        pos = df_m.index.get_loc(anchor_date)
    except KeyError:
        # If anchor date dropped out after dropna, fall back to last available row.
        pos = len(df_m) - 1
        anchor_close = float(df_m["close"].iloc[pos])

    if pos < SEQ:
        raise ValueError(
            f"Not enough history for model '{model_name}' anchor at {anchor_date.date()}."
        )

    pred_log_rets = inference.roll_forward_future_ar(
        model_name, df_m, feature_cols, anchor_pos=pos, steps=days
    )

    running_close = anchor_close
    points: list[ForecastPoint] = []
    for step, pred_lr in enumerate(pred_log_rets):
        running_close = running_close * float(np.exp(pred_lr))
        date = (anchor_date + pd.Timedelta(days=step + 1)).strftime("%Y-%m-%d")
        points.append(ForecastPoint(
            date=date,
            predicted_close=float(running_close),
            predicted_log_ret=float(pred_lr),
        ))

    return ForecastModelResult(features=feature_cols, points=points)


# ── Public API ─────────────────────────────────────────────────────────────

def historical_predictions(
    coin: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> HistoricalResult:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start >= end:
        raise ValueError("start must be earlier than end")

    result = HistoricalResult(
        coin=coin,
        timeframe="1d",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        training_note=config.TRAINING_NOTE,
    )

    for model_name in config.SUPPORTED_MODELS:
        logger.info("Historical inference: %s / %s", coin, model_name)
        try:
            result.models[model_name] = _historical_one_model(model_name, coin, start, end)
        except Exception as exc:
            logger.exception("Historical inference failed for %s/%s", coin, model_name)
            raise

    return result


def future_forecast(
    coin: str,
    days: int = 7,
) -> ForecastResult:
    if not (config.FORECAST_DAYS_MIN <= days <= config.FORECAST_DAYS_MAX):
        raise ValueError(f"days must be between {config.FORECAST_DAYS_MIN} and {config.FORECAST_DAYS_MAX}")

    # Load a broad df (all features union) to determine anchor.
    all_features = list({f for cols in config.BEST_FEATURE_SET.values() for f in cols})
    df_broad = _load_df(coin, all_features)

    anchor_date: pd.Timestamp = df_broad.index.max()
    anchor_close = float(df_broad["close"].iloc[-1])

    result = ForecastResult(
        coin=coin,
        timeframe="1d",
        days=days,
        last_known_date=anchor_date.strftime("%Y-%m-%d"),
        last_known_close=anchor_close,
        training_note=config.TRAINING_NOTE,
    )

    for model_name in config.SUPPORTED_MODELS:
        logger.info("Future inference: %s / %s / %d days", coin, model_name, days)
        try:
            result.models[model_name] = _forecast_one_model(
                model_name=model_name,
                coin=coin,
                days=days,
                anchor_close=anchor_close,
                anchor_date=anchor_date,
                df=df_broad,
            )
        except Exception as exc:
            logger.exception("Future inference failed for %s/%s", coin, model_name)
            raise

    return result
