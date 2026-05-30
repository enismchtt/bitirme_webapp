"""XGBoost forecasting — uses the team's research-best feature set.

For 1d the best combination is ``['rsi', 'macd', 'log_ret_close']``
(see ``config.BEST_FEATURE_SET``).  We mirror the team's training pipeline
(``src/forecast.py``):

* Target = ``log_ret_close`` with 7 past lags.
* Each past covariate (``rsi``, ``macd``) gets its own 7-step lag block.
* XGBoost hyperparameters come from ``config.XGB_KWARGS`` and match
  ``darts.models.XGBModel(lags=7, lags_past_covariates=7, …)``.

We drive XGBoost directly (no ``darts.historical_forecasts``) so each
request stays in the millisecond range.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from services import binance_fetcher

logger = logging.getLogger(__name__)

LAG = config.XGB_LAGS  # 7
TARGET = "log_ret_close"


@dataclass
class HistoricalPoint:
    date: str
    actual_close: float
    predicted_close: float
    actual_log_ret: float
    predicted_log_ret: float
    # Autoregressive (compound) variant: at t = start the model sees the
    # last 7 actual log-returns, but at t > start it also feeds its own
    # earlier predictions back as input. This drifts away from the actual
    # over time but produces a "real forecast" looking line that does not
    # appear to lag the actual price.
    predicted_close_ar: float
    predicted_log_ret_ar: float


@dataclass
class HistoricalResult:
    coin: str
    timeframe: str
    points: list[HistoricalPoint]
    rmse_log_ret: float
    rmse_price: float
    direction_accuracy: float
    mape: float
    features: list[str]


@dataclass
class ForecastPoint:
    date: str
    predicted_close: float
    predicted_log_ret: float


@dataclass
class ForecastResult:
    coin: str
    timeframe: str
    last_known_date: str
    last_known_close: float
    points: list[ForecastPoint]
    features: list[str]


def _features_for(timeframe: str) -> list[str]:
    feats = config.BEST_FEATURE_SET.get(timeframe)
    if not feats:
        raise ValueError(f"No BEST_FEATURE_SET configured for timeframe={timeframe}")
    return feats


def _xgb_model() -> xgb.XGBRegressor:
    kw = config.XGB_KWARGS
    return xgb.XGBRegressor(
        n_estimators=kw["n_estimators"],
        learning_rate=kw["learning_rate"],
        max_leaves=kw["max_leaves"],
        min_child_weight=kw["min_child_weight"],
        subsample=kw["subsample"],
        colsample_bytree=kw["colsample_bytree"],
        gamma=kw["gamma"],
        reg_alpha=kw["reg_alpha"],
        reg_lambda=kw["reg_lambda"],
        random_state=kw["random_state"],
        n_jobs=kw["n_jobs"],
        tree_method="hist",
        verbosity=0,
    )


def _load_clean(coin: str, timeframe: str, needed: list[str]) -> pd.DataFrame:
    df = binance_fetcher.get_candles(coin, timeframe)
    if df.empty:
        raise RuntimeError(f"No data for {coin}")
    # +60 row warmup for indicators (RSI 14, MACD 26, BBands 20).
    df = df.tail(config.HISTORY_DAYS + 60).copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns from fetcher: {missing}")
    df = df.dropna(subset=needed)
    return df


def _build_lag_matrix(
    df: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    lag: int = LAG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X[i] = [target lags 1..L] + [cov_1 lags 1..L] + …; y[i] = target[i+L]."""
    cols = [target] + [f for f in features if f != target]
    arr = df[cols].to_numpy(dtype=np.float64)
    n, ncol = arr.shape
    if n < lag + 1:
        raise ValueError(f"Need at least {lag + 1} rows, got {n}")
    parts = [np.lib.stride_tricks.sliding_window_view(arr[:, c], lag)[:-1]
             for c in range(ncol)]
    X = np.concatenate(parts, axis=1)
    y = arr[lag:, 0]
    target_dates = df.index.to_numpy()[lag:]
    return X, y, target_dates


def historical_predictions(
    coin: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str = "1d",
) -> HistoricalResult:
    features = _features_for(timeframe)
    logger.info("Predicting %s %s with features=%s", coin, timeframe, features)

    df = _load_clean(coin, timeframe, features)
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start >= end:
        raise ValueError("start must be earlier than end")

    X, y, target_dates = _build_lag_matrix(df, features)
    close_full = df["close"].to_numpy()

    in_range = np.where(
        (target_dates >= np.datetime64(start)) & (target_dates <= np.datetime64(end))
    )[0]
    if len(in_range) == 0:
        raise ValueError(
            f"No candles fall in {start.date()} .. {end.date()}. "
            f"Available range: {pd.Timestamp(target_dates[0]).date()} "
            f".. {pd.Timestamp(target_dates[-1]).date()}"
        )

    train_end = int(in_range[0])
    if train_end < 30:
        raise ValueError(
            f"Not enough history before {start.date()} to train (need ≥30 rows)."
        )

    model = _xgb_model()
    logger.info("Training XGBoost on %d samples × %d features (before %s)",
                train_end, X.shape[1], start.date())
    model.fit(X[:train_end], y[:train_end])

    sel = in_range
    preds = model.predict(X[sel])
    actual_lr = y[sel]
    actual_close_arr = close_full[LAG:][sel]
    prev_close_arr = close_full[LAG - 1 + sel]
    predicted_close = prev_close_arr * np.exp(preds)
    point_dates = target_dates[sel]

    # --- Autoregressive variant ---
    # Build a sliding window from the actual values right before `sel[0]`
    # then roll forward feeding the model its own previous predictions.
    cols = [TARGET] + [f for f in features if f != TARGET]
    arr_full = df[cols].to_numpy(dtype=np.float64)
    # Index of the first prediction day in `arr_full`:
    first_target_idx = LAG + int(sel[0])
    # The lag window is arr_full[first_target_idx - LAG : first_target_idx]
    ar_window = arr_full[first_target_idx - LAG: first_target_idx].copy()
    preds_ar: list[float] = []
    for _ in range(len(sel)):
        feats = np.concatenate([ar_window[:, c] for c in range(ar_window.shape[1])])
        p = float(model.predict(feats.reshape(1, -1))[0])
        preds_ar.append(p)
        new_row = ar_window[-1].copy()
        new_row[0] = p
        ar_window = np.vstack([ar_window[1:], new_row])
    preds_ar_arr = np.array(preds_ar)

    # Compound close starting from the last actual close before `start`.
    start_close = float(close_full[first_target_idx - 1])
    predicted_close_ar = start_close * np.exp(np.cumsum(preds_ar_arr))

    points = [
        HistoricalPoint(
            date=pd.Timestamp(d).strftime("%Y-%m-%d"),
            actual_close=float(ac),
            predicted_close=float(pc),
            actual_log_ret=float(ar),
            predicted_log_ret=float(pr),
            predicted_close_ar=float(pca),
            predicted_log_ret_ar=float(par),
        )
        for d, ac, pc, ar, pr, pca, par in zip(
            point_dates, actual_close_arr, predicted_close, actual_lr, preds,
            predicted_close_ar, preds_ar_arr,
        )
    ]

    rmse_log_ret = float(np.sqrt(np.mean((preds - actual_lr) ** 2)))
    direction_acc = float(np.mean(np.sign(preds) == np.sign(actual_lr)))
    mask = (actual_close_arr != 0) & ~np.isnan(actual_close_arr) & ~np.isnan(predicted_close)
    if mask.any():
        rmse_price = float(np.sqrt(np.mean((actual_close_arr[mask] - predicted_close[mask]) ** 2)))
        mape = float(np.mean(np.abs((actual_close_arr[mask] - predicted_close[mask]) / actual_close_arr[mask])) * 100)
    else:
        rmse_price = float("nan")
        mape = float("nan")

    return HistoricalResult(
        coin=coin,
        timeframe=timeframe,
        points=points,
        rmse_log_ret=rmse_log_ret,
        rmse_price=rmse_price,
        direction_accuracy=direction_acc,
        mape=mape,
        features=features,
    )


def future_forecast(
    coin: str,
    days: int = 7,
    timeframe: str = "1d",
) -> ForecastResult:
    if days < 1 or days > 30:
        raise ValueError("days must be between 1 and 30")

    features = _features_for(timeframe)
    logger.info("Forecasting %s %s with features=%s for %d days", coin, timeframe, features, days)

    df = _load_clean(coin, timeframe, features)
    X, y, _ = _build_lag_matrix(df, features)

    model = _xgb_model()
    logger.info("Training XGBoost on %d samples × %d features (future)",
                len(X), X.shape[1])
    model.fit(X, y)

    cols = [TARGET] + [f for f in features if f != TARGET]
    last_window = df[cols].to_numpy(dtype=np.float64)[-LAG:].copy()

    last_known_date = df.index.max()
    last_known_close = float(df["close"].iloc[-1])

    running_close = last_known_close
    points: list[ForecastPoint] = []
    for step in range(days):
        feats = np.concatenate([last_window[:, c] for c in range(last_window.shape[1])])
        pred_lr = float(model.predict(feats.reshape(1, -1))[0])

        running_close = running_close * float(np.exp(pred_lr))
        points.append(
            ForecastPoint(
                date=(last_known_date + pd.Timedelta(days=step + 1)).strftime("%Y-%m-%d"),
                predicted_close=running_close,
                predicted_log_ret=pred_lr,
            )
        )

        # Slide: target gets the new prediction, covariates carry forward
        # (a conservative choice since we have no future RSI/MACD).
        new_row = last_window[-1].copy()
        new_row[0] = pred_lr
        last_window = np.vstack([last_window[1:], new_row])

    return ForecastResult(
        coin=coin,
        timeframe=timeframe,
        last_known_date=last_known_date.strftime("%Y-%m-%d"),
        last_known_close=last_known_close,
        points=points,
        features=features,
    )
