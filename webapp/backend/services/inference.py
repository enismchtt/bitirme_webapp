"""Core inference helpers: windowed prediction + autoregressive roll-forward.

Window convention (mirrors train_models/dataset.py ``make_sequence_arrays``):
    For evaluation day T at DataFrame position t:
        X = df[feature_cols].iloc[t-30 : t]   ← 30 rows, last row = date T-1
        y = predicted log_ret_close[T]          ← log(close[T] / close[T-1])
        predicted_close[T] = close[T-1] * exp(y_pred)

The shared ``_ar_predict`` helper drives both the historical AR overlay and
the future autoregressive forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import config
from services import model_registry

SEQ = config.SEQUENCE_LENGTH


# ── Low-level prediction helpers ──────────────────────────────────────────

def _scale(window: np.ndarray, feature_scaler) -> np.ndarray:
    """Apply feature_scaler to a [SEQ, F] window; return same shape."""
    return feature_scaler.transform(window)


def _forward(model_name: str, payload: dict, X_scaled: np.ndarray) -> float:
    """Run model on a scaled [SEQ, F] window; return raw scaled output."""
    if model_name == "xg_boost":
        X_flat = X_scaled.flatten().reshape(1, -1)
        return float(payload["model"].predict(X_flat)[0])
    else:
        model = payload["_model_obj"]
        tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # [1, SEQ, F]
        with torch.no_grad():
            return float(model(tensor).squeeze())


def _inverse(pred_scaled: float, target_scaler) -> float:
    return float(target_scaler.inverse_transform([[pred_scaled]])[0, 0])


def predict_one_step(
    model_name: str,
    window_unscaled: np.ndarray,
) -> float:
    """Predict next-day log return from a [SEQ, F] unscaled feature window.

    Returns the predicted ``log_ret_close`` in real (unscaled) units.
    """
    payload = model_registry.get(model_name)
    X_scaled = _scale(window_unscaled, payload["feature_scaler"])
    raw = _forward(model_name, payload, X_scaled)
    return _inverse(raw, payload["target_scaler"])


# ── Autoregressive roll-forward ────────────────────────────────────────────

def _ar_predict(
    model_name: str,
    initial_window: np.ndarray,   # [SEQ, F] unscaled — same as 1-step window for first eval day
    anchor_features: np.ndarray,  # [F] features of the "anchor" day (first eval or last known)
    log_ret_col_idx: int,
    steps: int,
    first_slide_uses_actual: bool,
) -> list[float]:
    """Core AR loop.  Returns a list of *steps* predicted log returns.

    Slide rule:
    - Step 0 slide: if *first_slide_uses_actual* is True, add anchor_features
      as-is (the real first-eval-day features).  This keeps the AR and 1-step
      predictions identical for the first two output days.
    - Step k (k ≥ 1) slide: add anchor_features with log_ret_close overridden
      by the *previous* step's prediction (frozen covariates, predicted return).

    ``first_slide_uses_actual=True``  → historical AR (T_0 is a real date).
    ``first_slide_uses_actual=False`` → future AR (L+1 is entirely unknown).
    """
    payload = model_registry.get(model_name)
    window = initial_window.copy()
    pred_log_rets: list[float] = []
    prev_pred: float | None = None

    for step in range(steps):
        X_scaled = _scale(window, payload["feature_scaler"])
        raw = _forward(model_name, payload, X_scaled)
        pred_lr = _inverse(raw, payload["target_scaler"])
        pred_log_rets.append(pred_lr)

        # Build the row to slide into the window.
        if step == 0 and first_slide_uses_actual:
            new_row = anchor_features.copy()
        else:
            new_row = anchor_features.copy()
            # Override log_ret_close with the PREVIOUS step's prediction.
            prev_val = anchor_features[log_ret_col_idx] if prev_pred is None else prev_pred
            new_row[log_ret_col_idx] = prev_val

        window = np.vstack([window[1:], new_row])
        prev_pred = pred_lr

    return pred_log_rets


def roll_forward_historical_ar(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    eval_positions: list[int],
) -> list[float]:
    """Historical AR chain for a list of consecutive df row positions.

    *eval_positions* must be sorted ascending.  The first eval position t_0
    must satisfy ``t_0 >= SEQ + 1`` so there are enough rows before it.

    Returns one predicted log_ret per position (same length as eval_positions).
    """
    t_0 = eval_positions[0]
    lr_col_idx = feature_cols.index("log_ret_close")

    # Initial window = same 30 rows used by 1-step at t_0.
    initial_window = df[feature_cols].iloc[t_0 - SEQ: t_0].to_numpy(dtype=np.float32)
    # Anchor = actual features of the first eval day.
    anchor_features = df[feature_cols].iloc[t_0].to_numpy(dtype=np.float32)

    return _ar_predict(
        model_name=model_name,
        initial_window=initial_window,
        anchor_features=anchor_features,
        log_ret_col_idx=lr_col_idx,
        steps=len(eval_positions),
        first_slide_uses_actual=True,
    )


def roll_forward_future_ar(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    anchor_pos: int,
    steps: int,
) -> list[float]:
    """Future AR chain starting from the last known row (anchor_pos = N).

    Predicts log returns for N+1, N+2, …, N+steps.

    Returns a list of *steps* predicted log returns.
    """
    lr_col_idx = feature_cols.index("log_ret_close")

    # Initial window = last SEQ rows INCLUDING the anchor day L.
    initial_window = df[feature_cols].iloc[anchor_pos - SEQ + 1: anchor_pos + 1].to_numpy(dtype=np.float32)
    # Anchor features = actual row at L (frozen covariate source for future rows).
    anchor_features = df[feature_cols].iloc[anchor_pos].to_numpy(dtype=np.float32)

    return _ar_predict(
        model_name=model_name,
        initial_window=initial_window,
        anchor_features=anchor_features,
        log_ret_col_idx=lr_col_idx,
        steps=steps,
        first_slide_uses_actual=False,
    )
