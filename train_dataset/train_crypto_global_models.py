#!/usr/bin/env python3
"""
Train one global BTC+ETH forecasting model per architecture using correctly scaled data.

Expected input files by default:
    BTC_dataset.csv
    ETH_dataset.csv

Expected columns:
    date, symbol(optional), open, high, low, close, volume,
    log_ret_close, log_ret_vol, volatility, rsi, macd,
    target_log_ret_close_next_1d(optional)

The script will rebuild target_log_ret_close_next_1d per coin as:
    next day's log_ret_close = groupby(symbol)["log_ret_close"].shift(-1)

Models trained:
    1. xg_boost  -> XGBoost regressor on flattened sliding windows
    2. lstm      -> PyTorch LSTM regressor
    3. cnn_lstm  -> PyTorch 1D-CNN + LSTM regressor

Saved outputs:
    outputs/models/xg_boost.pt
    outputs/models/xg_boost.json
    outputs/models/lstm.pt
    outputs/models/cnn_lstm.pt
    outputs/results/metrics.csv

Important loading note:
    Some .pt files contain sklearn scalers and/or XGBoost objects.
    In newer PyTorch versions, load them with:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    XGBRegressor = None
    _XGB_IMPORT_ERROR = exc
else:
    _XGB_IMPORT_ERROR = None


BEST_FEATURE_SET: dict[str, list[str]] = {
    "xg_boost": ["rsi", "macd", "log_ret_close"],
    "lstm": ["rsi", "log_ret_close"],
    "cnn_lstm": ["volatility", "log_ret_close"],
}

TARGET_COL = "target_log_ret_close_next_1d"
COIN_COLS = ["coin_BTC", "coin_ETH"]


@dataclass
class PreparedData:
    model_name: str
    selected_feature_cols: list[str]
    full_feature_cols: list[str]
    target_col: str
    sequence_length: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    scaled_df: pd.DataFrame
    train_X_seq: np.ndarray
    train_y: np.ndarray
    val_X_seq: np.ndarray
    val_y: np.ndarray
    test_X_seq: np.ndarray
    test_y: np.ndarray
    train_meta: pd.DataFrame
    val_meta: pd.DataFrame
    test_meta: pd.DataFrame


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class CNNLSTMRegressor(nn.Module):
    """1D-CNN extracts short local patterns over time, LSTM models sequence dynamics."""

    def __init__(
        self,
        input_size: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            # Input to Conv1d will be [batch, features, time]
            nn.Conv1d(
                in_channels=input_size,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        x = x.transpose(1, 2)          # [batch, features, time]
        x = self.conv(x)               # [batch, conv_channels, time]
        x = x.transpose(1, 2)          # [batch, time, conv_channels]
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def normalize_symbol_from_path(path: str | Path, fallback: str) -> str:
    name = Path(path).stem.upper()
    if "BTC" in name:
        return "BTCUSDT"
    if "ETH" in name:
        return "ETHUSDT"
    return fallback


def read_coin_csv(path: str | Path, fallback_symbol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a 'date' column.")

    if "symbol" not in df.columns:
        df["symbol"] = normalize_symbol_from_path(path, fallback_symbol)
    else:
        df["symbol"] = df["symbol"].fillna(normalize_symbol_from_path(path, fallback_symbol))
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head()
        raise ValueError(f"{path} has invalid date values. Examples:\n{bad}")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_ret_close",
        "log_ret_vol",
        "volatility",
        "rsi",
        "macd",
        TARGET_COL,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_global_dataframe(btc_path: str, eth_path: str) -> pd.DataFrame:
    btc_df = read_coin_csv(btc_path, fallback_symbol="BTCUSDT")
    eth_df = read_coin_csv(eth_path, fallback_symbol="ETHUSDT")

    df = pd.concat([btc_df, eth_df], ignore_index=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    required = ["log_ret_close", "rsi", "macd", "volatility"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Replace infinities caused by log(volume/0) or division by zero.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Rebuild target safely per coin. This prevents ETH's first target from accidentally using BTC.
    df[TARGET_COL] = df.groupby("symbol")["log_ret_close"].shift(-1)

    # Coin identity is not scaled.
    df["coin_BTC"] = (df["symbol"] == "BTCUSDT").astype(np.float32)
    df["coin_ETH"] = (df["symbol"] == "ETHUSDT").astype(np.float32)

    return df


def add_time_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> pd.DataFrame:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    parts = []
    for symbol, coin_df in df.groupby("symbol", sort=False):
        coin_df = coin_df.sort_values("date").reset_index(drop=True)
        n = len(coin_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        split = np.empty(n, dtype=object)
        split[:train_end] = "train"
        split[train_end:val_end] = "val"
        split[val_end:] = "test"

        coin_df["split"] = split
        coin_df["row_in_coin"] = np.arange(n)
        parts.append(coin_df)

    return pd.concat(parts, ignore_index=True)


def build_sequences_for_split(
    df: pd.DataFrame,
    full_feature_cols: list[str],
    target_col: str,
    sequence_length: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    meta_rows: list[dict[str, Any]] = []

    for symbol, coin_df in df.groupby("symbol", sort=False):
        coin_df = coin_df.sort_values("date").reset_index(drop=True)

        features = coin_df[full_feature_cols].to_numpy(dtype=np.float32)
        targets = coin_df[target_col].to_numpy(dtype=np.float32)
        splits = coin_df["split"].to_numpy()
        dates = coin_df["date"].to_numpy()

        # end_idx is the current day. The target at end_idx is next day's log_ret_close.
        for end_idx in range(sequence_length - 1, len(coin_df)):
            if splits[end_idx] != split_name:
                continue

            start_idx = end_idx - sequence_length + 1
            X_window = features[start_idx : end_idx + 1]
            y_value = targets[end_idx]

            if np.isnan(X_window).any() or np.isnan(y_value):
                continue

            # target_date is the date whose log_ret_close is being predicted.
            # Since target is shift(-1), it is usually the next row's date.
            target_date = dates[end_idx + 1] if end_idx + 1 < len(coin_df) else pd.NaT

            X_list.append(X_window)
            y_list.append(float(y_value))
            meta_rows.append(
                {
                    "symbol": symbol,
                    "input_end_date": dates[end_idx],
                    "target_date": target_date,
                    "split": split_name,
                }
            )

    if not X_list:
        raise RuntimeError(
            f"No {split_name} sequences were created. "
            f"Try smaller --sequence-length or check NaN values."
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def prepare_data_for_model(
    raw_df: pd.DataFrame,
    model_name: str,
    selected_feature_cols: list[str],
    sequence_length: int,
    train_ratio: float,
    val_ratio: float,
    add_coin_id: bool = True,
) -> PreparedData:
    df = add_time_splits(raw_df, train_ratio=train_ratio, val_ratio=val_ratio)

    missing = [col for col in selected_feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{model_name} feature set contains missing columns: {missing}")

    full_feature_cols = selected_feature_cols + (COIN_COLS if add_coin_id else [])

    # Fit scalers only on rows in the training period and only for the selected feature set.
    train_mask = df["split"].eq("train")
    train_clean_mask = train_mask & df[selected_feature_cols + [TARGET_COL]].notna().all(axis=1)

    if train_clean_mask.sum() < sequence_length + 1:
        raise RuntimeError(
            f"Not enough clean training rows for {model_name}. "
            f"Clean train rows: {train_clean_mask.sum()}, sequence_length: {sequence_length}"
        )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    df_scaled = df.copy()
    feature_scaler.fit(df.loc[train_clean_mask, selected_feature_cols])
    target_scaler.fit(df.loc[train_clean_mask, [TARGET_COL]])

    # Transform continuous model features and target. Coin columns stay 0/1.
    df_scaled[selected_feature_cols] = feature_scaler.transform(df[selected_feature_cols])

    target_non_null = df_scaled[TARGET_COL].notna()
    df_scaled.loc[target_non_null, [TARGET_COL]] = target_scaler.transform(
        df.loc[target_non_null, [TARGET_COL]]
    )

    train_X, train_y, train_meta = build_sequences_for_split(
        df_scaled, full_feature_cols, TARGET_COL, sequence_length, "train"
    )
    val_X, val_y, val_meta = build_sequences_for_split(
        df_scaled, full_feature_cols, TARGET_COL, sequence_length, "val"
    )
    test_X, test_y, test_meta = build_sequences_for_split(
        df_scaled, full_feature_cols, TARGET_COL, sequence_length, "test"
    )

    return PreparedData(
        model_name=model_name,
        selected_feature_cols=selected_feature_cols,
        full_feature_cols=full_feature_cols,
        target_col=TARGET_COL,
        sequence_length=sequence_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        scaled_df=df_scaled,
        train_X_seq=train_X,
        train_y=train_y,
        val_X_seq=val_X,
        val_y=val_y,
        test_X_seq=test_X,
        test_y=test_y,
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta,
    )


def make_loaders(
    prepared: PreparedData,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        SequenceDataset(prepared.train_X_seq, prepared.train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceDataset(prepared.val_X_seq, prepared.val_y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        SequenceDataset(prepared.test_X_seq, prepared.test_y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                val_loss = loss_fn(preds, y_batch)
                val_losses.append(float(val_loss.item()))

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        print(f"Epoch {epoch:03d} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def inverse_target(values_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    values_scaled = np.asarray(values_scaled).reshape(-1, 1)
    return scaler.inverse_transform(values_scaled).reshape(-1)


def compute_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_scaler: StandardScaler,
) -> dict[str, float]:
    y_true = inverse_target(y_true_scaled, target_scaler)
    y_pred = inverse_target(y_pred_scaled, target_scaler)

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    direction_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "direction_accuracy": direction_acc,
        "n_samples": int(len(y_true)),
    }


def evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: StandardScaler,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            preds.append(pred)
            actuals.append(y_batch.numpy())

    y_pred_scaled = np.vstack(preds).reshape(-1, 1)
    y_true_scaled = np.vstack(actuals).reshape(-1, 1)
    metrics = compute_metrics(y_true_scaled, y_pred_scaled, target_scaler)
    return metrics, y_pred_scaled


def train_xgboost_model(
    prepared: PreparedData,
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is not installed. Install it with: pip install xgboost"
        ) from _XGB_IMPORT_ERROR

    X_train = prepared.train_X_seq.reshape(prepared.train_X_seq.shape[0], -1)
    X_val = prepared.val_X_seq.reshape(prepared.val_X_seq.shape[0], -1)
    X_test = prepared.test_X_seq.reshape(prepared.test_X_seq.shape[0], -1)

    y_train = prepared.train_y.reshape(-1)
    y_val = prepared.val_y.reshape(-1)
    y_test = prepared.test_y.reshape(-1)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_pred = model.predict(X_val).reshape(-1, 1)
    test_pred = model.predict(X_test).reshape(-1, 1)

    val_metrics = compute_metrics(y_val.reshape(-1, 1), val_pred, prepared.target_scaler)
    test_metrics = compute_metrics(y_test.reshape(-1, 1), test_pred, prepared.target_scaler)

    output_dir.mkdir(parents=True, exist_ok=True)

    # XGBoost native format is the safest for production reload.
    model.save_model(output_dir / "xg_boost.json")

    # User requested .pt. This stores the Python object plus scalers and metadata.
    # Load with: torch.load("xg_boost.pt", weights_only=False)
    torch.save(
        {
            "model_type": "xg_boost",
            "model": model,
            "selected_feature_cols": prepared.selected_feature_cols,
            "full_feature_cols": prepared.full_feature_cols,
            "target_col": prepared.target_col,
            "sequence_length": prepared.sequence_length,
            "feature_scaler": prepared.feature_scaler,
            "target_scaler": prepared.target_scaler,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        output_dir / "xg_boost.pt",
    )

    return val_metrics, test_metrics


def train_lstm_model(
    prepared: PreparedData,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    train_loader, val_loader, test_loader = make_loaders(prepared, batch_size=args.batch_size)

    model = LSTMRegressor(
        input_size=len(prepared.full_feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
    )

    model, history = train_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    val_metrics, _ = evaluate_torch_model(model, val_loader, prepared.target_scaler, device)
    test_metrics, _ = evaluate_torch_model(model, test_loader, prepared.target_scaler, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "lstm",
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": len(prepared.full_feature_cols),
                "hidden_size": args.hidden_size,
                "num_layers": args.lstm_layers,
                "dropout": args.dropout,
            },
            "selected_feature_cols": prepared.selected_feature_cols,
            "full_feature_cols": prepared.full_feature_cols,
            "target_col": prepared.target_col,
            "sequence_length": prepared.sequence_length,
            "feature_scaler": prepared.feature_scaler,
            "target_scaler": prepared.target_scaler,
            "history": history,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        output_dir / "lstm.pt",
    )

    return val_metrics, test_metrics


def train_cnn_lstm_model(
    prepared: PreparedData,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    train_loader, val_loader, test_loader = make_loaders(prepared, batch_size=args.batch_size)

    model = CNNLSTMRegressor(
        input_size=len(prepared.full_feature_cols),
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        hidden_size=args.hidden_size,
        num_layers=args.cnn_lstm_layers,
        dropout=args.dropout,
    )

    model, history = train_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    val_metrics, _ = evaluate_torch_model(model, val_loader, prepared.target_scaler, device)
    test_metrics, _ = evaluate_torch_model(model, test_loader, prepared.target_scaler, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "cnn_lstm",
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": len(prepared.full_feature_cols),
                "conv_channels": args.conv_channels,
                "kernel_size": args.kernel_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.cnn_lstm_layers,
                "dropout": args.dropout,
            },
            "selected_feature_cols": prepared.selected_feature_cols,
            "full_feature_cols": prepared.full_feature_cols,
            "target_col": prepared.target_col,
            "sequence_length": prepared.sequence_length,
            "feature_scaler": prepared.feature_scaler,
            "target_scaler": prepared.target_scaler,
            "history": history,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        output_dir / "cnn_lstm.pt",
    )

    return val_metrics, test_metrics


def save_prepared_split_preview(prepared: PreparedData, output_dir: Path) -> None:
    preview_dir = output_dir / "prepared_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "model_name": prepared.model_name,
        "selected_feature_cols": prepared.selected_feature_cols,
        "full_feature_cols": prepared.full_feature_cols,
        "target_col": prepared.target_col,
        "sequence_length": prepared.sequence_length,
        "train_X_shape": list(prepared.train_X_seq.shape),
        "val_X_shape": list(prepared.val_X_seq.shape),
        "test_X_shape": list(prepared.test_X_seq.shape),
        "train_y_shape": list(prepared.train_y.shape),
        "val_y_shape": list(prepared.val_y.shape),
        "test_y_shape": list(prepared.test_y.shape),
    }
    with open(preview_dir / f"{prepared.model_name}_dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)

    prepared.train_meta.head(20).to_csv(preview_dir / f"{prepared.model_name}_train_meta_head.csv", index=False)
    prepared.val_meta.head(20).to_csv(preview_dir / f"{prepared.model_name}_val_meta_head.csv", index=False)
    prepared.test_meta.head(20).to_csv(preview_dir / f"{prepared.model_name}_test_meta_head.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--btc-path", default="BTC_dataset.csv")
    parser.add_argument("--eth-path", default="ETH_dataset.csv")
    parser.add_argument("--output-dir", default="outputs")

    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--no-coin-id", action="store_true")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--cnn-lstm-layers", type=int, default=1)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["xg_boost", "lstm", "cnn_lstm"],
        choices=["xg_boost", "lstm", "cnn_lstm"],
        help="Which models to train.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_df = load_global_dataframe(args.btc_path, args.eth_path)
    print("Loaded rows:")
    print(raw_df.groupby("symbol").size())
    print("Date ranges:")
    print(raw_df.groupby("symbol")["date"].agg(["min", "max"]))

    all_metrics: list[dict[str, Any]] = []

    for model_name in args.models:
        print("\n" + "=" * 80)
        print(f"Preparing data for {model_name}")
        selected_features = BEST_FEATURE_SET[model_name]
        print(f"Selected features: {selected_features}")

        prepared = prepare_data_for_model(
            raw_df=raw_df,
            model_name=model_name,
            selected_feature_cols=selected_features,
            sequence_length=args.sequence_length,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            add_coin_id=not args.no_coin_id,
        )

        print(f"Full features used: {prepared.full_feature_cols}")
        print(f"Train X shape: {prepared.train_X_seq.shape}")
        print(f"Val   X shape: {prepared.val_X_seq.shape}")
        print(f"Test  X shape: {prepared.test_X_seq.shape}")
        save_prepared_split_preview(prepared, output_dir)

        if model_name == "xg_boost":
            val_metrics, test_metrics = train_xgboost_model(
                prepared=prepared,
                output_dir=models_dir,
                seed=args.seed,
            )
        elif model_name == "lstm":
            val_metrics, test_metrics = train_lstm_model(
                prepared=prepared,
                output_dir=models_dir,
                args=args,
                device=device,
            )
        elif model_name == "cnn_lstm":
            val_metrics, test_metrics = train_cnn_lstm_model(
                prepared=prepared,
                output_dir=models_dir,
                args=args,
                device=device,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"Validation metrics for {model_name}: {val_metrics}")
        print(f"Test metrics for {model_name}: {test_metrics}")

        all_metrics.append(
            {
                "model": model_name,
                "features": ",".join(selected_features),
                "full_features": ",".join(prepared.full_feature_cols),
                "sequence_length": prepared.sequence_length,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_direction_accuracy": val_metrics["direction_accuracy"],
                "val_n_samples": val_metrics["n_samples"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_direction_accuracy": test_metrics["direction_accuracy"],
                "test_n_samples": test_metrics["n_samples"],
            }
        )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Saved models to: {models_dir}")
    print(f"Saved metrics to: {metrics_path}")
    print(metrics_df)


if __name__ == "__main__":
    main()
