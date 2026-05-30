from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from config import CNN_LSTM_PARAMS, LSTM_PARAMS, XGBOOST_PARAMS


class XGBoostForecastModel:
    """
    XGBoost wrapper.

    This is not a PyTorch nn.Module, but it is still defined in models.py
    so model definitions are centralized.

    Input shape:
        [samples, 30 * num_features]

    The dataset code creates this by flattening:
        [samples, 30, num_features]
    """

    def __init__(self, **params: Any):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            ) from exc

        model_params = dict(XGBOOST_PARAMS)
        model_params.update(params)

        self.model = XGBRegressor(**model_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save_native(self, path: str) -> None:
        self.model.save_model(path)


class LSTMForecastModel(nn.Module):
    """
    LSTM model.

    Input shape:
        [batch_size, 30, input_size]

    Output shape:
        [batch_size, 1]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()

        hidden_size = hidden_size if hidden_size is not None else LSTM_PARAMS["hidden_size"]
        num_layers = num_layers if num_layers is not None else LSTM_PARAMS["num_layers"]
        dropout = dropout if dropout is not None else LSTM_PARAMS["dropout"]

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.head(last_step)


class CNNLSTMForecastModel(nn.Module):
    """
    1D CNN + LSTM model.

    Input shape:
        [batch_size, 30, input_size]

    The CNN reads short local temporal patterns,
    then the LSTM reads the transformed sequence.
    """

    def __init__(
        self,
        input_size: int,
        conv_channels: int | None = None,
        kernel_size: int | None = None,
        lstm_hidden_size: int | None = None,
        lstm_layers: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()

        conv_channels = (
            conv_channels
            if conv_channels is not None
            else CNN_LSTM_PARAMS["conv_channels"]
        )
        kernel_size = (
            kernel_size
            if kernel_size is not None
            else CNN_LSTM_PARAMS["kernel_size"]
        )
        lstm_hidden_size = (
            lstm_hidden_size
            if lstm_hidden_size is not None
            else CNN_LSTM_PARAMS["lstm_hidden_size"]
        )
        lstm_layers = (
            lstm_layers
            if lstm_layers is not None
            else CNN_LSTM_PARAMS["lstm_layers"]
        )
        dropout = dropout if dropout is not None else CNN_LSTM_PARAMS["dropout"]

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch, time, features]
        x = x.transpose(1, 2)
        # After transpose: [batch, features, time]

        x = self.conv(x)
        # After Conv1D: [batch, conv_channels, time]

        x = x.transpose(1, 2)
        # Back to: [batch, time, conv_channels]

        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]

        return self.head(last_step)


def create_model(model_name: str, input_size: int | None = None):
    """
    Factory for all models.

    XGBoost:
        create_model("xg_boost")

    LSTM:
        create_model("lstm", input_size=2)

    CNN-LSTM:
        create_model("cnn_lstm", input_size=2)
    """
    if model_name == "xg_boost":
        return XGBoostForecastModel()

    if model_name == "lstm":
        if input_size is None:
            raise ValueError("input_size is required for LSTM.")
        return LSTMForecastModel(input_size=input_size)

    if model_name == "cnn_lstm":
        if input_size is None:
            raise ValueError("input_size is required for CNN-LSTM.")
        return CNNLSTMForecastModel(input_size=input_size)

    raise ValueError(f"Unknown model_name: {model_name}")
