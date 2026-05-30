"""Lazy-load and in-process cache for trained model checkpoints.

Checkpoints live at ``outputs/models/{model_name}.pt`` relative to the repo
root.  XGBoost payloads include the fitted wrapper object directly.  LSTM /
CNN-LSTM payloads include ``state_dict`` + ``input_size``; we reconstruct the
``nn.Module`` on first load and store it in the payload under ``"_model_obj"``.

Call ``load_all()`` at startup to detect missing files early.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch

import config

logger = logging.getLogger(__name__)

_TORCH_MODELS = {"lstm", "cnn_lstm"}
_TRAIN_MODELS_DIR = config.REPO_ROOT / "train_models"

_cache: dict[str, dict[str, Any]] = {}


def _reconstruct_torch(model_name: str, payload: dict[str, Any]) -> None:
    """Reconstruct nn.Module from state_dict and store under ``_model_obj``."""
    if str(_TRAIN_MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(_TRAIN_MODELS_DIR))

    from models import create_model  # train_models/models.py

    model = create_model(model_name, input_size=payload["input_size"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    payload["_model_obj"] = model
    logger.info("Reconstructed %s (input_size=%d)", model_name, payload["input_size"])


def _load_payload(model_name: str) -> dict[str, Any]:
    path: Path = config.MODELS_DIR / f"{model_name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. "
            "Run train_models/train.py to generate checkpoints first."
        )

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)

    if model_name in _TORCH_MODELS:
        _reconstruct_torch(model_name, payload)

    logger.info(
        "Loaded %s | features=%s | seq=%s",
        model_name,
        payload.get("feature_cols"),
        payload.get("sequence_length"),
    )
    return payload


def get(model_name: str) -> dict[str, Any]:
    """Return (cached) payload for *model_name*, loading on first call."""
    if model_name not in config.SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Supported: {config.SUPPORTED_MODELS}")
    if model_name not in _cache:
        _cache[model_name] = _load_payload(model_name)
    return _cache[model_name]


def load_all() -> None:
    """Pre-load every checkpoint.  Raises RuntimeError listing all missing files."""
    errors: list[str] = []
    for name in config.SUPPORTED_MODELS:
        try:
            get(name)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("Missing model checkpoints:\n" + "\n".join(errors))
