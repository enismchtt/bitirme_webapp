"""Lazy-load and in-process cache for trained model checkpoints.

Checkpoints live at ``outputs/models/{model_name}.pt`` relative to the repo
root.  XGBoost payloads include the fitted wrapper object directly.  LSTM /
CNN-LSTM payloads include ``state_dict`` + ``input_size``; we reconstruct the
``nn.Module`` on first load and store it in the payload under ``"_model_obj"``.

Call ``load_all()`` at startup to detect missing files early.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import torch

import config

logger = logging.getLogger(__name__)

_TORCH_MODELS = {"lstm", "cnn_lstm", "tcn"}
_TRAIN_MODELS_DIR = config.REPO_ROOT / "train_models"

_cache: dict[str, dict[str, Any]] = {}
_train_models_bootstrapped = False


def _bootstrap_train_models_modules() -> None:
    """Load ``train_models/models.py`` without clobbering webapp ``config``.

    Training pickles reference ``models.XGBoostForecastModel``.  That module
    does ``from config import …`` which must resolve to ``train_models/config.py``,
    not ``webapp/backend/config.py`` (already imported as ``config``).
    """
    global _train_models_bootstrapped
    if _train_models_bootstrapped:
        return

    if not _TRAIN_MODELS_DIR.is_dir():
        raise FileNotFoundError(
            f"train_models directory not found: {_TRAIN_MODELS_DIR}. "
            "Expected repo layout: <repo>/train_models/models.py"
        )

    cfg_path = _TRAIN_MODELS_DIR / "config.py"
    models_path = _TRAIN_MODELS_DIR / "models.py"

    cfg_spec = importlib.util.spec_from_file_location("_train_models_config", cfg_path)
    if cfg_spec is None or cfg_spec.loader is None:
        raise ImportError(f"Cannot load train_models config from {cfg_path}")
    train_config = importlib.util.module_from_spec(cfg_spec)
    sys.modules["_train_models_config"] = train_config
    cfg_spec.loader.exec_module(train_config)

    webapp_config = sys.modules["config"]
    sys.modules["config"] = train_config
    try:
        models_spec = importlib.util.spec_from_file_location("models", models_path)
        if models_spec is None or models_spec.loader is None:
            raise ImportError(f"Cannot load train_models models from {models_path}")
        models_mod = importlib.util.module_from_spec(models_spec)
        sys.modules["models"] = models_mod
        models_spec.loader.exec_module(models_mod)
    finally:
        sys.modules["config"] = webapp_config

    _train_models_bootstrapped = True
    logger.debug("Bootstrapped train_models.models for checkpoint load")


def _reconstruct_torch(model_name: str, payload: dict[str, Any]) -> None:
    """Reconstruct nn.Module from state_dict and store under ``_model_obj``."""
    _bootstrap_train_models_modules()

    from models import create_model  # train_models/models.py  # noqa: E402

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

    # xg_boost.pt pickles XGBoostForecastModel from train_models/models.py
    _bootstrap_train_models_modules()
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
