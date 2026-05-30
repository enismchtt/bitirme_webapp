from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config
from dataset import prepare_data_for_feature_set
from models import XGBoostForecastModel, create_model


TORCH_MODELS = {"lstm", "cnn_lstm"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_scaler,
) -> dict[str, float]:
    """
    Calculates validation RMSE/MAE after inverse-transforming the target.

    So values are in real log-return units, not scaled units.
    """
    y_true = target_scaler.inverse_transform(
        y_true_scaled.reshape(-1, 1)
    ).reshape(-1)

    y_pred = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(-1)

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def predict_torch_scaled(
    model: torch.nn.Module,
    loader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            batch_preds = model(X_batch).detach().cpu().numpy().reshape(-1)
            batch_actuals = y_batch.detach().cpu().numpy().reshape(-1)

            preds.append(batch_preds)
            actuals.append(batch_actuals)

    return np.concatenate(preds), np.concatenate(actuals)


def train_xgboost(prepared, output_dir: Path) -> dict[str, Any]:
    model: XGBoostForecastModel = create_model("xg_boost")

    model.fit(
        X_train=prepared.X_train_flat,
        y_train=prepared.y_train,
        X_val=prepared.X_val_flat,
        y_val=prepared.y_val,
    )

    val_pred = model.predict(prepared.X_val_flat)

    val_metrics = calculate_metrics(
        y_true_scaled=prepared.y_val,
        y_pred_scaled=val_pred,
        target_scaler=prepared.target_scaler,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    native_path = output_dir / "xg_boost_native.json"
    model.save_native(str(native_path))

    payload = {
        "model_name": "xg_boost",
        "model": model,
        "feature_scaler": prepared.feature_scaler,
        "target_scaler": prepared.target_scaler,
        "feature_cols": prepared.feature_cols,
        "target_col": prepared.target_col,
        "sequence_length": prepared.sequence_length,
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "input_format": "[samples, 30 * num_features]",
        "native_model_path": str(native_path),
    }

    save_path = output_dir / "xg_boost.pt"
    torch.save(payload, save_path)

    return {
        "model_name": "xg_boost",
        "feature_cols": prepared.feature_cols,
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "saved_path": str(save_path),
    }


def train_torch_model(
    model_name: str,
    prepared,
    output_dir: Path,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_size = len(prepared.feature_cols)
    model = create_model(model_name, input_size=input_size).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )
    criterion = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    best_val_metrics = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in prepared.train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_pred_scaled, val_true_scaled = predict_torch_scaled(
            model=model,
            loader=prepared.val_loader,
            device=device,
        )

        val_metrics = calculate_metrics(
            y_true_scaled=val_true_scaled,
            y_pred_scaled=val_pred_scaled,
            target_scaler=prepared.target_scaler,
        )

        print(
            f"{model_name} | "
            f"epoch {epoch:03d} | "
            f"train_loss={np.mean(train_losses):.6f} | "
            f"val_rmse={val_metrics['rmse']:.6f} | "
            f"val_mae={val_metrics['mae']:.6f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_val_metrics = val_metrics
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.PATIENCE:
            print(f"{model_name} | early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_val_metrics is None:
        raise RuntimeError(f"{model_name} did not train correctly.")

    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{model_name}.pt"

    payload = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "input_size": input_size,
        "feature_scaler": prepared.feature_scaler,
        "target_scaler": prepared.target_scaler,
        "feature_cols": prepared.feature_cols,
        "target_col": prepared.target_col,
        "sequence_length": prepared.sequence_length,
        "val_rmse": best_val_metrics["rmse"],
        "val_mae": best_val_metrics["mae"],
        "input_format": "[samples, 30, num_features]",
    }

    torch.save(payload, save_path)

    return {
        "model_name": model_name,
        "feature_cols": prepared.feature_cols,
        "val_rmse": best_val_metrics["rmse"],
        "val_mae": best_val_metrics["mae"],
        "saved_path": str(save_path),
    }


def save_metadata(result: dict[str, Any], output_dir: Path) -> None:
    metadata_path = output_dir / f"{result['model_name']}_metadata.json"

    metadata = {
        "model_name": result["model_name"],
        "feature_cols": result["feature_cols"],
        "sequence_length": config.SEQUENCE_LENGTH,
        "target_col": config.TARGET_COL,
        "val_rmse": result["val_rmse"],
        "val_mae": result["val_mae"],
        "saved_path": result["saved_path"],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train_one_model(model_name: str) -> dict[str, Any]:
    if model_name not in config.BEST_FEATURE_SET:
        raise ValueError(f"No feature set defined for model: {model_name}")

    print("\n" + "=" * 80)
    print(f"Preparing dataset for model: {model_name}")
    print(f"Base features: {config.BEST_FEATURE_SET[model_name]}")

    prepared = prepare_data_for_feature_set(
        btc_path=config.BTC_DATASET_PATH,
        eth_path=config.ETH_DATASET_PATH,
        base_feature_cols=config.BEST_FEATURE_SET[model_name],
        include_coin_identity=config.INCLUDE_COIN_IDENTITY,
        sequence_length=config.SEQUENCE_LENGTH,
        batch_size=config.BATCH_SIZE,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
    )

    print(f"Final features used: {prepared.feature_cols}")
    print(f"Train sequence shape: {prepared.X_train_seq.shape}")
    print(f"Val sequence shape:   {prepared.X_val_seq.shape}")

    if model_name == "xg_boost":
        result = train_xgboost(prepared, config.OUTPUT_DIR)
    elif model_name in TORCH_MODELS:
        result = train_torch_model(model_name, prepared, config.OUTPUT_DIR)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    save_metadata(result, config.OUTPUT_DIR)

    print(
        f"Saved {model_name} | "
        f"val_rmse={result['val_rmse']:.6f} | "
        f"val_mae={result['val_mae']:.6f} | "
        f"path={result['saved_path']}"
    )

    return result


def main() -> None:
    set_seed(config.SEED)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Training configuration")
    print("-" * 80)
    print(f"BTC dataset: {config.BTC_DATASET_PATH}")
    print(f"ETH dataset: {config.ETH_DATASET_PATH}")
    print(f"Output dir:  {config.OUTPUT_DIR}")
    print(f"Lookback:    {config.SEQUENCE_LENGTH} days")
    print(f"Target:      {config.TARGET_COL}")
    print(f"Models:      {config.MODELS_TO_TRAIN}")
    print(f"Coin ID:     {config.INCLUDE_COIN_IDENTITY}")

    results = []

    for model_name in config.MODELS_TO_TRAIN:
        result = train_one_model(model_name)
        results.append(result)

    results_df = pd.DataFrame(results)
    metrics_path = config.OUTPUT_DIR / "validation_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 80)
    print("Validation metrics")
    print(results_df[["model_name", "feature_cols", "val_rmse", "val_mae", "saved_path"]])
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
