# BTC + ETH Global Forecasting Training

This version uses `config.py`, not command-line arguments.

## Project files

```text
config.py   -> dataset paths, feature sets, model choices, hyperparameters
dataset.py  -> loading, global scaling, time split, sliding-window creation
models.py   -> XGBoost, LSTM, CNN-LSTM model definitions
train.py    -> training, validation RMSE/MAE, saving models
```

## Expected input CSVs

Put these files in the same folder as the Python files:

```text
BTC_dataset.csv
ETH_dataset.csv
```

Expected columns:

```text
date
open
high
low
close
volume
log_ret_close
log_ret_vol
volatility
rsi
macd
target_log_ret_close_next_1d
```

The target column can exist, but `dataset.py` recomputes it safely per coin:

```python
df[TARGET_COL] = df.groupby("symbol")["log_ret_close"].shift(-1)
```

So BTC future rows never mix with ETH future rows.

## Forecasting rule

All models use:

```text
past 30 days of selected features
        ↓
predict next day's log_ret_close
```

This is controlled in `config.py`:

```python
SEQUENCE_LENGTH = 30
TARGET_COL = "target_log_ret_close_next_1d"
```

## Feature sets

In `config.py`:

```python
BEST_FEATURE_SET = {
    "xg_boost": ["rsi", "macd", "log_ret_close"],
    "lstm": ["rsi", "log_ret_close"],
    "cnn_lstm": ["volatility", "log_ret_close"],
}
```

By default, only these fields are used.

## Optional coin identity

In `config.py`:

```python
INCLUDE_COIN_IDENTITY = False
```

If you set it to `True`, the dataset adds:

```text
coin_BTC
coin_ETH
```

to each timestep.

## Shapes

LSTM / CNN-LSTM:

```text
[samples, 30, num_features]
```

XGBoost:

```text
[samples, 30 * num_features]
```

So XGBoost also sees the same 30-day backward period, but flattened.

## Install

```bash
pip install pandas numpy scikit-learn torch xgboost
```

## Train

```bash
python train.py
```

## Outputs

Saved into:

```text
outputs/models/
```

Example output files:

```text
xg_boost.pt
xg_boost_native.json
xg_boost_metadata.json
lstm.pt
lstm_metadata.json
cnn_lstm.pt
cnn_lstm_metadata.json
validation_metrics.csv
```

Only validation metrics are used:

```text
val_rmse
val_mae
```

They are calculated after inverse-transforming the target, so the values are in real log-return units.
