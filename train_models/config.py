from pathlib import Path


# ============================================================
# DATASET PATHS
# ============================================================

BTC_DATASET_PATH = "./train_dataset/BTC_dataset.csv"
ETH_DATASET_PATH = "./train_dataset/ETH_dataset.csv"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path("outputs/models")


# ============================================================
# FIXED FORECASTING SETUP
# ============================================================

# Every model uses the last 30 days to predict next day's log_ret_close.
SEQUENCE_LENGTH = 30

TARGET_COL = "target_log_ret_close_next_1d"


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Split is done separately per coin by time.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


# ============================================================
# TRAINING SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 32

EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 8


# ============================================================
# FEATURE SETS
# ============================================================

BEST_FEATURE_SET: dict[str, list[str]] = {
    "xg_boost": ["rsi", "macd", "log_ret_close"],
    "lstm": ["rsi", "log_ret_close"],
    "cnn_lstm": ["volatility", "log_ret_close"],
    "tcn": ["log_ret_close"],
}


# Choose which models to train.
MODELS_TO_TRAIN = [
    "xg_boost",
    "lstm",
    "cnn_lstm",
    "tcn",
]


# Optional.
# False means models use exactly BEST_FEATURE_SET columns.
# True adds coin_BTC and coin_ETH to every timestep.
INCLUDE_COIN_IDENTITY = False


# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": -1,
}

LSTM_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.10,
}

CNN_LSTM_PARAMS = {
    "conv_channels": 32,
    "kernel_size": 3,
    "lstm_hidden_size": 64,
    "lstm_layers": 1,
    "dropout": 0.10,
}

TCN_PARAMS = {
        "num_filters": 16,
        "kernel_size": 5,
        "num_layers": 4,
        "dilation_base": 4,
        "dropout": 0.10,
    }