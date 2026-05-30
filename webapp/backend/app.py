"""FastAPI entrypoint for the crypto prediction dashboard."""
from __future__ import annotations

import logging
from dataclasses import asdict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from services import binance_fetcher, forecaster, interpreter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("webapp")

app = FastAPI(
    title="Crypto Prediction Dashboard API",
    description="Multi-model 1-day forecasting with LLM interpretation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────

class CoinInfo(BaseModel):
    symbol: str
    earliest_date: str
    latest_date: str
    latest_close: float


# Historical

class HistoricalPointOut(BaseModel):
    date: str
    actual_close: float
    actual_log_ret: float
    predicted_close: float
    predicted_log_ret: float
    predicted_close_ar: float
    predicted_log_ret_ar: float


class HistoricalModelOut(BaseModel):
    features: list[str]
    rmse_log_ret: float
    rmse_price: float
    direction_accuracy: float
    mape: float
    points: list[HistoricalPointOut]


class HistoricalResponse(BaseModel):
    coin: str
    timeframe: str
    start: str
    end: str
    training_note: str | None
    models: dict[str, HistoricalModelOut]


# Forecast

class ForecastPointOut(BaseModel):
    date: str
    predicted_close: float
    predicted_log_ret: float


class ForecastModelOut(BaseModel):
    features: list[str]
    points: list[ForecastPointOut]


class ForecastResponse(BaseModel):
    coin: str
    timeframe: str
    days: int
    last_known_date: str
    last_known_close: float
    training_note: str | None
    models: dict[str, ForecastModelOut]


# Interpret

class InterpretRequest(BaseModel):
    coin: str
    recent: list[dict] = Field(default_factory=list)
    # All three models' forecast points, keyed by model name.
    models: dict[str, list[dict]] = Field(default_factory=dict)
    last_known_close: float = 0.0


class InterpretResponse(BaseModel):
    coin: str
    interpretation: str
    provider: str
    signal: str  # BUY | HOLD | SELL


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "ollama_url": config.OLLAMA_URL,
        "ollama_model": config.OLLAMA_MODEL,
    }


@app.get("/api/coins")
def list_coins() -> dict:
    """Lightweight coin list — does NOT touch Binance."""
    return {"coins": config.SUPPORTED_COINS}


@app.get("/api/coins/{symbol}", response_model=CoinInfo)
def coin_info(symbol: str) -> CoinInfo:
    symbol = symbol.upper()
    if symbol not in config.SUPPORTED_COINS:
        raise HTTPException(status_code=404, detail=f"Unsupported coin: {symbol}")
    try:
        df = binance_fetcher.get_candles(symbol)
    except Exception as exc:
        logger.exception("Failed to load candles for %s", symbol)
        raise HTTPException(status_code=502, detail=str(exc))
    return CoinInfo(
        symbol=symbol,
        earliest_date=df.index.min().strftime("%Y-%m-%d"),
        latest_date=df.index.max().strftime("%Y-%m-%d"),
        latest_close=float(df["close"].iloc[-1]),
    )


@app.get("/api/recent")
def recent_candles(
    coin: str = Query(..., min_length=2, max_length=8),
    days: int = Query(14, ge=1, le=90),
) -> dict:
    """Raw recent candles (no inference). Used as chart context + LLM input."""
    symbol = coin.upper()
    if symbol not in config.SUPPORTED_COINS:
        raise HTTPException(status_code=404, detail=f"Unsupported coin: {symbol}")
    try:
        df = binance_fetcher.get_candles(symbol)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    tail = df.tail(days)
    return {
        "coin": symbol,
        "points": [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "close": float(row["close"]),
                "log_ret": float(row["log_ret_close"]) if pd.notna(row["log_ret_close"]) else 0.0,
            }
            for idx, row in tail.iterrows()
        ],
    }


@app.get("/api/historical", response_model=HistoricalResponse)
def historical(
    coin: str = Query(..., min_length=2, max_length=8),
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
) -> HistoricalResponse:
    symbol = coin.upper()
    if symbol not in config.SUPPORTED_COINS:
        raise HTTPException(status_code=404, detail=f"Unsupported coin: {symbol}")
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {exc}")

    try:
        result = forecaster.historical_predictions(symbol, start_ts, end_ts)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Historical inference failed for %s", symbol)
        raise HTTPException(status_code=500, detail=str(exc))

    models_out: dict[str, HistoricalModelOut] = {}
    for name, mr in result.models.items():
        models_out[name] = HistoricalModelOut(
            features=mr.features,
            rmse_log_ret=mr.rmse_log_ret,
            rmse_price=mr.rmse_price,
            direction_accuracy=mr.direction_accuracy,
            mape=mr.mape,
            points=[HistoricalPointOut(**asdict(p)) for p in mr.points],
        )

    return HistoricalResponse(
        coin=result.coin,
        timeframe=result.timeframe,
        start=result.start,
        end=result.end,
        training_note=result.training_note,
        models=models_out,
    )


@app.get("/api/forecast", response_model=ForecastResponse)
def forecast(
    coin: str = Query(..., min_length=2, max_length=8),
    days: int = Query(7, ge=1, le=30),
) -> ForecastResponse:
    symbol = coin.upper()
    if symbol not in config.SUPPORTED_COINS:
        raise HTTPException(status_code=404, detail=f"Unsupported coin: {symbol}")
    try:
        result = forecaster.future_forecast(symbol, days=days)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Forecast failed for %s", symbol)
        raise HTTPException(status_code=500, detail=str(exc))

    models_out: dict[str, ForecastModelOut] = {}
    for name, mr in result.models.items():
        models_out[name] = ForecastModelOut(
            features=mr.features,
            points=[ForecastPointOut(**asdict(p)) for p in mr.points],
        )

    return ForecastResponse(
        coin=result.coin,
        timeframe=result.timeframe,
        days=result.days,
        last_known_date=result.last_known_date,
        last_known_close=result.last_known_close,
        training_note=result.training_note,
        models=models_out,
    )


@app.post("/api/interpret", response_model=InterpretResponse)
def interpret(req: InterpretRequest) -> InterpretResponse:
    symbol = req.coin.upper()
    text, provider, signal = interpreter.interpret(
        coin=symbol,
        recent=req.recent,
        models=req.models,
        last_known_close=req.last_known_close,
    )
    return InterpretResponse(
        coin=symbol,
        interpretation=text,
        provider=provider,
        signal=signal,
    )
