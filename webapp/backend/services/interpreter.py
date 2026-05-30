"""Gemini-based interpretation of a coin's recent history and 7-day forecast."""
from __future__ import annotations

import json
import logging

import config

logger = logging.getLogger(__name__)

try:  # The import is optional at runtime if the key is missing.
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore


def _build_prompt(coin: str, recent: list[dict], forecast: list[dict]) -> str:
    return f"""You are a cryptocurrency market analyst. Write a concise, balanced,
non-financial-advice interpretation in English of the upcoming {len(forecast)}-day
forecast for {coin}/USDT. Be specific with numbers and percent changes; mention
volatility, direction, and any noteworthy pattern.

Recent actual prices (last days):
{json.dumps(recent, ensure_ascii=False, indent=2)}

Model forecast (next {len(forecast)} days, autoregressive XGBoost):
{json.dumps(forecast, ensure_ascii=False, indent=2)}

Use Markdown with these section headings:

## Overview
1-2 sentence summary.

## Forecast detail
Bullet points (day-by-day or grouped). Include the expected total % change
and a comment on daily volatility.

## Risks and notes
Mention the model's limitations (XGBoost, lags=7, log-returns only), market
surprises that aren't modeled, and add a "this is not financial advice" note.

Keep it crisp: 180–250 words total.
"""


def _rule_based(coin: str, recent: list[dict], forecast: list[dict]) -> str:
    if not forecast:
        return "No forecast data is available, so no interpretation can be produced."
    start = forecast[0]["predicted_close"]
    end = forecast[-1]["predicted_close"]
    pct = (end - start) / start * 100 if start else 0.0
    direction = "upward" if pct > 0 else ("downward" if pct < 0 else "sideways")
    daily_changes = [p["predicted_log_ret"] for p in forecast]
    avg = sum(daily_changes) / len(daily_changes)
    vol = (
        (sum((x - avg) ** 2 for x in daily_changes) / len(daily_changes)) ** 0.5 * 100
    )
    last_actual = recent[-1]["close"] if recent else start
    gap_pct = (start - last_actual) / last_actual * 100 if last_actual else 0
    return (
        f"## Overview\n"
        f"For {coin}/USDT, the XGBoost model projects an approximately "
        f"**{pct:+.2f}%** {direction} move over the next {len(forecast)} days. "
        f"The gap between the last actual close and the first forecast point "
        f"is {gap_pct:+.2f}%.\n\n"
        f"## Forecast detail\n"
        f"- First forecast: {start:,.4f} USDT\n"
        f"- Final forecast: {end:,.4f} USDT\n"
        f"- Mean daily log return: {avg*100:+.3f}%\n"
        f"- Daily volatility (std of log returns): {vol:.3f}%\n\n"
        f"## Risks and notes\n"
        f"This output was produced by an XGBoost model with a 7-day lag, trained "
        f"only on historical log returns. News, liquidity, and regulatory shocks "
        f"are not reflected. **This is not financial advice.**\n\n"
        f"_(For richer commentary, set GEMINI_API_KEY in backend/.env.)_"
    )


def interpret(coin: str, recent: list[dict], forecast: list[dict]) -> str:
    """Return a markdown interpretation; falls back to a rule-based summary."""
    if not config.GEMINI_API_KEY or genai is None:
        logger.info("Using rule-based interpreter (no Gemini key).")
        return _rule_based(coin, recent, forecast)

    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        prompt = _build_prompt(coin, recent, forecast)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("Empty Gemini response")
        return text
    except Exception as exc:
        logger.warning("Gemini call failed (%s); falling back to rule-based.", exc)
        return _rule_based(coin, recent, forecast) + f"\n\n> Note: Gemini call failed ({exc})."
