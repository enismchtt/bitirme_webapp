"""Local-Ollama-based interpretation of multi-model forecasts.

Sends all three models' forecast series to Ollama and asks for a concise
buy/sell/hold-style commentary.  The model must emit a parseable ``SIGNAL:``
line; the backend extracts BUY / HOLD / SELL for the UI.

If Ollama is not reachable, a rule-based summary and consensus signal are returned.
"""
from __future__ import annotations

import json
import logging
import re

import requests

import config

logger = logging.getLogger(__name__)

VALID_SIGNALS = frozenset({"BUY", "HOLD", "SELL"})
_SIGNAL_LINE_RE = re.compile(
    r"^\s*SIGNAL\s*:\s*(BUY|HOLD|SELL)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SIGNAL_ANYWHERE_RE = re.compile(
    r"\bSIGNAL\s*:\s*(BUY|HOLD|SELL)\b",
    re.IGNORECASE,
)


# ── Prompt builder ─────────────────────────────────────────────────────────

def _build_prompt(
    coin: str,
    recent: list[dict],
    models: dict[str, list[dict]],
    last_known_close: float,
) -> str:
    days = max((len(v) for v in models.values()), default=0)

    model_summaries: list[str] = []
    for model_name, points in models.items():
        if not points:
            continue
        final_close = points[-1]["predicted_close"]
        total_pct = (final_close - last_known_close) / last_known_close * 100
        direction = "UP" if total_pct > 0 else ("DOWN" if total_pct < 0 else "FLAT")
        daily_log_rets = [p["predicted_log_ret"] for p in points]
        avg_daily = sum(daily_log_rets) / len(daily_log_rets) * 100
        model_summaries.append(
            f"- {model_name}: {direction} {total_pct:+.2f}% over {len(points)} days "
            f"(final ${final_close:,.2f}, avg daily {avg_daily:+.3f}%)"
        )

    recent_snippet = json.dumps(
        recent[-5:] if len(recent) > 5 else recent,
        ensure_ascii=False,
        indent=2,
    )

    return f"""You are a crypto market analyst. Three ML models forecast {coin}/USDT for the next {days} days.

Last known close: ${last_known_close:,.2f}

Model predictions:
{chr(10).join(model_summaries)}

Recent actual prices (last ≤5 days):
{recent_snippet}

STRICT OUTPUT FORMAT — follow exactly:

Line 1 (required, nothing else on this line):
SIGNAL: BUY
OR
SIGNAL: SELL
OR
SIGNAL: HOLD

Choose exactly one of BUY, SELL, or HOLD for a {days}-day horizon based on the model consensus.
- BUY = models mostly predict price up; actionable lean long.
- SELL = models mostly predict price down; actionable lean short/reduce.
- HOLD = mixed or flat; no clear edge.

Lines 2 onward: Short commentary (80-120 words):
- Which models agree or disagree (use % numbers).
- Why you chose that SIGNAL for the next {days} days.
- One concrete risk.

Do not write BUY, SELL, or HOLD anywhere except on line 1 after "SIGNAL:".
End the last line with: Not financial advice.
"""


# ── Signal parsing & consensus ─────────────────────────────────────────────

def _consensus_from_models(
    models: dict[str, list[dict]],
    last_known_close: float,
) -> str:
    """Sum per-model % changes; BUY if total > threshold, SELL if < -threshold, else HOLD.

    Example: xg_boost -1.60%, lstm -2.75%, cnn_lstm +1.71%
      → sum = -2.64%  → SELL (if threshold is e.g. 0.05%)
    This avoids majority-vote ties and weights larger predicted moves properly.
    """
    if last_known_close <= 0:
        return "HOLD"

    total_pct = 0.0
    n = 0

    for points in models.values():
        if not points:
            continue
        final_close = points[-1]["predicted_close"]
        total_pct += (final_close - last_known_close) / last_known_close * 100
        n += 1

    if n == 0:
        return "HOLD"

    thr = config.SIGNAL_CONSENSUS_PCT_THRESHOLD
    logger.debug(
        "Consensus: sum_pct=%.4f%% over %d models, threshold=%.4f%%",
        total_pct, n, thr,
    )

    if total_pct > thr:
        return "BUY"
    if total_pct < -thr:
        return "SELL"
    return "HOLD"


def extract_signal(text: str, fallback: str = "HOLD") -> str:
    """Parse BUY / HOLD / SELL from LLM text; use *fallback* if missing."""
    if not text:
        return fallback if fallback in VALID_SIGNALS else "HOLD"

    m = _SIGNAL_LINE_RE.search(text)
    if not m:
        m = _SIGNAL_ANYWHERE_RE.search(text)
    if m:
        return m.group(1).upper()

    # Last resort: first standalone keyword (SELL before BUY — longer match order)
    upper = text.upper()
    for word in ("SELL", "BUY", "HOLD"):
        if re.search(rf"\b{word}\b", upper):
            logger.warning("SIGNAL: line missing; inferred %s from body text.", word)
            return word

    fb = fallback.upper() if fallback.upper() in VALID_SIGNALS else "HOLD"
    logger.warning("No SIGNAL in LLM output; using fallback %s.", fb)
    return fb


# ── Ollama call ────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    url = f"{config.OLLAMA_URL}/api/generate"
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 320},
    }
    resp = requests.post(url, json=payload, timeout=config.OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Empty response from Ollama")
    return text


# ── Rule-based fallback ────────────────────────────────────────────────────

def _rule_based(
    coin: str,
    recent: list[dict],
    models: dict[str, list[dict]],
    last_known_close: float,
) -> tuple[str, str]:
    """Return ``(markdown_body, signal)``."""
    if not models or all(not v for v in models.values()):
        return "No forecast data available — cannot produce an interpretation.", "HOLD"

    signal = _consensus_from_models(models, last_known_close)
    days = max((len(v) for v in models.values()), default=0)

    lines: list[str] = [
        f"### {coin}/USDT · Model consensus\n",
        f"Rule-based consensus over **{days} days**: models imply **{signal}**.\n",
    ]
    votes_up = 0
    votes_total = 0

    for model_name, points in models.items():
        if not points:
            continue
        final_close = points[-1]["predicted_close"]
        pct = (final_close - last_known_close) / last_known_close * 100 if last_known_close else 0
        direction = "▲ UP" if pct > 0 else ("▼ DOWN" if pct < 0 else "→ FLAT")
        lines.append(f"- **{model_name}**: {direction} **{pct:+.2f}%** → ${final_close:,.2f}")
        if pct > 0:
            votes_up += 1
        votes_total += 1

    lines.append(
        f"\n({votes_up}/{votes_total} models project higher close vs last known ${last_known_close:,.2f}.)"
    )
    lines.append("\n_⚠ Ollama not reachable — rule-based summary. Start Ollama for LLM commentary._")
    lines.append("_⚠ Not financial advice._")
    return "\n".join(lines), signal


# ── Public entry point ─────────────────────────────────────────────────────

def interpret(
    coin: str,
    recent: list[dict],
    models: dict[str, list[dict]],
    last_known_close: float,
) -> tuple[str, str, str]:
    """Return ``(markdown_text, provider, signal)`` where signal is BUY|HOLD|SELL."""
    fallback_signal = _consensus_from_models(models, last_known_close)
    prompt = _build_prompt(coin, recent, models, last_known_close)

    try:
        text = _call_ollama(prompt)
        signal = extract_signal(text, fallback=fallback_signal)
        logger.info(
            "Ollama interpretation OK for %s (%s) → signal=%s",
            coin, config.OLLAMA_MODEL, signal,
        )
        return text, f"ollama/{config.OLLAMA_MODEL}", signal
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable at %s — using rule-based fallback.", config.OLLAMA_URL)
        text, signal = _rule_based(coin, recent, models, last_known_close)
        return text, "rule-based (ollama unreachable)", signal
    except Exception as exc:
        logger.warning("Ollama call failed (%s) — using rule-based fallback.", exc)
        text, signal = _rule_based(coin, recent, models, last_known_close)
        return text, f"rule-based (error: {exc})", signal
