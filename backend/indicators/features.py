"""Feature engineering on canonical OHLCV frames for the Trading AI Platform.

PAPER TRADING ONLY — research features; nothing here places orders.

``add_features`` enriches a canonical OHLCV frame with EXACTLY the 17
canonical feature columns defined in CONTRACTS.md; ``feature_summary``
produces a JSON-safe last-row snapshot (plus derived qualitative flags) used
to build the AI analyst prompt. Both are pure pandas/numpy with strictly no
lookahead: every feature at row ``t`` depends only on data at rows ``<= t``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from backend.indicators.technical import (
    atr,
    bollinger,
    ema,
    macd,
    momentum,
    returns,
    rsi,
    sma,
    volatility,
    vwap,
)

logger = logging.getLogger(__name__)

__all__ = ["FEATURE_COLUMNS", "add_features", "feature_summary"]

#: The 17 canonical feature columns, in canonical order (see CONTRACTS.md).
FEATURE_COLUMNS: tuple[str, ...] = (
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "atr_14",
    "vwap",
    "ret_1",
    "ret_5",
    "volatility_20",
    "momentum_10",
)

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` enriched with the 17 canonical feature columns.

    Added columns (exactly these, no others): ``sma_20, sma_50, ema_12,
    ema_26, rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_mid, bb_lower,
    atr_14, vwap, ret_1, ret_5, volatility_20, momentum_10``.

    All features are computed with backward-looking operations only
    (rolling / ewm / shift / per-day cumulative sums), so there is no
    lookahead: recomputing on a truncated frame yields identical values for
    the retained rows. Early rows contain NaN while indicators warm up.

    Args:
        df: Canonical OHLCV frame (UTC tz-aware ``DatetimeIndex``, float64
            columns ``open, high, low, close, volume``).

    Returns:
        A new DataFrame — the original is never mutated — with the same index
        and the input columns plus the 17 feature columns (float64).

    Raises:
        ValueError: If ``df`` is missing any required OHLCV column.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"add_features: DataFrame missing required columns: {missing}")

    out = df.copy()
    if out.empty:
        logger.warning("add_features called with an empty DataFrame")
        for col in FEATURE_COLUMNS:
            out[col] = pd.Series(dtype="float64", index=out.index)
        return out

    close = out["close"]

    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)
    out["rsi_14"] = rsi(close, 14)

    macd_line, macd_sig, macd_hist = macd(close, fast=12, slow=26, signal=9)
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    bb_upper, bb_mid, bb_lower = bollinger(close, window=20, num_std=2.0)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower

    out["atr_14"] = atr(out, 14)
    out["vwap"] = vwap(out)
    out["ret_1"] = returns(close, 1)
    out["ret_5"] = returns(close, 5)
    out["volatility_20"] = volatility(close, 20)
    out["momentum_10"] = momentum(close, 10)

    return out


def _json_safe_number(value: Any) -> float | None:
    """Round a numeric value to 6 decimals; map NaN/inf/missing to ``None``.

    Args:
        value: Raw numeric (or missing) scalar from a DataFrame row.

    Returns:
        A plain Python float rounded to 6 decimal places, or ``None`` when
        the value is missing or non-finite.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, 6)


def feature_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build a JSON-safe last-row snapshot of the canonical features.

    Intended as the structured market context handed to the AI analyst
    prompt. Includes every canonical feature column (rounded to 6 decimals,
    ``None`` for NaN), the current ``price`` (last close), and four derived
    qualitative flags:

    * ``price_vs_sma20``: ``"above"`` / ``"below"`` (close vs ``sma_20``).
    * ``macd_state``: ``"bullish"`` when ``macd > macd_signal`` else
      ``"bearish"``.
    * ``rsi_zone``: ``"oversold"`` (< 30) / ``"neutral"`` / ``"overbought"``
      (> 70).
    * ``bb_position``: ``"above_upper"``, ``"upper_half"``, ``"lower_half"``
      or ``"below_lower"`` relative to the Bollinger bands.

    Flags are ``None`` when their inputs are still NaN (warm-up rows).

    Args:
        df: Canonical OHLCV frame; if the 17 feature columns are not already
            present it is enriched via :func:`add_features` first.

    Returns:
        Dict with the 17 feature values, ``price``, and the 4 derived flags —
        every value a plain float, str or ``None`` (safe for ``json.dumps``).

    Raises:
        ValueError: If ``df`` is empty or missing required OHLCV columns.
    """
    if df.empty:
        raise ValueError("feature_summary requires a non-empty DataFrame")

    if any(col not in df.columns for col in FEATURE_COLUMNS):
        logger.debug("feature_summary: input not feature-enriched; running add_features")
        df = add_features(df)

    last = df.iloc[-1]

    summary: dict[str, Any] = {
        col: _json_safe_number(last.get(col)) for col in FEATURE_COLUMNS
    }
    price = _json_safe_number(last.get("close"))
    summary["price"] = price

    sma_20 = summary["sma_20"]
    summary["price_vs_sma20"] = (
        None
        if price is None or sma_20 is None
        else ("above" if price >= sma_20 else "below")
    )

    macd_val = summary["macd"]
    macd_sig = summary["macd_signal"]
    summary["macd_state"] = (
        None
        if macd_val is None or macd_sig is None
        else ("bullish" if macd_val > macd_sig else "bearish")
    )

    rsi_val = summary["rsi_14"]
    if rsi_val is None:
        summary["rsi_zone"] = None
    elif rsi_val < 30.0:
        summary["rsi_zone"] = "oversold"
    elif rsi_val > 70.0:
        summary["rsi_zone"] = "overbought"
    else:
        summary["rsi_zone"] = "neutral"

    bb_upper = summary["bb_upper"]
    bb_mid = summary["bb_mid"]
    bb_lower = summary["bb_lower"]
    if price is None or bb_upper is None or bb_mid is None or bb_lower is None:
        summary["bb_position"] = None
    elif price > bb_upper:
        summary["bb_position"] = "above_upper"
    elif price < bb_lower:
        summary["bb_position"] = "below_lower"
    elif price >= bb_mid:
        summary["bb_position"] = "upper_half"
    else:
        summary["bb_position"] = "lower_half"

    return summary
