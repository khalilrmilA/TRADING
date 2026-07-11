"""Feature engineering on canonical OHLCV frames for the Trading AI Platform.

PAPER TRADING ONLY — research features; nothing here places orders.

``add_features`` enriches a canonical OHLCV frame with EXACTLY the 48
canonical feature columns defined in CONTRACTS.md ("Indicator Pack v4" — the
original 17 columns unchanged and first); ``feature_summary`` produces a
JSON-safe last-row snapshot (plus derived qualitative flags and a grouped
``groups`` view) used to build the AI analyst prompt. Both are pure
pandas/numpy with strictly no lookahead: every feature at row ``t`` depends
only on data at rows ``<= t``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from backend.indicators.technical import (
    adx,
    aroon,
    atr,
    bollinger,
    bollinger_bandwidth,
    bollinger_pct_b,
    cci,
    cmf,
    donchian,
    ema,
    hull_ma,
    ichimoku,
    keltner,
    macd,
    mfi,
    momentum,
    obv,
    psar,
    returns,
    roc,
    rsi,
    sma,
    stoch_rsi,
    stochastic,
    supertrend,
    trix,
    volatility,
    vwap,
    vwma,
    williams_r,
)

logger = logging.getLogger(__name__)

__all__ = ["FEATURE_COLUMNS", "add_features", "feature_summary"]

#: The 48 canonical feature columns, in canonical order (see CONTRACTS.md,
#: "Indicator Pack v4"). The original 17 columns come first, unchanged.
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
    "adx_14",
    "di_plus_14",
    "di_minus_14",
    "stoch_k",
    "stoch_d",
    "stoch_rsi_k",
    "williams_r_14",
    "cci_20",
    "roc_10",
    "mfi_14",
    "obv",
    "cmf_20",
    "vwma_20",
    "rel_volume_20",
    "supertrend_10_3",
    "supertrend_dir",
    "psar",
    "aroon_up_25",
    "aroon_down_25",
    "donchian_upper_20",
    "donchian_lower_20",
    "keltner_upper_20",
    "keltner_lower_20",
    "bb_pct_b",
    "bb_bandwidth",
    "hull_20",
    "trix_15",
    "ichimoku_tenkan",
    "ichimoku_kijun",
    "ichimoku_senkou_a",
    "ichimoku_senkou_b",
)

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` enriched with the 48 canonical feature columns.

    Added columns (exactly these, no others): the original 17 — ``sma_20,
    sma_50, ema_12, ema_26, rsi_14, macd, macd_signal, macd_hist, bb_upper,
    bb_mid, bb_lower, atr_14, vwap, ret_1, ret_5, volatility_20,
    momentum_10`` — plus the Indicator Pack v4 columns ``adx_14, di_plus_14,
    di_minus_14, stoch_k, stoch_d, stoch_rsi_k, williams_r_14, cci_20,
    roc_10, mfi_14, obv, cmf_20, vwma_20, rel_volume_20, supertrend_10_3,
    supertrend_dir, psar, aroon_up_25, aroon_down_25, donchian_upper_20,
    donchian_lower_20, keltner_upper_20, keltner_lower_20, bb_pct_b,
    bb_bandwidth, hull_20, trix_15, ichimoku_tenkan, ichimoku_kijun,
    ichimoku_senkou_a, ichimoku_senkou_b``.

    All features are computed with backward-looking operations only
    (rolling / ewm / shift / per-day cumulative sums / forward-shifted
    Ichimoku spans / recursions seeded at bar 0), so there is no lookahead:
    recomputing on a truncated frame yields identical values for the
    retained rows. Early rows contain NaN while indicators warm up.

    Args:
        df: Canonical OHLCV frame (UTC tz-aware ``DatetimeIndex``, float64
            columns ``open, high, low, close, volume``).

    Returns:
        A new DataFrame — the original is never mutated — with the same index
        and the input columns plus the 48 feature columns (float64).

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

    # --- Indicator Pack v4 (all contract defaults) ---
    adx_line, di_plus, di_minus = adx(out)
    out["adx_14"] = adx_line
    out["di_plus_14"] = di_plus
    out["di_minus_14"] = di_minus

    stoch_k_line, stoch_d_line = stochastic(out)
    out["stoch_k"] = stoch_k_line
    out["stoch_d"] = stoch_d_line

    out["stoch_rsi_k"] = stoch_rsi(close)[0]
    out["williams_r_14"] = williams_r(out)
    out["cci_20"] = cci(out)
    out["roc_10"] = roc(close)
    out["mfi_14"] = mfi(out)
    out["obv"] = obv(out)
    out["cmf_20"] = cmf(out)
    out["vwma_20"] = vwma(out)

    # rel_volume_20 is a features-level derivation (not a technical function):
    # current volume relative to its 20-bar SMA; SMA of 0 or NaN -> NaN.
    volume_sma_20 = sma(out["volume"], 20)
    out["rel_volume_20"] = out["volume"] / volume_sma_20.where(volume_sma_20 != 0.0)

    supertrend_line, supertrend_dir = supertrend(out)
    out["supertrend_10_3"] = supertrend_line
    out["supertrend_dir"] = supertrend_dir

    out["psar"] = psar(out)

    aroon_up, aroon_down = aroon(out)
    out["aroon_up_25"] = aroon_up
    out["aroon_down_25"] = aroon_down

    donchian_upper, _donchian_mid, donchian_lower = donchian(out)
    out["donchian_upper_20"] = donchian_upper
    out["donchian_lower_20"] = donchian_lower

    keltner_upper, _keltner_mid, keltner_lower = keltner(out)
    out["keltner_upper_20"] = keltner_upper
    out["keltner_lower_20"] = keltner_lower

    # Reuse the Bollinger band columns computed above (do NOT recompute).
    out["bb_pct_b"] = bollinger_pct_b(close, out["bb_upper"], out["bb_lower"])
    out["bb_bandwidth"] = bollinger_bandwidth(
        out["bb_upper"], out["bb_mid"], out["bb_lower"]
    )

    out["hull_20"] = hull_ma(close)
    out["trix_15"] = trix(close)

    tenkan, kijun, senkou_a, senkou_b = ichimoku(out)
    out["ichimoku_tenkan"] = tenkan
    out["ichimoku_kijun"] = kijun
    out["ichimoku_senkou_a"] = senkou_a
    out["ichimoku_senkou_b"] = senkou_b

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
    ``None`` for NaN), the current ``price`` (last close), and derived
    qualitative flags. The original four flags:

    * ``price_vs_sma20``: ``"above"`` / ``"below"`` (close vs ``sma_20``).
    * ``macd_state``: ``"bullish"`` when ``macd > macd_signal`` else
      ``"bearish"``.
    * ``rsi_zone``: ``"oversold"`` (< 30) / ``"neutral"`` / ``"overbought"``
      (> 70).
    * ``bb_position``: ``"above_upper"``, ``"upper_half"``, ``"lower_half"``
      or ``"below_lower"`` relative to the Bollinger bands.

    Indicator Pack v4 adds the flags ``adx_trend, di_state, stoch_zone,
    williams_zone, cci_zone, mfi_zone, obv_trend, supertrend_side,
    psar_side, aroon_state, ichimoku_state, squeeze_on,
    donchian_position`` (thresholds pinned in CONTRACTS.md) and a nested
    ``groups`` dict that repeats ONLY the qualitative flags grouped by
    family (``trend`` / ``momentum`` / ``volume`` / ``volatility``) for
    compact scanning by the LLM.

    Flags are ``None`` when their inputs are still NaN (warm-up rows).

    Args:
        df: Canonical OHLCV frame; if the 48 feature columns are not already
            present it is enriched via :func:`add_features` first.

    Returns:
        Dict with the 48 feature values, ``price``, the 17 derived flags and
        ``groups`` — every value a plain float, str, bool or ``None`` (safe
        for ``json.dumps``).

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

    # --- Indicator Pack v4 flags (thresholds pinned in CONTRACTS.md) ---
    adx_val = summary["adx_14"]
    if adx_val is None:
        summary["adx_trend"] = None
    elif adx_val >= 25.0:
        summary["adx_trend"] = "strong"
    elif adx_val < 20.0:
        summary["adx_trend"] = "weak"
    else:
        summary["adx_trend"] = "moderate"

    di_plus = summary["di_plus_14"]
    di_minus = summary["di_minus_14"]
    summary["di_state"] = (
        None
        if di_plus is None or di_minus is None
        else ("bullish" if di_plus > di_minus else "bearish")
    )

    stoch_k_val = summary["stoch_k"]
    if stoch_k_val is None:
        summary["stoch_zone"] = None
    elif stoch_k_val < 20.0:
        summary["stoch_zone"] = "oversold"
    elif stoch_k_val > 80.0:
        summary["stoch_zone"] = "overbought"
    else:
        summary["stoch_zone"] = "neutral"

    williams_val = summary["williams_r_14"]
    if williams_val is None:
        summary["williams_zone"] = None
    elif williams_val < -80.0:
        summary["williams_zone"] = "oversold"
    elif williams_val > -20.0:
        summary["williams_zone"] = "overbought"
    else:
        summary["williams_zone"] = "neutral"

    cci_val = summary["cci_20"]
    if cci_val is None:
        summary["cci_zone"] = None
    elif cci_val > 100.0:
        summary["cci_zone"] = "overbought"
    elif cci_val < -100.0:
        summary["cci_zone"] = "oversold"
    else:
        summary["cci_zone"] = "neutral"

    mfi_val = summary["mfi_14"]
    if mfi_val is None:
        summary["mfi_zone"] = None
    elif mfi_val < 20.0:
        summary["mfi_zone"] = "oversold"
    elif mfi_val > 80.0:
        summary["mfi_zone"] = "overbought"
    else:
        summary["mfi_zone"] = "neutral"

    # obv_trend compares the last OBV against the 20-bar SMA of the obv
    # column, computed on the fly here (NOT a stored feature column).
    obv_last = summary["obv"]
    obv_sma_last = _json_safe_number(
        df["obv"].rolling(20, min_periods=20).mean().iloc[-1]
    )
    summary["obv_trend"] = (
        None
        if obv_last is None or obv_sma_last is None
        else ("rising" if obv_last > obv_sma_last else "falling")
    )

    supertrend_dir_val = summary["supertrend_dir"]
    summary["supertrend_side"] = (
        None
        if supertrend_dir_val is None
        else ("bullish" if supertrend_dir_val > 0.0 else "bearish")
    )

    psar_val = summary["psar"]
    summary["psar_side"] = (
        None
        if price is None or psar_val is None
        else ("bullish" if price > psar_val else "bearish")
    )

    aroon_up_val = summary["aroon_up_25"]
    aroon_down_val = summary["aroon_down_25"]
    if aroon_up_val is None or aroon_down_val is None:
        summary["aroon_state"] = None
    elif aroon_up_val > 70.0 and aroon_down_val < 30.0:
        summary["aroon_state"] = "bullish"
    elif aroon_down_val > 70.0 and aroon_up_val < 30.0:
        summary["aroon_state"] = "bearish"
    else:
        summary["aroon_state"] = "neutral"

    senkou_a_val = summary["ichimoku_senkou_a"]
    senkou_b_val = summary["ichimoku_senkou_b"]
    if price is None or senkou_a_val is None or senkou_b_val is None:
        summary["ichimoku_state"] = None
    else:
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bot = min(senkou_a_val, senkou_b_val)
        if price > cloud_top:
            summary["ichimoku_state"] = "above_cloud"
        elif price < cloud_bot:
            summary["ichimoku_state"] = "below_cloud"
        else:
            summary["ichimoku_state"] = "in_cloud"

    keltner_upper_val = summary["keltner_upper_20"]
    keltner_lower_val = summary["keltner_lower_20"]
    summary["squeeze_on"] = (
        None
        if bb_upper is None
        or bb_lower is None
        or keltner_upper_val is None
        or keltner_lower_val is None
        else bool(bb_upper < keltner_upper_val and bb_lower > keltner_lower_val)
    )

    donchian_upper_val = summary["donchian_upper_20"]
    donchian_lower_val = summary["donchian_lower_20"]
    if price is None or donchian_upper_val is None or donchian_lower_val is None:
        summary["donchian_position"] = None
    elif price >= donchian_upper_val:
        summary["donchian_position"] = "at_upper"
    elif price <= donchian_lower_val:
        summary["donchian_position"] = "at_lower"
    elif price >= (donchian_upper_val + donchian_lower_val) / 2.0:
        summary["donchian_position"] = "upper_half"
    else:
        summary["donchian_position"] = "lower_half"

    # Compact, family-grouped view of ONLY the qualitative flags (values
    # duplicate the flat keys — pinned membership per CONTRACTS.md).
    summary["groups"] = {
        "trend": {
            "price_vs_sma20": summary["price_vs_sma20"],
            "adx_trend": summary["adx_trend"],
            "di_state": summary["di_state"],
            "supertrend_side": summary["supertrend_side"],
            "psar_side": summary["psar_side"],
            "aroon_state": summary["aroon_state"],
            "ichimoku_state": summary["ichimoku_state"],
        },
        "momentum": {
            "macd_state": summary["macd_state"],
            "rsi_zone": summary["rsi_zone"],
            "stoch_zone": summary["stoch_zone"],
            "williams_zone": summary["williams_zone"],
            "cci_zone": summary["cci_zone"],
        },
        "volume": {
            "mfi_zone": summary["mfi_zone"],
            "obv_trend": summary["obv_trend"],
        },
        "volatility": {
            "bb_position": summary["bb_position"],
            "squeeze_on": summary["squeeze_on"],
            "donchian_position": summary["donchian_position"],
        },
    }

    return summary
