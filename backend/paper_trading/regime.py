"""Market-regime, cost-gate and shadow-flag utilities for the paper traders.

PAPER TRADING ONLY — this module informs *simulated* order gating. Nothing in
this module (or anywhere in the platform) can place a real order or touch a
private exchange API key.

Pure, read-only computation: market data is read ONLY via
:func:`backend.database.db.load_ohlcv` (cache-only — never fetches from an
exchange, never writes any table). No engine, trader or LLM imports — both
traders import THIS module. Every public function is cheap and NEVER raises:
any internal failure degrades to the conservative/neutral result and logs a
warning.

Public API (see CONTRACTS.md "Improvement Pack v3", section 1):
    - ``RegimeInfo``: frozen dataclass describing a symbol's 4h trend regime.
    - ``get_regime(source, symbol, ref_timeframe)``: EMA50/EMA200 regime on
      4h-resampled 1h closes.
    - ``regime_allows(side, symbol_regime, btc_regime)``: the single shared
      long/short gate predicate.
    - ``cost_gate(atr, price, commission_pct, slippage_pct, multiple)``: is
      the expected move worth the round-trip trading cost?
    - ``shadow_flags(df, entry_ts_utc_hour)``: pure-observation diagnostics
      attached to entries (never gates anything).

All ``*_pct`` values in the cost-gate math are fractions (0.006 = 0.6%),
consistent with ``settings.commission_rate`` / ``settings.slippage_rate``.
Exception: ``pct_change_7d`` / ``pct_change_30d`` on :class:`RegimeInfo` and
``atr_pctile`` in the shadow flags are human-scale percentages (−100..100 /
0..100), because they feed prompts and dashboards.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.database.db import load_ohlcv
from backend.indicators import technical

logger = logging.getLogger(__name__)

# --- Module constants (binding values — see CONTRACTS.md) ---
REF_TIMEFRAME = "1h"                    # source candles for the regime
RESAMPLE_RULE = "4h"                    # pandas resample rule applied to the 1h closes
REGIME_LOOKBACK_1H_BARS = 1500          # load_ohlcv limit (~62 days of 1h → ~375 4h bars)
MIN_4H_BARS = 220                       # fewer usable 4h closes → regime "neutral"
EMA_FAST = 50
EMA_SLOW = 200
SHADOW_VOL_MULT = 1.5                   # volume_ok threshold vs SMA20(volume)
SHADOW_ATR_WINDOW = 100                 # bars ranked for the ATR percentile
SHADOW_ATR_BAND = (20.0, 90.0)          # atr_in_band inclusive bounds
DEAD_ZONE_HOURS = (2, 5)                # dead_zone: 2 <= hour_utc < 5

# 1h bars in 7 / 30 days, for the human-scale percentage changes.
_PCT_CHANGE_7D_BARS = 168
_PCT_CHANGE_30D_BARS = 720

# Minimum finite observations before the shadow volume / ATR flags fire.
_SHADOW_MIN_ROWS = 20


@dataclass(frozen=True)
class RegimeInfo:
    """Snapshot of a symbol's higher-timeframe (4h) trend regime.

    Attributes:
        regime: ``"uptrend"``, ``"downtrend"`` or ``"neutral"``.
        close: Last 4h close used (None when no data).
        ema50: EMA50 of the 4h closes (None when fewer than ``MIN_4H_BARS``).
        ema200: EMA200 of the 4h closes (None when fewer than ``MIN_4H_BARS``).
        bars_used: Usable 4h closes after resample + dropna.
        pct_change_7d: Percent change vs 168 1h bars ago (human %, round 4;
            None when the frame is too short).
        pct_change_30d: Percent change vs 720 1h bars ago (human %, round 4;
            None when the frame is too short).
    """

    regime: str
    close: float | None
    ema50: float | None
    ema200: float | None
    bars_used: int
    pct_change_7d: float | None
    pct_change_30d: float | None


_NEUTRAL_INFO = RegimeInfo("neutral", None, None, None, 0, None, None)


def _pct_change(closes: pd.Series, n_bars: int) -> float | None:
    """Percent change of the last close vs ``n_bars`` bars earlier.

    Args:
        closes: Ascending close series (1h bars).
        n_bars: How many bars back the reference close sits.

    Returns:
        ``(last / close[-1 - n_bars] - 1) * 100`` rounded to 4 decimals, or
        None when the series is too short or either value is unusable.
    """
    if len(closes) < n_bars + 1:
        return None
    last = float(closes.iloc[-1])
    past = float(closes.iloc[-1 - n_bars])
    if not (math.isfinite(last) and math.isfinite(past)) or past == 0.0:
        return None
    return round((last / past - 1.0) * 100.0, 4)


def _finite_pos(value: float | None) -> bool:
    """True when ``value`` is a finite number greater than zero."""
    try:
        return value is not None and math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def get_regime(source: str, symbol: str, ref_timeframe: str = "1h") -> RegimeInfo:
    """Compute the 4h EMA50/EMA200 trend regime for one symbol.

    Loads up to ``REGIME_LOOKBACK_1H_BARS`` cached 1h candles (cache only —
    never fetches), resamples the closes to 4h (``label="right"``,
    ``closed="right"``, no unclosed-candle drop — observation-grade EMA
    smoothing) and classifies:

    - ``"uptrend"`` iff ``close > ema200 AND ema50 > ema200``
    - ``"downtrend"`` iff ``close < ema200 AND ema50 < ema200``
    - ``"neutral"`` otherwise, or when fewer than ``MIN_4H_BARS`` usable 4h
      closes exist (conservative by design: neutral blocks shorts, allows
      longs).

    Never raises — any failure degrades to the all-None neutral result with a
    logged warning. Does no caching of its own: callers cache results per
    cycle/tick.

    Args:
        source: Data source name (e.g. ``"binance"``).
        symbol: Trading pair symbol (e.g. ``"BTCUSDT"``).
        ref_timeframe: Timeframe of the source candles (default ``"1h"``).

    Returns:
        A :class:`RegimeInfo`; ``RegimeInfo("neutral", None, None, None, 0,
        None, None)`` when no cached data is available.

    Example:
        >>> info = get_regime("binance", "BTCUSDT")     # doctest: +SKIP
        >>> info.regime                                 # doctest: +SKIP
        'uptrend'
        >>> regime_allows("short", info.regime, info.regime)  # doctest: +SKIP
        False
    """
    try:
        df = load_ohlcv(source, symbol, ref_timeframe, limit=REGIME_LOOKBACK_1H_BARS)
        if df.empty:
            return _NEUTRAL_INFO

        closes_1h = df["close"]
        pct_change_7d = _pct_change(closes_1h, _PCT_CHANGE_7D_BARS)
        pct_change_30d = _pct_change(closes_1h, _PCT_CHANGE_30D_BARS)

        closes_4h = (
            closes_1h.resample(RESAMPLE_RULE, label="right", closed="right")
            .last()
            .dropna()
        )
        bars_used = int(len(closes_4h))
        close = float(closes_4h.iloc[-1]) if bars_used else None

        if bars_used < MIN_4H_BARS:
            return RegimeInfo(
                "neutral", close, None, None, bars_used, pct_change_7d, pct_change_30d
            )

        ema50 = float(technical.ema(closes_4h, EMA_FAST).iloc[-1])
        ema200 = float(technical.ema(closes_4h, EMA_SLOW).iloc[-1])
        if close > ema200 and ema50 > ema200:
            regime = "uptrend"
        elif close < ema200 and ema50 < ema200:
            regime = "downtrend"
        else:
            regime = "neutral"
        return RegimeInfo(
            regime, close, ema50, ema200, bars_used, pct_change_7d, pct_change_30d
        )
    except Exception:
        logger.warning(
            "get_regime(%s, %s, %s) failed — degrading to neutral",
            source,
            symbol,
            ref_timeframe,
            exc_info=True,
        )
        return _NEUTRAL_INFO


def regime_allows(side: str, symbol_regime: str, btc_regime: str) -> bool:
    """The single shared regime gate predicate (both traders MUST use it).

    SHORT is allowed only when the symbol AND BTCUSDT are both in a 4h
    downtrend; LONG is blocked only in that same double-downtrend state.
    Every other combination allows longs and blocks shorts. Never raises.

    Args:
        side: ``"long"`` or ``"short"``.
        symbol_regime: The symbol's regime string from :func:`get_regime`.
        btc_regime: BTCUSDT's regime string from :func:`get_regime`.

    Returns:
        True when the regime permits entering in that direction.

    Example:
        >>> regime_allows("short", "downtrend", "downtrend")
        True
        >>> regime_allows("long", "downtrend", "downtrend")
        False
        >>> regime_allows("short", "neutral", "downtrend")
        False
        >>> regime_allows("long", "uptrend", "neutral")
        True
    """
    both_down = (symbol_regime == "downtrend") and (btc_regime == "downtrend")
    return both_down if side == "short" else not both_down


def cost_gate(
    atr: float | None,
    price: float | None,
    commission_pct: float,
    slippage_pct: float,
    multiple: float,
) -> dict[str, Any]:
    """Check whether the expected move clears round-trip trading costs.

    ``round_trip = 2.0 * (commission_pct + slippage_pct)`` and
    ``needed_pct = max(0.0, multiple) * round_trip``; the expected move is
    ``atr / price`` (a fraction of price, like the fee rates). ``multiple <=
    0`` disables the gate (``passes=True``, ``needed_pct=0.0``); an invalid
    ATR or price fails conservatively. Never raises.

    Args:
        atr: Last-bar ATR in price units (None/non-finite/<= 0 → no expected
            move).
        price: Last close in price units (same validity rule).
        commission_pct: One-way commission as a fraction (e.g. 0.001).
        slippage_pct: One-way slippage as a fraction (e.g. 0.0005).
        multiple: How many round-trip costs the expected move must cover;
            values <= 0 disable the gate.

    Returns:
        ``{"passes": bool, "expected_move_pct": float | None,
        "needed_pct": float}`` — all pct values are fractions, not
        percentage points.

    Example:
        >>> gate = cost_gate(atr=150.0, price=30000.0, commission_pct=0.001,
        ...                  slippage_pct=0.001, multiple=3.0)
        >>> gate["passes"], gate["expected_move_pct"], gate["needed_pct"]
        (False, 0.005, 0.012)
        >>> cost_gate(atr=None, price=30000.0, commission_pct=0.001,
        ...           slippage_pct=0.001, multiple=0.0)["passes"]
        True
    """
    try:
        expected_move_pct: float | None = None
        if _finite_pos(atr) and _finite_pos(price):
            expected_move_pct = float(atr) / float(price)  # type: ignore[arg-type]

        if float(multiple) <= 0.0:
            return {
                "passes": True,
                "expected_move_pct": expected_move_pct,
                "needed_pct": 0.0,
            }

        round_trip = 2.0 * (float(commission_pct) + float(slippage_pct))
        needed_pct = max(0.0, float(multiple)) * round_trip
        passes = expected_move_pct is not None and expected_move_pct >= needed_pct
        return {
            "passes": bool(passes),
            "expected_move_pct": expected_move_pct,
            "needed_pct": float(needed_pct),
        }
    except Exception:
        logger.warning("cost_gate failed — failing conservatively", exc_info=True)
        return {"passes": False, "expected_move_pct": None, "needed_pct": 0.0}


def shadow_flags(df: pd.DataFrame, entry_ts_utc_hour: int) -> dict[str, Any]:
    """Pure-observation entry diagnostics — never gates, never raises.

    Computed on a feature-enriched canonical frame (1h for the bot, 15m for
    the scalper) at entry time and attached to the entry's detail payload so
    later research can slice fills by conditions. Uses the ``atr_14`` column
    when present, else computes ``technical.atr(df, 14)``.

    Args:
        df: Canonical OHLCV frame, ideally feature-enriched with ``atr_14``.
        entry_ts_utc_hour: ``datetime.now(timezone.utc).hour`` at entry time.

    Returns:
        ``{"volume_ok": bool | None, "atr_pctile": float | None,
        "atr_in_band": bool | None, "dead_zone": bool}`` where:

        - ``volume_ok``: last volume >= ``SHADOW_VOL_MULT`` * SMA20(volume);
          None when volume is missing or there are fewer than 20 rows.
        - ``atr_pctile``: 100 * (# of window ATR values <= current) / n over
          the last ``min(100, available)`` finite ``atr_14`` values including
          the current bar, rounded to 2; None when fewer than 20 finite ATR
          values exist.
        - ``atr_in_band``: ``20.0 <= atr_pctile <= 90.0``; None when
          ``atr_pctile`` is None.
        - ``dead_zone``: ``2 <= entry_ts_utc_hour < 5`` (UTC).

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
        >>> frame = pd.DataFrame({"volume": [100.0] * 29 + [200.0],
        ...                       "atr_14": [1.0] * 29 + [2.0]}, index=idx)
        >>> flags = shadow_flags(frame, entry_ts_utc_hour=3)
        >>> flags["volume_ok"], flags["atr_pctile"], flags["dead_zone"]
        (True, 100.0, True)
    """
    volume_ok: bool | None = None
    atr_pctile: float | None = None
    atr_in_band: bool | None = None

    try:
        dead_zone = bool(DEAD_ZONE_HOURS[0] <= int(entry_ts_utc_hour) < DEAD_ZONE_HOURS[1])
    except Exception:
        logger.warning("shadow_flags: bad entry hour %r", entry_ts_utc_hour, exc_info=True)
        dead_zone = False

    try:
        if df is not None and "volume" in df.columns and len(df) >= _SHADOW_MIN_ROWS:
            vol = df["volume"].astype("float64")
            sma20 = float(technical.sma(vol, 20).iloc[-1])
            last = float(vol.iloc[-1])
            if math.isfinite(sma20) and math.isfinite(last):
                volume_ok = bool(last >= SHADOW_VOL_MULT * sma20)
    except Exception:
        logger.warning("shadow_flags: volume check failed", exc_info=True)
        volume_ok = None

    try:
        if df is not None and len(df) > 0:
            if "atr_14" in df.columns:
                atr_series = df["atr_14"].astype("float64")
            else:
                atr_series = technical.atr(df, 14)
            values = atr_series.to_numpy(dtype="float64")
            finite = values[np.isfinite(values)]
            if finite.size >= _SHADOW_MIN_ROWS:
                window = finite[-min(SHADOW_ATR_WINDOW, int(finite.size)):]
                current = window[-1]
                atr_pctile = round(
                    100.0 * float(np.count_nonzero(window <= current)) / float(window.size),
                    2,
                )
                atr_in_band = bool(
                    SHADOW_ATR_BAND[0] <= atr_pctile <= SHADOW_ATR_BAND[1]
                )
    except Exception:
        logger.warning("shadow_flags: ATR percentile failed", exc_info=True)
        atr_pctile = None
        atr_in_band = None

    return {
        "volume_ok": volume_ok,
        "atr_pctile": atr_pctile,
        "atr_in_band": atr_in_band,
        "dead_zone": dead_zone,
    }
