"""Tests for backend/paper_trading/regime.py — Improvement Pack v3, section 1.

PAPER TRADING ONLY — these tests never touch a network socket and never call
Ollama. Market data is synthetic, seeded into the throwaway per-test SQLite
database (``conftest._fresh_db``) via ``upsert_ohlcv`` so ``get_regime`` reads
it back cache-only, exactly like production.

Covers the binding contract of the four public functions:

* ``get_regime``      — 4h uptrend / downtrend / neutral classification,
  insufficient-bars → neutral, empty cache → all-None neutral, 7d/30d
  percent changes, and the never-raises degradation rule.
* ``regime_allows``   — full truth table (shorts only in a double downtrend).
* ``cost_gate``       — fee/slippage round-trip math, exact boundary,
  disabled gate, invalid ATR/price conservatism.
* ``shadow_flags``    — volume_ok, atr_pctile window math, atr_in_band
  bounds, dead_zone hours, and small-frame/None degradation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import backend.database.db as db
from backend.database.db import upsert_ohlcv
from backend.paper_trading import regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_1h(
    closes: np.ndarray | list[float],
    symbol: str = "BTCUSDT",
    source: str = "binance",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Seed the cache with a canonical 1h OHLCV frame built from ``closes``.

    Args:
        closes: Close prices, one per hourly candle (all must stay positive).
        symbol: Symbol to store the candles under.
        source: Data source key.
        start: First candle timestamp (UTC).

    Returns:
        The seeded canonical frame.
    """
    close = np.asarray(closes, dtype=np.float64)
    n = len(close)
    index = pd.date_range(
        start, periods=n, freq="1h", tz="UTC", name="timestamp"
    ).as_unit("ms")
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 500.0),
        },
        index=index,
    ).astype("float64")
    assert (df["low"] > 0).all(), "test data must keep prices positive"
    upsert_ohlcv(source, symbol, "1h", df)
    return df


def _shadow_frame(
    atr_values: list[float] | np.ndarray,
    volumes: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """A minimal feature-enriched frame with a crafted ``atr_14`` column."""
    atr = np.asarray(atr_values, dtype=np.float64)
    n = len(atr)
    index = pd.date_range(
        "2024-03-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    close = np.full(n, 100.0)
    volume = (
        np.asarray(volumes, dtype=np.float64)
        if volumes is not None
        else np.full(n, 100.0)
    )
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    df["atr_14"] = atr
    return df


# ---------------------------------------------------------------------------
# Module constants (binding values)
# ---------------------------------------------------------------------------


def test_module_constants_match_contract() -> None:
    assert regime.REF_TIMEFRAME == "1h"
    assert regime.RESAMPLE_RULE == "4h"
    assert regime.REGIME_LOOKBACK_1H_BARS == 1500
    assert regime.MIN_4H_BARS == 220
    assert regime.EMA_FAST == 50
    assert regime.EMA_SLOW == 200
    assert regime.SHADOW_VOL_MULT == 1.5
    assert regime.SHADOW_ATR_WINDOW == 100
    assert regime.SHADOW_ATR_BAND == (20.0, 90.0)
    assert regime.DEAD_ZONE_HOURS == (2, 5)


# ---------------------------------------------------------------------------
# get_regime — classification on synthetic 4h trends
# ---------------------------------------------------------------------------


def test_get_regime_empty_cache_is_all_none_neutral() -> None:
    info = regime.get_regime("binance", "BTCUSDT")
    assert info.regime == "neutral"
    assert info.close is None
    assert info.ema50 is None
    assert info.ema200 is None
    assert info.bars_used == 0
    assert info.pct_change_7d is None
    assert info.pct_change_30d is None


def test_get_regime_uptrend() -> None:
    """A steadily rising series → close > EMA200 and EMA50 > EMA200."""
    closes = 100.0 + 0.1 * np.arange(1200)
    df = _seed_1h(closes)

    info = regime.get_regime("binance", "BTCUSDT")

    assert info.regime == "uptrend"
    assert info.bars_used >= regime.MIN_4H_BARS
    assert info.close == pytest.approx(closes[-1])
    assert info.ema50 is not None and info.ema200 is not None
    assert info.ema50 > info.ema200
    assert info.close > info.ema200

    # 7d / 30d percent changes come from the 1h closes (round 4).
    c = df["close"]
    expected_7d = (c.iloc[-1] / c.iloc[-169] - 1.0) * 100.0
    expected_30d = (c.iloc[-1] / c.iloc[-721] - 1.0) * 100.0
    assert info.pct_change_7d == pytest.approx(expected_7d, abs=1e-3)
    assert info.pct_change_30d == pytest.approx(expected_30d, abs=1e-3)


def test_get_regime_downtrend() -> None:
    """A steadily falling series → close < EMA200 and EMA50 < EMA200."""
    closes = 1000.0 - 0.5 * np.arange(1200)
    _seed_1h(closes)

    info = regime.get_regime("binance", "BTCUSDT")

    assert info.regime == "downtrend"
    assert info.bars_used >= regime.MIN_4H_BARS
    assert info.close == pytest.approx(closes[-1])
    assert info.ema50 is not None and info.ema200 is not None
    assert info.ema50 < info.ema200
    assert info.close < info.ema200
    assert info.pct_change_7d is not None and info.pct_change_7d < 0


def test_get_regime_mixed_signals_is_neutral() -> None:
    """Long rise then a sharp 8-bar crash → close < EMA200 < EMA50 → neutral.

    Verified numerically: with 1192 rising 1h bars (400 → 1000) and 8 crash
    bars at 100, the 4h EMAs are ~853 (EMA50) and ~787 (EMA200) while the
    close is 100 — neither the uptrend nor the downtrend condition holds.
    """
    closes = np.concatenate(
        [np.linspace(400.0, 1000.0, 1192), np.full(8, 100.0)]
    )
    _seed_1h(closes)

    info = regime.get_regime("binance", "BTCUSDT")

    assert info.bars_used >= regime.MIN_4H_BARS
    assert info.regime == "neutral"
    # Document WHY it is neutral: mixed close/EMA relationship.
    assert info.ema50 is not None and info.ema200 is not None
    assert info.close < info.ema200
    assert info.ema50 > info.ema200


def test_get_regime_insufficient_bars_is_neutral_with_none_emas() -> None:
    """< MIN_4H_BARS usable 4h closes → neutral, EMAs None, close still set."""
    closes = 100.0 + np.arange(400.0)  # strongly rising, but only ~101 4h bars
    df = _seed_1h(closes)

    info = regime.get_regime("binance", "BTCUSDT")

    assert 0 < info.bars_used < regime.MIN_4H_BARS
    assert info.regime == "neutral"
    assert info.ema50 is None
    assert info.ema200 is None
    assert info.close == pytest.approx(closes[-1])
    # 400 bars: enough for the 7d change (169 bars) but not 30d (721 bars).
    c = df["close"]
    expected_7d = (c.iloc[-1] / c.iloc[-169] - 1.0) * 100.0
    assert info.pct_change_7d == pytest.approx(expected_7d, abs=1e-3)
    assert info.pct_change_30d is None


def test_get_regime_never_raises_degrades_to_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any internal failure must degrade to the conservative neutral result."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated internal failure")

    if hasattr(regime, "load_ohlcv"):
        monkeypatch.setattr(regime, "load_ohlcv", _boom)
    else:  # implementation may call through the db module instead
        monkeypatch.setattr(db, "load_ohlcv", _boom)

    info = regime.get_regime("binance", "BTCUSDT")
    assert info.regime == "neutral"


# ---------------------------------------------------------------------------
# regime_allows — the single shared gate predicate
# ---------------------------------------------------------------------------

_REGIMES = ("uptrend", "downtrend", "neutral")


@pytest.mark.parametrize("btc_regime", _REGIMES)
@pytest.mark.parametrize("symbol_regime", _REGIMES)
def test_regime_allows_truth_table(symbol_regime: str, btc_regime: str) -> None:
    """Shorts ONLY in a double downtrend; longs blocked ONLY there."""
    both_down = symbol_regime == "downtrend" and btc_regime == "downtrend"
    assert regime.regime_allows("short", symbol_regime, btc_regime) == both_down
    assert regime.regime_allows("long", symbol_regime, btc_regime) == (
        not both_down
    )


# ---------------------------------------------------------------------------
# cost_gate — fee-aware expected-move math
# ---------------------------------------------------------------------------


def test_cost_gate_passes_when_move_covers_costs() -> None:
    # round_trip = 2 * (0.001 + 0.0005) = 0.003; needed = 3 * 0.003 = 0.009
    result = regime.cost_gate(1.0, 100.0, 0.001, 0.0005, 3.0)
    assert result["passes"] is True
    assert result["expected_move_pct"] == pytest.approx(0.01)
    assert result["needed_pct"] == pytest.approx(0.009)


def test_cost_gate_fails_when_move_too_small() -> None:
    result = regime.cost_gate(0.5, 100.0, 0.001, 0.0005, 3.0)
    assert result["passes"] is False
    assert result["expected_move_pct"] == pytest.approx(0.005)
    assert result["needed_pct"] == pytest.approx(0.009)


def test_cost_gate_exact_boundary_passes() -> None:
    """expected == needed → passes (>= comparison), float-exact by design.

    Powers of two keep every value exactly representable:
    needed = 4 * 2 * (2**-10 + 0) = 2**-7; expected = 1 / 128 = 2**-7.
    """
    result = regime.cost_gate(1.0, 128.0, 0.0009765625, 0.0, 4.0)
    assert result["expected_move_pct"] == result["needed_pct"]
    assert result["passes"] is True


@pytest.mark.parametrize("multiple", [0.0, -1.0, -5.0])
def test_cost_gate_disabled_when_multiple_not_positive(multiple: float) -> None:
    result = regime.cost_gate(0.0001, 100.0, 0.001, 0.0005, multiple)
    assert result["passes"] is True
    assert result["needed_pct"] == 0.0


@pytest.mark.parametrize("bad_atr", [None, 0.0, -1.0, math.nan, math.inf])
def test_cost_gate_invalid_atr_fails_conservatively(bad_atr: object) -> None:
    result = regime.cost_gate(bad_atr, 100.0, 0.001, 0.0005, 3.0)
    assert result["expected_move_pct"] is None
    assert result["passes"] is False


@pytest.mark.parametrize("bad_price", [None, 0.0, -100.0, math.nan])
def test_cost_gate_invalid_price_fails_conservatively(bad_price: object) -> None:
    result = regime.cost_gate(1.0, bad_price, 0.001, 0.0005, 3.0)
    assert result["expected_move_pct"] is None
    assert result["passes"] is False


def test_cost_gate_returns_required_keys() -> None:
    result = regime.cost_gate(1.0, 100.0, 0.001, 0.0005, 3.0)
    assert {"passes", "expected_move_pct", "needed_pct"} <= set(result)
    assert isinstance(result["passes"], bool)
    assert isinstance(result["needed_pct"], float)


# ---------------------------------------------------------------------------
# shadow_flags — pure observation diagnostics
# ---------------------------------------------------------------------------


def test_shadow_flags_required_keys_present() -> None:
    df = _shadow_frame(np.arange(1.0, 101.0))
    flags = regime.shadow_flags(df, 12)
    assert {"volume_ok", "atr_pctile", "atr_in_band", "dead_zone"} <= set(flags)


def test_shadow_flags_volume_ok_true_on_spike() -> None:
    # Last volume 400 vs a ~100 baseline → comfortably >= 1.5 × SMA20.
    volumes = [100.0] * 99 + [400.0]
    flags = regime.shadow_flags(_shadow_frame(np.arange(1.0, 101.0), volumes), 12)
    assert flags["volume_ok"] is True


def test_shadow_flags_volume_ok_false_on_quiet_bar() -> None:
    volumes = [100.0] * 99 + [50.0]
    flags = regime.shadow_flags(_shadow_frame(np.arange(1.0, 101.0), volumes), 12)
    assert flags["volume_ok"] is False


def test_shadow_flags_atr_pctile_window_math() -> None:
    """99 earlier values 1..99 + current 45.5 → 46 of 100 <= current → 46.0."""
    atr = list(np.arange(1.0, 100.0)) + [45.5]
    flags = regime.shadow_flags(_shadow_frame(atr), 12)
    assert flags["atr_pctile"] == pytest.approx(46.0, abs=0.01)
    assert flags["atr_in_band"] is True


def test_shadow_flags_atr_window_uses_last_100_values_only() -> None:
    """Values before the 100-value window must not affect the percentile."""
    atr = [10_000.0] * 50 + list(np.arange(1.0, 100.0)) + [45.5]
    flags = regime.shadow_flags(_shadow_frame(atr), 12)
    assert flags["atr_pctile"] == pytest.approx(46.0, abs=0.01)


def test_shadow_flags_atr_band_boundaries_inclusive() -> None:
    """pctile exactly 90.0 and exactly 20.0 are IN the band (inclusive)."""
    # 5 NaN warmup + 19 finite values 1..19 + current → exactly 20 finite.
    warmup = [math.nan] * 5
    base = list(np.arange(1.0, 20.0))

    # count(<= 17.5) = 17 + current = 18 → 100 * 18/20 = 90.0
    flags_hi = regime.shadow_flags(_shadow_frame(warmup + base + [17.5]), 12)
    assert flags_hi["atr_pctile"] == pytest.approx(90.0, abs=0.01)
    assert flags_hi["atr_in_band"] is True

    # count(<= 3.5) = 3 + current = 4 → 20.0
    flags_lo = regime.shadow_flags(_shadow_frame(warmup + base + [3.5]), 12)
    assert flags_lo["atr_pctile"] == pytest.approx(20.0, abs=0.01)
    assert flags_lo["atr_in_band"] is True

    # count(<= 0.5) = 1 → 5.0 → below the band
    flags_out = regime.shadow_flags(_shadow_frame(warmup + base + [0.5]), 12)
    assert flags_out["atr_pctile"] == pytest.approx(5.0, abs=0.01)
    assert flags_out["atr_in_band"] is False


def test_shadow_flags_fewer_than_20_finite_atr_is_none() -> None:
    # 5 NaN + 18 finite + current = 19 finite values → percentile undefined.
    atr = [math.nan] * 5 + list(np.arange(1.0, 19.0)) + [17.5]
    flags = regime.shadow_flags(_shadow_frame(atr), 12)
    assert flags["atr_pctile"] is None
    assert flags["atr_in_band"] is None


def test_shadow_flags_small_frame_degrades_to_none() -> None:
    flags = regime.shadow_flags(_shadow_frame(np.arange(1.0, 11.0)), 12)
    assert flags["volume_ok"] is None
    assert flags["atr_pctile"] is None
    assert flags["atr_in_band"] is None
    assert isinstance(flags["dead_zone"], bool)


def test_shadow_flags_missing_volume_column_is_none() -> None:
    df = _shadow_frame(np.arange(1.0, 101.0)).drop(columns=["volume"])
    flags = regime.shadow_flags(df, 12)
    assert flags["volume_ok"] is None


@pytest.mark.parametrize(
    ("hour", "expected"),
    [(0, False), (1, False), (2, True), (3, True), (4, True), (5, False),
     (12, False), (23, False)],
)
def test_shadow_flags_dead_zone_hours(hour: int, expected: bool) -> None:
    flags = regime.shadow_flags(_shadow_frame(np.arange(1.0, 101.0)), hour)
    assert flags["dead_zone"] is expected


def test_shadow_flags_computes_atr_when_column_missing(
    sample_df: pd.DataFrame,
) -> None:
    """Without an atr_14 column the flags fall back to technical.atr(df, 14)."""
    assert "atr_14" not in sample_df.columns
    flags = regime.shadow_flags(sample_df, 12)
    assert flags["atr_pctile"] is not None
    assert 0.0 <= flags["atr_pctile"] <= 100.0
    assert flags["atr_in_band"] in (True, False)
    assert flags["volume_ok"] in (True, False)


def test_shadow_flags_never_raises_on_empty_frame() -> None:
    flags = regime.shadow_flags(pd.DataFrame(), 3)
    assert flags["volume_ok"] is None
    assert flags["atr_pctile"] is None
    assert flags["atr_in_band"] is None
    assert flags["dead_zone"] is True
