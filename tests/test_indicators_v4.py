"""Tests for Indicator Pack v4 — the ~20 new pure indicators and the 48-column
feature pipeline (see CONTRACTS.md "## Indicator Pack v4", which is binding).

PAPER TRADING ONLY — offline tests over seeded synthetic frames; nothing here
touches a network socket or places orders.

Coverage per the contract's tests/test_indicators_v4.py scope:

* Hand-computed / pinned-formula reference values on small synthetic frames
  (OBV sign logic, donchian == rolling extremes, MFI 100/0/50 edges, CMF
  ``high == low`` rule, flat-window stochastic → NaN, aroon tie rule, ...).
* Range bounds: adx/di/stoch/stoch_rsi/mfi/aroon in [0, 100], williams_r in
  [-100, 0], cmf in [-1, 1]; band ordering for donchian and keltner.
* ``supertrend_dir`` in {+1.0, -1.0} after warm-up; psar flips to the opposite
  side of price after a reversal; the pinned ichimoku forward-shift identity.
* Strict no-lookahead prefix stability for EVERY new function and for the full
  48-column ``add_features`` output.
* NaN warm-up lengths (not pinned inside the first ``window`` bars of
  Wilder-smoothed outputs, per contract), input frames never mutated, and
  short/empty frames degrading to NaN instead of raising.
* ``feature_summary`` v4: all 48 numeric keys, the 13 new qualitative flags
  with pinned thresholds, the exact ``groups`` dict, JSON safety.
* Performance smoke: ``add_features`` on a 2,000-row random walk, median of 5
  runs asserted under the contract's generous 500 ms CI margin.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import pytest

from backend.indicators.features import FEATURE_COLUMNS, add_features, feature_summary
from backend.indicators.technical import (
    adl,
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
    mfi,
    momentum,
    obv,
    psar,
    roc,
    rsi,
    sma,
    stoch_rsi,
    stochastic,
    supertrend,
    trix,
    vwma,
    williams_r,
)

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

#: The canonical 48-name v4 feature list, in canonical order (CONTRACTS.md §2).
FEATURE_COLUMNS_V4 = [
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
]

#: New qualitative flags added by feature_summary v4 (CONTRACTS.md §3).
NEW_FLAGS = [
    "adx_trend",
    "di_state",
    "stoch_zone",
    "williams_zone",
    "cci_zone",
    "mfi_zone",
    "obv_trend",
    "supertrend_side",
    "psar_side",
    "aroon_state",
    "ichimoku_state",
    "squeeze_on",
    "donchian_position",
]

EXPECTED_GROUPS = {
    "trend": {
        "price_vs_sma20",
        "adx_trend",
        "di_state",
        "supertrend_side",
        "psar_side",
        "aroon_state",
        "ichimoku_state",
    },
    "momentum": {"macd_state", "rsi_zone", "stoch_zone", "williams_zone", "cci_zone"},
    "volume": {"mfi_zone", "obv_trend"},
    "volatility": {"bb_position", "squeeze_on", "donchian_position"},
}


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------


def make_ohlcv(
    close,
    high=None,
    low=None,
    open_=None,
    volume=None,
) -> pd.DataFrame:
    """Build a canonical OHLCV frame from explicit arrays (UTC 1h index)."""
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    if open_ is None:
        open_ = np.empty(n, dtype=np.float64)
        if n:
            open_[0] = close[0]
            open_[1:] = close[:-1]
    else:
        open_ = np.asarray(open_, dtype=np.float64)
    if high is None:
        high = np.maximum(open_, close) + 0.25
    else:
        high = np.asarray(high, dtype=np.float64)
    if low is None:
        low = np.minimum(open_, close) - 0.25
    else:
        low = np.asarray(low, dtype=np.float64)
    if volume is None:
        volume = np.full(n, 500.0)
    else:
        volume = np.asarray(volume, dtype=np.float64)
    index = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    ).as_unit("ms")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")


def uptrend_df(n: int = 200) -> pd.DataFrame:
    """Monotonic up-frame: close == high, +1 per bar (deterministic extremes)."""
    close = 100.0 + np.arange(n, dtype=np.float64)
    return make_ohlcv(close, high=close, low=close - 1.0, open_=close - 0.5)


def downtrend_df(n: int = 200) -> pd.DataFrame:
    """Monotonic down-frame: close == low, -1 per bar (mirror of uptrend_df)."""
    close = 1000.0 - np.arange(n, dtype=np.float64)
    return make_ohlcv(close, high=close + 1.0, low=close, open_=close + 0.5)


def flat_df(n: int = 30) -> pd.DataFrame:
    """Fully flat frame: open == high == low == close (flat rolling windows)."""
    close = np.full(n, 100.0)
    return make_ohlcv(close, high=close, low=close, open_=close)


def squeeze_df(n: int = 60) -> pd.DataFrame:
    """Flat close but wide bar range: Bollinger collapses inside Keltner."""
    close = np.full(n, 100.0)
    return make_ohlcv(close, high=close + 5.0, low=close - 5.0, open_=close)


def random_walk_df(n: int, seed: int = 7) -> pd.DataFrame:
    """Seeded random-walk canonical frame (conftest-style construction)."""
    rng = np.random.default_rng(seed)
    close = 500.0 + np.cumsum(rng.normal(0.05, 1.0, n))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = 500.0
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0.0, 0.5, n)) + 0.05
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(100.0, 1_000.0, n)
    df = make_ohlcv(close, high=high, low=low, open_=open_, volume=volume)
    assert (df["low"] > 0).all()
    return df


@pytest.fixture()
def features_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    return add_features(sample_df)


# ---------------------------------------------------------------------------
# Every new function, normalized to df -> tuple[pd.Series, ...]
# ---------------------------------------------------------------------------


def _pct_b_from_df(df: pd.DataFrame) -> pd.Series:
    upper, _mid, lower = bollinger(df["close"], 20, 2.0)
    return bollinger_pct_b(df["close"], upper, lower)


def _bandwidth_from_df(df: pd.DataFrame) -> pd.Series:
    upper, mid, lower = bollinger(df["close"], 20, 2.0)
    return bollinger_bandwidth(upper, mid, lower)


ALL_FUNCS: list[tuple[str, object]] = [
    ("adx", lambda df: adx(df)),
    ("stochastic", lambda df: stochastic(df)),
    ("stoch_rsi", lambda df: stoch_rsi(df["close"])),
    ("williams_r", lambda df: williams_r(df)),
    ("cci", lambda df: cci(df)),
    ("roc", lambda df: roc(df["close"])),
    ("mfi", lambda df: mfi(df)),
    ("obv", lambda df: obv(df)),
    ("cmf", lambda df: cmf(df)),
    ("adl", lambda df: adl(df)),
    ("vwma", lambda df: vwma(df)),
    ("supertrend", lambda df: supertrend(df)),
    ("psar", lambda df: psar(df)),
    ("aroon", lambda df: aroon(df)),
    ("donchian", lambda df: donchian(df)),
    ("keltner", lambda df: keltner(df)),
    ("hull_ma", lambda df: hull_ma(df["close"])),
    ("trix", lambda df: trix(df["close"])),
    ("ichimoku", lambda df: ichimoku(df)),
    ("bollinger_pct_b", _pct_b_from_df),
    ("bollinger_bandwidth", _bandwidth_from_df),
]

#: Functions whose default warm-up exceeds 3 bars → all-NaN on a 3-row frame.
ALL_NAN_ON_3_ROWS = {
    "stochastic",
    "stoch_rsi",
    "williams_r",
    "cci",
    "roc",
    "mfi",
    "cmf",
    "vwma",
    "supertrend",
    "aroon",
    "donchian",
    "hull_ma",
    "ichimoku",
    "bollinger_pct_b",
    "bollinger_bandwidth",
}


def _as_tuple(result) -> tuple[pd.Series, ...]:
    return result if isinstance(result, tuple) else (result,)


# ---------------------------------------------------------------------------
# Range bounds & band ordering (contract §5 scope)
# ---------------------------------------------------------------------------


def test_range_bounds(sample_df: pd.DataFrame) -> None:
    close = sample_df["close"]
    adx_line, di_plus, di_minus = adx(sample_df)
    stoch_k, stoch_d = stochastic(sample_df)
    srsi_k, srsi_d = stoch_rsi(close)
    aroon_up, aroon_down = aroon(sample_df)
    checks = [
        ("adx", adx_line, 0.0, 100.0),
        ("di_plus", di_plus, 0.0, 100.0),
        ("di_minus", di_minus, 0.0, 100.0),
        ("stoch_k", stoch_k, 0.0, 100.0),
        ("stoch_d", stoch_d, 0.0, 100.0),
        ("stoch_rsi_k", srsi_k, 0.0, 100.0),
        ("stoch_rsi_d", srsi_d, 0.0, 100.0),
        ("mfi", mfi(sample_df), 0.0, 100.0),
        ("aroon_up", aroon_up, 0.0, 100.0),
        ("aroon_down", aroon_down, 0.0, 100.0),
        ("williams_r", williams_r(sample_df), -100.0, 0.0),
        ("cmf", cmf(sample_df), -1.0, 1.0),
    ]
    for name, series, lo, hi in checks:
        assert len(series) == len(sample_df), name
        valid = series.dropna()
        assert len(valid) > 0, f"{name}: no valid values"
        assert ((valid >= lo) & (valid <= hi)).all(), f"{name} out of [{lo}, {hi}]"


def test_donchian_matches_rolling_extremes(sample_df: pd.DataFrame) -> None:
    upper, mid, lower = donchian(sample_df)
    ref_upper = sample_df["high"].rolling(20, min_periods=20).max()
    ref_lower = sample_df["low"].rolling(20, min_periods=20).min()
    pd.testing.assert_series_equal(upper, ref_upper, check_names=False)
    pd.testing.assert_series_equal(lower, ref_lower, check_names=False)
    pd.testing.assert_series_equal(mid, (ref_upper + ref_lower) / 2, check_names=False)
    valid = upper.notna() & mid.notna() & lower.notna()
    assert valid.any()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_keltner_identity_and_ordering(sample_df: pd.DataFrame) -> None:
    upper, mid, lower = keltner(sample_df)
    ref_mid = ema(sample_df["close"], 20)  # EMA mid — pinned
    ref_atr = atr(sample_df, 10)
    pd.testing.assert_series_equal(mid, ref_mid, check_names=False)
    pd.testing.assert_series_equal(upper, ref_mid + 2.0 * ref_atr, check_names=False)
    pd.testing.assert_series_equal(lower, ref_mid - 2.0 * ref_atr, check_names=False)
    valid = upper.notna() & mid.notna() & lower.notna()
    assert valid.any()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


# ---------------------------------------------------------------------------
# Hand-computed / pinned-formula reference values
# ---------------------------------------------------------------------------


def test_obv_sign_logic_hand_values() -> None:
    df = make_ohlcv(
        close=[10.0, 11.0, 11.0, 10.0, 12.0],
        volume=[10.0, 20.0, 30.0, 40.0, 50.0],
    )
    result = obv(df)
    # signs: first bar 0, up, tie, down, up → 0, +20, +20-0, -40, +50
    assert np.allclose(result.to_numpy(), [0.0, 20.0, 20.0, -20.0, 30.0])
    assert result.iloc[0] == 0.0
    assert result.notna().all()


def test_adl_hand_values_and_high_low_rule() -> None:
    df = make_ohlcv(
        close=[2.0, 1.0, 5.0, 4.0],
        high=[2.0, 3.0, 5.0, 6.0],
        low=[0.0, 1.0, 5.0, 2.0],
        open_=[1.0, 2.0, 5.0, 3.0],
        volume=[10.0, 20.0, 30.0, 40.0],
    )
    result = adl(df)
    # mfm: 1, -1, 0 (high == low pinned), 0 → mfv: 10, -20, 0, 0
    assert np.allclose(result.to_numpy(), [10.0, -10.0, -10.0, -10.0])


def test_cmf_hand_value_and_high_low_rule() -> None:
    df = make_ohlcv(
        close=[2.0, 1.0, 5.0, 4.0],
        high=[2.0, 3.0, 5.0, 6.0],
        low=[0.0, 1.0, 5.0, 2.0],
        open_=[1.0, 2.0, 5.0, 3.0],
        volume=[10.0, 20.0, 30.0, 40.0],
    )
    result = cmf(df, 4)
    assert result.iloc[:3].isna().all()
    # sum(mfv) = 10 - 20 + 0 + 0 = -10; sum(volume) = 100
    assert result.iloc[3] == pytest.approx(-0.1)


def test_cmf_close_at_high_is_plus_one() -> None:
    close = 100.0 + np.arange(30, dtype=np.float64)
    df = make_ohlcv(close, high=close, low=close - 1.0, open_=close - 0.5)
    valid = cmf(df).dropna()
    assert len(valid) > 0
    assert np.allclose(valid.to_numpy(), 1.0)


def test_mfi_pinned_edges() -> None:
    # Monotonic rising typical price → neg flow 0 → pinned 100.
    up = mfi(uptrend_df(20), 5).dropna()
    assert len(up) > 0
    assert np.allclose(up.to_numpy(), 100.0)
    # Monotonic falling typical price → pos flow 0 → pinned 0.
    down = mfi(downtrend_df(20), 5).dropna()
    assert len(down) > 0
    assert np.allclose(down.to_numpy(), 0.0)
    # Constant typical price → both flows 0 → pinned 50 (RSI-like neutrality).
    flat = mfi(flat_df(12), 3).dropna()
    assert len(flat) > 0
    assert np.allclose(flat.to_numpy(), 50.0)


def test_mfi_matches_pinned_formula(sample_df: pd.DataFrame) -> None:
    tp = (sample_df["high"] + sample_df["low"] + sample_df["close"]) / 3
    rmf = tp * sample_df["volume"]
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    ref = (100.0 * pos / (pos + neg)).where(~(pos.eq(0.0) & neg.eq(0.0)), 50.0)
    pd.testing.assert_series_equal(mfi(sample_df), ref, check_names=False)


def test_stochastic_flat_window_nan_and_hand_values() -> None:
    # HH == LL → pinned NaN, not a division error.
    k_flat, d_flat = stochastic(flat_df(30))
    assert k_flat.isna().all()
    assert d_flat.isna().all()
    # close == high monotonic → raw_k = 100 exactly, smoothing preserves it.
    k_up, d_up = stochastic(uptrend_df(40))
    assert np.allclose(k_up.dropna().to_numpy(), 100.0)
    assert np.allclose(d_up.dropna().to_numpy(), 100.0)


def test_stochastic_matches_pinned_formula(sample_df: pd.DataFrame) -> None:
    hh = sample_df["high"].rolling(14, min_periods=14).max()
    ll = sample_df["low"].rolling(14, min_periods=14).min()
    width = hh - ll
    raw_k = (100.0 * (sample_df["close"] - ll) / width).where(width != 0)
    ref_k = raw_k.rolling(3, min_periods=3).mean()
    ref_d = ref_k.rolling(3, min_periods=3).mean()
    k, d = stochastic(sample_df)
    pd.testing.assert_series_equal(k, ref_k, check_names=False)
    pd.testing.assert_series_equal(d, ref_d, check_names=False)


def test_stoch_rsi_matches_pinned_formula(sample_df: pd.DataFrame) -> None:
    close = sample_df["close"]
    r = rsi(close, 14)
    rmin = r.rolling(14, min_periods=14).min()
    rmax = r.rolling(14, min_periods=14).max()
    width = rmax - rmin
    raw = (100.0 * (r - rmin) / width).where(width != 0)
    ref_k = raw.rolling(3, min_periods=3).mean()
    ref_d = ref_k.rolling(3, min_periods=3).mean()
    k, d = stoch_rsi(close)
    pd.testing.assert_series_equal(k, ref_k, check_names=False)
    pd.testing.assert_series_equal(d, ref_d, check_names=False)


def test_williams_r_hand_values_and_formula(sample_df: pd.DataFrame) -> None:
    # close == HH → 0; close == LL → -100; flat window → NaN.
    assert np.allclose(williams_r(uptrend_df(40)).dropna().to_numpy(), 0.0)
    assert np.allclose(williams_r(downtrend_df(40)).dropna().to_numpy(), -100.0)
    assert williams_r(flat_df(30)).isna().all()

    hh = sample_df["high"].rolling(14, min_periods=14).max()
    ll = sample_df["low"].rolling(14, min_periods=14).min()
    width = hh - ll
    ref = (-100.0 * (hh - sample_df["close"]) / width).where(width != 0)
    pd.testing.assert_series_equal(williams_r(sample_df), ref, check_names=False)


def test_cci_reference_and_flat_nan(sample_df: pd.DataFrame) -> None:
    tp = (sample_df["high"] + sample_df["low"] + sample_df["close"]) / 3
    tp_sma = tp.rolling(20, min_periods=20).mean()
    mad = tp.rolling(20, min_periods=20).apply(
        lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True
    )
    ref = ((tp - tp_sma) / (0.015 * mad)).where(mad != 0)
    pd.testing.assert_series_equal(cci(sample_df), ref, check_names=False)
    # Constant series → MAD == 0 → pinned NaN.
    assert cci(flat_df(40)).isna().all()


def test_roc_is_percent_scale() -> None:
    s = pd.Series(np.arange(1.0, 21.0))
    result = roc(s, 10)
    assert result.iloc[:10].isna().all()
    assert result.iloc[10] == pytest.approx(1000.0)  # 100 * (11/1 - 1)
    assert result.iloc[19] == pytest.approx(100.0)  # 100 * (20/10 - 1)


def test_roc_equals_100x_momentum(sample_df: pd.DataFrame) -> None:
    close = sample_df["close"]
    got = roc(close).dropna()
    ref = (100.0 * momentum(close, 10)).dropna()
    assert np.allclose(got.to_numpy(), ref.to_numpy())


def test_vwma_hand_values_and_constant_volume() -> None:
    df = make_ohlcv(close=[1.0, 3.0, 5.0], volume=[1.0, 3.0, 1.0])
    result = vwma(df, 2)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(2.5)  # (1*1 + 3*3) / 4
    assert result.iloc[2] == pytest.approx(3.5)  # (3*3 + 5*1) / 4

    # Constant volume → vwma degenerates to the SMA.
    walk = random_walk_df(60)
    walk["volume"] = 500.0
    got = vwma(walk, 20).dropna()
    ref = sma(walk["close"], 20).dropna()
    assert np.allclose(got.to_numpy(), ref.to_numpy())


def test_vwma_zero_volume_is_nan() -> None:
    df = flat_df(30)
    df["volume"] = 0.0
    assert vwma(df, 5).isna().all()


def test_trix_matches_pinned_formula(sample_df: pd.DataFrame) -> None:
    close = sample_df["close"]
    t3 = ema(ema(ema(close, 15), 15), 15)
    ref = 100.0 * (t3 / t3.shift(1) - 1)
    result = trix(close)
    pd.testing.assert_series_equal(result, ref, check_names=False)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].notna().all()


def test_hull_ma_matches_wma_reference(sample_df: pd.DataFrame) -> None:
    def wma_ref(s: pd.Series, window: int) -> pd.Series:
        weights = np.arange(1, window + 1, dtype=np.float64)
        return s.rolling(window, min_periods=window).apply(
            lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
        )

    close = sample_df["close"]
    # window=20 → half=10, sqrt_w=4 (pinned floor/round conventions).
    ref = wma_ref(2.0 * wma_ref(close, 10) - wma_ref(close, 20), 4)
    got = hull_ma(close, 20)
    assert np.allclose(
        got.dropna().to_numpy(), ref.dropna().to_numpy(), rtol=1e-9, atol=1e-9
    )
    assert got.iloc[:22].isna().all()
    assert got.iloc[22:].notna().all()


def test_adx_monotonic_uptrend_extremes() -> None:
    df = uptrend_df(100)
    adx_line, di_plus, di_minus = adx(df)
    for series in (adx_line, di_plus, di_minus):
        valid = series.dropna()
        assert len(valid) > 0
        assert ((valid >= 0.0) & (valid <= 100.0)).all()
    # Perfect uptrend: -DM is always 0 → di_minus 0, dx 100, adx → 100.
    assert di_minus.dropna().iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert di_plus.iloc[-1] > di_minus.iloc[-1]
    assert adx_line.iloc[-1] > 90.0


def test_adx_matches_pinned_formula(sample_df: pd.DataFrame) -> None:
    high, low = sample_df["high"], sample_df["low"]
    up = high.diff()
    down = -low.diff()
    dm_plus = up.where((up > down) & (up > 0), 0.0)
    dm_minus = down.where((down > up) & (down > 0), 0.0)

    def wilder(s: pd.Series) -> pd.Series:
        return s.ewm(alpha=1.0 / 14.0, adjust=False).mean()

    atr_14 = atr(sample_df, 14)
    ref_plus = (100.0 * wilder(dm_plus) / atr_14).where(atr_14 != 0)
    ref_minus = (100.0 * wilder(dm_minus) / atr_14).where(atr_14 != 0)
    denom = ref_plus + ref_minus
    dx = (100.0 * (ref_plus - ref_minus).abs() / denom).where(denom != 0)
    ref_adx = wilder(dx)

    got_adx, got_plus, got_minus = adx(sample_df)
    # Wilder outputs are unsettled early — compare only past the warm-up.
    for got, ref in ((got_adx, ref_adx), (got_plus, ref_plus), (got_minus, ref_minus)):
        assert np.allclose(
            got.iloc[30:].to_numpy(),
            ref.iloc[30:].to_numpy(),
            rtol=1e-7,
            atol=1e-9,
            equal_nan=True,
        )


def test_aroon_extremes_and_tie_rule() -> None:
    # Monotonic up: highest high is the current bar (up=100), lowest low the
    # oldest bar in the window (down=0).
    up_frame = aroon(uptrend_df(60))
    assert np.allclose(up_frame[0].dropna().to_numpy(), 100.0)
    assert np.allclose(up_frame[1].dropna().to_numpy(), 0.0)
    # Pinned tie rule: most recent extreme wins → flat window yields 100/100.
    flat_up, flat_down = aroon(flat_df(40))
    assert np.allclose(flat_up.dropna().to_numpy(), 100.0)
    assert np.allclose(flat_down.dropna().to_numpy(), 100.0)


def test_aroon_matches_most_recent_extreme_reference(sample_df: pd.DataFrame) -> None:
    window = 25

    def bars_since_max(x: np.ndarray) -> float:
        return float(np.argmax(x[::-1]))

    def bars_since_min(x: np.ndarray) -> float:
        return float(np.argmin(x[::-1]))

    since_high = sample_df["high"].rolling(window + 1, min_periods=window + 1).apply(
        bars_since_max, raw=True
    )
    since_low = sample_df["low"].rolling(window + 1, min_periods=window + 1).apply(
        bars_since_min, raw=True
    )
    ref_up = 100.0 * (window - since_high) / window
    ref_down = 100.0 * (window - since_low) / window
    got_up, got_down = aroon(sample_df)
    pd.testing.assert_series_equal(got_up, ref_up, check_names=False)
    pd.testing.assert_series_equal(got_down, ref_down, check_names=False)


def test_supertrend_warmup_direction_and_sides() -> None:
    up = uptrend_df(60)
    line_up, dir_up = supertrend(up)
    # Pinned warm-up: rows 0..window-1 NaN for BOTH outputs.
    assert line_up.iloc[:10].isna().all()
    assert dir_up.iloc[:10].isna().all()
    assert line_up.iloc[10:].notna().all()
    assert dir_up.iloc[10:].notna().all()
    assert dir_up.dropna().isin([1.0, -1.0]).all()
    # Steady uptrend: bullish, line below price.
    assert (dir_up.iloc[-20:] == 1.0).all()
    assert (line_up.iloc[-20:] < up["close"].iloc[-20:]).all()

    down = downtrend_df(60)
    line_down, dir_down = supertrend(down)
    assert dir_down.dropna().isin([1.0, -1.0]).all()
    assert (dir_down.iloc[-20:] == -1.0).all()
    assert (line_down.iloc[-20:] > down["close"].iloc[-20:]).all()


def test_supertrend_dir_values_on_sample(sample_df: pd.DataFrame) -> None:
    _line, direction = supertrend(sample_df)
    valid = direction.dropna()
    assert len(valid) > 0
    assert valid.isin([1.0, -1.0]).all()


def test_psar_uptrend_below_price_then_flips_on_reversal() -> None:
    n_up, n_down = 30, 30
    close = np.concatenate(
        [100.0 + np.arange(n_up), 129.0 - 2.0 * np.arange(1, n_down + 1)]
    )
    df = make_ohlcv(close, high=close + 0.5, low=close - 0.5)
    result = psar(df)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].notna().all()
    # Mid-uptrend: SAR trails below price.
    assert (result.iloc[10:25] < df["close"].iloc[10:25]).all()
    # After the sharp reversal, SAR must flip to the opposite side of price.
    assert result.iloc[-1] > df["close"].iloc[-1]
    side = np.sign(result.iloc[1:] - df["close"].iloc[1:])
    assert (side == -1.0).any() and (side == 1.0).any()


def test_psar_needs_two_bars() -> None:
    one_row = make_ohlcv([100.0])
    assert psar(one_row).isna().all()


def test_ichimoku_pinned_shift_identity_and_warmups(sample_df: pd.DataFrame) -> None:
    result = ichimoku(sample_df)
    assert len(result) == 4
    tenkan, kijun, senkou_a, senkou_b = result

    high, low = sample_df["high"], sample_df["low"]
    ref_tenkan = (
        high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()
    ) / 2
    ref_kijun = (
        high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()
    ) / 2
    senkou_a_raw = (ref_tenkan + ref_kijun) / 2
    senkou_b_raw = (
        high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()
    ) / 2

    pd.testing.assert_series_equal(tenkan, ref_tenkan, check_names=False)
    pd.testing.assert_series_equal(kijun, ref_kijun, check_names=False)
    # PINNED convention: spans shifted FORWARD by 26 — senkou_a[t] equals the
    # raw span computed at t-26 (strictly backward-looking).
    pd.testing.assert_series_equal(senkou_a, senkou_a_raw.shift(26), check_names=False)
    pd.testing.assert_series_equal(senkou_b, senkou_b_raw.shift(26), check_names=False)

    # Pinned warm-up lengths: 8 / 25 / 51 / 77.
    for series, n_nan in ((tenkan, 8), (kijun, 25), (senkou_a, 51), (senkou_b, 77)):
        assert series.iloc[:n_nan].isna().all()
        assert series.iloc[n_nan:].notna().all()


def test_bollinger_pct_b_hand_values_not_clipped() -> None:
    close = pd.Series([9.0, 10.0, 11.0, 13.0])
    upper = pd.Series([12.0] * 4)
    lower = pd.Series([10.0] * 4)
    result = bollinger_pct_b(close, upper, lower)
    # Values outside [0, 1] are meaningful band breaks — NOT clipped.
    assert np.allclose(result.to_numpy(), [-0.5, 0.0, 0.5, 1.5])
    # Zero band width → NaN.
    zero_width = bollinger_pct_b(close, lower, lower)
    assert zero_width.isna().all()


def test_bollinger_bandwidth_hand_values() -> None:
    upper = pd.Series([12.0, 12.0])
    mid = pd.Series([11.0, 0.0])
    lower = pd.Series([10.0, 10.0])
    result = bollinger_bandwidth(upper, mid, lower)
    assert result.iloc[0] == pytest.approx(2.0 / 11.0)
    assert np.isnan(result.iloc[1])  # mid == 0 → NaN


# ---------------------------------------------------------------------------
# Warm-up NaN lengths (not pinned inside Wilder-smoothed settling windows)
# ---------------------------------------------------------------------------


def test_warmup_nan_lengths(sample_df: pd.DataFrame) -> None:
    close = sample_df["close"]
    stoch_k, stoch_d = stochastic(sample_df)
    donch_upper, donch_mid, donch_lower = donchian(sample_df)
    st_line, st_dir = supertrend(sample_df)
    aroon_up, aroon_down = aroon(sample_df)

    # (name, series, nan_until, valid_from)
    checks = [
        ("williams_r", williams_r(sample_df), 13, 13),
        ("cci", cci(sample_df), 19, 19),
        ("vwma", vwma(sample_df), 19, 19),
        ("cmf", cmf(sample_df), 19, 19),
        # bar-0 flow convention unpinned → index 13 may be either.
        ("mfi", mfi(sample_df), 13, 14),
        ("roc", roc(close), 10, 10),
        ("stoch_k", stoch_k, 15, 15),
        ("stoch_d", stoch_d, 17, 17),
        ("aroon_up", aroon_up, 25, 25),
        ("aroon_down", aroon_down, 25, 25),
        ("donchian_upper", donch_upper, 19, 19),
        ("donchian_mid", donch_mid, 19, 19),
        ("donchian_lower", donch_lower, 19, 19),
        ("supertrend_line", st_line, 10, 10),
        ("supertrend_dir", st_dir, 10, 10),
        ("hull_ma", hull_ma(close), 22, 22),
    ]
    for name, series, nan_until, valid_from in checks:
        assert series.iloc[:nan_until].isna().all(), f"{name}: warm-up too short"
        assert series.iloc[valid_from:].notna().all(), f"{name}: NaN after warm-up"

    # Cumulative indicators are valid from bar 0.
    assert obv(sample_df).notna().all()
    assert adl(sample_df).notna().all()
    # First-bar NaN only.
    for name, series in (("psar", psar(sample_df)), ("trix", trix(close))):
        assert np.isnan(series.iloc[0]), name
        assert series.iloc[1:].notna().all(), name
    # Wilder-smoothed adx family: dx is undefined at bar 0 (both DI lines are
    # 0 there, so the masked 0-denominator yields NaN) — the adx line starts
    # NaN. The DI lines themselves start at the formula-forced 0.0: +DM/-DM
    # pin to 0 on bar 0 (diff is NaN, `where(..., 0.0)` takes the else branch)
    # while the existing atr degrades to high - low > 0 on bar 0. Per the
    # contract, no artificial NaN masking beyond what the formulas produce.
    adx_line, di_plus, di_minus = adx(sample_df)
    assert np.isnan(adx_line.iloc[0])
    assert di_plus.iloc[0] == 0.0
    assert di_minus.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# No lookahead — prefix stability for every new function
# ---------------------------------------------------------------------------

_CUT = 150


@pytest.mark.parametrize(
    "func", [f for _, f in ALL_FUNCS], ids=[name for name, _ in ALL_FUNCS]
)
def test_no_lookahead_per_function(sample_df: pd.DataFrame, func) -> None:
    """Recomputing on a truncated frame must reproduce the retained rows."""
    full = _as_tuple(func(sample_df))
    trunc = _as_tuple(func(sample_df.iloc[:_CUT].copy()))
    assert len(full) == len(trunc)
    for full_series, trunc_series in zip(full, trunc):
        pd.testing.assert_series_equal(
            full_series.iloc[:_CUT],
            trunc_series,
            check_names=False,
            check_freq=False,
            rtol=1e-9,
            atol=1e-9,
        )


def test_no_lookahead_add_features_all_48(sample_df: pd.DataFrame) -> None:
    """Prefix stability must hold over the FULL 48-column feature set."""
    full = add_features(sample_df)
    head = add_features(sample_df.iloc[:120].copy())
    assert list(head.columns) == list(full.columns)
    pd.testing.assert_frame_equal(
        full.iloc[:120],
        head,
        check_freq=False,
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Purity: inputs never mutated; short/empty frames degrade to NaN
# ---------------------------------------------------------------------------


def test_functions_do_not_mutate_input(sample_df: pd.DataFrame) -> None:
    snapshot = sample_df.copy(deep=True)
    for _name, func in ALL_FUNCS:
        func(sample_df)
    add_features(sample_df)
    feature_summary(sample_df)
    pd.testing.assert_frame_equal(sample_df, snapshot)


@pytest.mark.parametrize("n_rows", [0, 1, 3])
def test_short_and_empty_frames_never_raise(n_rows: int) -> None:
    if n_rows == 0:
        df = make_ohlcv([])
    else:
        df = make_ohlcv(100.0 + np.arange(n_rows, dtype=np.float64))
    for name, func in ALL_FUNCS:
        result = _as_tuple(func(df))
        for series in result:
            assert isinstance(series, pd.Series), name
            assert len(series) == n_rows, f"{name}: wrong length on {n_rows} rows"
        if n_rows == 3 and name in ALL_NAN_ON_3_ROWS:
            for series in result:
                assert series.isna().all(), f"{name}: expected all-NaN on 3 rows"
        if n_rows == 1 and name in ("psar", "supertrend"):
            for series in result:
                assert series.isna().all(), f"{name}: expected all-NaN on 1 row"


def test_add_features_short_frame_never_raises() -> None:
    df = make_ohlcv([100.0, 101.0, 102.0])
    out = add_features(df)
    assert len(out) == 3
    assert list(out.columns) == OHLCV_COLUMNS + FEATURE_COLUMNS_V4


# ---------------------------------------------------------------------------
# add_features v4 — canonical 48 columns, order, wiring, empty path
# ---------------------------------------------------------------------------


def test_feature_columns_v4_canonical_order() -> None:
    assert list(FEATURE_COLUMNS) == FEATURE_COLUMNS_V4
    assert len(FEATURE_COLUMNS) == 48
    # Original 17 unchanged and FIRST (backward compatibility).
    assert list(FEATURE_COLUMNS[:17]) == FEATURE_COLUMNS_V4[:17]


def test_add_features_exposes_48_columns_in_order(
    sample_df: pd.DataFrame, features_df: pd.DataFrame
) -> None:
    assert list(features_df.columns) == OHLCV_COLUMNS + FEATURE_COLUMNS_V4
    assert features_df.index.equals(sample_df.index)
    for col in FEATURE_COLUMNS_V4:
        assert features_df[col].dtype == np.float64, col


def test_add_features_empty_frame_has_48_empty_columns() -> None:
    out = add_features(make_ohlcv([]))
    assert len(out) == 0
    assert list(out.columns) == OHLCV_COLUMNS + FEATURE_COLUMNS_V4
    for col in FEATURE_COLUMNS_V4:
        assert out[col].dtype == np.float64, col


def test_add_features_wiring_matches_functions(
    sample_df: pd.DataFrame, features_df: pd.DataFrame
) -> None:
    """Each new column must equal its contract-pinned default-args call."""
    close = sample_df["close"]
    adx_line, di_plus, di_minus = adx(sample_df)
    stoch_k, stoch_d = stochastic(sample_df)
    st_line, st_dir = supertrend(sample_df)
    aroon_up, aroon_down = aroon(sample_df)
    donch_upper, _donch_mid, donch_lower = donchian(sample_df)
    kelt_upper, _kelt_mid, kelt_lower = keltner(sample_df)
    tenkan, kijun, senkou_a, senkou_b = ichimoku(sample_df)

    expected = {
        "adx_14": adx_line,
        "di_plus_14": di_plus,
        "di_minus_14": di_minus,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "stoch_rsi_k": stoch_rsi(close)[0],
        "williams_r_14": williams_r(sample_df),
        "cci_20": cci(sample_df),
        "roc_10": roc(close),
        "mfi_14": mfi(sample_df),
        "obv": obv(sample_df),
        "cmf_20": cmf(sample_df),
        "vwma_20": vwma(sample_df),
        "rel_volume_20": sample_df["volume"] / sma(sample_df["volume"], 20),
        "supertrend_10_3": st_line,
        "supertrend_dir": st_dir,
        "psar": psar(sample_df),
        "aroon_up_25": aroon_up,
        "aroon_down_25": aroon_down,
        "donchian_upper_20": donch_upper,
        "donchian_lower_20": donch_lower,
        "keltner_upper_20": kelt_upper,
        "keltner_lower_20": kelt_lower,
        "bb_pct_b": bollinger_pct_b(
            close, features_df["bb_upper"], features_df["bb_lower"]
        ),
        "bb_bandwidth": bollinger_bandwidth(
            features_df["bb_upper"], features_df["bb_mid"], features_df["bb_lower"]
        ),
        "hull_20": hull_ma(close),
        "trix_15": trix(close),
        "ichimoku_tenkan": tenkan,
        "ichimoku_kijun": kijun,
        "ichimoku_senkou_a": senkou_a,
        "ichimoku_senkou_b": senkou_b,
    }
    assert set(expected) == set(FEATURE_COLUMNS_V4[17:])
    for col, ref in expected.items():
        pd.testing.assert_series_equal(
            features_df[col], ref, check_names=False, rtol=1e-9, atol=1e-12
        )


def test_rel_volume_constant_and_zero_volume() -> None:
    df = flat_df(30)  # constant volume 500 → rel_volume exactly 1
    rel = add_features(df)["rel_volume_20"]
    assert np.allclose(rel.dropna().to_numpy(), 1.0)

    zero = flat_df(30)
    zero["volume"] = 0.0
    rel_zero = add_features(zero)["rel_volume_20"]
    assert rel_zero.isna().all()  # SMA 0 → NaN, pinned


# ---------------------------------------------------------------------------
# feature_summary v4 — flags with pinned thresholds + groups dict
# ---------------------------------------------------------------------------


def _expected_flags(summary: dict, features_df: pd.DataFrame) -> dict:
    """Recompute every new flag from the summary's own numeric values using
    the contract's pinned thresholds."""
    exp: dict = {}
    price = summary["price"]

    adx_v = summary["adx_14"]
    if adx_v is None:
        exp["adx_trend"] = None
    elif adx_v >= 25:
        exp["adx_trend"] = "strong"
    elif adx_v < 20:
        exp["adx_trend"] = "weak"
    else:
        exp["adx_trend"] = "moderate"

    dip, dim = summary["di_plus_14"], summary["di_minus_14"]
    exp["di_state"] = (
        None if dip is None or dim is None else ("bullish" if dip > dim else "bearish")
    )

    def zone(value, low_cut, high_cut, low_label, high_label):
        if value is None:
            return None
        if value < low_cut:
            return low_label
        if value > high_cut:
            return high_label
        return "neutral"

    exp["stoch_zone"] = zone(summary["stoch_k"], 20, 80, "oversold", "overbought")
    exp["williams_zone"] = zone(
        summary["williams_r_14"], -80, -20, "oversold", "overbought"
    )
    exp["cci_zone"] = zone(summary["cci_20"], -100, 100, "oversold", "overbought")
    exp["mfi_zone"] = zone(summary["mfi_14"], 20, 80, "oversold", "overbought")

    if len(features_df) < 20:
        exp["obv_trend"] = None
    else:
        obv_col = features_df["obv"]
        obv_sma = obv_col.rolling(20, min_periods=20).mean().iloc[-1]
        exp["obv_trend"] = "rising" if float(obv_col.iloc[-1]) > obv_sma else "falling"

    st_dir = summary["supertrend_dir"]
    exp["supertrend_side"] = (
        None if st_dir is None else ("bullish" if st_dir > 0 else "bearish")
    )

    psar_v = summary["psar"]
    exp["psar_side"] = (
        None
        if psar_v is None or price is None
        else ("bullish" if price > psar_v else "bearish")
    )

    a_up, a_down = summary["aroon_up_25"], summary["aroon_down_25"]
    if a_up is None or a_down is None:
        exp["aroon_state"] = None
    elif a_up > 70 and a_down < 30:
        exp["aroon_state"] = "bullish"
    elif a_down > 70 and a_up < 30:
        exp["aroon_state"] = "bearish"
    else:
        exp["aroon_state"] = "neutral"

    span_a, span_b = summary["ichimoku_senkou_a"], summary["ichimoku_senkou_b"]
    if span_a is None or span_b is None or price is None:
        exp["ichimoku_state"] = None
    elif price > max(span_a, span_b):
        exp["ichimoku_state"] = "above_cloud"
    elif price < min(span_a, span_b):
        exp["ichimoku_state"] = "below_cloud"
    else:
        exp["ichimoku_state"] = "in_cloud"

    bbu, bbl = summary["bb_upper"], summary["bb_lower"]
    ku, kl = summary["keltner_upper_20"], summary["keltner_lower_20"]
    if bbu is None or bbl is None or ku is None or kl is None:
        exp["squeeze_on"] = None
    else:
        exp["squeeze_on"] = bool(bbu < ku and bbl > kl)

    du, dl = summary["donchian_upper_20"], summary["donchian_lower_20"]
    if du is None or dl is None or price is None:
        exp["donchian_position"] = None
    elif price >= du:
        exp["donchian_position"] = "at_upper"
    elif price <= dl:
        exp["donchian_position"] = "at_lower"
    elif price >= (du + dl) / 2:
        exp["donchian_position"] = "upper_half"
    else:
        exp["donchian_position"] = "lower_half"

    return exp


def test_feature_summary_has_all_keys_and_is_json_safe(
    features_df: pd.DataFrame,
) -> None:
    summary = feature_summary(features_df)
    for col in FEATURE_COLUMNS_V4:
        assert col in summary, f"missing numeric key {col}"
        assert summary[col] is None or isinstance(summary[col], float), col
    assert "price" in summary
    # Existing 4 flags unchanged, 13 new flags present.
    for flag in ["price_vs_sma20", "macd_state", "rsi_zone", "bb_position"] + NEW_FLAGS:
        assert flag in summary, f"missing flag {flag}"
    assert isinstance(summary["squeeze_on"], (bool, type(None)))
    assert isinstance(summary["groups"], dict)
    json.dumps(summary)  # must not raise


def test_feature_summary_flags_match_pinned_thresholds(
    sample_df: pd.DataFrame,
) -> None:
    frames = [sample_df, uptrend_df(200), downtrend_df(200), squeeze_df(60)]
    for frame in frames:
        enriched = add_features(frame)
        summary = feature_summary(enriched)
        expected = _expected_flags(summary, enriched)
        for flag, want in expected.items():
            assert summary[flag] == want, (
                f"{flag}: got {summary[flag]!r}, expected {want!r} "
                f"(frame len {len(frame)})"
            )


def test_feature_summary_directed_uptrend_flags() -> None:
    summary = feature_summary(add_features(uptrend_df(200)))
    assert summary["adx_trend"] == "strong"
    assert summary["di_state"] == "bullish"
    assert summary["stoch_zone"] == "overbought"
    assert summary["williams_zone"] == "overbought"
    assert summary["cci_zone"] == "overbought"
    assert summary["mfi_zone"] == "overbought"
    assert summary["obv_trend"] == "rising"
    assert summary["supertrend_side"] == "bullish"
    assert summary["psar_side"] == "bullish"
    assert summary["aroon_state"] == "bullish"
    assert summary["ichimoku_state"] == "above_cloud"
    assert summary["donchian_position"] == "at_upper"
    assert summary["squeeze_on"] is False


def test_feature_summary_directed_downtrend_flags() -> None:
    summary = feature_summary(add_features(downtrend_df(200)))
    assert summary["di_state"] == "bearish"
    assert summary["stoch_zone"] == "oversold"
    assert summary["williams_zone"] == "oversold"
    assert summary["cci_zone"] == "oversold"
    assert summary["mfi_zone"] == "oversold"
    assert summary["obv_trend"] == "falling"
    assert summary["supertrend_side"] == "bearish"
    assert summary["psar_side"] == "bearish"
    assert summary["aroon_state"] == "bearish"
    assert summary["ichimoku_state"] == "below_cloud"
    assert summary["donchian_position"] == "at_lower"


def test_feature_summary_squeeze_on_true_case() -> None:
    # Flat close + wide bar ranges → Bollinger fully inside Keltner.
    summary = feature_summary(add_features(squeeze_df(60)))
    assert summary["squeeze_on"] is True
    assert summary["cci_zone"] is None  # flat close → MAD 0 → CCI NaN → None


def test_feature_summary_warmup_flags_none() -> None:
    summary = feature_summary(add_features(random_walk_df(5)))
    for flag in [
        "stoch_zone",
        "williams_zone",
        "cci_zone",
        "mfi_zone",
        "aroon_state",
        "ichimoku_state",
        "donchian_position",
        "squeeze_on",
        "obv_trend",
        "supertrend_side",
    ]:
        assert summary[flag] is None, f"{flag} should be None during warm-up"
    # groups mirror the flat values, including None.
    assert summary["groups"]["momentum"]["stoch_zone"] is None
    assert summary["groups"]["volatility"]["squeeze_on"] is None


def test_feature_summary_groups_exact_membership(features_df: pd.DataFrame) -> None:
    summary = feature_summary(features_df)
    groups = summary["groups"]
    assert set(groups) == set(EXPECTED_GROUPS)
    for group_name, keys in EXPECTED_GROUPS.items():
        assert set(groups[group_name]) == keys, group_name
        for key in keys:
            assert groups[group_name][key] == summary[key], f"{group_name}.{key}"


# ---------------------------------------------------------------------------
# Performance smoke (contract §4: ~150 ms budget, assert < 500 ms CI margin)
# ---------------------------------------------------------------------------


def test_add_features_performance_smoke() -> None:
    df = random_walk_df(2000)
    add_features(df)  # warm-up run (imports, caches) excluded from timing
    times = []
    for _ in range(5):
        start = time.perf_counter()
        out = add_features(df)
        times.append(time.perf_counter() - start)
        assert len(out) == 2000
    median = sorted(times)[2]
    logger.info(
        "add_features on 2000 rows: median %.1f ms over 5 runs (budget 500 ms)",
        median * 1000.0,
    )
    assert median < 0.5, (
        f"add_features on 2000 rows took {median * 1000.0:.1f} ms median "
        f"(contract budget: < 500 ms)"
    )
