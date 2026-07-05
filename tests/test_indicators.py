"""Tests for backend/indicators — pure-function technical indicators.

Covers hand-checked SMA/EMA values, RSI bounds, the MACD identity, Bollinger
band ordering, ATR positivity, the exact ``add_features`` column set and the
no-lookahead property (prefix stability) of the feature pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.indicators.features import add_features
from backend.indicators.technical import atr, bollinger, ema, macd, rsi, sma

FEATURE_COLUMNS = [
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
]
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def test_sma_hand_checked() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(s, 3)
    assert len(result) == len(s)
    assert np.allclose(result.iloc[2:].to_numpy(), [2.0, 3.0, 4.0])


def test_ema_hand_checked() -> None:
    # ewm(span=3, adjust=False) → alpha = 0.5:
    # 1, 1.5, 2.25, 3.125, 4.0625
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ema(s, 3)
    assert np.allclose(result.to_numpy(), [1.0, 1.5, 2.25, 3.125, 4.0625])


def test_rsi_bounds(sample_df: pd.DataFrame) -> None:
    result = rsi(sample_df["close"], 14)
    assert len(result) == len(sample_df)
    valid = result.dropna()
    assert len(valid) > 0
    assert ((valid >= 0.0) & (valid <= 100.0)).all()


def test_macd_relation(sample_df: pd.DataFrame) -> None:
    close = sample_df["close"]
    macd_line, signal_line, hist = macd(close)

    line_err = (macd_line - (ema(close, 12) - ema(close, 26))).abs().dropna()
    assert (line_err < 1e-8).all()

    hist_err = (hist - (macd_line - signal_line)).abs().dropna()
    assert (hist_err < 1e-8).all()


def test_bollinger_ordering(sample_df: pd.DataFrame) -> None:
    upper, mid, lower = bollinger(sample_df["close"], 20, 2.0)
    valid = upper.notna() & mid.notna() & lower.notna()
    assert valid.any()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_atr_positive(sample_df: pd.DataFrame) -> None:
    result = atr(sample_df, 14)
    assert len(result) == len(sample_df)
    valid = result.dropna()
    assert len(valid) > 0
    assert (valid > 0).all()


def test_add_features_exact_columns(sample_df: pd.DataFrame) -> None:
    features = add_features(sample_df)
    assert set(features.columns) == set(OHLCV_COLUMNS) | set(FEATURE_COLUMNS)
    assert features.index.equals(sample_df.index)
    # Input frame must not be mutated.
    assert list(sample_df.columns) == OHLCV_COLUMNS


def test_no_lookahead(sample_df: pd.DataFrame) -> None:
    """Features computed on df[:100] must equal the first 100 rows of the
    features computed on the full frame — i.e. row t only uses data <= t."""
    full = add_features(sample_df)
    head = add_features(sample_df.iloc[:100].copy())
    pd.testing.assert_frame_equal(
        full.iloc[:100],
        head,
        check_freq=False,
        rtol=1e-9,
        atol=1e-9,
    )
