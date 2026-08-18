"""Tests for backend/database/db.py — upsert idempotency and load roundtrips.

The autouse ``_fresh_db`` conftest fixture points ``settings.db_path`` at a
per-test temporary file and runs ``init_db()``, so every test starts empty.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.database.db import (
    list_cached,
    load_ohlcv,
    ohlcv_coverage,
    upsert_ohlcv,
)

SOURCE = "binance"
SYMBOL = "TESTUSDT"
TIMEFRAME = "1h"


def test_upsert_is_idempotent(sample_df: pd.DataFrame) -> None:
    assert upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, sample_df) == len(sample_df)
    assert upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, sample_df) == len(sample_df)
    loaded = load_ohlcv(SOURCE, SYMBOL, TIMEFRAME)
    assert len(loaded) == len(sample_df)  # no duplicate rows


def test_upsert_replaces_existing_candle(sample_df: pd.DataFrame) -> None:
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, sample_df)
    patched = sample_df.iloc[[10]].copy()
    patched.loc[:, "close"] = 123.456
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, patched)
    loaded = load_ohlcv(SOURCE, SYMBOL, TIMEFRAME)
    assert len(loaded) == len(sample_df)
    assert loaded["close"].iloc[10] == pytest.approx(123.456)


def test_load_roundtrip_preserves_values_and_tz(sample_df: pd.DataFrame) -> None:
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, sample_df)
    loaded = load_ohlcv(SOURCE, SYMBOL, TIMEFRAME)

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert str(loaded.index.tz) == "UTC"
    assert loaded.index.name == "timestamp"
    assert loaded.index.is_monotonic_increasing
    assert loaded.index.is_unique
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert all(str(dtype) == "float64" for dtype in loaded.dtypes)

    pd.testing.assert_frame_equal(loaded, sample_df, check_freq=False)


def test_load_limit_returns_most_recent_rows_ascending(
    sample_df: pd.DataFrame,
) -> None:
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, sample_df)
    loaded = load_ohlcv(SOURCE, SYMBOL, TIMEFRAME, limit=50)
    assert len(loaded) == 50
    assert loaded.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(loaded, sample_df.iloc[-50:], check_freq=False)


def test_load_start_end_filters(sample_df: pd.DataFrame) -> None:
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, sample_df)
    start = sample_df.index[100]
    end = sample_df.index[199]
    loaded = load_ohlcv(SOURCE, SYMBOL, TIMEFRAME, start=start, end=end)
    assert len(loaded) == 100
    pd.testing.assert_frame_equal(loaded, sample_df.iloc[100:200], check_freq=False)


def test_load_missing_symbol_returns_empty_canonical_frame() -> None:
    loaded = load_ohlcv(SOURCE, "NOPEUSDT", TIMEFRAME)
    assert loaded.empty
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert str(loaded.index.tz) == "UTC"
    assert loaded.index.name == "timestamp"


def test_ohlcv_coverage(sample_df: pd.DataFrame) -> None:
    assert ohlcv_coverage(SOURCE, SYMBOL, TIMEFRAME) is None
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, sample_df)
    coverage = ohlcv_coverage(SOURCE, SYMBOL, TIMEFRAME)
    assert coverage is not None
    first, last = coverage
    assert pd.Timestamp(first) == sample_df.index[0]
    assert pd.Timestamp(last) == sample_df.index[-1]


def test_list_cached(sample_df: pd.DataFrame) -> None:
    assert list_cached() == []
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, sample_df)
    items = list_cached()
    assert len(items) == 1
    item = items[0]
    assert item["source"] == SOURCE
    assert item["symbol"] == SYMBOL
    assert item["timeframe"] == TIMEFRAME
    assert item["rows"] == len(sample_df)
    assert pd.Timestamp(item["first_ts"]) == sample_df.index[0]
    assert pd.Timestamp(item["last_ts"]) == sample_df.index[-1]
