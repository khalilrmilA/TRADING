"""Tests for backend/backtest — engine determinism, commission drag,
metric keys, drawdown sign convention and walk-forward structure.

Uses a synthetic monotonic uptrend so ``trend_following`` MUST be profitable.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from backend.backtest.engine import Backtester, BacktestConfig
from backend.backtest.metrics import compute_metrics, periods_per_year_for
from backend.backtest.walk_forward import walk_forward
from backend.indicators.features import add_features
from backend.strategies import get_strategy

EXPECTED_METRIC_KEYS = {
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "num_trades",
    "avg_trade_pnl",
    "avg_win",
    "avg_loss",
    "exposure",
}

TIMEFRAME_PERIODS = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}


def _uptrend_df(n: int = 300, start_price: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """Deterministic monotonic uptrend canonical OHLCV frame (1h UTC)."""
    index = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    close = start_price + step * np.arange(n, dtype=np.float64)
    open_ = close - step / 2.0
    high = close + step
    low = open_ - step
    volume = np.full(n, 500.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")


@pytest.fixture()
def uptrend_features() -> pd.DataFrame:
    return add_features(_uptrend_df())


def test_trend_following_profits_on_uptrend(uptrend_features: pd.DataFrame) -> None:
    config = BacktestConfig()
    result = Backtester(config).run(
        uptrend_features, get_strategy("trend_following"), symbol="TESTUSDT"
    )
    assert len(result.trades) > 0
    assert result.metrics["total_return"] > 0
    assert result.equity_curve.iloc[-1] > config.initial_capital


def test_backtest_is_deterministic(uptrend_features: pd.DataFrame) -> None:
    r1 = Backtester().run(uptrend_features, get_strategy("trend_following"), symbol="T")
    r2 = Backtester().run(uptrend_features, get_strategy("trend_following"), symbol="T")
    pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)
    assert r1.metrics == r2.metrics
    assert len(r1.trades) == len(r2.trades)


def test_equity_curve_shape(uptrend_features: pd.DataFrame) -> None:
    config = BacktestConfig()
    result = Backtester(config).run(
        uptrend_features, get_strategy("trend_following"), symbol="TESTUSDT"
    )
    assert result.equity_curve.index.equals(uptrend_features.index)
    assert result.equity_curve.iloc[0] == pytest.approx(config.initial_capital)


def test_commissions_reduce_pnl(uptrend_features: pd.DataFrame) -> None:
    strat = get_strategy("trend_following")
    free = Backtester(BacktestConfig(commission_rate=0.0)).run(
        uptrend_features, strat, symbol="TESTUSDT"
    )
    costly = Backtester(BacktestConfig(commission_rate=0.005)).run(
        uptrend_features, strat, symbol="TESTUSDT"
    )
    assert len(free.trades) > 0
    assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
    assert costly.metrics["total_return"] < free.metrics["total_return"]


def test_metrics_keys_exact_and_json_safe(uptrend_features: pd.DataFrame) -> None:
    result = Backtester().run(
        uptrend_features, get_strategy("trend_following"), symbol="TESTUSDT"
    )
    assert set(result.metrics.keys()) == EXPECTED_METRIC_KEYS
    json.dumps(result.metrics)  # must not raise
    for key, value in result.metrics.items():
        assert isinstance(value, (int, float)), f"{key} is not numeric"
        assert math.isfinite(float(value)), f"{key} is not finite"


def test_trade_fields_valid(uptrend_features: pd.DataFrame) -> None:
    result = Backtester().run(
        uptrend_features, get_strategy("trend_following"), symbol="TESTUSDT"
    )
    assert len(result.trades) > 0
    for trade in result.trades:
        assert trade.side in {"long", "short"}
        assert trade.exit_reason in {"signal", "stop_loss", "take_profit", "end_of_data"}
        assert trade.qty > 0
        assert trade.entry_price > 0
        assert trade.exit_price > 0


def test_result_to_dict_json_safe(uptrend_features: pd.DataFrame) -> None:
    result = Backtester().run(
        uptrend_features, get_strategy("trend_following"), symbol="TESTUSDT"
    )
    payload = result.to_dict()
    json.dumps(payload)  # must not raise
    assert set(payload["equity_curve"].keys()) == {"timestamps", "values"}
    assert len(payload["equity_curve"]["timestamps"]) == len(uptrend_features)
    assert isinstance(payload["trades"], list)
    assert set(payload["metrics"].keys()) == EXPECTED_METRIC_KEYS


def test_max_drawdown_positive_fraction() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    equity = pd.Series(
        [100_000.0, 110_000.0, 99_000.0, 105_000.0, 120_000.0], index=index
    )
    metrics = compute_metrics(equity, [], 8_760)
    # Peak 110k → trough 99k = 10% drawdown, reported as a POSITIVE fraction.
    assert metrics["max_drawdown"] == pytest.approx(0.1)
    assert metrics["max_drawdown"] >= 0
    assert metrics["total_return"] == pytest.approx(0.2)
    assert metrics["num_trades"] == 0


def test_periods_per_year_for() -> None:
    for timeframe, expected in TIMEFRAME_PERIODS.items():
        assert periods_per_year_for(timeframe) == expected


def test_walk_forward_fold_count() -> None:
    df = _uptrend_df(n=1_200)
    result = walk_forward(df, "trend_following", n_splits=4)
    assert len(result["folds"]) == 4
    for fold in result["folds"]:
        assert {"fold", "train_start", "test_start", "test_end", "metrics"} <= set(
            fold.keys()
        )
        assert isinstance(fold["metrics"], dict)
    assert isinstance(result["aggregate"], dict)
    assert len(result["aggregate"]) > 0
