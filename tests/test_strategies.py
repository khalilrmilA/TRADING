"""Tests for backend/strategies — registry, signal contract and voting math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.indicators.features import add_features
from backend.strategies import STRATEGIES, get_strategy, list_strategies
from backend.strategies.base import BaseStrategy

EXPECTED_NAMES = {
    "trend_following",
    "mean_reversion",
    "breakout",
    "rsi_macd",
    "ensemble",
}


@pytest.fixture()
def features(sample_df: pd.DataFrame) -> pd.DataFrame:
    return add_features(sample_df)


def test_registry_completeness() -> None:
    assert set(STRATEGIES.keys()) == EXPECTED_NAMES
    for name, cls in STRATEGIES.items():
        assert issubclass(cls, BaseStrategy)
        assert cls.name == name


def test_get_strategy_returns_instance() -> None:
    strat = get_strategy("trend_following")
    assert isinstance(strat, BaseStrategy)
    assert strat.name == "trend_following"


def test_get_strategy_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_strategy("definitely_not_a_strategy")


def test_list_strategies_shape() -> None:
    items = list_strategies()
    assert isinstance(items, list)
    assert {item["name"] for item in items} == EXPECTED_NAMES
    for item in items:
        assert "description" in item
        assert "default_params" in item


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_signal_values_and_dtype(name: str, features: pd.DataFrame) -> None:
    out = get_strategy(name).generate_signals(features)
    assert "signal" in out.columns
    assert str(out["signal"].dtype) == "int8"
    assert out.index.equals(features.index)
    values = {int(v) for v in np.unique(out["signal"].to_numpy())}
    assert values.issubset({-1, 0, 1})


def test_generate_signals_returns_copy(features: pd.DataFrame) -> None:
    before_cols = list(features.columns)
    out = get_strategy("trend_following").generate_signals(features)
    assert out is not features
    assert list(features.columns) == before_cols
    assert "signal" not in features.columns


def test_voting_math(features: pd.DataFrame) -> None:
    """Ensemble = sum of the 4 component signals: >=2 -> 1, <=-2 -> -1, else 0."""
    strat = get_strategy("ensemble")
    breakdown = strat.vote_breakdown(features)

    assert "ensemble" in breakdown.columns
    components = [c for c in breakdown.columns if c != "ensemble"]
    assert len(components) == 4

    total = breakdown[components].sum(axis=1).to_numpy()
    expected = np.where(total >= 2, 1, np.where(total <= -2, -1, 0))
    actual = breakdown["ensemble"].to_numpy().astype(np.int64)
    assert np.array_equal(actual, expected.astype(np.int64))

    signals = strat.generate_signals(features)["signal"].to_numpy().astype(np.int64)
    assert np.array_equal(signals, actual)
