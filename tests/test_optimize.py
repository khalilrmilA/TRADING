"""Tests for scripts/optimize.py — objective wiring, under-trading penalty,
cache-only data loading, coarse search-space steps, seeded reproducibility
and the train/test report structure.

Fast smoke tests only: tiny synthetic frames, <= 5 optuna trials, no network
(the optimizer reads CACHED candles exclusively; conftest blocks sockets).
"""

from __future__ import annotations

import math

import optuna
import pandas as pd
import pytest

from backend.database.db import upsert_ohlcv
from backend.indicators.features import add_features
from scripts.optimize import (
    FITNESS_METRIC,
    OVERFIT_TEST_FRACTION,
    UNDER_TRADED_PENALTY,
    fitness,
    is_overfit,
    load_candles,
    make_objective,
    run_optimization,
    split_point,
    suggest_config,
)

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


# --------------------------------------------------------------------------- #
# Fitness / flags                                                             #
# --------------------------------------------------------------------------- #


def test_fitness_metric_is_sortino() -> None:
    # metrics.py provides sortino, so the optimizer must maximize it.
    assert FITNESS_METRIC == "sortino"


def test_fitness_penalizes_under_trading() -> None:
    metrics = {"sortino": 5.0, "num_trades": 3.0}
    score = fitness(metrics, min_trades=20)
    assert score == UNDER_TRADED_PENALTY + 3.0
    assert score < -900.0  # large negative — cannot win by barely trading


def test_fitness_returns_sortino_when_enough_trades() -> None:
    metrics = {"sortino": 1.25, "num_trades": 25.0}
    assert fitness(metrics, min_trades=20) == pytest.approx(1.25)


def test_fitness_penalty_keeps_gradient_toward_more_trades() -> None:
    few = fitness({"sortino": 9.0, "num_trades": 2.0}, min_trades=20)
    more = fitness({"sortino": -9.0, "num_trades": 10.0}, min_trades=20)
    assert more > few  # more trades score less badly, sortino irrelevant


def test_is_overfit_flag_logic() -> None:
    train = {FITNESS_METRIC: 2.0}
    collapsed = {FITNESS_METRIC: OVERFIT_TEST_FRACTION * 2.0 - 0.1}
    held_up = {FITNESS_METRIC: OVERFIT_TEST_FRACTION * 2.0 + 0.1}
    assert is_overfit(train, collapsed) is True
    assert is_overfit(train, held_up) is False
    # A negative train score never flags (nothing was "won" in-sample).
    assert is_overfit({FITNESS_METRIC: -1.0}, {FITNESS_METRIC: -5.0}) is False


def test_split_point_defaults_to_70_30_and_clamps() -> None:
    assert split_point(300, 0.7) == 210
    assert split_point(300, 0.99) == 270  # clamped to 0.9
    assert split_point(300, 0.01) == 150  # clamped to 0.5


# --------------------------------------------------------------------------- #
# Cache-only data loading                                                     #
# --------------------------------------------------------------------------- #


def test_load_candles_errors_clearly_when_cache_empty() -> None:
    with pytest.raises(ValueError, match="No cached candles"):
        load_candles("binance", "BTCUSDT", "1h")


def test_load_candles_errors_when_too_few_rows(sample_df: pd.DataFrame) -> None:
    upsert_ohlcv("binance", "BTCUSDT", "1h", sample_df.iloc[:50])
    with pytest.raises(ValueError, match="need at least"):
        load_candles("binance", "BTCUSDT", "1h")


def test_load_candles_roundtrip(sample_df: pd.DataFrame) -> None:
    upsert_ohlcv("binance", "BTCUSDT", "1h", sample_df)
    df = load_candles("binance", "BTCUSDT", "1h")
    assert len(df) == len(sample_df)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None


# --------------------------------------------------------------------------- #
# Objective wiring (tiny optuna smoke — 5 trials, no network)                 #
# --------------------------------------------------------------------------- #


def test_objective_smoke_on_synthetic_frame(sample_df: pd.DataFrame) -> None:
    featured = add_features(sample_df)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=7)
    )
    study.optimize(
        make_objective(featured, "ensemble", min_trades=1, symbol="TESTUSDT"),
        n_trials=5,
    )
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    assert len(completed) == 5
    for trial in completed:
        assert math.isfinite(trial.value)
        assert set(trial.user_attrs["train_metrics"]) == EXPECTED_METRIC_KEYS
        config = trial.user_attrs["config"]
        assert set(config["strategy_params"]) == {
            "threshold",
            "mean_reversion",
            "breakout",
            "rsi_macd",
        }
        # Coarse, docs-recommended steps: multipliers land on 0.5 grid etc.
        assert (2 * config["atr_stop_multiplier"]) == int(
            2 * config["atr_stop_multiplier"]
        )
        assert 1.0 <= config["atr_stop_multiplier"] <= 4.0
        assert 1.5 <= config["atr_take_profit_multiplier"] <= 6.0
        assert config["strategy_params"]["threshold"] in (2, 3, 4)
        assert config["strategy_params"]["breakout"]["entry_window"] % 5 == 0


def test_suggest_config_trend_following_has_only_atr_params() -> None:
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=1)
    )
    trial = study.ask()
    config = suggest_config(trial, "trend_following")
    assert config["strategy_params"] == {}
    assert set(trial.params) == {
        "atr_stop_multiplier",
        "atr_take_profit_multiplier",
    }


def test_seeded_study_is_reproducible(sample_df: pd.DataFrame) -> None:
    featured = add_features(sample_df)

    def _best_params() -> dict:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            make_objective(featured, "trend_following", min_trades=1),
            n_trials=5,
        )
        return study.best_trial.params

    assert _best_params() == _best_params()


# --------------------------------------------------------------------------- #
# End-to-end pipeline report                                                  #
# --------------------------------------------------------------------------- #


def test_run_optimization_report_structure(sample_df: pd.DataFrame) -> None:
    report = run_optimization(
        sample_df,
        "trend_following",
        symbol="TESTUSDT",
        timeframe="1h",
        trials=5,
        train_ratio=0.7,
        min_trades=1,
        top_n=2,
        seed=42,
    )
    assert report["symbol"] == "TESTUSDT"
    assert report["strategy"] == "trend_following"
    assert report["fitness_metric"] == FITNESS_METRIC
    assert report["train_rows"] == split_point(len(sample_df), 0.7)
    assert report["train_rows"] + report["test_rows"] == report["rows"]

    baseline = report["baseline"]
    assert set(baseline["train_metrics"]) == EXPECTED_METRIC_KEYS
    assert set(baseline["test_metrics"]) == EXPECTED_METRIC_KEYS
    assert isinstance(baseline["overfit"], bool)

    assert 1 <= len(report["candidates"]) <= 2
    seen_params: list[dict] = []
    for cand in report["candidates"]:
        assert set(cand["train_metrics"]) == EXPECTED_METRIC_KEYS
        assert set(cand["test_metrics"]) == EXPECTED_METRIC_KEYS
        assert isinstance(cand["overfit"], bool)
        assert math.isfinite(cand["score"])
        assert cand["params"] not in seen_params  # deduplicated configs
        seen_params.append(cand["params"])

    # Chronological split: test range starts after the train range ends.
    assert report["train_range"][1] < report["test_range"][0]


def test_run_optimization_rejects_unknown_strategy(
    sample_df: pd.DataFrame,
) -> None:
    with pytest.raises(KeyError, match="Unknown strategy"):
        run_optimization(sample_df, "does_not_exist", trials=1)
