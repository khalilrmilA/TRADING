"""Optuna hyperparameter optimization over the existing backtester.

PAPER TRADING / RESEARCH ONLY — pure simulation over CACHED candles. This
script NEVER fetches from the network: if the requested (source, symbol,
timeframe) is not already in the local SQLite cache it errors out with a hint
to fetch data through the API/dashboard first.

Method (overfitting-aware by construction):
    * Chronological train/test split (default 70/30). The optimizer only ever
      sees the TRAIN window.
    * Search space: the ensemble vote threshold, each sub-strategy's tunable
      parameters (as exposed by the strategy classes' ``default_params``) and
      the backtester's ATR stop/take-profit multipliers. All parameters are
      stepped COARSELY on purpose — a docs-recommended guardrail against
      overfitting to noise.
    * Fitness: the Sortino ratio when ``backend.backtest.metrics`` provides
      it (it does), else Sharpe. Configurations with fewer than
      ``--min-trades`` (default 20) trades score a large negative penalty so
      the optimizer cannot win by barely trading.
    * Honesty check: the best N configs are re-run on the held-out TEST
      window and reported train-vs-test side by side; a config whose test
      score is below 50% of its train score is flagged OVERFIT. A defaults
      baseline row is included for reference.
    * Optuna TPE sampler with a fixed seed — runs are reproducible.

Usage:
    python scripts/optimize.py --symbol BTCUSDT --source binance
                               --timeframe 1h --trials 100
                               [--strategy ensemble] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/optimize.py`
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import pandas as pd

from backend.backtest.engine import Backtester, BacktestConfig
from backend.backtest.metrics import METRIC_KEYS
from backend.database.db import load_ohlcv
from backend.indicators.features import add_features
from backend.strategies import STRATEGIES, get_strategy
from config.settings import settings

#: Metric maximized on the TRAIN window (contract-guaranteed key).
FITNESS_METRIC: str = "sortino" if "sortino" in METRIC_KEYS else "sharpe"

#: Score assigned to under-traded configs (plus num_trades, so the TPE
#: sampler still sees a gradient toward "trades more").
UNDER_TRADED_PENALTY: float = -1_000.0

#: A config is flagged OVERFIT when test score < this fraction of train score.
OVERFIT_TEST_FRACTION: float = 0.5

#: Minimum cached rows required for a meaningful split + indicator warm-up.
MIN_ROWS: int = 200


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #


def load_candles(source: str, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load cached candles from the local DB — never from the network.

    Args:
        source: Data source key (``binance`` / ``bybit`` / ``yahoo``).
        symbol: Symbol in the source's native format, e.g. ``BTCUSDT``.
        timeframe: Candle timeframe, e.g. ``1h``.

    Returns:
        Canonical OHLCV frame with at least :data:`MIN_ROWS` rows.

    Raises:
        ValueError: When nothing (or too little) is cached for the triple.
    """
    df = load_ohlcv(source, symbol, timeframe)
    if df.empty:
        raise ValueError(
            f"No cached candles for {source}:{symbol}:{timeframe} in "
            f"{settings.db_path} - fetch data first (POST /api/data/fetch or "
            "the dashboard sidebar). This tool never fetches from the network."
        )
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Only {len(df)} cached candles for {source}:{symbol}:{timeframe} "
            f"- need at least {MIN_ROWS} for a train/test split with "
            "indicator warm-up. Fetch more history first."
        )
    return df


def split_point(n_rows: int, train_ratio: float) -> int:
    """Return the chronological train/test boundary row index.

    Args:
        n_rows: Total number of candles.
        train_ratio: Fraction of rows in the train window (clamped to
            ``[0.5, 0.9]`` so both windows stay meaningful).

    Returns:
        Index of the first TEST row (``train`` is ``[0, split)``).
    """
    ratio = min(max(float(train_ratio), 0.5), 0.9)
    return int(round(n_rows * ratio))


# --------------------------------------------------------------------------- #
# Search space                                                                #
# --------------------------------------------------------------------------- #


def suggest_config(trial: optuna.Trial, strategy_name: str) -> dict[str, Any]:
    """Suggest one coarsely-stepped configuration for ``strategy_name``.

    The space covers the ensemble vote threshold, every tunable parameter the
    strategy classes expose in ``default_params``, and the backtester's ATR
    stop/take-profit multipliers. Steps are deliberately coarse (0.25-5.0
    increments) — the standard guardrail against fitting noise.

    Args:
        trial: The optuna trial proposing values.
        strategy_name: Registry key (``ensemble`` or a single strategy).

    Returns:
        ``{"strategy_params": {...}, "atr_stop_multiplier": float,
        "atr_take_profit_multiplier": float}`` ready for
        :func:`backtest_window`.
    """
    strategy_params: dict[str, Any] = {}

    def _mean_reversion(prefix: str) -> dict[str, float]:
        return {
            "rsi_oversold": trial.suggest_float(
                f"{prefix}rsi_oversold", 20.0, 40.0, step=5.0
            ),
            "rsi_overbought": trial.suggest_float(
                f"{prefix}rsi_overbought", 60.0, 80.0, step=5.0
            ),
        }

    def _breakout(prefix: str) -> dict[str, Any]:
        return {
            "entry_window": trial.suggest_int(
                f"{prefix}entry_window", 10, 40, step=5
            ),
            "exit_window": trial.suggest_int(
                f"{prefix}exit_window", 5, 20, step=5
            ),
            "volume_mult": trial.suggest_float(
                f"{prefix}volume_mult", 1.0, 2.5, step=0.25
            ),
        }

    def _rsi_macd(prefix: str) -> dict[str, float]:
        return {
            "rsi_threshold": trial.suggest_float(
                f"{prefix}rsi_threshold", 40.0, 60.0, step=5.0
            ),
        }

    if strategy_name == "ensemble":
        strategy_params["threshold"] = trial.suggest_int("threshold", 2, 4)
        strategy_params["mean_reversion"] = _mean_reversion("mr_")
        strategy_params["breakout"] = _breakout("bo_")
        strategy_params["rsi_macd"] = _rsi_macd("rm_")
    elif strategy_name == "mean_reversion":
        strategy_params = _mean_reversion("")
    elif strategy_name == "breakout":
        strategy_params = _breakout("")
    elif strategy_name == "rsi_macd":
        strategy_params = _rsi_macd("")
    # trend_following exposes no tunables — only the ATR geometry below.

    return {
        "strategy_params": strategy_params,
        "atr_stop_multiplier": trial.suggest_float(
            "atr_stop_multiplier", 1.0, 4.0, step=0.5
        ),
        "atr_take_profit_multiplier": trial.suggest_float(
            "atr_take_profit_multiplier", 1.5, 6.0, step=0.5
        ),
    }


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #


def backtest_window(
    window_df: pd.DataFrame,
    strategy_name: str,
    config: dict[str, Any],
    symbol: str = "",
) -> dict[str, float]:
    """Backtest one config over one feature-enriched candle window.

    Args:
        window_df: Feature-enriched frame for the window (train or test).
        strategy_name: Registry key for ``get_strategy``.
        config: Dict from :func:`suggest_config` (``strategy_params`` +
            ATR multipliers).
        symbol: Symbol label stamped on trades (informational).

    Returns:
        The contract metrics dict from :class:`Backtester`.
    """
    bt_config = BacktestConfig(
        atr_stop_multiplier=float(config["atr_stop_multiplier"]),
        atr_take_profit_multiplier=float(config["atr_take_profit_multiplier"]),
    )
    strategy = get_strategy(strategy_name, **config.get("strategy_params", {}))
    result = Backtester(bt_config).run(window_df, strategy, symbol=symbol)
    return result.metrics


def fitness(metrics: dict[str, float], min_trades: int) -> float:
    """Score a train-window result (higher is better).

    Args:
        metrics: Contract metrics dict from a backtest.
        min_trades: Minimum trade count; below it the config scores
            :data:`UNDER_TRADED_PENALTY` + num_trades so the optimizer cannot
            win by barely trading.

    Returns:
        The :data:`FITNESS_METRIC` value, or the large negative penalty.
    """
    num_trades = int(metrics.get("num_trades", 0.0))
    if num_trades < min_trades:
        return UNDER_TRADED_PENALTY + float(num_trades)
    return float(metrics.get(FITNESS_METRIC, 0.0))


def make_objective(
    featured_train: pd.DataFrame,
    strategy_name: str,
    min_trades: int,
    symbol: str = "",
):
    """Build the optuna objective closure over the TRAIN window only.

    Args:
        featured_train: Feature-enriched train-window frame.
        strategy_name: Registry key for ``get_strategy``.
        min_trades: Under-trading penalty threshold (see :func:`fitness`).
        symbol: Symbol label for trade records.

    Returns:
        ``objective(trial) -> float`` storing ``config`` and
        ``train_metrics`` in the trial's user attrs.
    """

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial, strategy_name)
        metrics = backtest_window(
            featured_train, strategy_name, config, symbol=symbol
        )
        trial.set_user_attr("config", config)
        trial.set_user_attr("train_metrics", metrics)
        return fitness(metrics, min_trades)

    return objective


def is_overfit(
    train_metrics: dict[str, float], test_metrics: dict[str, float]
) -> bool:
    """Flag a config whose test score collapses versus its train score.

    Args:
        train_metrics: Train-window metrics.
        test_metrics: Test-window metrics.

    Returns:
        True when train score is positive and test score is below
        :data:`OVERFIT_TEST_FRACTION` of it.
    """
    train_score = float(train_metrics.get(FITNESS_METRIC, 0.0))
    test_score = float(test_metrics.get(FITNESS_METRIC, 0.0))
    return train_score > 0.0 and test_score < OVERFIT_TEST_FRACTION * train_score


def top_candidates(
    study: optuna.Study, top_n: int
) -> list[optuna.trial.FrozenTrial]:
    """Return the best ``top_n`` completed trials with distinct params.

    Args:
        study: Finished optuna study (maximize direction).
        top_n: Number of distinct configs to keep.

    Returns:
        Frozen trials sorted best-first, deduplicated on their param dicts.
    """
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda t: t.value, reverse=True)
    seen: set[str] = set()
    unique: list[optuna.trial.FrozenTrial] = []
    for trial in completed:
        key = json.dumps(trial.params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(trial)
        if len(unique) >= top_n:
            break
    return unique


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #


def run_optimization(
    raw: pd.DataFrame,
    strategy_name: str,
    symbol: str = "",
    timeframe: str = "1h",
    trials: int = 100,
    train_ratio: float = 0.7,
    min_trades: int = 20,
    top_n: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Optimize on the train window, then verify the best configs on test.

    Args:
        raw: Canonical raw OHLCV frame (cached candles, ascending).
        strategy_name: Registry key (default caller value: ``ensemble``).
        symbol: Symbol label for trades and the report header.
        timeframe: Timeframe label for the report header.
        trials: Optuna trial budget (clamped to >= 1).
        train_ratio: Chronological split fraction (default 0.7 = 70/30).
        min_trades: Under-trading penalty threshold.
        top_n: Number of best distinct configs re-run on the test window.
        seed: TPE sampler seed (reproducible runs).

    Returns:
        JSON-safe report dict: run header, baseline row and ``candidates``
        with train/test metrics side by side and per-config overfit flags.

    Raises:
        KeyError: When ``strategy_name`` is not a registered strategy.
        ValueError: When the frame is too small to split.
    """
    if strategy_name not in STRATEGIES:
        valid = ", ".join(sorted(STRATEGIES))
        raise KeyError(
            f"Unknown strategy {strategy_name!r}. Valid strategies: {valid}"
        )
    n_rows = len(raw)
    if n_rows < MIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_ROWS} candles, got {n_rows}"
        )
    trials = max(int(trials), 1)
    top_n = max(int(top_n), 1)
    min_trades = max(int(min_trades), 0)

    split = split_point(n_rows, train_ratio)
    # Features are hyperparameter-independent, so each window is enriched
    # ONCE and reused across all trials. No lookahead either way: the train
    # frame only sees train rows, and the test slice's indicator warm-up
    # happens inside the train region (same scheme as walk_forward).
    featured_train = add_features(raw.iloc[:split])
    featured_full = add_features(raw)
    test_df = featured_full.iloc[split:]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        make_objective(featured_train, strategy_name, min_trades, symbol),
        n_trials=trials,
    )

    # Defaults baseline: default strategy params + settings ATR multipliers.
    baseline_config: dict[str, Any] = {
        "strategy_params": {},
        "atr_stop_multiplier": settings.atr_stop_multiplier,
        "atr_take_profit_multiplier": settings.atr_take_profit_multiplier,
    }
    baseline_train = backtest_window(
        featured_train, strategy_name, baseline_config, symbol=symbol
    )
    baseline_test = backtest_window(
        test_df, strategy_name, baseline_config, symbol=symbol
    )

    candidates: list[dict[str, Any]] = []
    for rank, trial in enumerate(top_candidates(study, top_n), start=1):
        config = trial.user_attrs["config"]
        train_metrics = trial.user_attrs["train_metrics"]
        test_metrics = backtest_window(
            test_df, strategy_name, config, symbol=symbol
        )
        candidates.append(
            {
                "rank": rank,
                "trial": trial.number,
                "score": float(trial.value),
                "params": dict(trial.params),
                "config": config,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "overfit": is_overfit(train_metrics, test_metrics),
            }
        )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "rows": n_rows,
        "train_rows": split,
        "test_rows": n_rows - split,
        "train_range": [raw.index[0].isoformat(), raw.index[split - 1].isoformat()],
        "test_range": [raw.index[split].isoformat(), raw.index[-1].isoformat()],
        "fitness_metric": FITNESS_METRIC,
        "min_trades": min_trades,
        "trials": trials,
        "seed": seed,
        "baseline": {
            "config": baseline_config,
            "train_metrics": baseline_train,
            "test_metrics": baseline_test,
            "overfit": is_overfit(baseline_train, baseline_test),
        },
        "candidates": candidates,
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def _metric_cells(metrics: dict[str, float]) -> str:
    """Format one window's metrics as fixed-width table cells."""
    return (
        f"{metrics.get('sortino', 0.0):>8.3f} {metrics.get('sharpe', 0.0):>7.3f} "
        f"{100 * metrics.get('total_return', 0.0):>7.2f}% "
        f"{int(metrics.get('num_trades', 0.0)):>4d} "
        f"{100 * metrics.get('max_drawdown', 0.0):>6.2f}%"
    )


def print_report(report: dict[str, Any]) -> None:
    """Print the console report for a finished optimization run."""
    # Console output stays ASCII — Windows consoles often decode cp1252.
    print(
        f"# Optuna hyperopt - {report['symbol']} {report['timeframe']} - "
        f"strategy '{report['strategy']}' (PAPER/RESEARCH ONLY)"
    )
    print(
        f"rows: {report['rows']} | train: {report['train_rows']} "
        f"[{report['train_range'][0]} .. {report['train_range'][1]}] | "
        f"test: {report['test_rows']} "
        f"[{report['test_range'][0]} .. {report['test_range'][1]}]"
    )
    print(
        f"fitness: {report['fitness_metric']} (min {report['min_trades']} "
        f"trades) | sampler: TPE(seed={report['seed']}) | "
        f"trials: {report['trials']}"
    )

    header = (
        f"{'rank':<5} {'score':>9} | "
        f"{'sortino':>8} {'sharpe':>7} {'return':>8} {'trds':>4} {'maxDD':>7} | "
        f"{'sortino':>8} {'sharpe':>7} {'return':>8} {'trds':>4} {'maxDD':>7} | flag"
    )
    print("\n## Train vs test - the honest overfitting check")
    print(f"{'':<5} {'':>9} | {'TRAIN':^39} | {'TEST':^39} |")
    print(header)

    def _row(label: str, score: str, entry: dict[str, Any]) -> None:
        flag = "OVERFIT" if entry["overfit"] else "ok"
        print(
            f"{label:<5} {score:>9} | {_metric_cells(entry['train_metrics'])} | "
            f"{_metric_cells(entry['test_metrics'])} | {flag}"
        )

    _row("BASE", "-", report["baseline"])
    for cand in report["candidates"]:
        _row(str(cand["rank"]), f"{cand['score']:.3f}", cand)

    print(
        f"\nOVERFIT = test {report['fitness_metric']} < "
        f"{int(100 * OVERFIT_TEST_FRACTION)}% of train "
        f"{report['fitness_metric']} (train > 0). BASE = default parameters."
    )
    for cand in report["candidates"]:
        print(f"\n## Rank {cand['rank']} (trial #{cand['trial']}) params")
        print(f"  {json.dumps(cand['params'], sort_keys=True)}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Optuna hyperopt over the backtester using CACHED candles only "
            "(paper/research; never fetches from the network)."
        )
    )
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--strategy",
        default="ensemble",
        choices=sorted(STRATEGIES),
        help="strategy to optimize (default: ensemble)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--min-trades",
        type=int,
        default=20,
        help="train-window trade count below which a config is penalized",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json", type=Path, help="also write the full report as JSON"
    )
    args = parser.parse_args()

    try:
        raw = load_candles(args.source, args.symbol, args.timeframe)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    report = run_optimization(
        raw,
        args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        trials=args.trials,
        train_ratio=args.train_ratio,
        min_trades=args.min_trades,
        top_n=args.top,
        seed=args.seed,
    )
    print_report(report)
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
        )
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
