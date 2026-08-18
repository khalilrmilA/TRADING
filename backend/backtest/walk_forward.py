"""Walk-forward evaluation of rule-based strategies.

PAPER TRADING / RESEARCH ONLY — pure simulation over historical candles.

Scheme (kept deliberately simple, anchored/expanding):
    * The final ``(1 - train_ratio) * len(df)`` rows form the overall TEST
      region; it is split into ``n_splits`` equal contiguous test windows.
    * Fold ``i`` "trains" on rows ``[0, window_start_i)`` — informational
      only, since the strategies are rule-based with fixed parameters — and is
      evaluated on test window ``i``.
    * For each fold, features are recomputed with ``add_features`` on the
      train+test slice ``df.iloc[0:window_end_i]`` (so indicator warm-up
      happens inside the training region, with no leakage from future rows),
      but the backtest and its metrics are computed ONLY on the test rows.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.backtest.engine import Backtester, BacktestConfig

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
_AGGREGATE_METRICS: tuple[str, ...] = (
    "sharpe",
    "total_return",
    "max_drawdown",
    "win_rate",
)


def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    n_splits: int = 4,
    train_ratio: float = 0.7,
    config: BacktestConfig | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an anchored walk-forward backtest over contiguous test folds.

    Args:
        df: Canonical raw OHLCV frame (UTC tz-aware DatetimeIndex, ascending;
            columns ``open, high, low, close, volume``). Extra columns are
            ignored — features are recomputed per fold.
        strategy_name: Registry key for ``backend.strategies.get_strategy``.
        n_splits: Number of contiguous test folds (>= 1).
        train_ratio: Fraction of rows reserved before the overall test region
            (0 < train_ratio < 1).
        config: Backtest cost/risk config; ``None`` uses settings defaults.
        params: Strategy constructor parameters.

    Returns:
        ``{"folds": [{"fold": i, "train_start": iso, "test_start": iso,
        "test_end": iso, "metrics": {...}}, ...],
        "aggregate": {"<metric>_mean"/"<metric>_std" for sharpe,
        total_return, max_drawdown, win_rate}}``.
        Aggregate std uses population std (ddof=0), so it is 0.0 for a
        single fold.

    Raises:
        ValueError: On bad ``n_splits``/``train_ratio``, a malformed frame,
            or too few rows to form ``n_splits`` non-empty test windows.
        KeyError: From the strategy registry when ``strategy_name`` is
            unknown.
    """
    # Imported here so this module never imports other agents' modules at
    # import time (they are being developed concurrently).
    from backend.indicators.features import add_features
    from backend.strategies import get_strategy

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("walk_forward requires a non-empty DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a UTC tz-aware pd.DatetimeIndex")
    missing = [col for col in _OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required OHLCV columns {missing}")
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            f"train_ratio must be strictly between 0 and 1, got {train_ratio}"
        )

    raw = df.loc[:, list(_OHLCV_COLUMNS)]
    n_rows = len(raw)
    test_region_start = int(round(n_rows * train_ratio))
    test_region_len = n_rows - test_region_start
    if test_region_len < n_splits:
        raise ValueError(
            f"Not enough rows for walk-forward: {n_rows} rows with "
            f"train_ratio={train_ratio} leave {test_region_len} test rows "
            f"for n_splits={n_splits}"
        )

    # Equal contiguous window boundaries over the test region. With
    # test_region_len >= n_splits every window has at least one row.
    boundaries = (
        np.linspace(test_region_start, n_rows, n_splits + 1).round().astype(int)
    )

    folds: list[dict[str, Any]] = []
    for fold_idx in range(n_splits):
        window_start = int(boundaries[fold_idx])
        window_end = int(boundaries[fold_idx + 1])

        # Features recomputed on the train+test slice only (no future rows).
        featured = add_features(raw.iloc[:window_end])
        test_df = featured.iloc[window_start:window_end]

        strategy = get_strategy(strategy_name, **(params or {}))
        result = Backtester(config).run(test_df, strategy)

        folds.append(
            {
                "fold": fold_idx,
                "train_start": raw.index[0].isoformat(),
                "test_start": raw.index[window_start].isoformat(),
                "test_end": raw.index[window_end - 1].isoformat(),
                "metrics": result.metrics,
            }
        )
        logger.info(
            "Walk-forward fold %d/%d: strategy=%s test_rows=%d sharpe=%.3f "
            "total_return=%.4f",
            fold_idx + 1,
            n_splits,
            strategy_name,
            window_end - window_start,
            result.metrics["sharpe"],
            result.metrics["total_return"],
        )

    aggregate: dict[str, float] = {}
    for metric in _AGGREGATE_METRICS:
        values = [fold["metrics"][metric] for fold in folds]
        aggregate[f"{metric}_mean"] = float(np.mean(values))
        aggregate[f"{metric}_std"] = float(np.std(values))

    return {"folds": folds, "aggregate": aggregate}
