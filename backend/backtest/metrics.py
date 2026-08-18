"""Performance metrics for backtest results.

PAPER TRADING ONLY — these metrics evaluate simulated equity curves and
simulated trades. Nothing here touches an exchange.

All returned values are plain Python ``float`` (JSON-safe, no numpy scalar
types). Undefined quantities (e.g. Sharpe with zero return variance, win rate
with zero trades) are reported as ``0.0`` per the platform contract.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid a runtime circular import with engine.py
    from backend.backtest.engine import Trade

logger = logging.getLogger(__name__)

#: Exact metric keys, in contract order.
METRIC_KEYS: tuple[str, ...] = (
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
)

#: Bars per year for each supported timeframe (contract-mandated mapping).
_PERIODS_PER_YEAR: dict[str, int] = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}

#: profit_factor reported when there are winning trades but zero losses.
_PROFIT_FACTOR_CAP: float = 100.0


def periods_per_year_for(timeframe: str) -> int:
    """Return the number of bars per (365-day) year for a timeframe.

    Args:
        timeframe: One of ``"1m" | "5m" | "15m" | "1h" | "4h" | "1d"``.

    Returns:
        Bars per year, e.g. ``8760`` for ``"1h"``.

    Raises:
        ValueError: If the timeframe is not one of the supported values.
    """
    try:
        return _PERIODS_PER_YEAR[timeframe]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; expected one of "
            f"{sorted(_PERIODS_PER_YEAR)}"
        ) from None


def _finite_or_zero(value: float) -> float:
    """Return ``float(value)`` or ``0.0`` when it is NaN/inf (JSON safety)."""
    result = float(value)
    return result if math.isfinite(result) else 0.0


def _zero_metrics() -> dict[str, float]:
    """Return an all-zero metrics dict with the exact contract keys."""
    return {key: 0.0 for key in METRIC_KEYS}


def _compute_exposure(equity_curve: pd.Series, trades: Sequence["Trade"]) -> float:
    """Fraction of bars during which a position was open.

    Each trade is assumed to occupy the bars in the half-open interval
    ``[entry_time, exit_time)`` (a trade entered and exited within the same
    bar counts as one bar). Trades never overlap (one position at a time), so
    the per-trade bar counts are simply summed and clamped to ``[0, 1]``.

    Args:
        equity_curve: Per-bar equity series (defines the bar index).
        trades: Closed trades from the backtest.

    Returns:
        Exposure as a fraction in ``[0, 1]``.
    """
    n_bars = len(equity_curve)
    if n_bars == 0 or not trades:
        return 0.0
    index = equity_curve.index
    open_bars = 0
    for trade in trades:
        start = int(index.searchsorted(trade.entry_time, side="left"))
        end = int(index.searchsorted(trade.exit_time, side="left"))
        open_bars += max(end - start, 1)
    return float(min(open_bars / n_bars, 1.0))


def compute_metrics(
    equity_curve: pd.Series,
    trades: Sequence["Trade"],
    periods_per_year: int,
) -> dict[str, float]:
    """Compute the contract metric set from an equity curve and trade list.

    Definitions:
        * ``total_return``: ``end_equity / start_equity - 1``.
        * ``cagr``: geometric annual growth using ``len - 1`` elapsed bars at
          ``periods_per_year`` bars/year; ``0.0`` when undefined, ``-1.0``
          when equity is wiped out.
        * ``sharpe``: ``mean / std`` of per-bar equity pct-changes scaled by
          ``sqrt(periods_per_year)`` (rf = 0); ``0.0`` when fewer than 2
          return observations or the std is zero.
        * ``sortino``: same numerator, denominator is the downside deviation
          about MAR=0 over ALL per-bar returns (root mean square of the
          negative returns, non-negative bars counting as 0); ``0.0`` when
          undefined (no negative returns → zero downside deviation).
        * ``max_drawdown``: max peak-to-trough decline from the running peak,
          as a POSITIVE fraction (``0.12`` means -12%).
        * ``win_rate``: fraction of trades with ``pnl > 0``.
        * ``profit_factor``: gross wins / gross losses; ``100.0`` when there
          are wins but no losses; ``0.0`` with no wins.
        * ``num_trades``: trade count (as float for JSON uniformity).
        * ``avg_trade_pnl`` / ``avg_win`` / ``avg_loss``: mean net PnL over
          all / winning / losing trades (``avg_loss`` is negative); ``0.0``
          when the corresponding set is empty.
        * ``exposure``: fraction of bars with an open position.

    Args:
        equity_curve: Per-bar mark-to-market equity, indexed like the input
            candles, starting at initial capital.
        trades: Closed :class:`~backend.backtest.engine.Trade` records.
        periods_per_year: Bars per year (see :func:`periods_per_year_for`).

    Returns:
        Dict with EXACTLY the contract keys, every value a JSON-safe float.
    """
    if equity_curve is None or len(equity_curve) == 0:
        logger.debug("compute_metrics called with empty equity curve")
        return _zero_metrics()

    values = equity_curve.to_numpy(dtype=float)
    n_bars = len(values)
    start_equity = float(values[0])
    end_equity = float(values[-1])

    # --- Returns-based metrics ------------------------------------------------
    total_return = (
        end_equity / start_equity - 1.0 if start_equity > 0 else 0.0
    )

    elapsed_periods = n_bars - 1
    if elapsed_periods <= 0 or start_equity <= 0 or periods_per_year <= 0:
        cagr = 0.0
    elif end_equity <= 0:
        cagr = -1.0
    else:
        years = elapsed_periods / float(periods_per_year)
        cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0

    sharpe = 0.0
    sortino = 0.0
    bar_returns = equity_curve.pct_change().dropna()
    if len(bar_returns) >= 2:
        mean_ret = float(bar_returns.mean())
        std_ret = float(bar_returns.std())
        annualizer = math.sqrt(float(periods_per_year))
        if math.isfinite(std_ret) and std_ret > 0.0:
            sharpe = mean_ret / std_ret * annualizer
        downside_dev = math.sqrt(
            float((bar_returns.clip(upper=0.0) ** 2).mean())
        )
        if math.isfinite(downside_dev) and downside_dev > 0.0:
            sortino = mean_ret / downside_dev * annualizer

    running_peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(
            running_peak > 0, (running_peak - values) / running_peak, 0.0
        )
    max_drawdown = float(np.nanmax(drawdowns)) if drawdowns.size else 0.0
    if not math.isfinite(max_drawdown) or max_drawdown < 0:
        max_drawdown = 0.0

    # --- Trade-based metrics ---------------------------------------------------
    pnls = [float(trade.pnl) for trade in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    num_trades = float(len(pnls))

    win_rate = len(wins) / len(pnls) if pnls else 0.0

    gross_wins = float(sum(wins))
    gross_losses = float(-sum(losses))  # positive magnitude
    if gross_losses > 0.0:
        profit_factor = gross_wins / gross_losses
    elif gross_wins > 0.0:
        profit_factor = _PROFIT_FACTOR_CAP
    else:
        profit_factor = 0.0

    avg_trade_pnl = float(np.mean(pnls)) if pnls else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    exposure = _compute_exposure(equity_curve, trades)

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "avg_trade_pnl": avg_trade_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "exposure": exposure,
    }
    return {key: _finite_or_zero(value) for key, value in metrics.items()}
