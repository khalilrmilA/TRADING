"""Bar-by-bar backtesting engine.

PAPER TRADING / RESEARCH ONLY — all fills are simulated against historical
candles. This module never talks to an exchange and never places orders.

Engine semantics (contract-mandated):
    * A strategy signal at bar ``t`` is the TARGET STANCE for the next bar and
      is acted upon at bar ``t+1`` OPEN — never on the signal bar itself.
    * Entry fill = next open with ADVERSE slippage (buys fill higher, sells
      fill lower). Commission = ``notional * commission_rate`` per fill, on
      both entry and exit.
    * Position size: ``qty = (equity * risk_per_trade) /
      (atr_stop_multiplier * atr_14[t])`` using the ATR of the SIGNAL bar,
      capped so notional <= 95% of equity. Entry is skipped when the ATR is
      NaN or <= 0.
    * Stop = entry -/+ ``atr_stop_multiplier * atr_14``; take-profit =
      entry +/- ``atr_take_profit_multiplier * atr_14`` (signs by side, same
      signal-bar ATR).
    * Intrabar: if a bar's high/low touches the stop or take-profit level, the
      position exits at that level with adverse slippage — the STOP is checked
      FIRST (conservative when both levels are touched in one bar).
    * Signal flip: exit at the next open, then immediately open the new side
      at that same open.
    * Equity is marked-to-market on every bar CLOSE.
    * Any position still open on the final bar is force-closed at the final
      close (``exit_reason="end_of_data"``).
    * One position at a time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from backend.backtest.metrics import compute_metrics
from config.settings import settings

if TYPE_CHECKING:  # runtime import avoided: strategies module owned elsewhere
    from backend.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_SECONDS_PER_YEAR: int = 365 * 24 * 3600
_DEFAULT_PERIODS_PER_YEAR: int = 8_760  # 1h bars — safe fallback
_REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
_ATR_COLUMN: str = "atr_14"
_MAX_NOTIONAL_FRACTION: float = 0.95


@dataclass
class BacktestConfig:
    """Backtest cost/risk parameters; every default comes from settings."""

    initial_capital: float = field(
        default_factory=lambda: settings.initial_capital
    )
    commission_rate: float = field(
        default_factory=lambda: settings.commission_rate
    )
    slippage_rate: float = field(default_factory=lambda: settings.slippage_rate)
    risk_per_trade: float = field(
        default_factory=lambda: settings.risk_per_trade
    )
    atr_stop_multiplier: float = field(
        default_factory=lambda: settings.atr_stop_multiplier
    )
    atr_take_profit_multiplier: float = field(
        default_factory=lambda: settings.atr_take_profit_multiplier
    )
    allow_short: bool = True


@dataclass
class Trade:
    """One closed simulated round-trip trade."""

    symbol: str
    side: str  # "long" | "short"
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    commission: float  # entry + exit commission, total
    pnl: float  # net of all commissions
    pnl_pct: float  # pnl / entry notional (fraction)
    exit_reason: str  # "signal" | "stop_loss" | "take_profit" | "end_of_data"


@dataclass
class BacktestResult:
    """Output of one backtest run."""

    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation.

        Returns:
            Dict with ``equity_curve`` as ``{"timestamps": [iso...],
            "values": [...]}``, ``trades`` as a list of dicts with ISO-8601
            times, and ``metrics`` as-is (already JSON-safe floats).
        """
        return {
            "equity_curve": {
                "timestamps": [ts.isoformat() for ts in self.equity_curve.index],
                "values": [float(v) for v in self.equity_curve.to_numpy()],
            },
            "trades": [
                {
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "qty": float(trade.qty),
                    "entry_time": trade.entry_time.isoformat(),
                    "entry_price": float(trade.entry_price),
                    "exit_time": trade.exit_time.isoformat(),
                    "exit_price": float(trade.exit_price),
                    "commission": float(trade.commission),
                    "pnl": float(trade.pnl),
                    "pnl_pct": float(trade.pnl_pct),
                    "exit_reason": trade.exit_reason,
                }
                for trade in self.trades
            ],
            "metrics": self.metrics,
        }


@dataclass
class _Position:
    """Internal open-position state (one at a time)."""

    direction: int  # 1 = long, -1 = short
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    stop: float
    take_profit: float
    entry_commission: float


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer bars-per-year from the median bar spacing of the index.

    Reproduces the contract mapping exactly for the supported timeframes
    (1m -> 525600, ..., 1d -> 365). Falls back to hourly (8760) when the
    index has fewer than 2 rows or a non-positive median spacing.

    Args:
        index: The candle DatetimeIndex.

    Returns:
        Estimated bars per 365-day year (>= 1).
    """
    if len(index) < 2:
        return _DEFAULT_PERIODS_PER_YEAR
    # Unit-independent (works for datetime64[ns] and datetime64[us] indexes).
    delta_seconds = (index[1:] - index[:-1]).total_seconds()
    median_seconds = float(np.median(np.asarray(delta_seconds, dtype=float)))
    if not math.isfinite(median_seconds) or median_seconds <= 0:
        return _DEFAULT_PERIODS_PER_YEAR
    return max(int(round(_SECONDS_PER_YEAR / median_seconds)), 1)


class Backtester:
    """Single-position, next-open-fill backtesting engine (simulation only)."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Initialize the engine.

        Args:
            config: Cost/risk parameters. ``None`` uses settings defaults.
        """
        self.config = config if config is not None else BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        strategy: "BaseStrategy",
        symbol: str = "",
    ) -> BacktestResult:
        """Run the strategy over the candles and simulate fills.

        Args:
            df: Feature-enriched canonical OHLCV frame (UTC tz-aware
                DatetimeIndex; must include ``atr_14``).
            strategy: A ``BaseStrategy`` whose ``generate_signals(df)``
                returns a copy of ``df`` with a ``signal`` column
                (1 long / -1 short / 0 flat — target stance for the NEXT bar).
            symbol: Symbol label stamped on trades (informational).

        Returns:
            A :class:`BacktestResult` with the per-bar equity curve (starting
            at ``initial_capital``), the closed trades, and the metric dict.

        Raises:
            ValueError: If ``df`` is empty, lacks required columns, has a
                non-DatetimeIndex, or the strategy output has no ``signal``
                column.
        """
        self._validate_df(df)
        cfg = self.config

        signal_frame = strategy.generate_signals(df)
        if "signal" not in signal_frame.columns:
            raise ValueError(
                "strategy.generate_signals() returned a frame without a "
                "'signal' column"
            )
        signals = np.sign(
            signal_frame["signal"]
            .reindex(df.index)
            .fillna(0)
            .to_numpy(dtype=float)
        ).astype(np.int64)

        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        atr_values = df[_ATR_COLUMN].to_numpy(dtype=float)
        index = df.index
        n_bars = len(df)

        cash: float = float(cfg.initial_capital)
        position: _Position | None = None
        equity = np.empty(n_bars, dtype=float)
        trades: list[Trade] = []

        def close_position(
            pos: _Position, raw_price: float, ts: pd.Timestamp, reason: str
        ) -> None:
            """Exit ``pos`` at ``raw_price`` with adverse slippage."""
            nonlocal cash, position
            if pos.direction == 1:  # selling to close a long -> fill lower
                fill = raw_price * (1.0 - cfg.slippage_rate)
            else:  # buying to cover a short -> fill higher
                fill = raw_price * (1.0 + cfg.slippage_rate)
            notional = pos.qty * fill
            exit_commission = notional * cfg.commission_rate
            if pos.direction == 1:
                cash += notional - exit_commission
                gross_pnl = pos.qty * (fill - pos.entry_price)
            else:
                cash -= notional + exit_commission
                gross_pnl = pos.qty * (pos.entry_price - fill)
            total_commission = pos.entry_commission + exit_commission
            pnl = gross_pnl - total_commission
            entry_notional = pos.qty * pos.entry_price
            pnl_pct = pnl / entry_notional if entry_notional > 0 else 0.0
            trades.append(
                Trade(
                    symbol=symbol,
                    side="long" if pos.direction == 1 else "short",
                    qty=float(pos.qty),
                    entry_time=pos.entry_time,
                    entry_price=float(pos.entry_price),
                    exit_time=ts,
                    exit_price=float(fill),
                    commission=float(total_commission),
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct),
                    exit_reason=reason,
                )
            )
            position = None

        def open_position(
            direction: int, raw_open: float, atr_value: float, ts: pd.Timestamp
        ) -> None:
            """Enter at ``raw_open`` with adverse slippage and ATR sizing."""
            nonlocal cash, position
            if not math.isfinite(atr_value) or atr_value <= 0.0:
                logger.debug(
                    "Skipping entry at %s: invalid ATR %r", ts, atr_value
                )
                return
            if direction == 1:  # buying -> fill higher
                fill = raw_open * (1.0 + cfg.slippage_rate)
            else:  # selling short -> fill lower
                fill = raw_open * (1.0 - cfg.slippage_rate)
            if not math.isfinite(fill) or fill <= 0.0:
                logger.debug(
                    "Skipping entry at %s: invalid fill price %r", ts, fill
                )
                return
            equity_now = cash  # flat at this point, so cash == equity
            stop_distance = cfg.atr_stop_multiplier * atr_value
            if stop_distance <= 0.0:
                return
            qty = (equity_now * cfg.risk_per_trade) / stop_distance
            qty = min(qty, _MAX_NOTIONAL_FRACTION * equity_now / fill)
            if not math.isfinite(qty) or qty <= 0.0:
                logger.debug("Skipping entry at %s: qty %r", ts, qty)
                return
            notional = qty * fill
            entry_commission = notional * cfg.commission_rate
            tp_distance = cfg.atr_take_profit_multiplier * atr_value
            if direction == 1:
                cash -= notional + entry_commission
                stop = fill - stop_distance
                take_profit = fill + tp_distance
            else:
                cash += notional - entry_commission
                stop = fill + stop_distance
                take_profit = fill - tp_distance
            position = _Position(
                direction=direction,
                qty=float(qty),
                entry_price=float(fill),
                entry_time=ts,
                stop=float(stop),
                take_profit=float(take_profit),
                entry_commission=float(entry_commission),
            )

        for i in range(n_bars):
            ts = index[i]

            # 1) Act on the signal from bar i-1 at THIS bar's open.
            if i > 0:
                desired = int(signals[i - 1])
                current = position.direction if position is not None else 0
                if desired != current:
                    if position is not None:
                        close_position(position, opens[i], ts, "signal")
                    if desired != 0 and (desired == 1 or cfg.allow_short):
                        open_position(desired, opens[i], atr_values[i - 1], ts)

            # 2) Intrabar stop/take-profit — stop checked FIRST (conservative).
            if position is not None:
                if position.direction == 1:
                    if lows[i] <= position.stop:
                        close_position(position, position.stop, ts, "stop_loss")
                    elif highs[i] >= position.take_profit:
                        close_position(
                            position, position.take_profit, ts, "take_profit"
                        )
                else:
                    if highs[i] >= position.stop:
                        close_position(position, position.stop, ts, "stop_loss")
                    elif lows[i] <= position.take_profit:
                        close_position(
                            position, position.take_profit, ts, "take_profit"
                        )

            # 3) Force-close anything still open on the final bar.
            if i == n_bars - 1 and position is not None:
                close_position(position, closes[i], ts, "end_of_data")

            # 4) Mark-to-market on the bar close.
            if position is None:
                equity[i] = cash
            else:
                equity[i] = cash + position.direction * position.qty * closes[i]

        equity_curve = pd.Series(equity, index=index, name="equity")
        periods_per_year = _infer_periods_per_year(index)
        metrics = compute_metrics(equity_curve, trades, periods_per_year)
        logger.info(
            "Backtest done: strategy=%s symbol=%s bars=%d trades=%d "
            "final_equity=%.2f total_return=%.4f",
            getattr(strategy, "name", type(strategy).__name__),
            symbol or "?",
            n_bars,
            len(trades),
            float(equity[-1]),
            metrics["total_return"],
        )
        return BacktestResult(
            equity_curve=equity_curve, trades=trades, metrics=metrics
        )

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        """Validate the input frame shape.

        Args:
            df: Candidate feature-enriched OHLCV frame.

        Raises:
            ValueError: On empty input, missing columns, or a bad index.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Backtester.run requires a non-empty DataFrame")
        missing = [
            col
            for col in (*_REQUIRED_COLUMNS, _ATR_COLUMN)
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"df is missing required columns {missing}; pass a "
                "feature-enriched frame (see backend.indicators.features."
                "add_features)"
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "df.index must be a UTC tz-aware pd.DatetimeIndex"
            )
