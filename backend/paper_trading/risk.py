"""Risk management for the paper-trading engine.

PAPER TRADING ONLY — these checks gate *simulated* orders. Nothing in this
module (or anywhere in the platform) can place a real order or touch a
private exchange API key.

Public API (see CONTRACTS.md):
    - ``RiskCheckResult``: NamedTuple ``(allowed: bool, reason: str)``.
    - ``RiskManager.check_order(portfolio, open_positions)``: pre-trade gate.
    - ``RiskManager.position_size(equity, atr_value, price)``: ATR-based sizing.
"""

from __future__ import annotations

import logging
import math
from typing import Any, NamedTuple

from config.settings import settings

logger = logging.getLogger(__name__)


class RiskCheckResult(NamedTuple):
    """Outcome of a pre-trade risk check.

    Attributes:
        allowed: True when the order may proceed.
        reason: Human-readable explanation ("ok" when allowed; otherwise the
            rule that rejected the order).
    """

    allowed: bool
    reason: str


class RiskManager:
    """Pre-trade risk checks and ATR-based position sizing.

    The manager is stateless — all account state lives in the ``portfolio``
    dict produced by :meth:`PaperTradingEngine.get_portfolio` (persisted in
    the ``account_state`` table), so checks survive process restarts. The
    circuit breaker is *sticky*: once ``circuit_breaker_active`` is set on the
    portfolio it keeps rejecting orders until ``engine.reset()`` clears it.
    """

    def check_order(
        self,
        portfolio: dict[str, Any],
        open_positions: list[dict[str, Any]],
        side: str | None = None,
        qty: float | None = None,
        symbol: str | None = None,
        source: str | None = None,
        timeframe: str | None = None,
    ) -> RiskCheckResult:
        """Decide whether a new order may be accepted.

        Rejects (in order of severity) when:
            1. The circuit breaker is active, or drawdown from peak equity is
               at/above ``settings.circuit_breaker_drawdown`` — stays tripped
               until a manual ``engine.reset()``.
            2. Daily PnL (realized + unrealized, i.e. equity vs day-start
               equity) is at/below ``-settings.daily_loss_limit``.
            3. Trading is otherwise flagged halted on the portfolio.
            4. Open positions already number ``settings.max_open_positions``
               or more.

        Halts and the position cap gate NEW exposure only: when the optional
        order context (``side``/``qty``/``symbol``) is provided and the order
        fully nets against existing opposite-side open positions in that
        symbol (i.e. it only closes/reduces risk), the order is allowed —
        matching ``close_position``, which is never blocked. The netting scope
        mirrors the engine's ``_apply_fill``: when ``source``/``timeframe``
        are supplied, only opposite-side positions in the SAME
        (symbol, source, timeframe) bucket count as reducible — an order the
        engine would execute as a brand-new position in another timeframe is
        never classified as risk-reducing.

        Args:
            portfolio: Portfolio snapshot as returned by
                ``PaperTradingEngine.get_portfolio()`` (keys such as
                ``equity``, ``peak_equity``, ``drawdown``, ``daily_pnl_pct``,
                ``circuit_breaker_active``, ``trading_halted``).
            open_positions: Currently open position dicts.
            side: Optional order side (``"buy"``/``"sell"``) for the
                risk-reducing classification.
            qty: Optional order quantity (``None`` → treated as opening).
            symbol: Optional order symbol for the classification.
            source: Optional data source restricting the netting scope.
            timeframe: Optional candle timeframe restricting the netting
                scope (the engine nets FIFO per (symbol, source, timeframe)).

        Returns:
            RiskCheckResult: ``(True, "ok")`` when allowed, otherwise
            ``(False, reason)``.
        """
        # Orders that only close/reduce existing exposure are always allowed.
        if self._is_risk_reducing(open_positions, side, qty, symbol, source, timeframe):
            return RiskCheckResult(True, "ok")

        equity = self._as_float(portfolio.get("equity"), 0.0)

        # --- 1) circuit breaker (sticky until engine.reset()) -----------------
        drawdown = portfolio.get("drawdown")
        if drawdown is None:
            peak = self._as_float(portfolio.get("peak_equity"), 0.0)
            drawdown = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
        drawdown = self._as_float(drawdown, 0.0)
        if bool(portfolio.get("circuit_breaker_active", False)) or (
            drawdown >= settings.circuit_breaker_drawdown
        ):
            reason = (
                f"circuit_breaker: drawdown {drawdown:.2%} >= limit "
                f"{settings.circuit_breaker_drawdown:.2%} from peak equity — "
                "all new orders blocked until engine.reset()"
            )
            logger.warning("Order rejected — %s", reason)
            return RiskCheckResult(False, reason)

        # --- 2) daily loss limit vs day-start equity ---------------------------
        daily_pnl_pct = portfolio.get("daily_pnl_pct")
        if daily_pnl_pct is None:
            day_start = self._as_float(portfolio.get("day_start_equity"), 0.0)
            daily_pnl_pct = (equity - day_start) / day_start if day_start > 0 else 0.0
        daily_pnl_pct = self._as_float(daily_pnl_pct, 0.0)
        if daily_pnl_pct <= -settings.daily_loss_limit:
            reason = (
                f"daily_loss_limit: daily PnL {daily_pnl_pct:.2%} <= limit "
                f"-{settings.daily_loss_limit:.2%} of day-start equity — "
                "no new entries until the next UTC day"
            )
            logger.warning("Order rejected — %s", reason)
            return RiskCheckResult(False, reason)

        # --- 3) any other persisted halt flag ----------------------------------
        if bool(portfolio.get("trading_halted", False)):
            reason = str(portfolio.get("halt_reason") or "trading_halted")
            logger.warning("Order rejected — %s", reason)
            return RiskCheckResult(False, reason)

        # --- 4) max concurrent open positions ----------------------------------
        n_open = len(open_positions or [])
        if n_open >= settings.max_open_positions:
            reason = (
                f"max_open_positions: {n_open} open positions >= limit "
                f"{settings.max_open_positions}"
            )
            logger.warning("Order rejected — %s", reason)
            return RiskCheckResult(False, reason)

        return RiskCheckResult(True, "ok")

    def position_size(self, equity: float, atr_value: float, price: float) -> float:
        """Compute the risk-based order quantity from ATR volatility.

        Formula: ``qty = (equity * risk_per_trade) / (atr_stop_multiplier * atr)``,
        capped so the notional (``qty * price``) never exceeds 95% of equity.

        Args:
            equity: Current account equity.
            atr_value: Latest ATR(14) of the instrument.
            price: Latest price, used for the notional cap.

        Returns:
            float: Quantity to trade; ``0.0`` when ATR/price/equity are
            invalid (None, NaN, non-finite or non-positive).
        """
        try:
            equity_f = float(equity)
            atr_f = float(atr_value)
            price_f = float(price)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(atr_f) or atr_f <= 0.0:
            return 0.0
        if not math.isfinite(price_f) or price_f <= 0.0:
            return 0.0
        if not math.isfinite(equity_f) or equity_f <= 0.0:
            return 0.0

        qty = (equity_f * settings.risk_per_trade) / (
            settings.atr_stop_multiplier * atr_f
        )
        max_qty = (0.95 * equity_f) / price_f
        return max(0.0, min(qty, max_qty))

    @staticmethod
    def _is_risk_reducing(
        open_positions: list[dict[str, Any]] | None,
        side: str | None,
        qty: float | None,
        symbol: str | None,
        source: str | None = None,
        timeframe: str | None = None,
    ) -> bool:
        """True when the order fully nets against opposite-side open positions.

        A ``sell`` covering no more than the open long quantity (or a ``buy``
        covering no more than the open short quantity) closes or reduces
        exposure and never opens new risk. The netting scope matches the
        engine's ``_apply_fill``: when ``source``/``timeframe`` are provided,
        only positions in the same (symbol, source, timeframe) bucket count —
        a "cover" in a different timeframe would actually OPEN a new position
        and must re-pass every gate. Missing/invalid context conservatively
        returns False (order treated as an entry).
        """
        if side not in ("buy", "sell") or qty is None or not symbol:
            return False
        try:
            qty_f = float(qty)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(qty_f) or qty_f <= 0.0:
            return False
        opposite = "short" if side == "buy" else "long"
        available = 0.0
        for pos in open_positions or []:
            if pos.get("status", "open") != "open":
                continue
            if pos.get("symbol") != symbol or pos.get("side") != opposite:
                continue
            if source is not None and pos.get("source") != source:
                continue
            if timeframe is not None and pos.get("timeframe") != timeframe:
                continue
            try:
                available += float(pos.get("qty", 0.0))
            except (TypeError, ValueError):
                continue
        return qty_f <= available + 1e-9

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        """Coerce ``value`` to a finite float, falling back to ``default``."""
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default
