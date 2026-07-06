"""Risk management for the paper-trading engine.

PAPER TRADING ONLY — these checks gate *simulated* orders. Nothing in this
module (or anywhere in the platform) can place a real order or touch a
private exchange API key.

Public API (see CONTRACTS.md):
    - ``RiskCheckResult``: NamedTuple ``(allowed: bool, reason: str)``.
    - ``RiskManager.check_order(portfolio, open_positions, ...)``: pre-trade
      gate (circuit breaker, daily loss, halt flag, position cap, direction
      cap, portfolio heat cap).
    - ``RiskManager.position_size(equity, atr_value, price)``: ATR-based sizing.
    - ``RiskManager.portfolio_heat(open_positions)``: dollars of stop-distance
      risk across open positions.
    - ``RiskManager.soft_daily_stop_active(...)``: True at 80% of the daily
      loss limit (the "slow down before the hard halt" threshold).
    - ``RiskManager.derisk_multiplier(...)``: drawdown ratchet (0.8 per 10%
      of drawdown) plus soft-daily-stop halving, applied to position sizes.
    - ``NO_STOP_RISK_FRACTION``: stop-distance reference (2% of notional) for
      positions without a stop-loss.
"""

from __future__ import annotations

import logging
import math
from typing import Any, NamedTuple

from config.settings import settings

logger = logging.getLogger(__name__)

# Stop-distance reference for positions without a stop-loss: treat them as
# risking 2% of their notional, so they still count toward portfolio heat.
NO_STOP_RISK_FRACTION = 0.02

# Fallback defaults mirroring the CONTRACTS.md "Improvement Pack v3" settings
# additions — used via getattr so this module works even before/without the
# new config fields landing in config/settings.py.
_DEFAULT_HEAT_CAP_FRACTION = 0.06
_DEFAULT_MAX_SAME_DIRECTION = 5


def _heat_cap_fraction() -> float:
    """Return ``settings.heat_cap_fraction`` with a defensive fallback."""
    try:
        value = float(getattr(settings, "heat_cap_fraction", _DEFAULT_HEAT_CAP_FRACTION))
    except (TypeError, ValueError):
        return _DEFAULT_HEAT_CAP_FRACTION
    return value if math.isfinite(value) else _DEFAULT_HEAT_CAP_FRACTION


def _max_same_direction() -> int:
    """Return ``settings.max_same_direction`` with a defensive fallback."""
    try:
        return int(getattr(settings, "max_same_direction", _DEFAULT_MAX_SAME_DIRECTION))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_SAME_DIRECTION


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
        price: float | None = None,
        stop_loss: float | None = None,
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
            5. Direction cap: open positions in the order's direction
               (platform-wide, all sources/timeframes) already number
               ``settings.max_same_direction`` or more. Skipped when ``side``
               is None.
            6. Heat cap: the total stop-distance risk of all open positions
               plus this order (see :meth:`portfolio_heat`) would exceed
               ``settings.heat_cap_fraction`` of equity. When ``price`` /
               ``stop_loss`` are missing the new order's risk degrades to
               ``price * qty * NO_STOP_RISK_FRACTION`` (or 0.0 with no price
               — existing heat only). Skipped when ``side`` is None.

        Halts and the position/direction/heat caps gate NEW exposure only:
        when the optional order context (``side``/``qty``/``symbol``) is
        provided and the order fully nets against existing opposite-side open
        positions in that symbol (i.e. it only closes/reduces risk), the
        order is allowed — matching ``close_position``, which is never
        blocked. The netting scope mirrors the engine's ``_apply_fill``: when
        ``source``/``timeframe`` are supplied, only opposite-side positions
        in the SAME (symbol, source, timeframe) bucket count as reducible —
        an order the engine would execute as a brand-new position in another
        timeframe is never classified as risk-reducing.

        Args:
            portfolio: Portfolio snapshot as returned by
                ``PaperTradingEngine.get_portfolio()`` (keys such as
                ``equity``, ``peak_equity``, ``drawdown``, ``daily_pnl_pct``,
                ``circuit_breaker_active``, ``trading_halted``).
            open_positions: Currently open position dicts.
            side: Optional order side (``"buy"``/``"sell"``) for the
                risk-reducing classification and the direction/heat caps.
            qty: Optional order quantity (``None`` → treated as opening).
            symbol: Optional order symbol for the classification.
            source: Optional data source restricting the netting scope.
            timeframe: Optional candle timeframe restricting the netting
                scope (the engine nets FIFO per (symbol, source, timeframe)).
            price: Optional expected fill price for the heat-cap projection.
            stop_loss: Optional stop-loss price for the heat-cap projection.

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

        # Checks 5 and 6 need an order direction; without one (legacy callers)
        # the engine still gets the first four checks unchanged.
        if side is None:
            return RiskCheckResult(True, "ok")

        # --- 5) direction cap: max open positions in ONE direction -------------
        direction = "long" if side == "buy" else "short"
        max_same = _max_same_direction()
        n_same = 0
        for pos in open_positions or []:
            try:
                if pos.get("status", "open") != "open":
                    continue
                if pos.get("side") == direction:
                    n_same += 1
            except AttributeError:
                continue
        if n_same >= max_same:
            reason = (
                f"direction_cap: {n_same} {direction} positions already open "
                f">= limit {max_same} — too much exposure in one direction"
            )
            logger.warning("Order rejected — %s", reason)
            return RiskCheckResult(False, reason)

        # --- 6) heat cap: total stop-distance risk vs equity --------------------
        new_risk = 0.0
        qty_f = self._as_float(qty, math.nan)
        price_f = self._as_float(price, math.nan)
        stop_f = self._as_float(stop_loss, math.nan)
        if math.isfinite(qty_f) and qty_f > 0.0:
            if (
                math.isfinite(price_f)
                and math.isfinite(stop_f)
                and abs(price_f - stop_f) > 0.0
            ):
                new_risk = abs(price_f - stop_f) * qty_f
            elif math.isfinite(price_f):
                new_risk = max(0.0, price_f * qty_f * NO_STOP_RISK_FRACTION)
            # No price at all → unknown context; degraded check on existing
            # heat only (new_risk stays 0.0).
        cap_fraction = _heat_cap_fraction()
        projected = self.portfolio_heat(open_positions) + new_risk
        if projected > cap_fraction * equity + 1e-9:
            ratio = projected / equity if equity > 0 else math.inf
            reason = (
                f"heat_cap: total stop-distance risk would reach {ratio:.2%} of "
                f"equity, above the {cap_fraction:.2%} cap — the account would "
                "lose too much if every stop was hit at once"
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
    def portfolio_heat(open_positions: list[dict[str, Any]]) -> float:
        """Total stop-distance risk, in account-currency DOLLARS, of open positions.

        Per position: ``|entry_price - stop_loss| * qty`` when the stop-loss
        is finite and the stop distance is > 0; otherwise the position is
        treated as risking ``entry_price * qty * NO_STOP_RISK_FRACTION``
        (2% of its notional). This is the amount the account would lose if
        every open position's stop was hit at once.

        Args:
            open_positions: Open position dicts (``entry_price``, ``qty``,
                optional ``stop_loss``). Malformed rows contribute 0.

        Returns:
            float: Sum of per-position stop-distance risk in dollars
            (``0.0`` for an empty/invalid list). Never raises.
        """
        try:
            rows = list(open_positions or [])
        except TypeError:
            return 0.0
        total = 0.0
        for pos in rows:
            try:
                if pos.get("status", "open") != "open":
                    continue
                entry = float(pos.get("entry_price"))
                qty = float(pos.get("qty"))
                if not math.isfinite(entry) or entry <= 0.0:
                    continue
                if not math.isfinite(qty) or qty <= 0.0:
                    continue
                stop_raw = pos.get("stop_loss")
                try:
                    stop = float(stop_raw) if stop_raw is not None else math.nan
                except (TypeError, ValueError):
                    stop = math.nan
                distance = abs(entry - stop) if math.isfinite(stop) else 0.0
                if distance > 0.0:
                    total += distance * qty
                else:
                    total += entry * qty * NO_STOP_RISK_FRACTION
            except (TypeError, ValueError, AttributeError):
                continue
        return total

    @staticmethod
    def soft_daily_stop_active(
        equity: float, realized_pnl_today: float, daily_limit_fraction: float
    ) -> bool:
        """True when today's realized loss has reached 80% of the daily limit.

        The *soft* stop fires before the hard daily-loss halt so traders can
        stand down (scalper) or halve their size (bot) instead of running
        straight into the halt. Base is CURRENT equity — the closest thing
        available from these arguments.

        Args:
            equity: Current account equity.
            realized_pnl_today: Today's realized PnL in dollars (losses
                negative).
            daily_limit_fraction: The daily loss limit as a fraction
                (e.g. ``settings.daily_loss_limit`` = 0.03).

        Returns:
            bool: ``equity > 0 and realized_pnl_today <=
            -0.8 * daily_limit_fraction * equity``; False on invalid inputs.
        """
        try:
            equity_f = float(equity)
            pnl_f = float(realized_pnl_today)
            limit_f = float(daily_limit_fraction)
        except (TypeError, ValueError):
            return False
        if not (
            math.isfinite(equity_f) and math.isfinite(pnl_f) and math.isfinite(limit_f)
        ):
            return False
        return equity_f > 0.0 and pnl_f <= -0.8 * limit_f * equity_f

    @staticmethod
    def derisk_multiplier(
        equity: float,
        peak_equity: float,
        realized_pnl_today: float,
        daily_limit_fraction: float,
    ) -> float:
        """Position-size multiplier that shrinks as the account draws down.

        Drawdown ratchet: sizes shrink by 20% for every full 10% of drawdown
        from peak equity (``0.8 ** floor(drawdown_pct / 10)``), and are halved
        on top of that while the soft daily stop is active
        (:meth:`soft_daily_stop_active`). In plain English: the more the
        account is losing, the smaller every new trade gets.

        Args:
            equity: Current account equity.
            peak_equity: All-time-high equity (``portfolio["peak_equity"]``).
            realized_pnl_today: Today's realized PnL in dollars.
            daily_limit_fraction: Daily loss limit as a fraction
                (e.g. ``settings.daily_loss_limit``).

        Returns:
            float: Multiplier in ``(0, 1]`` to apply to a computed quantity;
            ``1.0`` when there is no drawdown or the inputs are invalid.
        """
        try:
            equity_f = float(equity)
            peak_f = float(peak_equity)
        except (TypeError, ValueError):
            return 1.0
        if not (math.isfinite(equity_f) and math.isfinite(peak_f)):
            return 1.0
        dd_pct = (
            100.0 * max(0.0, (peak_f - equity_f) / peak_f) if peak_f > 0.0 else 0.0
        )
        multiplier = 0.8 ** int(dd_pct // 10)
        if RiskManager.soft_daily_stop_active(
            equity_f, realized_pnl_today, daily_limit_fraction
        ):
            multiplier *= 0.5
        return min(multiplier, 1.0)

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
