"""Paper-trading engine — SIMULATION ONLY.

Every fill is priced off cached candles in the local SQLite database
(``backend.database.db.load_ohlcv``). No exchange order endpoint is ever
called, no private API keys exist anywhere in this project, and no real money
can ever move. This module exists purely for research.

State is fully persisted so it survives restarts:
    - cash / peak equity / day-start equity / halt flags → ``account_state``
      key-value table (JSON-encoded values);
    - positions / orders / trades → ``paper_positions`` / ``paper_orders`` /
      ``paper_trades`` tables;
    - one row in ``equity_snapshots`` per :meth:`PaperTradingEngine.process_tick`.

Fill semantics mirror the backtester: market and stop fills pay adverse
slippage (``settings.slippage_rate``); every fill pays commission
(``settings.commission_rate`` × notional). Selling into an open long closes
it FIFO rather than opening a short; a sell with no open long opens a short
(symmetric for buys covering shorts).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd

from backend.database.db import get_conn, init_db, load_ohlcv, utc_now_ms
from backend.paper_trading.risk import RiskManager
from config.settings import settings

logger = logging.getLogger(__name__)

_EPS = 1e-9

_SIDES = ("buy", "sell")
_ORDER_TYPES = ("market", "limit", "stop")


def _ms_to_iso(ms: int | None) -> str | None:
    """Convert epoch milliseconds to an ISO-8601 UTC string (None-safe)."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _utc_today() -> str:
    """Current UTC calendar date as an ISO string (daily-anchor key)."""
    return datetime.now(timezone.utc).date().isoformat()


def _clamp(price: float, low: float, high: float) -> float:
    """Clamp a fill price into the candle's traded range ``[low, high]``.

    Simulated fills must never happen at a price the candle did not trade.
    """
    return min(max(price, low), high)


def _last_candle_ms(df: pd.DataFrame) -> int:
    """Epoch-ms UTC timestamp of the most recent candle in a canonical frame."""
    return int(df.index[-1].value // 1_000_000)


def _drop_wrong_side_levels(
    side: str,
    reference_price: float,
    stop_loss: float | None,
    take_profit: float | None,
) -> tuple[float | None, float | None, list[str]]:
    """Drop protective levels sitting on the WRONG side of a reference price.

    A level on the wrong side of the entry would trigger on the very next
    evaluated candle regardless of where price goes (stops additionally pay
    adverse slippage): for a LONG position a ``stop_loss`` at or above the
    reference price, or a ``take_profit`` at or below it, is invalid —
    mirrored for a SHORT.

    Args:
        side: Position side, ``"long"`` or ``"short"``.
        reference_price: The fill/mark price the levels must straddle.
        stop_loss: Proposed stop level (``None`` → nothing to check).
        take_profit: Proposed target level (``None`` → nothing to check).

    Returns:
        ``(stop_loss, take_profit, reasons)`` — each invalid level replaced
        by ``None``, with one human-readable reason per dropped level.
    """
    reasons: list[str] = []
    if stop_loss is not None:
        level = float(stop_loss)
        wrong = level >= reference_price if side == "long" else level <= reference_price
        if wrong:
            reasons.append(
                f"invalid stop_loss {level:.8g} dropped: at or "
                f"{'above' if side == 'long' else 'below'} the {side} "
                f"reference price {reference_price:.8g} — it would trigger "
                "immediately"
            )
            stop_loss = None
    if take_profit is not None:
        level = float(take_profit)
        wrong = level <= reference_price if side == "long" else level >= reference_price
        if wrong:
            reasons.append(
                f"invalid take_profit {level:.8g} dropped: at or "
                f"{'below' if side == 'long' else 'above'} the {side} "
                f"reference price {reference_price:.8g} — it would trigger "
                "immediately"
            )
            take_profit = None
    return stop_loss, take_profit, reasons


def _watermark_key(source: str, symbol: str, timeframe: str) -> str:
    """account_state key for the last-processed-candle watermark."""
    return f"last_candle_ms:{source}:{symbol}:{timeframe}"


def _range_key(source: str, symbol: str, timeframe: str) -> str:
    """account_state key for the already-evaluated range of the current candle.

    Stored as ``{"ms": candle_ms, "low": float, "high": float}``. The data
    service keeps re-fetching the in-progress candle, so the newest cached
    candle repaints for its whole period; tracking which part of its range has
    already been evaluated lets :meth:`PaperTradingEngine.process_tick`
    re-check SL/TP and pending fills on EVERY tick (whenever the range grows)
    instead of only once when the candle first appears.
    """
    return f"evaluated_range:{source}:{symbol}:{timeframe}"


#: account_state key stamped by :meth:`PaperTradingEngine.reset` (epoch ms).
LAST_RESET_MS_KEY = "last_reset_ms"


def _extension_triggers(
    evaluated: dict[str, Any], low: float, high: float
) -> tuple[float, float]:
    """Trigger bounds for the UNSEEN extension of an already-evaluated candle.

    A side that did not extend beyond the evaluated range is disabled by
    returning a bound no price level can satisfy (``+inf`` for the low-side
    ``trigger_low <= level`` tests, ``-inf`` for the high-side
    ``trigger_high >= level`` tests).

    Args:
        evaluated: The stored ``{"ms", "low", "high"}`` evaluated-range record.
        low: The candle's current low.
        high: The candle's current high.

    Returns:
        ``(trigger_low, trigger_high)`` to use in touch tests.
    """
    trigger_low = low if low < float(evaluated["low"]) else math.inf
    trigger_high = high if high > float(evaluated["high"]) else -math.inf
    return trigger_low, trigger_high


def _wilder_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Fallback ATR (Wilder smoothing) used if the indicators module is absent.

    Args:
        df: Canonical OHLCV frame.
        window: ATR lookback window.

    Returns:
        pd.Series: ATR values aligned to ``df``'s index.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


class PaperTradingEngine:
    """Facade over the simulated (paper) trading account.

    All prices come from the local OHLCV cache; all state lives in SQLite.
    Thread-safe for use as a singleton inside the FastAPI app (a re-entrant
    lock serializes mutating operations).
    """

    def __init__(self) -> None:
        """Initialise the database schema and seed persisted account state."""
        init_db()
        self.risk = RiskManager()
        self._lock = threading.RLock()
        with self._connect() as conn:
            self._ensure_state(conn)
        logger.info(
            "PaperTradingEngine ready (SIMULATION ONLY, initial capital %.2f)",
            settings.initial_capital,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        source: str = "binance",
        timeframe: str = "1h",
    ) -> dict[str, Any]:
        """Submit a simulated order.

        Market orders fill immediately at the latest cached close ± adverse
        slippage; limit/stop orders stay ``pending`` until a later
        :meth:`process_tick` candle touches their price. ``qty=None``
        auto-sizes the order from the latest ATR(14) via
        :meth:`RiskManager.position_size`. :meth:`RiskManager.check_order`
        runs BEFORE any fill/accept; rejected orders are recorded with
        ``status='rejected'`` (reason in ``note``) and a ``ValueError`` is
        raised so the API returns 400.

        Args:
            symbol: Instrument symbol in the source's native format.
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"market"``, ``"limit"`` or ``"stop"``.
            qty: Quantity; ``None`` → ATR risk-based auto-size.
            limit_price: Required for limit orders.
            stop_price: Required for stop orders.
            stop_loss: Optional protective stop attached to the position.
                A level on the wrong side of the fill price (e.g. a long's
                stop at/above entry) is dropped at fill time — the fill
                stands, the reason is appended to the order's ``note``.
            take_profit: Optional profit target attached to the position.
                Wrong-side levels are dropped the same way.
            source: Data source (``binance``/``bybit``/``yahoo``).
            timeframe: Candle timeframe used for pricing.

        Returns:
            dict: The persisted order record.

        Raises:
            ValueError: Invalid parameters, no cached data, zero auto-size,
                or risk-manager rejection (recorded first).
        """
        symbol = str(symbol or "").strip()
        side = str(side or "").strip().lower()
        order_type = str(order_type or "").strip().lower()
        if not symbol:
            raise ValueError("symbol is required")
        if side not in _SIDES:
            raise ValueError(f"invalid side {side!r} — expected one of {_SIDES}")
        if order_type not in _ORDER_TYPES:
            raise ValueError(
                f"invalid order_type {order_type!r} — expected one of {_ORDER_TYPES}"
            )
        if order_type == "limit" and (limit_price is None or float(limit_price) <= 0):
            raise ValueError("limit orders require a positive limit_price")
        if order_type == "stop" and (stop_price is None or float(stop_price) <= 0):
            raise ValueError("stop orders require a positive stop_price")
        if qty is not None and float(qty) <= 0:
            raise ValueError("qty must be positive (or None for auto-sizing)")

        with self._lock, self._connect() as conn:
            self._roll_day(conn)
            portfolio = self._refresh_portfolio(conn)
            open_positions = [
                self._position_dict(r) for r in self._open_positions(conn)
            ]

            # Risk gate BEFORE any fill/accept. The order context lets the
            # risk manager wave through orders that only close/reduce
            # existing exposure (halts and the position cap gate entries).
            check = self.risk.check_order(
                portfolio,
                open_positions,
                side=side,
                qty=float(qty) if qty is not None else None,
                symbol=symbol,
                source=source,
                timeframe=timeframe,
            )
            if not check.allowed:
                self._record_rejection(
                    conn,
                    symbol,
                    source,
                    timeframe,
                    side,
                    order_type,
                    float(qty) if qty is not None else 0.0,
                    limit_price,
                    stop_price,
                    stop_loss,
                    take_profit,
                    check.reason,
                )

            df = load_ohlcv(source, symbol, timeframe, limit=100)
            if df.empty:
                self._record_rejection(
                    conn,
                    symbol,
                    source,
                    timeframe,
                    side,
                    order_type,
                    float(qty) if qty is not None else 0.0,
                    limit_price,
                    stop_price,
                    stop_loss,
                    take_profit,
                    f"no cached candles for {source}:{symbol}:{timeframe} — "
                    "fetch market data before trading",
                )
            last_close = float(df["close"].iloc[-1])

            # Anchor the candle watermark so candles that predate this order
            # can never fill it (or trigger its SL/TP) retroactively. The
            # evaluated range is anchored too: the forming candle's price
            # action BEFORE this order existed counts as already seen.
            wm_key = _watermark_key(source, symbol, timeframe)
            if self._state_get(conn, wm_key, None) is None:
                anchor_ms = _last_candle_ms(df)
                self._state_set(conn, wm_key, anchor_ms)
                self._state_set(
                    conn,
                    _range_key(source, symbol, timeframe),
                    {
                        "ms": anchor_ms,
                        "low": float(df["low"].iloc[-1]),
                        "high": float(df["high"].iloc[-1]),
                    },
                )

            if qty is None:
                atr_value = self._latest_atr(df)
                qty = self.risk.position_size(
                    portfolio["equity"], atr_value, last_close
                )
                if qty <= _EPS:
                    self._record_rejection(
                        conn,
                        symbol,
                        source,
                        timeframe,
                        side,
                        order_type,
                        0.0,
                        limit_price,
                        stop_price,
                        stop_loss,
                        take_profit,
                        "auto position sizing returned qty=0 "
                        f"(atr_14={atr_value!r}, price={last_close}) — "
                        "invalid ATR or insufficient equity",
                    )
            qty = float(qty)

            # Buying-power gate: the portion of qty that would OPEN exposure
            # (i.e. not net FIFO against opposite-side open positions) may not
            # exceed 95% of equity in notional — the same cap used by
            # RiskManager.position_size for auto-sized orders.
            opposite = "short" if side == "buy" else "long"
            reducible_qty = sum(
                float(p["qty"])
                for p in open_positions
                if p.get("side") == opposite
                and p.get("symbol") == symbol
                and p.get("source") == source
                and p.get("timeframe") == timeframe
            )
            opening_qty = max(0.0, qty - reducible_qty)
            if order_type == "limit" and limit_price is not None:
                ref_price = float(limit_price)
            elif order_type == "stop" and stop_price is not None:
                ref_price = float(stop_price)
            else:
                ref_price = last_close
            max_notional = 0.95 * float(portfolio["equity"])
            if opening_qty * ref_price > max_notional * (1.0 + 1e-9):
                self._record_rejection(
                    conn,
                    symbol,
                    source,
                    timeframe,
                    side,
                    order_type,
                    qty,
                    limit_price,
                    stop_price,
                    stop_loss,
                    take_profit,
                    f"buying_power: opening notional {opening_qty * ref_price:.2f} "
                    f"exceeds 95% of equity ({max_notional:.2f})",
                )

            now = utc_now_ms()
            order_id = self._insert_order(
                conn,
                symbol,
                source,
                timeframe,
                side,
                order_type,
                qty,
                limit_price,
                stop_price,
                stop_loss,
                take_profit,
                status="pending",
                note="",
                created_ms=now,
            )

            opened_position_id: int | None = None
            if order_type == "market":
                slip = settings.slippage_rate
                fill_price = (
                    last_close * (1.0 + slip)
                    if side == "buy"
                    else last_close * (1.0 - slip)
                )
                slippage_cost = qty * last_close * slip
                fill_result = self._apply_fill(
                    conn,
                    order_id,
                    symbol,
                    source,
                    timeframe,
                    side,
                    qty,
                    fill_price,
                    slippage_cost,
                    stop_loss,
                    take_profit,
                    now,
                )
                opened = fill_result.get("opened")
                if opened is not None:
                    opened_position_id = int(opened["id"])
                conn.execute(
                    "UPDATE paper_orders SET status='filled', filled_at=?, "
                    "fill_price=? WHERE id=?",
                    (now, fill_price, order_id),
                )
                self._refresh_portfolio(conn)
                logger.info(
                    "Market %s %s x%.8g filled at %.8g (paper)",
                    side,
                    symbol,
                    qty,
                    fill_price,
                )
            else:
                logger.info(
                    "%s %s %s x%.8g accepted as pending (paper)",
                    order_type,
                    side,
                    symbol,
                    qty,
                )

            row = conn.execute(
                "SELECT * FROM paper_orders WHERE id=?", (order_id,)
            ).fetchone()
            record = self._order_dict(row)
            # Id of the position this fill OPENED (None when the fill only
            # netted against existing positions, or for pending orders) — lets
            # callers (e.g. the scalper) attribute ownership without racy
            # before/after position diffs.
            record["opened_position_id"] = opened_position_id
            return record

    def process_tick(
        self,
        symbol: str,
        source: str = "binance",
        timeframe: str = "1h",
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Advance the simulation using the latest cached candle for ``symbol``.

        Steps: optionally refresh the cache from the public data service,
        fill pending limit/stop orders whose price the candle range touched,
        evaluate open positions' stop-loss/take-profit intrabar (stop first —
        conservative), mark all matching positions to market on the close,
        update halt flags, and write an equity snapshot.

        Guarantees: each PRICE RANGE triggers fills/SL/TP at most once — a
        per-(source, symbol, timeframe) watermark plus an evaluated-range
        record in ``account_state`` track how much of the current (forming)
        candle has already been checked, so a range EXTENSION on a later tick
        is re-evaluated (the forming candle repaints for its whole period)
        while already-seen price action — including action predating the
        oldest order — is never re-applied. Fill prices are clamped into the
        candle's traded ``[low, high]`` range; fills that would open or
        increase exposure re-pass the risk manager (fills that only
        reduce/close positions are always allowed).

        Args:
            symbol: Instrument to tick.
            source: Data source of the cached candles.
            timeframe: Candle timeframe.
            refresh: When True, call ``backend.data.service.update_symbol``
                first (public market-data endpoints only; failures are logged
                and ignored).

        Returns:
            dict: ``{"filled": [order dicts], "closed": [position dicts],
            "equity": float}``.
        """
        if refresh:
            try:
                from backend.data.service import update_symbol

                update_symbol(source, symbol, timeframe)
            except Exception as exc:  # noqa: BLE001 — refresh is best-effort
                logger.warning(
                    "Data refresh failed for %s:%s:%s — %s",
                    source,
                    symbol,
                    timeframe,
                    exc,
                )

        with self._lock, self._connect() as conn:
            self._roll_day(conn)
            filled: list[dict[str, Any]] = []
            closed: list[dict[str, Any]] = []

            df = load_ohlcv(source, symbol, timeframe, limit=2)
            if df.empty:
                logger.warning(
                    "process_tick: no cached candles for %s:%s:%s",
                    source,
                    symbol,
                    timeframe,
                )
                portfolio = self._refresh_portfolio(conn)
                self._snapshot_equity(conn, portfolio)
                return {"filled": [], "closed": [], "equity": portfolio["equity"]}

            candle = df.iloc[-1]
            high = float(candle["high"])
            low = float(candle["low"])
            close = float(candle["close"])
            now = utc_now_ms()

            # Temporal guard: any given slice of price action triggers
            # fills/SL/TP exactly once. The watermark tracks the newest candle
            # ever processed; the evaluated-range record tracks how much of
            # that candle's (still repainting) range has already been checked,
            # so a later tick re-evaluates only the UNSEEN range extension —
            # never price action that predates an order, and never the same
            # extreme twice.
            candle_ms = _last_candle_ms(df)
            wm_key = _watermark_key(source, symbol, timeframe)
            range_key = _range_key(source, symbol, timeframe)
            last_processed = self._state_get(conn, wm_key, None)
            is_new_candle = last_processed is None or candle_ms > int(last_processed)

            prev_range = self._state_get(conn, range_key, None)
            if not (
                isinstance(prev_range, dict)
                and {"ms", "low", "high"} <= set(prev_range)
            ):
                prev_range = None

            if is_new_candle:
                # 1a) The watermark candle may have extended its range after
                # the last tick that looked at it (its final minutes were
                # never evaluated) — check that unseen extension first.
                if (
                    last_processed is not None
                    and len(df) >= 2
                    and prev_range is not None
                    and int(prev_range["ms"]) == int(last_processed)
                    and int(df.index[-2].value // 1_000_000) == int(last_processed)
                ):
                    prev_candle = df.iloc[-2]
                    p_low = float(prev_candle["low"])
                    p_high = float(prev_candle["high"])
                    t_low, t_high = _extension_triggers(prev_range, p_low, p_high)
                    if math.isfinite(t_low) or math.isfinite(t_high):
                        self._evaluate_candle_fills(
                            conn,
                            symbol,
                            source,
                            timeframe,
                            p_low,
                            p_high,
                            float(prev_candle["close"]),
                            t_low,
                            t_high,
                            now,
                            filled,
                            closed,
                        )
                # 1b) The fresh candle: its whole range is unseen.
                self._evaluate_candle_fills(
                    conn,
                    symbol,
                    source,
                    timeframe,
                    low,
                    high,
                    close,
                    low,
                    high,
                    now,
                    filled,
                    closed,
                )
                self._state_set(conn, wm_key, candle_ms)
                self._state_set(
                    conn, range_key, {"ms": candle_ms, "low": low, "high": high}
                )
            elif int(last_processed) == candle_ms:
                # Same forming candle revisited: re-evaluate ONLY the range
                # extension since the last look. Price action already inside
                # the evaluated range had its chance on an earlier tick (and
                # may predate orders/positions opened mid-candle).
                if prev_range is not None and int(prev_range["ms"]) == candle_ms:
                    t_low, t_high = _extension_triggers(prev_range, low, high)
                    if math.isfinite(t_low) or math.isfinite(t_high):
                        self._evaluate_candle_fills(
                            conn,
                            symbol,
                            source,
                            timeframe,
                            low,
                            high,
                            close,
                            t_low,
                            t_high,
                            now,
                            filled,
                            closed,
                        )
                    self._state_set(
                        conn,
                        range_key,
                        {
                            "ms": candle_ms,
                            "low": min(low, float(prev_range["low"])),
                            "high": max(high, float(prev_range["high"])),
                        },
                    )
                else:
                    # Upgrade path (watermark written by an older version):
                    # record the current range without triggering — identical
                    # to the historical behaviour for this candle.
                    self._state_set(
                        conn, range_key, {"ms": candle_ms, "low": low, "high": high}
                    )
            # else: the candle predates the watermark — never re-apply it.

            # --- 3) mark-to-market on the candle close -------------------------
            conn.execute(
                "UPDATE paper_positions SET last_price=? WHERE status='open' "
                "AND symbol=? AND source=? AND timeframe=?",
                (close, symbol, source, timeframe),
            )

            # --- 4) refresh flags and snapshot equity ---------------------------
            portfolio = self._refresh_portfolio(conn)
            self._snapshot_equity(conn, portfolio)
            return {"filled": filled, "closed": closed, "equity": portfolio["equity"]}

    def _evaluate_candle_fills(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        source: str,
        timeframe: str,
        low: float,
        high: float,
        close: float,
        trigger_low: float,
        trigger_high: float,
        now: int,
        filled: list[dict[str, Any]],
        closed: list[dict[str, Any]],
    ) -> None:
        """Fill pending orders and evaluate SL/TP against one candle range.

        ``trigger_low``/``trigger_high`` are the extremes allowed to TRIGGER
        touches this pass (``+inf``/``-inf`` disable a side whose range was
        already evaluated on an earlier tick); ``low``/``high`` remain the
        candle's full traded range, used only to clamp fill prices.

        Args:
            conn: Open transaction connection.
            symbol / source / timeframe: Instrument scope.
            low / high / close: The candle's traded range and close.
            trigger_low / trigger_high: Unseen extremes used in touch tests.
            now: Fill timestamp (epoch ms UTC).
            filled: Output list — filled order dicts are appended.
            closed: Output list — closed position dicts are appended.
        """
        slip = settings.slippage_rate

        # --- 1) fill pending limit/stop orders touched by the candle ------
        pending = conn.execute(
            "SELECT * FROM paper_orders WHERE status='pending' AND symbol=? "
            "AND source=? AND timeframe=? ORDER BY id ASC",
            (symbol, source, timeframe),
        ).fetchall()
        for order in pending:
            oside = order["side"]
            oqty = float(order["qty"])
            fill_price: float | None = None
            slippage_cost = 0.0
            # Fill prices are clamped into the candle's traded range
            # [low, high]: an order can never fill at a price the market
            # did not trade (e.g. a buy stop far below the range).
            if order["order_type"] == "limit":
                lp = order["limit_price"]
                if lp is None:
                    continue
                lp = float(lp)
                if oside == "buy" and trigger_low <= lp:
                    fill_price = _clamp(lp, low, high)
                elif oside == "sell" and trigger_high >= lp:
                    fill_price = _clamp(lp, low, high)
            elif order["order_type"] == "stop":
                sp = order["stop_price"]
                if sp is None:
                    continue
                sp = float(sp)
                if oside == "buy" and trigger_high >= sp:
                    base = _clamp(sp, low, high)
                    fill_price = base * (1.0 + slip)
                    slippage_cost = oqty * base * slip
                elif oside == "sell" and trigger_low <= sp:
                    base = _clamp(sp, low, high)
                    fill_price = base * (1.0 - slip)
                    slippage_cost = oqty * base * slip
            else:  # defensive: a pending market order fills at the close
                fill_price = (
                    close * (1.0 + slip) if oside == "buy" else close * (1.0 - slip)
                )
                slippage_cost = oqty * close * slip
            if fill_price is None:
                continue

            # Risk gate at fill time: fills that would open or increase
            # exposure must still pass the risk manager (position cap,
            # daily-loss halt, circuit breaker); purely risk-reducing
            # fills always pass. Rejected orders stay pending.
            check = self.risk.check_order(
                self._refresh_portfolio(conn),
                [self._position_dict(r) for r in self._open_positions(conn)],
                side=oside,
                qty=oqty,
                symbol=symbol,
                source=source,
                timeframe=timeframe,
            )
            if not check.allowed:
                logger.warning(
                    "Pending order #%s left unfilled — %s",
                    order["id"],
                    check.reason,
                )
                continue

            result = self._apply_fill(
                conn,
                int(order["id"]),
                symbol,
                source,
                timeframe,
                oside,
                oqty,
                fill_price,
                slippage_cost,
                order["stop_loss"],
                order["take_profit"],
                now,
            )
            conn.execute(
                "UPDATE paper_orders SET status='filled', filled_at=?, "
                "fill_price=? WHERE id=?",
                (now, fill_price, order["id"]),
            )
            updated = conn.execute(
                "SELECT * FROM paper_orders WHERE id=?", (order["id"],)
            ).fetchone()
            filled.append(self._order_dict(updated))
            closed.extend(result["closed"])
            logger.info(
                "Pending %s order #%s filled at %.8g (paper)",
                order["order_type"],
                order["id"],
                fill_price,
            )

        # --- 2) stop-loss / take-profit on open positions (stop first) ----
        for pos in self._open_positions(conn, symbol, source, timeframe):
            pside = pos["side"]
            pqty = float(pos["qty"])
            sl = pos["stop_loss"]
            tp = pos["take_profit"]
            exit_price: float | None = None
            exit_reason = ""
            slippage_cost = 0.0
            if pside == "long":
                if sl is not None and trigger_low <= float(sl):
                    base = _clamp(float(sl), low, high)
                    exit_price = base * (1.0 - slip)
                    slippage_cost = pqty * base * slip
                    exit_reason = "stop_loss"
                elif tp is not None and trigger_high >= float(tp):
                    exit_price = _clamp(float(tp), low, high)
                    exit_reason = "take_profit"
            else:  # short
                if sl is not None and trigger_high >= float(sl):
                    base = _clamp(float(sl), low, high)
                    exit_price = base * (1.0 + slip)
                    slippage_cost = pqty * base * slip
                    exit_reason = "stop_loss"
                elif tp is not None and trigger_low <= float(tp):
                    exit_price = _clamp(float(tp), low, high)
                    exit_reason = "take_profit"
            if exit_price is None:
                continue

            close_side = "sell" if pside == "long" else "buy"
            result = self._apply_fill(
                conn,
                None,
                symbol,
                source,
                timeframe,
                close_side,
                pqty,
                exit_price,
                slippage_cost,
                None,
                None,
                now,
                only_position_id=int(pos["id"]),
            )
            for pos_dict in result["closed"]:
                pos_dict["exit_reason"] = exit_reason
                closed.append(pos_dict)
            logger.info(
                "Position #%s closed by %s at %.8g (paper)",
                pos["id"],
                exit_reason,
                exit_price,
            )

    def close_position(self, position_id: int) -> dict[str, Any]:
        """Market-close an open position at the latest cached price.

        Args:
            position_id: Id of an open row in ``paper_positions``.

        Returns:
            dict: The closed position record (includes ``exit_reason="manual"``).

        Raises:
            ValueError: Unknown/already-closed position, or no cached candles
                to price the close.
        """
        with self._lock, self._connect() as conn:
            self._roll_day(conn)
            pos = conn.execute(
                "SELECT * FROM paper_positions WHERE id=?", (int(position_id),)
            ).fetchone()
            if pos is None:
                raise ValueError(f"position {position_id} not found")
            if pos["status"] != "open":
                raise ValueError(f"position {position_id} is already closed")

            df = load_ohlcv(pos["source"], pos["symbol"], pos["timeframe"], limit=1)
            if df.empty:
                raise ValueError(
                    f"no cached candles for {pos['source']}:{pos['symbol']}:"
                    f"{pos['timeframe']} — cannot price the close"
                )
            last_close = float(df["close"].iloc[-1])
            slip = settings.slippage_rate
            close_side = "sell" if pos["side"] == "long" else "buy"
            fill_price = (
                last_close * (1.0 + slip)
                if close_side == "buy"
                else last_close * (1.0 - slip)
            )
            qty = float(pos["qty"])
            slippage_cost = qty * last_close * slip
            now = utc_now_ms()

            order_id = self._insert_order(
                conn,
                pos["symbol"],
                pos["source"],
                pos["timeframe"],
                close_side,
                "market",
                qty,
                None,
                None,
                None,
                None,
                status="filled",
                note=f"close_position({int(position_id)})",
                created_ms=now,
                filled_ms=now,
                fill_price=fill_price,
            )
            result = self._apply_fill(
                conn,
                order_id,
                pos["symbol"],
                pos["source"],
                pos["timeframe"],
                close_side,
                qty,
                fill_price,
                slippage_cost,
                None,
                None,
                now,
                only_position_id=int(position_id),
            )
            self._refresh_portfolio(conn)

            if result["closed"]:
                out = result["closed"][0]
            else:  # pragma: no cover — defensive refetch
                row = conn.execute(
                    "SELECT * FROM paper_positions WHERE id=?", (int(position_id),)
                ).fetchone()
                out = self._position_dict(row)
            out["exit_reason"] = "manual"
            logger.info(
                "Position #%s manually closed at %.8g (paper)",
                position_id,
                fill_price,
            )
            return out

    def update_protective_levels(
        self,
        position_id: int,
        stop_loss: float | None = None,
        clear_take_profit: bool = False,
    ) -> dict[str, Any]:
        """Tighten an open position's protective levels (trailing-stop support).

        TIGHTEN-ONLY by construction — this method can never widen risk:
        a long's stop may only move UP, a short's stop only DOWN; a proposed
        stop that loosens (or merely equals) the current one is silently
        ignored. ``clear_take_profit`` removes the take-profit so a winner
        can keep running while the (tightened) trailing stop protects the
        gain. Used by the auto-trader's "let winners run" trailing logic.

        Args:
            position_id: Id of an OPEN row in ``paper_positions``.
            stop_loss: Proposed new stop price (None → leave the stop alone).
                Must be a positive finite number to be considered. When the
                position has NO stop yet, the proposal must also sit on the
                correct side of the last mark price (below it for a long,
                above it for a short) — a wrong-side first install is
                refused with ``stop_moved: False``.
            clear_take_profit: True → set ``take_profit`` to NULL.

        Returns:
            dict: The (possibly updated) open position record, with an extra
            ``"stop_moved": bool`` key telling whether the stop changed.

        Raises:
            ValueError: Unknown or already-closed position.
        """
        with self._lock, self._connect() as conn:
            pos = conn.execute(
                "SELECT * FROM paper_positions WHERE id=?", (int(position_id),)
            ).fetchone()
            if pos is None:
                raise ValueError(f"position {position_id} not found")
            if pos["status"] != "open":
                raise ValueError(f"position {position_id} is already closed")

            side = str(pos["side"])
            current_stop = pos["stop_loss"]
            new_stop: float | None = None
            if stop_loss is not None:
                try:
                    candidate = float(stop_loss)
                except (TypeError, ValueError):
                    candidate = math.nan
                if math.isfinite(candidate) and candidate > 0.0:
                    if current_stop is None:
                        # A FIRST stop install has no tighten anchor, so its
                        # side is validated against the last mark price — a
                        # wrong-side stop would trigger immediately. Invalid
                        # proposals reuse the not-moved signalling below.
                        last = pos["last_price"]
                        reference = float(
                            last if last is not None else pos["entry_price"]
                        )
                        checked, _, reasons = _drop_wrong_side_levels(
                            side, reference, candidate, None
                        )
                        if checked is None:
                            logger.warning(
                                "Position #%s: %s (paper)",
                                position_id,
                                reasons[0],
                            )
                        else:
                            new_stop = candidate
                    elif side == "long" and candidate > float(current_stop):
                        new_stop = candidate
                    elif side == "short" and candidate < float(current_stop):
                        new_stop = candidate

            if new_stop is None and not clear_take_profit:
                out = self._position_dict(pos)
                out["stop_moved"] = False
                return out

            sets: list[str] = []
            params: list[Any] = []
            if new_stop is not None:
                sets.append("stop_loss=?")
                params.append(new_stop)
            if clear_take_profit:
                sets.append("take_profit=NULL")
            params.append(int(position_id))
            conn.execute(
                f"UPDATE paper_positions SET {', '.join(sets)} WHERE id=?", params
            )

            row = conn.execute(
                "SELECT * FROM paper_positions WHERE id=?", (int(position_id),)
            ).fetchone()
            out = self._position_dict(row)
            out["stop_moved"] = new_stop is not None
            if new_stop is not None:
                logger.info(
                    "Position #%s trailing stop tightened to %.8g (paper)",
                    position_id,
                    new_stop,
                )
            return out

    def get_portfolio(self) -> dict[str, Any]:
        """Return the current account snapshot.

        Returns:
            dict with keys ``equity``, ``cash``, ``unrealized_pnl``,
            ``realized_pnl_today``, ``daily_pnl_pct``, ``open_positions``,
            ``peak_equity``, ``drawdown``, ``circuit_breaker_active``,
            ``trading_halted``, ``halt_reason``.
        """
        with self._lock, self._connect() as conn:
            self._roll_day(conn)
            return self._refresh_portfolio(conn)

    def get_positions(self, status: str = "open") -> list[dict[str, Any]]:
        """List positions, most recent first.

        Args:
            status: ``"open"`` (default), ``"closed"``, or ``None``/``"all"``
                for every position.

        Returns:
            list[dict]: Position records (open ones include ``unrealized_pnl``).
        """
        query = "SELECT * FROM paper_positions"
        params: tuple[Any, ...] = ()
        if status not in (None, "", "all"):
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._position_dict(r) for r in rows]

    def get_orders(self, status: str | None = None) -> list[dict[str, Any]]:
        """List orders, most recent first, optionally filtered by status.

        Args:
            status: ``pending``/``filled``/``cancelled``/``rejected`` or
                ``None`` for all.

        Returns:
            list[dict]: Order records.
        """
        query = "SELECT * FROM paper_orders"
        params: tuple[Any, ...] = ()
        if status not in (None, "", "all"):
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._order_dict(r) for r in rows]

    def get_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        """List executed fills, most recent first.

        Args:
            limit: Maximum number of trades to return.

        Returns:
            list[dict]: Trade records.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._trade_dict(r) for r in rows]

    def get_equity_history(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Return equity snapshots in ascending time order.

        Args:
            limit: Maximum number of (most recent) snapshots to return.

        Returns:
            list[dict]: ``[{"ts": iso, "equity": float}, ...]``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, equity FROM equity_snapshots "
                "ORDER BY ts DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {"ts": _ms_to_iso(r["ts"]), "equity": float(r["equity"])}
            for r in reversed(rows)
        ]

    def reset(self) -> None:
        """Wipe all paper-trading state and restore the initial capital.

        Clears orders, positions, trades, equity snapshots and every account
        flag (including the sticky circuit breaker).
        """
        with self._lock, self._connect() as conn:
            for table in (
                "paper_trades",
                "paper_orders",
                "paper_positions",
                "equity_snapshots",
                "account_state",
            ):
                conn.execute(f"DELETE FROM {table}")  # noqa: S608 — fixed names
            self._ensure_state(conn, force=True)
            # Stamp the reset time so readers of historical logs kept in other
            # tables (e.g. the scalper's bot_activity trade ledger) can ignore
            # everything that happened before the account was wiped.
            self._state_set(conn, LAST_RESET_MS_KEY, utc_now_ms())
        logger.info(
            "Paper account reset — capital restored to %.2f (SIMULATION ONLY)",
            settings.initial_capital,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and always closes."""
        conn = get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_state(self, conn: sqlite3.Connection, force: bool = False) -> None:
        """Seed the ``account_state`` defaults (idempotent unless ``force``)."""
        defaults: dict[str, Any] = {
            "cash": settings.initial_capital,
            "peak_equity": settings.initial_capital,
            "day_start_equity": settings.initial_capital,
            "day_anchor_date": _utc_today(),
            "realized_pnl_today": 0.0,
            "circuit_breaker_active": False,
            "trading_halted": False,
            "halt_reason": "",
        }
        for key, value in defaults.items():
            if force or self._state_get(conn, key, None) is None:
                self._state_set(conn, key, value)

    @staticmethod
    def _state_get(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
        """Read a JSON-encoded value from ``account_state``."""
        row = conn.execute(
            "SELECT value FROM account_state WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            logger.warning("Corrupt account_state value for key %r", key)
            return default

    @staticmethod
    def _state_set(conn: sqlite3.Connection, key: str, value: Any) -> None:
        """Write a JSON-encoded value to ``account_state``."""
        conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )

    def _open_positions(
        self,
        conn: sqlite3.Connection,
        symbol: str | None = None,
        source: str | None = None,
        timeframe: str | None = None,
        side: str | None = None,
    ) -> list[sqlite3.Row]:
        """Open positions in FIFO (insertion) order, optionally filtered."""
        query = "SELECT * FROM paper_positions WHERE status='open'"
        params: list[Any] = []
        for column, value in (
            ("symbol", symbol),
            ("source", source),
            ("timeframe", timeframe),
            ("side", side),
        ):
            if value is not None:
                query += f" AND {column}=?"
                params.append(value)
        query += " ORDER BY id ASC"
        return conn.execute(query, params).fetchall()

    def _equity(self, conn: sqlite3.Connection) -> tuple[float, float]:
        """Compute (equity, unrealized_pnl) from cash + open-position marks."""
        cash = float(self._state_get(conn, "cash", settings.initial_capital))
        market_value = 0.0
        unrealized = 0.0
        for row in self._open_positions(conn):
            last = row["last_price"] if row["last_price"] is not None else row["entry_price"]
            last = float(last)
            qty = float(row["qty"])
            direction = 1.0 if row["side"] == "long" else -1.0
            market_value += direction * qty * last
            unrealized += direction * (last - float(row["entry_price"])) * qty
        return cash + market_value, unrealized

    def _roll_day(self, conn: sqlite3.Connection) -> None:
        """Reset the daily anchor (day-start equity, daily PnL) on UTC date change."""
        today = _utc_today()
        if self._state_get(conn, "day_anchor_date", "") == today:
            return
        equity, _ = self._equity(conn)
        self._state_set(conn, "day_anchor_date", today)
        self._state_set(conn, "day_start_equity", equity)
        self._state_set(conn, "realized_pnl_today", 0.0)
        logger.info(
            "Daily anchor rolled to %s (day-start equity %.2f)", today, equity
        )

    def _refresh_portfolio(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Recompute equity/peak/drawdown, update halt flags, return portfolio."""
        equity, unrealized = self._equity(conn)
        cash = float(self._state_get(conn, "cash", settings.initial_capital))

        peak = max(
            float(self._state_get(conn, "peak_equity", settings.initial_capital)),
            equity,
        )
        self._state_set(conn, "peak_equity", peak)
        drawdown = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0

        breaker = bool(self._state_get(conn, "circuit_breaker_active", False))
        if not breaker and drawdown >= settings.circuit_breaker_drawdown:
            breaker = True
            logger.warning(
                "CIRCUIT BREAKER tripped: drawdown %.2f%% >= %.2f%% — trading "
                "halted until reset()",
                drawdown * 100,
                settings.circuit_breaker_drawdown * 100,
            )
        self._state_set(conn, "circuit_breaker_active", breaker)

        day_start = float(
            self._state_get(conn, "day_start_equity", settings.initial_capital)
        )
        daily_pnl = equity - day_start
        daily_pnl_pct = daily_pnl / day_start if day_start > 0 else 0.0
        daily_loss_hit = (
            day_start > 0 and daily_pnl <= -settings.daily_loss_limit * day_start
        )

        halted = breaker or daily_loss_hit
        if breaker:
            halt_reason = (
                f"circuit_breaker: drawdown {drawdown:.2%} >= "
                f"{settings.circuit_breaker_drawdown:.2%} from peak equity — "
                "reset() required"
            )
        elif daily_loss_hit:
            halt_reason = (
                f"daily_loss_limit: daily PnL {daily_pnl_pct:.2%} <= "
                f"-{settings.daily_loss_limit:.2%} of day-start equity"
            )
        else:
            halt_reason = ""
        self._state_set(conn, "trading_halted", halted)
        self._state_set(conn, "halt_reason", halt_reason)

        n_open = len(self._open_positions(conn))
        realized_today = float(self._state_get(conn, "realized_pnl_today", 0.0))
        return {
            "equity": float(equity),
            "cash": cash,
            "unrealized_pnl": float(unrealized),
            "realized_pnl_today": realized_today,
            "daily_pnl_pct": float(daily_pnl_pct),
            "open_positions": n_open,
            "peak_equity": float(peak),
            "drawdown": float(drawdown),
            "circuit_breaker_active": breaker,
            "trading_halted": halted,
            "halt_reason": halt_reason,
        }

    def _snapshot_equity(
        self, conn: sqlite3.Connection, portfolio: dict[str, Any]
    ) -> None:
        """Append a row to ``equity_snapshots``."""
        conn.execute(
            "INSERT INTO equity_snapshots (ts, equity, cash, unrealized_pnl, "
            "realized_pnl_today) VALUES (?, ?, ?, ?, ?)",
            (
                utc_now_ms(),
                portfolio["equity"],
                portfolio["cash"],
                portfolio["unrealized_pnl"],
                portfolio["realized_pnl_today"],
            ),
        )

    def _latest_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        """Latest ATR value; prefers the shared indicators module.

        The import lives inside the method so this module never depends on
        the indicators package at import time.
        """
        try:
            from backend.indicators.technical import atr as atr_fn

            series = atr_fn(df, window)
        except Exception as exc:  # noqa: BLE001 — fall back to local ATR
            logger.debug(
                "Indicators module unavailable (%s); using internal Wilder ATR",
                exc,
            )
            series = _wilder_atr(df, window)
        if series is None or len(series) == 0:
            return 0.0
        value = series.iloc[-1]
        return float(value) if pd.notna(value) else 0.0

    def _record_rejection(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        source: str,
        timeframe: str,
        side: str,
        order_type: str,
        qty: float,
        limit_price: float | None,
        stop_price: float | None,
        stop_loss: float | None,
        take_profit: float | None,
        reason: str,
    ) -> None:
        """Persist a rejected order (status='rejected') then raise ValueError.

        The insert is committed before raising so the audit row survives the
        transaction rollback triggered by the exception; the API layer maps
        the ``ValueError`` to a 400 response carrying ``reason``.
        """
        self._insert_order(
            conn,
            symbol,
            source,
            timeframe,
            side,
            order_type,
            qty,
            limit_price,
            stop_price,
            stop_loss,
            take_profit,
            status="rejected",
            note=reason,
        )
        conn.commit()
        logger.warning("Order rejected (%s %s %s): %s", side, symbol, order_type, reason)
        raise ValueError(reason)

    def _insert_order(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        source: str,
        timeframe: str,
        side: str,
        order_type: str,
        qty: float,
        limit_price: float | None,
        stop_price: float | None,
        stop_loss: float | None,
        take_profit: float | None,
        status: str,
        note: str,
        created_ms: int | None = None,
        filled_ms: int | None = None,
        fill_price: float | None = None,
    ) -> int:
        """Insert a row into ``paper_orders`` and return its id."""
        cur = conn.execute(
            "INSERT INTO paper_orders (created_at, symbol, source, timeframe, "
            "side, order_type, qty, limit_price, stop_price, stop_loss, "
            "take_profit, status, filled_at, fill_price, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                created_ms if created_ms is not None else utc_now_ms(),
                symbol,
                source,
                timeframe,
                side,
                order_type,
                float(qty),
                limit_price,
                stop_price,
                stop_loss,
                take_profit,
                status,
                filled_ms,
                fill_price,
                note,
            ),
        )
        return int(cur.lastrowid)

    @staticmethod
    def _append_order_note(
        conn: sqlite3.Connection, order_id: int, text: str
    ) -> None:
        """Append ``text`` to an order's ``note`` column ("; "-separated)."""
        conn.execute(
            "UPDATE paper_orders SET note = CASE WHEN note IS NULL OR note = '' "
            "THEN ? ELSE note || '; ' || ? END WHERE id=?",
            (text, text, int(order_id)),
        )

    def _insert_trade(
        self,
        conn: sqlite3.Connection,
        order_id: int | None,
        position_id: int | None,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        commission: float,
        slippage: float,
        executed_ms: int,
    ) -> int:
        """Insert a row into ``paper_trades`` and return its id."""
        cur = conn.execute(
            "INSERT INTO paper_trades (order_id, position_id, symbol, side, "
            "qty, price, commission, slippage, executed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order_id,
                position_id,
                symbol,
                side,
                float(qty),
                float(price),
                float(commission),
                float(slippage),
                executed_ms,
            ),
        )
        return int(cur.lastrowid)

    def _apply_fill(
        self,
        conn: sqlite3.Connection,
        order_id: int | None,
        symbol: str,
        source: str,
        timeframe: str,
        side: str,
        qty: float,
        fill_price: float,
        slippage_cost: float,
        stop_loss: float | None,
        take_profit: float | None,
        executed_ms: int,
        only_position_id: int | None = None,
    ) -> dict[str, Any]:
        """Apply an executed fill: net FIFO against opposite positions, open the rest.

        A buy first covers open shorts oldest-first; any remaining quantity
        opens a long (symmetric for sells closing longs / opening shorts).
        Commission and slippage are allocated pro-rata across the touched
        positions; every allocation writes a ``paper_trades`` row. Realized
        PnL (net of both legs' commissions) accrues to ``realized_pnl_today``.

        Args:
            conn: Open transaction connection.
            order_id: Originating order id (None for SL/TP auto-closes).
            symbol / source / timeframe: Instrument scope for FIFO netting.
            side: Fill side, ``"buy"`` or ``"sell"``.
            qty: Filled quantity.
            fill_price: Executed price (already slippage-adjusted).
            slippage_cost: Total slippage cost of the fill (for the audit row).
            stop_loss / take_profit: Levels attached to a newly opened
                position. Levels on the wrong side of ``fill_price`` are
                dropped (with a note on the order) rather than installed.
            executed_ms: Execution time (epoch ms UTC).
            only_position_id: When set, net ONLY against this position and
                never open a remainder (used by close_position and SL/TP).

        Returns:
            dict: ``{"closed": [position dicts], "opened": position dict | None}``.
        """
        cash = float(self._state_get(conn, "cash", settings.initial_capital))
        realized_today = float(self._state_get(conn, "realized_pnl_today", 0.0))
        total_commission = qty * fill_price * settings.commission_rate

        if only_position_id is not None:
            row = conn.execute(
                "SELECT * FROM paper_positions WHERE id=? AND status='open'",
                (only_position_id,),
            ).fetchone()
            opposite_rows = [row] if row is not None else []
        else:
            opposite_side = "short" if side == "buy" else "long"
            opposite_rows = self._open_positions(
                conn, symbol, source, timeframe, side=opposite_side
            )

        remaining = float(qty)
        closed: list[dict[str, Any]] = []
        for pos in opposite_rows:
            if remaining <= _EPS:
                break
            pos_id = int(pos["id"])
            pos_qty = float(pos["qty"])
            entry_price = float(pos["entry_price"])
            close_qty = min(remaining, pos_qty)
            fraction = close_qty / qty if qty > 0 else 0.0
            commission = total_commission * fraction
            slip_share = slippage_cost * fraction
            entry_commission = close_qty * entry_price * settings.commission_rate
            direction = 1.0 if pos["side"] == "long" else -1.0
            pnl = (
                direction * (fill_price - entry_price) * close_qty
                - commission
                - entry_commission
            )
            if side == "buy":  # buying to cover a short
                cash -= close_qty * fill_price + commission
            else:  # selling out of a long
                cash += close_qty * fill_price - commission
            realized_today += pnl

            # pnl accumulates across partial closes so a position exited in
            # several fills reports its TOTAL realized PnL, not the last slice.
            if close_qty >= pos_qty - _EPS:
                conn.execute(
                    "UPDATE paper_positions SET status='closed', closed_at=?, "
                    "exit_price=?, pnl=COALESCE(pnl, 0) + ?, last_price=? "
                    "WHERE id=?",
                    (executed_ms, fill_price, pnl, fill_price, pos_id),
                )
                updated = conn.execute(
                    "SELECT * FROM paper_positions WHERE id=?", (pos_id,)
                ).fetchone()
                closed.append(self._position_dict(updated))
            else:
                conn.execute(
                    "UPDATE paper_positions SET qty=?, last_price=?, "
                    "pnl=COALESCE(pnl, 0) + ? WHERE id=?",
                    (pos_qty - close_qty, fill_price, pnl, pos_id),
                )
            self._insert_trade(
                conn,
                order_id,
                pos_id,
                symbol,
                side,
                close_qty,
                fill_price,
                commission,
                slip_share,
                executed_ms,
            )
            remaining -= close_qty

        opened: dict[str, Any] | None = None
        if remaining > _EPS and only_position_id is None:
            fraction = remaining / qty if qty > 0 else 0.0
            commission = total_commission * fraction
            slip_share = slippage_cost * fraction
            new_side = "long" if side == "buy" else "short"
            # Fill-time guard: a protective level on the wrong side of the
            # fill price would insta-trigger on the next evaluated candle
            # (with adverse slippage for stops). Drop ONLY the offending
            # level — the fill itself still stands (least-disruptive).
            stop_loss, take_profit, dropped = _drop_wrong_side_levels(
                new_side, fill_price, stop_loss, take_profit
            )
            if dropped:
                explanation = "; ".join(dropped)
                logger.warning(
                    "Order #%s (%s %s x%.8g at %.8g): %s (paper)",
                    order_id,
                    side,
                    symbol,
                    remaining,
                    fill_price,
                    explanation,
                )
                if order_id is not None:
                    self._append_order_note(conn, order_id, explanation)
            if side == "buy":
                cash -= remaining * fill_price + commission
            else:
                cash += remaining * fill_price - commission
            cur = conn.execute(
                "INSERT INTO paper_positions (symbol, source, timeframe, side, "
                "qty, entry_price, stop_loss, take_profit, opened_at, status, "
                "last_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                (
                    symbol,
                    source,
                    timeframe,
                    new_side,
                    remaining,
                    fill_price,
                    stop_loss,
                    take_profit,
                    executed_ms,
                    fill_price,
                ),
            )
            new_pos_id = int(cur.lastrowid)
            self._insert_trade(
                conn,
                order_id,
                new_pos_id,
                symbol,
                side,
                remaining,
                fill_price,
                commission,
                slip_share,
                executed_ms,
            )
            new_row = conn.execute(
                "SELECT * FROM paper_positions WHERE id=?", (new_pos_id,)
            ).fetchone()
            opened = self._position_dict(new_row)

        self._state_set(conn, "cash", cash)
        self._state_set(conn, "realized_pnl_today", realized_today)
        return {"closed": closed, "opened": opened}

    # ------------------------------------------------------------------ #
    # Row serializers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _order_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Serialize a ``paper_orders`` row (epoch ms → ISO strings)."""
        record = dict(row)
        record["created_at"] = _ms_to_iso(record.get("created_at"))
        record["filled_at"] = _ms_to_iso(record.get("filled_at"))
        return record

    @staticmethod
    def _position_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Serialize a ``paper_positions`` row, adding ``unrealized_pnl``."""
        record = dict(row)
        record["opened_at"] = _ms_to_iso(record.get("opened_at"))
        record["closed_at"] = _ms_to_iso(record.get("closed_at"))
        if record.get("status") == "open":
            last = record.get("last_price")
            last = float(last) if last is not None else float(record["entry_price"])
            direction = 1.0 if record.get("side") == "long" else -1.0
            record["unrealized_pnl"] = (
                direction * (last - float(record["entry_price"])) * float(record["qty"])
            )
        else:
            record["unrealized_pnl"] = 0.0
        return record

    @staticmethod
    def _trade_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Serialize a ``paper_trades`` row (epoch ms → ISO strings)."""
        record = dict(row)
        record["executed_at"] = _ms_to_iso(record.get("executed_at"))
        return record
