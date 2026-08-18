"""Tests for backend/paper_trading — the simulated (paper-only) engine.

NO network: every test pre-seeds the ``ohlcv`` table via ``upsert_ohlcv`` so
fills happen against known local candles only. The autouse conftest fixtures
give each test a fresh database and block sockets outright.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.database.db import upsert_ohlcv
from backend.paper_trading.engine import PaperTradingEngine
from config.settings import settings

SOURCE = "binance"
SYMBOL = "TESTUSDT"
TIMEFRAME = "1h"

PORTFOLIO_KEYS = {
    "equity",
    "cash",
    "unrealized_pnl",
    "realized_pnl_today",
    "daily_pnl_pct",
    "open_positions",
    "peak_equity",
    "drawdown",
    "circuit_breaker_active",
    "trading_halted",
    "halt_reason",
}


def _seed_flat_candles(
    n: int = 30, close: float = 100.0, start: str = "2024-01-01"
) -> pd.DataFrame:
    """Seed n flat candles (close constant, high/low ±1) into the DB cache."""
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="timestamp")
    closes = np.full(n, close, dtype=np.float64)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(n, 250.0),
        },
        index=index,
    )
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, df)
    return df


def _append_candle(
    ts: str, open_: float, high: float, low: float, close: float
) -> None:
    """Append one crafted candle (e.g. one whose low pierces a stop)."""
    index = pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")], name="timestamp")
    df = pd.DataFrame(
        {
            "open": [open_],
            "high": [high],
            "low": [low],
            "close": [close],
            "volume": [250.0],
        },
        index=index,
    )
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, df)


def test_market_buy_fills_with_slippage_and_commission() -> None:
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(SYMBOL, "buy", "market", qty=2.0)

    expected_fill = 100.0 * (1.0 + settings.slippage_rate)
    assert order["status"] == "filled"
    assert order["fill_price"] == pytest.approx(expected_fill)

    positions = engine.get_positions("open")
    assert len(positions) == 1
    position = positions[0]
    assert position["side"] == "long"
    assert position["qty"] == pytest.approx(2.0)
    assert position["entry_price"] == pytest.approx(expected_fill)

    trades = engine.get_trades()
    assert len(trades) == 1
    expected_commission = 2.0 * expected_fill * settings.commission_rate
    assert trades[0]["commission"] == pytest.approx(expected_commission)


def test_limit_order_fills_only_when_touched() -> None:
    # 30 candles: last one at 2024-01-02 05:00 UTC.
    _seed_flat_candles(n=30, close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(SYMBOL, "buy", "limit", qty=1.0, limit_price=95.0)
    assert order["status"] == "pending"

    # Candle whose range (96..101) does NOT touch the 95 limit → stays pending.
    _append_candle("2024-01-02 06:00", 100.0, 101.0, 96.0, 100.0)
    result = engine.process_tick(SYMBOL, source=SOURCE, timeframe=TIMEFRAME)
    assert {"filled", "closed", "equity"} <= set(result.keys())
    assert len(result["filled"]) == 0
    assert engine.get_positions("open") == []

    # Candle whose low (94) pierces the limit → fills at the limit price
    # (slippage applies to market/stop fills only).
    _append_candle("2024-01-02 07:00", 100.0, 101.0, 94.0, 96.0)
    result = engine.process_tick(SYMBOL, source=SOURCE, timeframe=TIMEFRAME)
    assert len(result["filled"]) == 1

    filled = engine.get_orders(status="filled")
    assert len(filled) == 1
    assert filled[0]["fill_price"] == pytest.approx(95.0)
    assert len(engine.get_positions("open")) == 1


def test_stop_loss_closes_position() -> None:
    _seed_flat_candles(n=30, close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(SYMBOL, "buy", "market", qty=1.0, stop_loss=95.0)
    assert order["status"] == "filled"
    assert len(engine.get_positions("open")) == 1

    # Crafted candle whose low (90) pierces the 95 stop.
    _append_candle("2024-01-02 06:00", 100.0, 100.5, 90.0, 92.0)
    result = engine.process_tick(SYMBOL, source=SOURCE, timeframe=TIMEFRAME)

    assert len(result["closed"]) == 1
    assert engine.get_positions("open") == []

    closed = engine.get_positions("closed")
    assert len(closed) == 1
    position = closed[0]
    # Stop fills at the stop level with adverse slippage (sell lower).
    expected_exit = 95.0 * (1.0 - settings.slippage_rate)
    assert position["exit_price"] == pytest.approx(expected_exit)
    assert position["pnl"] < 0


def test_reset_restores_initial_capital() -> None:
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    engine.submit_order(SYMBOL, "buy", "market", qty=1.0)
    assert engine.get_portfolio()["open_positions"] == 1

    engine.reset()

    portfolio = engine.get_portfolio()
    assert portfolio["equity"] == pytest.approx(settings.initial_capital)
    assert portfolio["cash"] == pytest.approx(settings.initial_capital)
    assert portfolio["open_positions"] == 0
    assert engine.get_positions("open") == []
    assert engine.get_orders() == []
    assert engine.get_trades() == []


def test_portfolio_keys_present() -> None:
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()
    portfolio = engine.get_portfolio()
    assert PORTFOLIO_KEYS <= set(portfolio.keys())
    assert portfolio["equity"] == pytest.approx(settings.initial_capital)
    assert portfolio["trading_halted"] is False
