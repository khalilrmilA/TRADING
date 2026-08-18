"""Tests for backend/paper_trading/risk.py — pre-trade risk checks and sizing.

Pure unit tests: portfolio snapshots are handcrafted dicts matching the
``PaperTradingEngine.get_portfolio()`` contract shape.
"""

from __future__ import annotations

import math

import pytest

from backend.paper_trading.risk import RiskCheckResult, RiskManager
from config.settings import settings


def _portfolio(**overrides: object) -> dict:
    """A healthy default portfolio snapshot; override fields per scenario."""
    portfolio: dict = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "unrealized_pnl": 0.0,
        "realized_pnl_today": 0.0,
        "daily_pnl_pct": 0.0,
        "open_positions": 0,
        "peak_equity": 100_000.0,
        "drawdown": 0.0,
        "circuit_breaker_active": False,
        "trading_halted": False,
        "halt_reason": "",
    }
    portfolio.update(overrides)
    return portfolio


def _open_positions(n: int) -> list[dict]:
    return [
        {
            "id": i + 1,
            "symbol": "TESTUSDT",
            "side": "long",
            "qty": 1.0,
            "entry_price": 100.0,
            "status": "open",
        }
        for i in range(n)
    ]


def test_check_order_allows_healthy_account() -> None:
    result = RiskManager().check_order(_portfolio(), [])
    assert isinstance(result, RiskCheckResult)
    assert result.allowed is True


def test_check_order_rejects_at_max_open_positions() -> None:
    n = settings.max_open_positions  # a new order would be position n+1
    result = RiskManager().check_order(
        _portfolio(open_positions=n), _open_positions(n)
    )
    assert result.allowed is False
    assert result.reason != ""


def test_check_order_rejects_on_daily_loss_limit() -> None:
    # -4% on the day (limit is -3%); drawdown kept below the circuit breaker.
    portfolio = _portfolio(
        equity=96_000.0,
        realized_pnl_today=-4_000.0,
        unrealized_pnl=0.0,
        daily_pnl_pct=-0.04,
        peak_equity=100_000.0,
        drawdown=0.04,
    )
    result = RiskManager().check_order(portfolio, [])
    assert result.allowed is False
    assert result.reason != ""


def test_check_order_rejects_on_circuit_breaker() -> None:
    # 12% drawdown from peak (breaker fires at 10%).
    portfolio = _portfolio(
        equity=88_000.0,
        peak_equity=100_000.0,
        drawdown=0.12,
        circuit_breaker_active=True,
        trading_halted=True,
        halt_reason="circuit breaker",
    )
    result = RiskManager().check_order(portfolio, [])
    assert result.allowed is False
    assert result.reason != ""


def test_position_size_formula() -> None:
    equity, atr_value, price = 100_000.0, 2.0, 100.0
    expected = (equity * settings.risk_per_trade) / (
        settings.atr_stop_multiplier * atr_value
    )
    qty = RiskManager().position_size(equity, atr_value, price)
    assert qty == pytest.approx(expected)
    # Uncapped in this scenario: notional stays under 95% of equity.
    assert qty * price <= 0.95 * equity


def test_position_size_capped_at_95_pct_notional() -> None:
    # Tiny ATR → raw formula wants a huge position → capped at 95% notional.
    equity, atr_value, price = 100_000.0, 0.01, 100.0
    qty = RiskManager().position_size(equity, atr_value, price)
    assert qty == pytest.approx(0.95 * equity / price)


@pytest.mark.parametrize("bad_atr", [0.0, -1.0, math.nan])
def test_position_size_invalid_atr_returns_zero(bad_atr: float) -> None:
    qty = RiskManager().position_size(100_000.0, bad_atr, 100.0)
    assert qty == 0.0
