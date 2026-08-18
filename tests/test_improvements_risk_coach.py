"""Tests for Improvement Pack v3 risk & coach rules (sections 3 and 7).

PAPER TRADING ONLY — pure unit tests: no network, no Ollama, handcrafted
portfolio/position dicts matching the ``PaperTradingEngine.get_portfolio()``
contract shape, tmp-path SQLite via the shared conftest fixtures.

Covers the binding contract of:

* ``RiskManager.portfolio_heat``       — stop-distance dollars, the
  ``NO_STOP_RISK_FRACTION`` fallback, malformed-row tolerance.
* ``RiskManager.check_order``          — the appended direction cap (check 5)
  and heat cap (check 6), including the exact boundary, the risk-reducing
  skip and the side-None skip.
* ``RiskManager.soft_daily_stop_active`` — the −80%-of-daily-limit trigger.
* ``RiskManager.derisk_multiplier``    — drawdown tiers (0% / 10% / 25%),
  soft-stop halving, the combined case and invalid-input degradation.
* ``Coach._sanitize_proposal``         — shrink-only evidence gates at
  n = 5 / 15 / 25 closed trades (size-up needs 20, side-bias needs 10,
  bench stays allowed at 3) with the ``shrink_only_clamped`` audit list.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import pytest

from backend.paper_trading import coach as coach_module
from backend.paper_trading.coach import Coach
from backend.paper_trading.risk import (
    NO_STOP_RISK_FRACTION,
    RiskCheckResult,
    RiskManager,
)
from config.settings import settings


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _v3_limits(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin every limit these tests depend on to the contract defaults.

    A local ``.env`` may override any Settings field; the v3 fields are set
    with ``raising=False`` so the pin also works while ``config/settings.py``
    is still being extended by its owner.
    """
    monkeypatch.setattr(settings, "heat_cap_fraction", 0.06, raising=False)
    monkeypatch.setattr(settings, "max_same_direction", 5, raising=False)
    monkeypatch.setattr(settings, "max_open_positions", 5)
    monkeypatch.setattr(settings, "daily_loss_limit", 0.03)
    monkeypatch.setattr(settings, "circuit_breaker_drawdown", 0.10)
    yield


def _portfolio(**overrides: Any) -> dict[str, Any]:
    """A healthy default portfolio snapshot; override fields per scenario."""
    portfolio: dict[str, Any] = {
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


def _position(
    symbol: str = "TESTUSDT",
    side: str = "long",
    qty: float = 1.0,
    entry_price: float = 100.0,
    stop_loss: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """One handcrafted open-position dict."""
    position: dict[str, Any] = {
        "id": 1,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "status": "open",
        "source": "binance",
        "timeframe": "1h",
    }
    position.update(extra)
    return position


# ---------------------------------------------------------------------------
# portfolio_heat
# ---------------------------------------------------------------------------


def test_no_stop_risk_fraction_constant() -> None:
    assert NO_STOP_RISK_FRACTION == 0.02


def test_portfolio_heat_empty_is_zero() -> None:
    assert RiskManager.portfolio_heat([]) == 0.0


def test_portfolio_heat_uses_stop_distance() -> None:
    positions = [_position(qty=2.0, entry_price=100.0, stop_loss=95.0)]
    assert RiskManager.portfolio_heat(positions) == pytest.approx(10.0)


def test_portfolio_heat_no_stop_uses_fraction_of_notional() -> None:
    positions = [_position(qty=3.0, entry_price=100.0, stop_loss=None)]
    expected = 100.0 * 3.0 * NO_STOP_RISK_FRACTION  # 6.0
    assert RiskManager.portfolio_heat(positions) == pytest.approx(expected)


@pytest.mark.parametrize("bad_stop", [math.nan, math.inf])
def test_portfolio_heat_nonfinite_stop_falls_back(bad_stop: float) -> None:
    positions = [_position(qty=2.0, entry_price=100.0, stop_loss=bad_stop)]
    expected = 100.0 * 2.0 * NO_STOP_RISK_FRACTION
    assert RiskManager.portfolio_heat(positions) == pytest.approx(expected)


def test_portfolio_heat_zero_distance_stop_falls_back() -> None:
    # stop == entry → distance 0 → treated like a missing stop.
    positions = [_position(qty=2.0, entry_price=50.0, stop_loss=50.0)]
    expected = 50.0 * 2.0 * NO_STOP_RISK_FRACTION  # 2.0
    assert RiskManager.portfolio_heat(positions) == pytest.approx(expected)


def test_portfolio_heat_malformed_rows_contribute_zero_and_never_raise() -> None:
    malformed: list[dict[str, Any]] = [
        {"entry_price": "not-a-number", "qty": "x", "stop_loss": None},
        {},
    ]
    assert RiskManager.portfolio_heat(malformed) == 0.0


def test_portfolio_heat_sums_across_positions() -> None:
    positions = [
        _position(qty=2.0, entry_price=100.0, stop_loss=95.0),  # 10.0
        _position(qty=3.0, entry_price=100.0, stop_loss=None),  # 6.0
        _position(qty=2.0, entry_price=50.0, stop_loss=50.0),   # 2.0
        {"entry_price": "junk"},                                 # 0.0
    ]
    assert RiskManager.portfolio_heat(positions) == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# check_order — direction cap (check 5)
# ---------------------------------------------------------------------------


def test_direction_cap_rejects_buy_at_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_same_direction", 3, raising=False)
    positions = [
        _position(symbol=f"COIN{i}USDT", side="long", qty=1.0,
                  entry_price=100.0, stop_loss=99.0)
        for i in range(3)
    ]
    result = RiskManager().check_order(
        _portfolio(open_positions=3),
        positions,
        side="buy",
        qty=1.0,
        symbol="NEWUSDT",
        price=100.0,
        stop_loss=99.0,
    )
    assert isinstance(result, RiskCheckResult)
    assert result.allowed is False
    expected_prefix = (
        "direction_cap: 3 long positions already open >= limit 3 — "
        "too much exposure in one direction"
    )
    assert result.reason.startswith(expected_prefix)


def test_direction_cap_rejects_sell_on_short_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_same_direction", 3, raising=False)
    positions = [
        _position(symbol=f"COIN{i}USDT", side="short", qty=1.0,
                  entry_price=100.0, stop_loss=101.0)
        for i in range(3)
    ]
    result = RiskManager().check_order(
        _portfolio(open_positions=3),
        positions,
        side="sell",
        qty=1.0,
        symbol="NEWUSDT",
        price=100.0,
        stop_loss=101.0,
    )
    assert result.allowed is False
    assert result.reason.startswith("direction_cap: 3 short positions")


def test_direction_cap_counts_only_matching_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 shorts must not block a BUY (its direction is long, count 0)."""
    monkeypatch.setattr(settings, "max_same_direction", 3, raising=False)
    positions = [
        _position(symbol=f"COIN{i}USDT", side="short", qty=1.0,
                  entry_price=100.0, stop_loss=101.0)
        for i in range(3)
    ]
    result = RiskManager().check_order(
        _portfolio(open_positions=3),
        positions,
        side="buy",
        qty=1.0,
        symbol="NEWUSDT",
        price=100.0,
        stop_loss=99.0,
    )
    assert result.allowed is True


def test_direction_cap_allows_below_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_same_direction", 3, raising=False)
    positions = [
        _position(symbol=f"COIN{i}USDT", side="long", qty=1.0,
                  entry_price=100.0, stop_loss=99.0)
        for i in range(2)
    ]
    result = RiskManager().check_order(
        _portfolio(open_positions=2),
        positions,
        side="buy",
        qty=1.0,
        symbol="NEWUSDT",
        price=100.0,
        stop_loss=99.0,
    )
    assert result.allowed is True


def test_new_checks_skipped_when_side_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy call shape (no side) must not hit the new caps."""
    monkeypatch.setattr(settings, "max_same_direction", 3, raising=False)
    positions = [
        _position(symbol=f"COIN{i}USDT", side="long", qty=1.0,
                  entry_price=100.0, stop_loss=99.0)
        for i in range(3)
    ]
    result = RiskManager().check_order(_portfolio(open_positions=3), positions)
    assert result.allowed is True


# ---------------------------------------------------------------------------
# check_order — heat cap (check 6)
# ---------------------------------------------------------------------------


def test_heat_cap_allows_exactly_at_boundary() -> None:
    """projected == cap must pass — rejection needs projected > cap + 1e-9.

    Existing heat 5000 (|100−50|·100) + new risk 1000 (|100−90|·100) lands
    exactly on the 6% × 100,000 = 6000 cap.
    """
    positions = [
        _position(symbol="AAAUSDT", qty=100.0, entry_price=100.0, stop_loss=50.0)
    ]
    result = RiskManager().check_order(
        _portfolio(open_positions=1),
        positions,
        side="buy",
        qty=100.0,
        symbol="NEWUSDT",
        price=100.0,
        stop_loss=90.0,
    )
    assert result.allowed is True


def test_heat_cap_rejects_just_over_boundary() -> None:
    """One extra unit of qty (projected 6010 > 6000) must reject."""
    positions = [
        _position(symbol="AAAUSDT", qty=100.0, entry_price=100.0, stop_loss=50.0)
    ]
    result = RiskManager().check_order(
        _portfolio(open_positions=1),
        positions,
        side="buy",
        qty=101.0,
        symbol="NEWUSDT",
        price=100.0,
        stop_loss=90.0,
    )
    assert result.allowed is False
    projected = 5_000.0 + 1_010.0
    expected_prefix = (
        f"heat_cap: total stop-distance risk would reach "
        f"{projected / 100_000.0:.2%} of equity, above the "
        f"{settings.heat_cap_fraction:.2%} cap — the account would "
        f"lose too much if every stop was hit at once"
    )
    assert result.reason.startswith(expected_prefix)


def test_heat_cap_new_order_without_stop_uses_fraction() -> None:
    """No stop on the new order → price·qty·0.02 counts as its risk."""
    result = RiskManager().check_order(
        _portfolio(),
        [],
        side="buy",
        qty=3_200.0,  # 100 · 3200 · 0.02 = 6400 > 6000 cap
        symbol="NEWUSDT",
        price=100.0,
        stop_loss=None,
    )
    assert result.allowed is False
    assert result.reason.startswith("heat_cap:")


def test_heat_cap_degraded_check_on_existing_heat_when_price_unknown() -> None:
    """price=None → new_risk 0, but existing heat over the cap still rejects."""
    positions = [
        _position(symbol="AAAUSDT", qty=100.0, entry_price=100.0, stop_loss=30.0)
    ]  # heat 7000 > 6000 cap
    result = RiskManager().check_order(
        _portfolio(open_positions=1),
        positions,
        side="buy",
        qty=1.0,
        symbol="NEWUSDT",
        price=None,
        stop_loss=None,
    )
    assert result.allowed is False
    assert result.reason.startswith("heat_cap:")


def test_heat_cap_skipped_for_risk_reducing_order() -> None:
    """A sell fully netting an open long is allowed despite over-cap heat."""
    positions = [
        _position(symbol="TESTUSDT", side="long", qty=200.0,
                  entry_price=100.0, stop_loss=50.0)
    ]  # heat 10000, way over the 6000 cap
    result = RiskManager().check_order(
        _portfolio(open_positions=1),
        positions,
        side="sell",
        qty=150.0,
        symbol="TESTUSDT",
        price=100.0,
        stop_loss=None,
    )
    assert result.allowed is True
    assert result.reason == "ok"


# ---------------------------------------------------------------------------
# soft_daily_stop_active
# ---------------------------------------------------------------------------


def test_soft_daily_stop_inactive_when_flat() -> None:
    assert RiskManager.soft_daily_stop_active(100_000.0, 0.0, 0.03) is False


def test_soft_daily_stop_active_beyond_80_pct_of_limit() -> None:
    # threshold = -0.8 · 0.03 · 100000 = -2400
    assert RiskManager.soft_daily_stop_active(100_000.0, -2_500.0, 0.03) is True


def test_soft_daily_stop_inactive_just_inside_limit() -> None:
    assert RiskManager.soft_daily_stop_active(100_000.0, -2_300.0, 0.03) is False


def test_soft_daily_stop_boundary_is_inclusive() -> None:
    """realized == −0.8·limit·equity triggers (<=). Float-exact by design:
    0.8 · 0.625 · 1000 is exactly 500.0 in IEEE-754 doubles."""
    assert RiskManager.soft_daily_stop_active(1_000.0, -500.0, 0.625) is True


@pytest.mark.parametrize("bad_equity", [0.0, -1.0])
def test_soft_daily_stop_requires_positive_equity(bad_equity: float) -> None:
    assert RiskManager.soft_daily_stop_active(bad_equity, -5_000.0, 0.03) is False


# ---------------------------------------------------------------------------
# derisk_multiplier
# ---------------------------------------------------------------------------


def test_derisk_multiplier_no_drawdown_is_one() -> None:
    m = RiskManager.derisk_multiplier(100_000.0, 100_000.0, 0.0, 0.03)
    assert m == 1.0


def test_derisk_multiplier_10_pct_drawdown() -> None:
    # dd = 10.0 → tier int(10 // 10) = 1 → 0.8
    m = RiskManager.derisk_multiplier(90_000.0, 100_000.0, 0.0, 0.03)
    assert m == pytest.approx(0.8)


def test_derisk_multiplier_25_pct_drawdown() -> None:
    # dd = 25.0 → tier 2 → 0.8 ** 2 = 0.64
    m = RiskManager.derisk_multiplier(75_000.0, 100_000.0, 0.0, 0.03)
    assert m == pytest.approx(0.64)


def test_derisk_multiplier_just_below_10_pct_is_one() -> None:
    m = RiskManager.derisk_multiplier(90_001.0, 100_000.0, 0.0, 0.03)
    assert m == 1.0


def test_derisk_multiplier_soft_stop_halves() -> None:
    # No drawdown, but the day's realized loss is past 80% of the limit.
    m = RiskManager.derisk_multiplier(100_000.0, 100_000.0, -2_500.0, 0.03)
    assert m == pytest.approx(0.5)


def test_derisk_multiplier_combined_drawdown_and_soft_stop() -> None:
    # 25% dd (0.64) AND soft stop (×0.5); threshold at 75k equity is -1800.
    m = RiskManager.derisk_multiplier(75_000.0, 100_000.0, -1_900.0, 0.03)
    assert m == pytest.approx(0.32)


def test_derisk_multiplier_deep_drawdown_tiers() -> None:
    # dd = 55.0 → tier 5 → 0.8 ** 5 = 0.32768
    m = RiskManager.derisk_multiplier(45_000.0, 100_000.0, 0.0, 0.03)
    assert m == pytest.approx(0.32768)


@pytest.mark.parametrize(
    ("equity", "peak"),
    [(math.nan, 100_000.0), (100_000.0, math.nan), (100_000.0, 0.0),
     (100_000.0, -5.0)],
)
def test_derisk_multiplier_invalid_inputs_return_one(
    equity: float, peak: float
) -> None:
    assert RiskManager.derisk_multiplier(equity, peak, 0.0, 0.03) == 1.0


@pytest.mark.parametrize(
    ("equity", "peak", "realized"),
    [
        (100_000.0, 100_000.0, 0.0),
        (90_000.0, 100_000.0, -2_500.0),
        (40_000.0, 100_000.0, -5_000.0),
    ],
)
def test_derisk_multiplier_always_in_unit_interval(
    equity: float, peak: float, realized: float
) -> None:
    m = RiskManager.derisk_multiplier(equity, peak, realized, 0.03)
    assert 0.0 < m <= 1.0


# ---------------------------------------------------------------------------
# Coach — shrink-only evidence gates (_sanitize_proposal)
# ---------------------------------------------------------------------------


@pytest.fixture()
def coach() -> Coach:
    """A Coach against the per-test tmp DB (conftest patches settings.db_path)."""
    return Coach()


def _current(**overrides: Any) -> dict[str, Any]:
    """A default (neutral) current playbook entry."""
    entry: dict[str, Any] = {
        "bench": False,
        "side_bias": "both",
        "size_multiplier": 1.0,
        "min_vote_override": None,
    }
    entry.update(overrides)
    return entry


def test_coach_evidence_constants_match_contract() -> None:
    assert coach_module.COACH_SIZE_UP_MIN_TRADES == 20
    assert coach_module.COACH_SIDE_BIAS_MIN_TRADES == 10
    assert coach_module.MIN_TRADES_FOR_CHANGE == 3  # unchanged by v3


def test_sanitize_n5_blocks_size_up_and_side_bias_allows_bench(
    coach: Coach,
) -> None:
    """5 trades: bench OK (≥3), side_bias forced 'both', size capped at 1.0."""
    entry, reasoning, clamped = coach._sanitize_proposal(
        {
            "bench": True,
            "side_bias": "long_only",
            "size_multiplier": 1.4,
            "min_vote_override": 3,
            "reasoning": "small hot streak",
        },
        _current(),
        5,
    )
    assert entry["bench"] is True
    assert entry["side_bias"] == "both"
    assert entry["size_multiplier"] == pytest.approx(1.0)
    assert entry["min_vote_override"] == 3
    assert reasoning == "small hot streak"
    assert set(clamped) == {"size_multiplier", "side_bias"}


def test_sanitize_n5_shrinking_proposal_passes_untouched(coach: Coach) -> None:
    """Shrink-only means shrinking stays allowed: 0.5 size at 5 trades is fine."""
    entry, _reasoning, clamped = coach._sanitize_proposal(
        {"bench": False, "side_bias": "both", "size_multiplier": 0.5,
         "reasoning": "de-risk"},
        _current(),
        5,
    )
    assert entry["size_multiplier"] == pytest.approx(0.5)
    assert entry["side_bias"] == "both"
    assert clamped == []


def test_sanitize_n5_forces_stored_side_bias_back_to_both(coach: Coach) -> None:
    """The gate applies to the RESULT — a stored non-'both' bias with thin
    evidence is forced back to 'both' even when the proposal omits it."""
    entry, _reasoning, clamped = coach._sanitize_proposal(
        {"reasoning": "no changes proposed"},
        _current(side_bias="long_only", size_multiplier=1.3),
        5,
    )
    assert entry["side_bias"] == "both"
    assert entry["size_multiplier"] == pytest.approx(1.0)
    assert set(clamped) == {"side_bias", "size_multiplier"}


def test_sanitize_n15_side_bias_allowed_size_up_still_blocked(
    coach: Coach,
) -> None:
    """15 trades: ≥10 unlocks side_bias, but size-up still needs 20."""
    entry, _reasoning, clamped = coach._sanitize_proposal(
        {"side_bias": "long_only", "size_multiplier": 1.4,
         "reasoning": "longs work"},
        _current(),
        15,
    )
    assert entry["side_bias"] == "long_only"
    assert entry["size_multiplier"] == pytest.approx(1.0)
    assert clamped == ["size_multiplier"]


def test_sanitize_n25_everything_allowed(coach: Coach) -> None:
    """25 trades: both gates satisfied — the proposal lands as sent."""
    entry, _reasoning, clamped = coach._sanitize_proposal(
        {"side_bias": "short_only", "size_multiplier": 1.4,
         "reasoning": "earned it"},
        _current(),
        25,
    )
    assert entry["side_bias"] == "short_only"
    assert entry["size_multiplier"] == pytest.approx(1.4)
    assert clamped == []


def test_sanitize_n25_bounds_clamp_is_not_shrink_only(coach: Coach) -> None:
    """The pre-existing [0.25, 1.5] bounds clamp still applies at 25 trades
    but is NOT a shrink-only event — the audit list stays empty."""
    entry, _reasoning, clamped = coach._sanitize_proposal(
        {"size_multiplier": 2.5, "reasoning": "greedy"},
        _current(),
        25,
    )
    assert entry["size_multiplier"] == pytest.approx(1.5)
    assert clamped == []


def test_sanitize_boundary_trades_10_unlocks_side_bias_only(
    coach: Coach,
) -> None:
    entry, _reasoning, clamped = coach._sanitize_proposal(
        {"side_bias": "long_only", "size_multiplier": 1.2,
         "reasoning": "boundary"},
        _current(),
        10,
    )
    assert entry["side_bias"] == "long_only"
    assert entry["size_multiplier"] == pytest.approx(1.0)
    assert clamped == ["size_multiplier"]


def test_sanitize_boundary_trades_20_unlocks_size_up(coach: Coach) -> None:
    entry, _reasoning, clamped = coach._sanitize_proposal(
        {"side_bias": "long_only", "size_multiplier": 1.2,
         "reasoning": "boundary"},
        _current(),
        20,
    )
    assert entry["side_bias"] == "long_only"
    assert entry["size_multiplier"] == pytest.approx(1.2)
    assert clamped == []


def test_sanitize_returns_entry_reasoning_and_audit_list(coach: Coach) -> None:
    result = coach._sanitize_proposal(
        {"size_multiplier": 1.4, "reasoning": "shape check"},
        _current(),
        5,
    )
    assert isinstance(result, tuple) and len(result) == 3
    entry, reasoning, clamped = result
    assert isinstance(entry, dict)
    assert {"bench", "side_bias", "size_multiplier", "min_vote_override"} <= set(
        entry
    )
    assert isinstance(reasoning, str)
    assert isinstance(clamped, list)
    assert all(isinstance(name, str) for name in clamped)
