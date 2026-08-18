"""Tests for the AutoTrader trade protections (paper only).

Covers the four protections added to ``backend/paper_trading/auto_trader.py``:

* Per-pair loss cooldown — a symbol that just closed a LOSING bot position is
  skipped (``protection_cooldown``) until ``cooldown_minutes_after_loss``
  expires; scalper-owned losses never count and 0 disables the rule.
* Stop-streak stand-aside — ``stop_streak_limit`` bot ``stop_loss`` exits
  inside ``stop_streak_window_hours`` write ONE ``halt`` row
  (``protection_stop_streak``), persist the pause in ``account_state`` (it
  survives restarts without spamming a halt row per cycle) and block new
  entries until the pause expires; scalper ``scalp_exit`` rows never count.
* Time-stop — a bot position older than ``time_stop_bars`` bars of its own
  timeframe is market-closed during position management with an ``exit`` row
  (reason ``time_stop``) through the usual ``_log_exit`` path; scalper-owned
  positions are untouched.
* Fallback-verdict gate — a primary AI confirmation answered by the FALLBACK
  model is conservatively skipped (``ai_fallback``, with the ``ai`` payload
  preserved for calibration) unless ``accept_fallback_verdicts`` is True; a
  differently-tagged install of the requested primary is NOT a fallback.

Everything runs offline and deterministically: sockets are blocked by the
autouse conftest fixture, ``update_symbol`` is a monkeypatched no-op,
``analyze_market`` is stubbed where needed, and all candles are seeded
synthetic frames in the per-test tmp SQLite cache (same fixtures/patterns as
``tests/test_auto_trader.py``).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

import backend.ai.analyst as analyst_mod
import backend.data.service as data_service_mod
import backend.paper_trading.auto_trader as auto_trader_mod
from backend.ai.analyst import MarketAnalysis
from backend.database.db import get_conn, upsert_ohlcv, utc_now_ms
from backend.paper_trading.auto_trader import AutoTrader, BotConfig
from backend.paper_trading.engine import PaperTradingEngine
from config.settings import settings

SOURCE = "binance"
TIMEFRAME = "1h"
TREND_SYMBOL = "TRENDUSDT"
SCALP_SYMBOL = "SCALPUSDT"
FRESH_SYMBOL = "FRESHUSDT"

MINUTE_MS = 60_000
HOUR_MS = 3_600_000


# --------------------------------------------------------------------------- #
# Synthetic market data (seeded — fully deterministic)                          #
# --------------------------------------------------------------------------- #


def _make_ohlcv(close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    """Wrap a close/volume path into a canonical hourly UTC OHLCV frame.

    The frame ends at the most recent CLOSED hourly candle so the bot's
    closed-candle and data-freshness gates treat it as live data.
    """
    n = len(close)
    last_closed = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    start = last_closed - pd.Timedelta(hours=n - 1)
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    index = pd.date_range(
        start, periods=n, freq="1h", tz="UTC", name="timestamp"
    ).as_unit("ms")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")


def _trending_df(n: int = 240) -> pd.DataFrame:
    """Strongly up-trending frame with a +2 last-bar strategy vote sum.

    Identical construction to ``tests/test_auto_trader.py`` (breakout close
    on a volume spike), which pins the vote sum at +2 — enough for the
    ``min_vote=2`` baseline below to shortlist a long candidate.
    """
    rng = np.random.default_rng(7)
    i = np.arange(n, dtype=np.float64)
    base = 100.0 + 0.6 * np.sin(i[: n // 2] / 5.0)
    trend = base[-1] + 0.45 * (i[: n - n // 2] + 1) + 0.3 * np.sin(i[: n - n // 2] / 4.0)
    close = np.concatenate([base, trend]) + rng.normal(0.0, 0.05, n)
    close[-1] = close[-2] * 1.02
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    volume[-1] = 800.0
    return _make_ohlcv(close, volume)


def _choppy_df(n: int = 240) -> pd.DataFrame:
    """Range-bound frame whose last bar never qualifies for the shortlist."""
    rng = np.random.default_rng(11)
    i = np.arange(n, dtype=np.float64)
    close = 100.0 + 0.8 * np.sin(i / 7.0 + 4.9) + rng.normal(0.0, 0.05, n)
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    return _make_ohlcv(close, volume)


def _seed_symbol(symbol: str, df: pd.DataFrame) -> None:
    """Pre-seed the tmp-DB candle cache for one symbol."""
    upsert_ohlcv(SOURCE, symbol, TIMEFRAME, df)


# --------------------------------------------------------------------------- #
# Bot / DB helpers                                                              #
# --------------------------------------------------------------------------- #


def _make_trader() -> tuple[PaperTradingEngine, AutoTrader]:
    """Build a fresh engine + trader pair against the per-test tmp database."""
    engine = PaperTradingEngine()
    return engine, AutoTrader(engine)


def _configure(trader: AutoTrader, **overrides: Any) -> BotConfig:
    """Apply the offline test baseline config plus per-test overrides."""
    updates: dict[str, Any] = {
        "watchlist": [TREND_SYMBOL],
        "source": SOURCE,
        "timeframe": TIMEFRAME,
        "use_ai": False,
        # Sockets are blocked, so list_models() is always "unknown" — keep
        # the judge out of these tests (it has its own acceptance suite).
        "use_second_judge": False,
        "min_vote": 2,
        "allow_short": False,
        # The v3 cost gate would veto the low-ATR fixtures before the
        # protections under test run (0 disables, per its contract knob).
        "cost_gate_multiple": 0.0,
    }
    updates.update(overrides)
    return trader.set_config(updates)


def _patch_lookup(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: Callable[..., Any],
    source_module: Any,
) -> None:
    """Patch a callable at its source module AND where auto_trader looked it up."""
    monkeypatch.setattr(source_module, name, replacement, raising=True)
    if hasattr(auto_trader_mod, name):
        monkeypatch.setattr(auto_trader_mod, name, replacement)


def _patch_no_data_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every data refresh a no-op so scans run purely off the seeded cache."""

    def _noop_update(*args: Any, **kwargs: Any) -> int:
        return 0

    _patch_lookup(monkeypatch, "update_symbol", _noop_update, data_service_mod)


def _forbid_analyst(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``analyze_market`` with a stub that fails the test when called."""

    def _boom(*args: Any, **kwargs: Any) -> MarketAnalysis:
        raise AssertionError("analyze_market must not be called when use_ai=False")

    _patch_lookup(monkeypatch, "analyze_market", _boom, analyst_mod)


def _patch_analyst_with_model(
    monkeypatch: pytest.MonkeyPatch,
    model_used: str,
    sentiment: str = "bullish",
    confidence: int = 90,
) -> list[str]:
    """Stub ``analyze_market`` with a fixed verdict carrying ``model_used``."""
    calls: list[str] = []

    def _stub(
        symbol: str,
        timeframe: str = TIMEFRAME,
        df: pd.DataFrame | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MarketAnalysis:
        calls.append(str(symbol))
        return MarketAnalysis(
            sentiment=sentiment,  # type: ignore[arg-type]
            confidence=confidence,
            risk_commentary="synthetic stub commentary",
            key_indicators=[],
            reasoning="synthetic stub reasoning",
            model_used=model_used,
            symbol=str(symbol),
            timeframe=str(timeframe),
        )

    _patch_lookup(monkeypatch, "analyze_market", _stub, analyst_mod)
    return calls


def _bot_activity(action: str | None = None) -> list[dict[str, Any]]:
    """Read the ``bot_activity`` audit log (oldest first), decoding detail JSON."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT ts, symbol, action, detail FROM bot_activity ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            detail = json.loads(row["detail"] or "{}")
        except (TypeError, ValueError):
            detail = {}
        items.append(
            {
                "ts": int(row["ts"]),
                "symbol": row["symbol"],
                "action": row["action"],
                "detail": detail,
            }
        )
    if action is not None:
        items = [item for item in items if item["action"] == action]
    return items


def _skips_with_reason(reason: str) -> list[dict[str, Any]]:
    """All ``skip`` rows whose detail reason equals ``reason``."""
    return [
        item
        for item in _bot_activity("skip")
        if str(item["detail"].get("reason") or "") == reason
    ]


def _halts_with_reason(reason: str) -> list[dict[str, Any]]:
    """All ``halt`` rows whose detail reason equals ``reason``."""
    return [
        item
        for item in _bot_activity("halt")
        if str(item["detail"].get("reason") or "") == reason
    ]


def _insert_closed_position(
    symbol: str, pnl: float, closed_ago_ms: int
) -> int:
    """Seed one CLOSED ``paper_positions`` row and return its id."""
    now_ms = utc_now_ms()
    closed_at = now_ms - int(closed_ago_ms)
    entry_price = 100.0
    conn = get_conn()
    try:
        cursor = conn.execute(
            "INSERT INTO paper_positions "
            "(symbol, source, timeframe, side, qty, entry_price, stop_loss, "
            " take_profit, opened_at, closed_at, exit_price, pnl, status) "
            "VALUES (?, ?, ?, 'long', 1.0, ?, NULL, NULL, ?, ?, ?, ?, 'closed')",
            (
                symbol,
                SOURCE,
                TIMEFRAME,
                entry_price,
                closed_at - HOUR_MS,
                closed_at,
                entry_price + float(pnl),
                float(pnl),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def _set_closed_ago(position_id: int, closed_ago_ms: int) -> None:
    """Move an existing closed position's ``closed_at`` relative to now."""
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE paper_positions SET closed_at=? WHERE id=?",
            (utc_now_ms() - int(closed_ago_ms), int(position_id)),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_exit_rows(
    n: int, ago_ms: int, action: str = "exit", reason: str = "stop_loss"
) -> None:
    """Fabricate ``n`` bot_activity exit rows ``ago_ms`` in the past."""
    now_ms = utc_now_ms()
    conn = get_conn()
    try:
        for i in range(n):
            detail = {"reason": reason, "pnl": -10.0, "pnl_pct": -0.01}
            conn.execute(
                "INSERT INTO bot_activity (ts, symbol, action, detail) "
                "VALUES (?, ?, ?, ?)",
                (now_ms - int(ago_ms) - i * 1000, TREND_SYMBOL, action, json.dumps(detail)),
            )
        conn.commit()
    finally:
        conn.close()


def _shift_exit_rows(action: str, ago_ms: int) -> None:
    """Re-timestamp every ``action`` activity row to ``ago_ms`` in the past."""
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE bot_activity SET ts=? WHERE action=?",
            (utc_now_ms() - int(ago_ms), action),
        )
        conn.commit()
    finally:
        conn.close()


def _state_get(key: str) -> Any:
    """Read a JSON value from ``account_state`` (None when absent)."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM account_state WHERE key=?", (key,)
        ).fetchone()
    finally:
        conn.close()
    return json.loads(row["value"]) if row is not None else None


def _state_set(key: str, value: Any) -> None:
    """Write a JSON value into ``account_state``."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()
    finally:
        conn.close()


def _mark_scalper_owned(position_id: int) -> None:
    """Register a position id as scalper-owned (ownership-set contract key)."""
    _state_set("scalper_position_ids", [int(position_id)])


def _backdate_opened_at(position_id: int, minutes: int) -> None:
    """Shift a position's ``opened_at`` back by ``minutes`` (age it)."""
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE paper_positions SET opened_at = opened_at - ? WHERE id=?",
            (int(minutes) * MINUTE_MS, int(position_id)),
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Config: defaults, clamps and persistence round-trip                           #
# --------------------------------------------------------------------------- #


def test_protection_config_defaults_clamps_and_roundtrip() -> None:
    """New fields default per spec, clamp instead of reject, and persist."""
    engine, trader = _make_trader()

    cfg = trader.get_config()
    assert cfg.cooldown_minutes_after_loss == 90
    assert cfg.stop_streak_limit == 3
    assert cfg.stop_streak_window_hours == 12
    assert cfg.stop_streak_pause_hours == 6
    assert cfg.time_stop_bars == 0
    assert cfg.accept_fallback_verdicts is False

    # Out-of-range values are CLAMPED (config style), never rejected.
    updated = trader.set_config(
        {
            "cooldown_minutes_after_loss": 5000,
            "stop_streak_limit": 99,
            "stop_streak_window_hours": 0,
            "stop_streak_pause_hours": 100,
            "time_stop_bars": 10_000,
            "accept_fallback_verdicts": True,
        }
    )
    assert updated.cooldown_minutes_after_loss == 1440
    assert updated.stop_streak_limit == 20
    assert updated.stop_streak_window_hours == 1
    assert updated.stop_streak_pause_hours == 72
    assert updated.time_stop_bars == 500
    assert updated.accept_fallback_verdicts is True

    # Below-range clamps to the floor (0 = disabled where supported).
    updated = trader.set_config(
        {"cooldown_minutes_after_loss": -5, "time_stop_bars": -1}
    )
    assert updated.cooldown_minutes_after_loss == 0
    assert updated.time_stop_bars == 0

    # Round-trip: a brand-new trader over the same DB sees the persisted
    # values — the same account_state path every other field uses.
    cfg2 = AutoTrader(engine).get_config()
    assert cfg2.stop_streak_limit == 20
    assert cfg2.stop_streak_pause_hours == 72
    assert cfg2.accept_fallback_verdicts is True
    assert cfg2.cooldown_minutes_after_loss == 0


# --------------------------------------------------------------------------- #
# Protection 1 — per-pair loss cooldown                                         #
# --------------------------------------------------------------------------- #


def test_cooldown_blocks_recent_loss_then_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh losing close blocks re-entry; an expired one no longer does."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)
    loss_id = _insert_closed_position(TREND_SYMBOL, pnl=-25.0, closed_ago_ms=30 * MINUTE_MS)

    engine, trader = _make_trader()
    _configure(trader)  # cooldown_minutes_after_loss default: 90

    trader.run_cycle()

    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    skips = _skips_with_reason("protection_cooldown")
    assert len(skips) == 1
    assert skips[0]["symbol"] == TREND_SYMBOL
    detail = skips[0]["detail"]
    assert float(detail["pnl"]) == pytest.approx(-25.0)
    assert int(detail["cooldown_minutes"]) == 90
    assert isinstance(detail["explanation"], str) and detail["explanation"]

    # Age the loss beyond the window — the cooldown has expired.
    _set_closed_ago(loss_id, 120 * MINUTE_MS)
    trader.run_cycle()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["symbol"] == TREND_SYMBOL
    assert len(_bot_activity("enter")) == 1
    assert len(_skips_with_reason("protection_cooldown")) == 1  # no new skip


def test_cooldown_zero_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """cooldown_minutes_after_loss=0 turns the protection off entirely."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)
    _insert_closed_position(TREND_SYMBOL, pnl=-25.0, closed_ago_ms=5 * MINUTE_MS)

    engine, trader = _make_trader()
    _configure(trader, cooldown_minutes_after_loss=0)

    trader.run_cycle()

    assert len(engine.get_positions("open")) == 1
    assert _skips_with_reason("protection_cooldown") == []


def test_cooldown_ignores_scalper_owned_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recent loss on a SCALPER-owned position never cools the bot down."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)
    scalp_loss = _insert_closed_position(
        TREND_SYMBOL, pnl=-25.0, closed_ago_ms=5 * MINUTE_MS
    )
    _mark_scalper_owned(scalp_loss)

    engine, trader = _make_trader()
    _configure(trader)

    trader.run_cycle()

    assert len(engine.get_positions("open")) == 1
    assert _skips_with_reason("protection_cooldown") == []


def test_cooldown_still_active_in_research_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Research mode bypasses the AI/signal gates, never the risk brakes."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)
    _insert_closed_position(TREND_SYMBOL, pnl=-25.0, closed_ago_ms=10 * MINUTE_MS)

    engine, trader = _make_trader()
    _configure(trader, research_mode=True)

    trader.run_cycle()

    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    assert len(_skips_with_reason("protection_cooldown")) == 1


# --------------------------------------------------------------------------- #
# Protection 2 — stop-streak stand-aside                                        #
# --------------------------------------------------------------------------- #


def test_stop_streak_halts_once_persists_and_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 stop-outs in-window → ONE halt row; the pause persists, then expires."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)
    _insert_exit_rows(3, ago_ms=HOUR_MS)  # bot 'exit' rows, reason stop_loss

    engine, trader = _make_trader()
    _configure(trader)  # limit 3, window 12h, pause 6h

    summary = trader.run_cycle()
    assert summary["status"] == "halted"
    assert summary["halt_reason"] == "protection_stop_streak"
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    halts = _halts_with_reason("protection_stop_streak")
    assert len(halts) == 1
    detail = halts[0]["detail"]
    assert int(detail["stop_losses"]) == 3
    assert isinstance(detail["explanation"], str) and detail["explanation"]

    pause_until = _state_get("bot_stop_streak_pause_until")
    assert isinstance(pause_until, int) and pause_until > utc_now_ms()

    # A RESTARTED bot (fresh instance, same DB) is still standing aside —
    # and does NOT spam another halt row while the pause runs.
    trader2 = AutoTrader(engine)
    summary2 = trader2.run_cycle()
    assert summary2["status"] == "halted"
    assert summary2["halt_reason"] == "protection_stop_streak"
    assert _bot_activity("enter") == []
    assert len(_halts_with_reason("protection_stop_streak")) == 1

    # Expire the pause AND age the stop-outs beyond the window → trading on.
    _state_set("bot_stop_streak_pause_until", utc_now_ms() - 1)
    _shift_exit_rows("exit", ago_ms=13 * HOUR_MS)
    summary3 = trader2.run_cycle()
    assert summary3["status"] == "ok"
    assert len(engine.get_positions("open")) == 1
    assert len(_bot_activity("enter")) == 1
    assert len(_halts_with_reason("protection_stop_streak")) == 1


def test_stop_streak_ignores_scalper_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scalper ``scalp_exit`` stop-outs never trigger the bot's stand-aside."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)
    _insert_exit_rows(3, ago_ms=HOUR_MS, action="scalp_exit")

    engine, trader = _make_trader()
    _configure(trader)

    summary = trader.run_cycle()
    assert summary["status"] == "ok"
    assert len(engine.get_positions("open")) == 1
    assert _halts_with_reason("protection_stop_streak") == []


def test_stop_streak_zero_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """stop_streak_limit=0 turns the stand-aside off entirely."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)
    _insert_exit_rows(5, ago_ms=HOUR_MS)

    engine, trader = _make_trader()
    _configure(trader, stop_streak_limit=0)

    summary = trader.run_cycle()
    assert summary["status"] == "ok"
    assert len(engine.get_positions("open")) == 1
    assert _halts_with_reason("protection_stop_streak") == []


# --------------------------------------------------------------------------- #
# Protection 3 — time-stop                                                      #
# --------------------------------------------------------------------------- #


def test_time_stop_closes_old_position_with_exit_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overaged bot position is closed (reason time_stop); scalper-owned
    and young positions are left alone."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _seed_symbol(SCALP_SYMBOL, _choppy_df())
    _seed_symbol(FRESH_SYMBOL, _choppy_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)

    engine, trader = _make_trader()
    # min_vote=4 keeps the cycle from opening anything new; time_stop_bars=5
    # on 1h candles → positions older than 300 minutes must be closed.
    _configure(trader, min_vote=4, time_stop_bars=5)

    old_pos = engine.submit_order(TREND_SYMBOL, "buy", "market", qty=5.0)
    old_id = int(
        next(
            p["id"]
            for p in engine.get_positions("open")
            if p["symbol"] == TREND_SYMBOL
        )
    )
    assert old_pos["status"] == "filled"
    scalp_order = engine.submit_order(SCALP_SYMBOL, "buy", "market", qty=5.0)
    assert scalp_order["status"] == "filled"
    scalp_id = int(
        next(
            p["id"]
            for p in engine.get_positions("open")
            if p["symbol"] == SCALP_SYMBOL
        )
    )
    fresh_order = engine.submit_order(FRESH_SYMBOL, "buy", "market", qty=5.0)
    assert fresh_order["status"] == "filled"

    _backdate_opened_at(old_id, minutes=400)  # > 5 bars × 60 min
    _backdate_opened_at(scalp_id, minutes=400)
    _mark_scalper_owned(scalp_id)

    trader.run_cycle()

    open_symbols = {p["symbol"] for p in engine.get_positions("open")}
    assert TREND_SYMBOL not in open_symbols  # overaged bot position closed
    assert SCALP_SYMBOL in open_symbols  # scalper-owned: untouched
    assert FRESH_SYMBOL in open_symbols  # young bot position: untouched

    closed = [
        p for p in engine.get_positions("closed") if int(p["id"]) == old_id
    ]
    assert len(closed) == 1

    exits = [
        item
        for item in _bot_activity("exit")
        if item["detail"].get("reason") == "time_stop"
    ]
    assert len(exits) == 1
    assert exits[0]["symbol"] == TREND_SYMBOL
    detail = exits[0]["detail"]
    # Same _log_exit shape as every other bot exit (R diagnostics included;
    # both null here — the position carried no protective stop).
    assert {"reason", "pnl", "pnl_pct", "designed_r", "realized_r", "explanation"} <= set(
        detail.keys()
    )
    assert isinstance(detail["explanation"], str) and "time" in detail["explanation"].lower()


def test_time_stop_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With time_stop_bars=0 (default) an ancient position stays open."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)

    engine, trader = _make_trader()
    _configure(trader, min_vote=4)  # defaults: time_stop_bars=0

    engine.submit_order(TREND_SYMBOL, "buy", "market", qty=5.0)
    old_id = int(engine.get_positions("open")[0]["id"])
    _backdate_opened_at(old_id, minutes=100_000)

    trader.run_cycle()

    assert {p["symbol"] for p in engine.get_positions("open")} == {TREND_SYMBOL}
    assert [
        item
        for item in _bot_activity("exit")
        if item["detail"].get("reason") == "time_stop"
    ] == []


# --------------------------------------------------------------------------- #
# Protection 4 — fallback-verdict gate                                          #
# --------------------------------------------------------------------------- #


def _pin_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the primary/fallback model names the gate compares against."""
    monkeypatch.setattr(settings, "ollama_model", "primary-model:latest")
    monkeypatch.setattr(settings, "ollama_fallback_model", "fallback-model")


def test_fallback_verdict_skipped_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verdict answered by the fallback model is skipped (ai_fallback)."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _pin_models(monkeypatch)
    # Any tag of the fallback model counts (base-name match).
    calls = _patch_analyst_with_model(
        monkeypatch, model_used="fallback-model:latest", sentiment="bullish", confidence=90
    )

    engine, trader = _make_trader()
    _configure(trader, use_ai=True, min_ai_confidence=60)

    trader.run_cycle()

    assert calls == [TREND_SYMBOL]  # the primary gate DID run the analysis
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    skips = _skips_with_reason("ai_fallback")
    assert len(skips) == 1
    assert skips[0]["symbol"] == TREND_SYMBOL
    detail = skips[0]["detail"]
    assert detail["model_used"] == "fallback-model:latest"
    assert detail["model_requested"] == "primary-model:latest"
    # The ai payload is preserved for calibration recovery (ai_gate shape).
    assert detail["ai"]["sentiment"] == "bullish"
    assert int(detail["ai"]["confidence"]) == 90
    assert isinstance(detail["explanation"], str) and detail["explanation"]


def test_fallback_verdict_accepted_when_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """accept_fallback_verdicts=True lets a fallback-model verdict gate entries."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _pin_models(monkeypatch)
    calls = _patch_analyst_with_model(
        monkeypatch, model_used="fallback-model", sentiment="bullish", confidence=90
    )

    engine, trader = _make_trader()
    _configure(
        trader, use_ai=True, min_ai_confidence=60, accept_fallback_verdicts=True
    )

    trader.run_cycle()

    assert calls == [TREND_SYMBOL]
    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["symbol"] == TREND_SYMBOL
    assert _skips_with_reason("ai_fallback") == []
    enters = _bot_activity("enter")
    assert len(enters) == 1
    assert enters[0]["detail"]["ai"]["sentiment"] == "bullish"


def test_primary_model_other_tag_is_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A differently-tagged install of the REQUESTED model never gates."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _pin_models(monkeypatch)
    # chat() resolves the requested name to the installed tag — same base
    # name, different tag: the verdict came from the requested model.
    _patch_analyst_with_model(
        monkeypatch, model_used="primary-model:q4_k_m", sentiment="bullish", confidence=90
    )

    engine, trader = _make_trader()
    _configure(trader, use_ai=True, min_ai_confidence=60)

    trader.run_cycle()

    assert len(engine.get_positions("open")) == 1
    assert _skips_with_reason("ai_fallback") == []
    assert len(_bot_activity("enter")) == 1
