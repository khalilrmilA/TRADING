"""Tests for backend/paper_trading/auto_trader.py — the AI auto-trader (paper only).

Everything runs offline and deterministically:
    * NO network — the autouse conftest fixture blocks sockets outright, and
      every test additionally monkeypatches ``update_symbol`` so the bot's
      refresh step never even tries to fetch.
    * NO Ollama — ``analyze_market`` is monkeypatched to fixed
      ``MarketAnalysis`` stubs (patched both at its source module and on the
      ``auto_trader`` module, wherever the bot looked the name up).
    * Synthetic candles only — two seeded (``default_rng``) frames are
      pre-seeded into the per-test tmp SQLite cache: TRENDUSDT (strongly
      trending, last-bar strategy vote sum == +2, empirically verified below
      in ``test_synthetic_frames_have_expected_votes``) and CHOPUSDT
      (range-bound, last-bar |vote sum| == 1).

The autouse conftest fixtures give every test a fresh tmp database and pin
``settings.initial_capital`` to the contract default of 100,000.
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
from backend.ai.ollama_client import OllamaError
from backend.database.db import get_conn, upsert_ohlcv
from backend.indicators.features import add_features
from backend.paper_trading.auto_trader import AutoTrader, BotConfig
from backend.paper_trading.engine import PaperTradingEngine
from backend.strategies.voting import VotingStrategy
from config.settings import settings

SOURCE = "binance"
TIMEFRAME = "1h"
TREND_SYMBOL = "TRENDUSDT"
CHOP_SYMBOL = "CHOPUSDT"

#: The four member strategies whose last-bar votes the bot sums for the scan.
MEMBER_STRATEGIES = ("trend_following", "mean_reversion", "breakout", "rsi_macd")

#: Empirically verified last-bar vote sums of the two synthetic frames
#: (see test_synthetic_frames_have_expected_votes, which re-derives them).
TREND_VOTE_SUM = 2
CHOP_VOTE_SUM = 1


# --------------------------------------------------------------------------- #
# Synthetic market data (seeded — fully deterministic)                         #
# --------------------------------------------------------------------------- #


def _make_ohlcv(
    close: np.ndarray, volume: np.ndarray, start: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    """Wrap a close/volume path into a canonical hourly UTC OHLCV frame.

    By default the frame ends at the most recent CLOSED hourly candle, so the
    bot's closed-candle and data-freshness gates treat it as live data. The
    strategy votes are date-independent (no member strategy reads the
    calendar), so the pinned vote sums below do not depend on the start time.

    Args:
        close: Closing prices (all positive).
        volume: Per-bar volumes (all positive).
        start: Timestamp of the first candle; ``None`` → anchored so the last
            candle is the most recent fully closed hour.

    Returns:
        Canonical OHLCV frame (ms-resolution index so DB roundtrips are exact).
    """
    n = len(close)
    if start is None:
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
    """Strongly up-trending frame whose LAST bar draws a +2 strategy vote sum.

    Flat base for the first half (indicator warm-up), then a steady climb
    ending in a decisive +2% breakout close on a 4x volume spike. Last-bar
    votes (verified via ``VotingStrategy.vote_breakdown``): trend_following
    +1, breakout +1, rsi_macd +1, mean_reversion -1 (fading the extension)
    → sum ``TREND_VOTE_SUM`` (= 2, the default ``min_vote``).
    """
    rng = np.random.default_rng(7)
    i = np.arange(n, dtype=np.float64)
    base = 100.0 + 0.6 * np.sin(i[: n // 2] / 5.0)
    trend = base[-1] + 0.45 * (i[: n - n // 2] + 1) + 0.3 * np.sin(i[: n - n // 2] / 4.0)
    close = np.concatenate([base, trend]) + rng.normal(0.0, 0.05, n)
    close[-1] = close[-2] * 1.02  # decisive breakout close on the last bar
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    volume[-1] = 800.0  # volume spike confirms the breakout
    return _make_ohlcv(close, volume)


def _choppy_df(n: int = 240) -> pd.DataFrame:
    """Range-bound frame whose LAST bar draws only a +1 vote sum (< min_vote).

    A slow sine oscillation around 100; the phase is chosen so the last bar's
    votes cancel to ``CHOP_VOTE_SUM`` (= 1), below the default ``min_vote`` of
    2 (verified via ``VotingStrategy.vote_breakdown``).
    """
    rng = np.random.default_rng(11)
    i = np.arange(n, dtype=np.float64)
    close = 100.0 + 0.8 * np.sin(i / 7.0 + 4.9) + rng.normal(0.0, 0.05, n)
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    return _make_ohlcv(close, volume)


def _last_bar_votes(df: pd.DataFrame) -> dict[str, int]:
    """Per-strategy votes on the last bar of a raw OHLCV frame."""
    breakdown = VotingStrategy().vote_breakdown(add_features(df))
    last = breakdown.iloc[-1]
    return {name: int(last[name]) for name in MEMBER_STRATEGIES}


def _seed_symbol(symbol: str, df: pd.DataFrame) -> None:
    """Pre-seed the tmp-DB candle cache for one symbol."""
    upsert_ohlcv(SOURCE, symbol, TIMEFRAME, df)


# --------------------------------------------------------------------------- #
# Bot helpers                                                                  #
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
        # This module tests the pre-judge pipeline (the second judge has its
        # own acceptance suite in test_second_judge.py). Sockets are blocked
        # here, so list_models() is always "unknown" ([]) — with the judge
        # left on, the fail-closed availability rule would route survivors
        # through the judge stub and break the pinned primary-gate call
        # counts.
        "use_second_judge": False,
        "min_vote": 2,
        "allow_short": False,
        # This module tests the pre-v3 entry pipeline; the Improvement Pack
        # v3 cost gate is disabled via its contract knob (0 disables) — the
        # low-ATR fixtures here would otherwise be cost-gated before the
        # behaviors under test run. The v3 gates have their own binding
        # suite in tests/test_improvements_traders.py.
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
    """Patch a callable at its source module AND where auto_trader looked it up.

    Covers both import styles: ``from x import name`` (rebinds the name into
    the ``auto_trader`` module namespace) and ``import x; x.name(...)``
    (resolved on the source module at call time).
    """
    monkeypatch.setattr(source_module, name, replacement, raising=True)
    if hasattr(auto_trader_mod, name):
        monkeypatch.setattr(auto_trader_mod, name, replacement)


def _patch_no_data_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every data refresh a no-op so scans run purely off the seeded cache."""

    def _noop_update(*args: Any, **kwargs: Any) -> int:
        return 0

    _patch_lookup(monkeypatch, "update_symbol", _noop_update, data_service_mod)


class _AnalystStub:
    """Recording ``analyze_market`` replacement returning a fixed verdict."""

    def __init__(
        self,
        sentiment: str = "bullish",
        confidence: int = 90,
        error: Exception | None = None,
    ) -> None:
        self.sentiment = sentiment
        self.confidence = confidence
        self.error = error
        self.calls: list[str] = []

    def __call__(
        self,
        symbol: str,
        timeframe: str = TIMEFRAME,
        df: pd.DataFrame | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MarketAnalysis:
        self.calls.append(str(symbol))
        if self.error is not None:
            raise self.error
        return MarketAnalysis(
            sentiment=self.sentiment,  # type: ignore[arg-type]
            confidence=self.confidence,
            risk_commentary="synthetic stub commentary",
            key_indicators=[],
            reasoning="synthetic stub reasoning",
            model_used="stub-model",
            symbol=str(symbol),
            timeframe=str(timeframe),
        )


def _patch_analyst(
    monkeypatch: pytest.MonkeyPatch,
    sentiment: str = "bullish",
    confidence: int = 90,
    error: Exception | None = None,
) -> _AnalystStub:
    """Replace ``analyze_market`` (wherever the bot looks it up) with a stub."""
    stub = _AnalystStub(sentiment=sentiment, confidence=confidence, error=error)
    _patch_lookup(monkeypatch, "analyze_market", stub, analyst_mod)
    return stub


def _forbid_analyst(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``analyze_market`` with a stub that fails the test when called."""

    def _boom(*args: Any, **kwargs: Any) -> MarketAnalysis:
        raise AssertionError("analyze_market must not be called when use_ai=False")

    _patch_lookup(monkeypatch, "analyze_market", _boom, analyst_mod)


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


def _trip_circuit_breaker() -> None:
    """Persist an active circuit breaker so the account reads as halted."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
            ("circuit_breaker_active", json.dumps(True)),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _pin_contract_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every settings knob the sizing/risk assertions depend on.

    A local ``.env`` may override any of these; the tests do exact math with
    the contract defaults, so pin them per test (monkeypatch restores).
    """
    monkeypatch.setattr(settings, "commission_rate", 0.001)
    monkeypatch.setattr(settings, "slippage_rate", 0.0005)
    monkeypatch.setattr(settings, "risk_per_trade", 0.01)
    monkeypatch.setattr(settings, "max_open_positions", 5)
    monkeypatch.setattr(settings, "daily_loss_limit", 0.03)
    monkeypatch.setattr(settings, "circuit_breaker_drawdown", 0.10)
    monkeypatch.setattr(settings, "atr_stop_multiplier", 2.0)
    monkeypatch.setattr(settings, "atr_take_profit_multiplier", 3.0)


# --------------------------------------------------------------------------- #
# Fixture self-check — the synthetic frames must vote the way the tests assume #
# --------------------------------------------------------------------------- #


def test_synthetic_frames_have_expected_votes() -> None:
    """Pin the last-bar vote sums the whole module's assertions rely on."""
    trend_votes = _last_bar_votes(_trending_df())
    chop_votes = _last_bar_votes(_choppy_df())

    assert sum(trend_votes.values()) == TREND_VOTE_SUM, (
        f"trending frame must vote sum {TREND_VOTE_SUM} on its last bar, "
        f"got {trend_votes}"
    )
    # A strong up-move: no member may vote short except the mean-reverter.
    assert trend_votes["trend_following"] == 1
    assert trend_votes["breakout"] == 1
    assert trend_votes["rsi_macd"] == 1

    assert abs(sum(chop_votes.values())) == CHOP_VOTE_SUM, (
        f"choppy frame must vote |sum| {CHOP_VOTE_SUM} on its last bar, "
        f"got {chop_votes}"
    )


# --------------------------------------------------------------------------- #
# Votes-only entries (use_ai=False)                                            #
# --------------------------------------------------------------------------- #


def test_votes_only_entry_when_ai_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With use_ai=False the bot enters purely on votes and never calls the AI."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)

    engine, trader = _make_trader()
    _configure(trader, use_ai=False)

    result = trader.run_cycle()
    assert isinstance(result, dict)

    positions = engine.get_positions("open")
    assert len(positions) == 1
    position = positions[0]
    assert position["symbol"] == TREND_SYMBOL
    assert position["side"] == "long"
    # ATR-based protective levels must bracket the entry (long side).
    assert position["stop_loss"] is not None
    assert position["take_profit"] is not None
    assert (
        float(position["stop_loss"])
        < float(position["entry_price"])
        < float(position["take_profit"])
    )

    enters = _bot_activity("enter")
    assert len(enters) == 1
    assert enters[0]["symbol"] == TREND_SYMBOL
    detail = enters[0]["detail"]
    assert {"side", "qty", "price", "notional", "vote_sum", "explanation"} <= set(
        detail.keys()
    )
    assert detail["ai"] is None  # no AI consulted → transparency field is null
    assert isinstance(detail["explanation"], str) and detail["explanation"]
    assert int(detail["vote_sum"]) == TREND_VOTE_SUM


def test_shortlist_respects_min_vote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only symbols with |vote_sum| >= min_vote are entered; weaker ones are not."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _seed_symbol(CHOP_SYMBOL, _choppy_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)

    engine, trader = _make_trader()
    _configure(trader, watchlist=[TREND_SYMBOL, CHOP_SYMBOL], min_vote=2)

    trader.run_cycle()

    positions = engine.get_positions("open")
    assert [p["symbol"] for p in positions] == [TREND_SYMBOL]

    enters = _bot_activity("enter")
    assert {item["symbol"] for item in enters} == {TREND_SYMBOL}

    # Both watchlist symbols were scanned and their vote sums logged.
    scans = {item["symbol"]: item["detail"] for item in _bot_activity("scan")}
    assert {TREND_SYMBOL, CHOP_SYMBOL} <= set(scans.keys())
    assert int(scans[TREND_SYMBOL]["vote_sum"]) == TREND_VOTE_SUM
    assert int(scans[CHOP_SYMBOL]["vote_sum"]) == CHOP_VOTE_SUM


def test_min_vote_above_available_votes_blocks_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raising min_vote above the strongest vote sum shortlists nothing."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)

    engine, trader = _make_trader()
    _configure(trader, min_vote=TREND_VOTE_SUM + 1)

    trader.run_cycle()

    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []


# --------------------------------------------------------------------------- #
# Position sizing                                                              #
# --------------------------------------------------------------------------- #


def test_position_notional_respects_max_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opened notional never exceeds max_position_fraction × equity.

    With a tight ATR the pure risk-based size would be ~85% of equity in
    notional, so the 5% fraction cap set here is the binding constraint —
    this genuinely exercises the cap rather than passing vacuously.
    """
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)

    engine, trader = _make_trader()
    fraction = 0.05
    _configure(trader, max_position_fraction=fraction)

    equity_before = engine.get_portfolio()["equity"]
    assert equity_before == pytest.approx(settings.initial_capital)

    # Sanity: risk-based sizing alone would blow far past the fraction cap.
    featured = add_features(_trending_df())
    atr_value = float(featured["atr_14"].iloc[-1])
    price = float(featured["close"].iloc[-1])
    risk_qty = engine.risk.position_size(equity_before, atr_value, price)
    assert risk_qty * price > fraction * equity_before

    trader.run_cycle()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    position = positions[0]
    notional = float(position["qty"]) * float(position["entry_price"])
    assert notional > 1.0  # not dust
    # Small tolerance: the fill price includes adverse slippage on top of the
    # close the quantity was sized against.
    assert notional <= fraction * equity_before * 1.02

    detail = _bot_activity("enter")[0]["detail"]
    assert float(detail["notional"]) <= fraction * equity_before * 1.02


# --------------------------------------------------------------------------- #
# AI confirmation gate (analyze_market monkeypatched — no Ollama)              #
# --------------------------------------------------------------------------- #


def test_ai_gate_blocks_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bullish verdict below min_ai_confidence must not open a position."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    stub = _patch_analyst(monkeypatch, sentiment="bullish", confidence=30)

    engine, trader = _make_trader()
    _configure(trader, use_ai=True, min_ai_confidence=60)

    trader.run_cycle()

    assert stub.calls == [TREND_SYMBOL]  # the AI was consulted exactly once
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []

    skips = [item for item in _bot_activity("skip") if item["symbol"] == TREND_SYMBOL]
    assert skips, "low-confidence candidate must be logged as 'skip'"
    assert "reason" in skips[-1]["detail"]


def test_ai_gate_blocks_wrong_direction_sentiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high-confidence BEARISH verdict must block a LONG candidate."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    stub = _patch_analyst(monkeypatch, sentiment="bearish", confidence=95)

    engine, trader = _make_trader()
    _configure(trader, use_ai=True, min_ai_confidence=60)

    trader.run_cycle()

    assert stub.calls == [TREND_SYMBOL]
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    skips = [item for item in _bot_activity("skip") if item["symbol"] == TREND_SYMBOL]
    assert skips, "wrong-direction candidate must be logged as 'skip'"


def test_ai_gate_allows_confident_matching_sentiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bullish at/above min_ai_confidence opens the position with full transparency."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    stub = _patch_analyst(monkeypatch, sentiment="bullish", confidence=85)

    engine, trader = _make_trader()
    _configure(trader, use_ai=True, min_ai_confidence=60)

    trader.run_cycle()

    assert stub.calls == [TREND_SYMBOL]
    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["symbol"] == TREND_SYMBOL
    assert positions[0]["side"] == "long"

    enters = _bot_activity("enter")
    assert len(enters) == 1
    detail = enters[0]["detail"]
    # Full transparency payload mandated by CONTRACTS.md.
    assert {
        "side",
        "qty",
        "price",
        "notional",
        "stop_loss",
        "take_profit",
        "vote_sum",
        "votes",
        "indicators",
        "ai",
        "explanation",
    } <= set(detail.keys())
    assert set(detail["votes"].keys()) == set(MEMBER_STRATEGIES)
    assert detail["ai"]["sentiment"] == "bullish"
    assert int(detail["ai"]["confidence"]) == 85
    assert isinstance(detail["explanation"], str) and detail["explanation"]


def test_ai_error_skips_candidate_conservatively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OllamaError from the analyst logs 'error' and skips the entry (no raise)."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    stub = _patch_analyst(
        monkeypatch, error=OllamaError("model unreachable (synthetic)")
    )

    engine, trader = _make_trader()
    _configure(trader, use_ai=True)

    result = trader.run_cycle()  # must not raise
    assert isinstance(result, dict)

    assert stub.calls == [TREND_SYMBOL]
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    assert _bot_activity("error"), "the analyst failure must be logged as 'error'"


# --------------------------------------------------------------------------- #
# Halt path                                                                    #
# --------------------------------------------------------------------------- #


def test_halt_logs_and_places_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the circuit breaker pre-tripped the cycle logs 'halt' and trades nothing."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)

    engine, trader = _make_trader()
    _configure(trader, use_ai=False)
    _trip_circuit_breaker()
    assert engine.get_portfolio()["trading_halted"] is True

    result = trader.run_cycle()  # must not raise
    assert isinstance(result, dict)

    assert _bot_activity("halt"), "a halted cycle must log a 'halt' action"
    assert _bot_activity("enter") == []
    assert engine.get_orders() == []  # nothing placed, not even rejected orders
    assert engine.get_positions("open") == []


# --------------------------------------------------------------------------- #
# Robustness                                                                   #
# --------------------------------------------------------------------------- #


def test_run_cycle_never_raises_when_update_symbol_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing data refresh is contained per symbol — run_cycle never raises."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _forbid_analyst(monkeypatch)

    def _explode(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("synthetic data-service outage")

    _patch_lookup(monkeypatch, "update_symbol", _explode, data_service_mod)

    _, trader = _make_trader()
    _configure(trader, use_ai=False)

    result = trader.run_cycle()  # the whole point: no exception escapes
    assert isinstance(result, dict)
    actions = {item["action"] for item in _bot_activity()}
    assert "cycle_end" in actions, "the cycle must still run to completion"


# --------------------------------------------------------------------------- #
# Config persistence                                                           #
# --------------------------------------------------------------------------- #


def test_config_roundtrip_persistence() -> None:
    """set_config persists to account_state and survives a fresh AutoTrader."""
    engine, trader = _make_trader()

    # Contract defaults.
    cfg = trader.get_config()
    assert isinstance(cfg, BotConfig)
    assert cfg.min_vote == 2
    assert cfg.use_ai is True
    assert cfg.min_ai_confidence == 60
    assert cfg.allow_short is False
    assert cfg.max_position_fraction == pytest.approx(0.35)
    assert cfg.running is False
    assert len(cfg.watchlist) == 10 and "BTCUSDT" in cfg.watchlist

    updated = trader.set_config(
        {
            "min_vote": 4,
            "use_ai": False,
            "watchlist": ["BTCUSDT"],
            "max_position_fraction": 0.2,
        }
    )
    assert updated.min_vote == 4
    assert updated.use_ai is False
    assert updated.watchlist == ["BTCUSDT"]
    assert updated.max_position_fraction == pytest.approx(0.2)

    # Persisted under the contract-mandated account_state key.
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM account_state WHERE key='bot_config'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    persisted = json.loads(row["value"])
    assert persisted["min_vote"] == 4
    assert persisted["use_ai"] is False

    # A brand-new trader instance sees the persisted config, not the defaults.
    trader2 = AutoTrader(engine)
    cfg2 = trader2.get_config()
    assert cfg2.min_vote == 4
    assert cfg2.use_ai is False
    assert cfg2.watchlist == ["BTCUSDT"]
    assert cfg2.max_position_fraction == pytest.approx(0.2)


def test_start_stop_persist_running_flag() -> None:
    """start()/stop() flip and persist the running flag; status() reports it."""
    engine, trader = _make_trader()

    started = trader.start()
    assert started.running is True
    status = trader.status()
    assert {"running", "last_cycle_ts", "watchlist", "equity", "open_positions"} <= set(
        status.keys()
    )
    assert status["running"] is True

    # The flag survives a restart (fresh instance over the same DB).
    assert AutoTrader(engine).get_config().running is True

    stopped = trader.stop()
    assert stopped.running is False
    assert trader.status()["running"] is False
    assert AutoTrader(engine).get_config().running is False
