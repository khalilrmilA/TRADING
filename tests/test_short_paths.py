"""SHORT-trade path coverage — engine lifecycle + AutoTrader short gates.

An audit found the short side of the platform effectively untested. This
module closes those gaps (paper only, like everything else):

* Engine short lifecycle — open a short via a ``sell`` market order (stop
  ABOVE entry, target BELOW), mark-to-market/equity move the right way when
  the price falls, buy-to-cover realizes ``(entry - exit) * qty`` minus both
  legs' commissions, a short STOP triggers on a candle HIGH touch and a
  short TP triggers on a candle LOW touch.
* AutoTrader AI gate, short direction — a bearish high-confidence verdict
  lets a short candidate through; a bullish verdict skips it (``ai_gate``).
* News veto, short side — fresh sentiment at/above +50 blocks a short with
  skip reason ``news_positive``.
* Short sizing with deployed cash (regression) — the free-cash cap applies
  to LONGS only; a short must still size by the equity/risk caps even when
  most cash is tied up in an open long.
* Short trailing stop — the stop is only ever LOWERED (tighten-only) as the
  price falls, never raised on a bounce, and the ``trail`` row says
  "lowered".

Everything runs offline and deterministically: NO network (conftest blocks
sockets), NO Ollama (``analyze_market`` is stubbed or forbidden), synthetic
seeded candles only. The autouse conftest fixtures give every test a fresh
tmp database and pin ``settings.initial_capital`` to 100,000.
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
from backend.indicators.features import add_features
from backend.paper_trading.auto_trader import AutoTrader, BotConfig
from backend.paper_trading.engine import PaperTradingEngine
from backend.strategies.voting import VotingStrategy
from config.settings import settings

SOURCE = "binance"
TIMEFRAME = "1h"

#: Symbol used by the pure-engine short-lifecycle tests (flat candles).
SHORT_SYMBOL = "SHORTUSDT"
#: Down-trending bot candidate (last-bar vote sum pinned to -2 below).
BEAR_SYMBOL = "BEARUSDT"
#: Symbol whose long position soaks up most of the free cash.
CASH_SYMBOL = "CASHUSDT"

#: The four member strategies whose last-bar votes the bot sums.
MEMBER_STRATEGIES = ("trend_following", "mean_reversion", "breakout", "rsi_macd")

#: Empirically pinned last-bar vote sum of the bearish frame (re-derived in
#: the self-check test below).
BEAR_VOTE_SUM = -2


# --------------------------------------------------------------------------- #
# Synthetic market data (seeded / arithmetic — fully deterministic)            #
# --------------------------------------------------------------------------- #


def _seed_flat_candles(
    symbol: str, n: int = 30, close: float = 100.0, start: str = "2024-01-01"
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
    upsert_ohlcv(SOURCE, symbol, TIMEFRAME, df)
    return df


def _append_candle(
    symbol: str, ts: str, open_: float, high: float, low: float, close: float
) -> None:
    """Append one crafted candle (e.g. one whose high pierces a short's stop)."""
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
    upsert_ohlcv(SOURCE, symbol, TIMEFRAME, df)


def _make_ohlcv(close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    """Wrap a close/volume path into a canonical hourly UTC OHLCV frame.

    Anchored so the last candle is the most recent fully CLOSED hour — the
    bot's closed-candle and data-freshness gates treat it as live data
    (same construction as tests/test_auto_trader.py).
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
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")
    assert (df["low"] > 0).all()
    return df


def _bearish_df(n: int = 240) -> pd.DataFrame:
    """Down-trending 1h frame whose LAST bar draws a -2 strategy vote sum.

    Exact mirror of the trending fixture in tests/test_improvements_traders:
    flat warm-up base, steady decline, -2% breakdown close on a 4x volume
    spike (vote sum pinned in ``test_bearish_frame_votes_short``).
    """
    rng = np.random.default_rng(7)
    i = np.arange(n, dtype=np.float64)
    base = 200.0 + 0.6 * np.sin(i[: n // 2] / 5.0)
    trend = base[-1] - 0.45 * (i[: n - n // 2] + 1) - 0.3 * np.sin(i[: n - n // 2] / 4.0)
    close = np.concatenate([base, trend]) + rng.normal(0.0, 0.05, n)
    close[-1] = close[-2] * 0.98  # decisive breakdown close on the last bar
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    volume[-1] = 800.0
    return _make_ohlcv(close, volume)


def _seed_symbol(symbol: str, df: pd.DataFrame) -> None:
    """Pre-seed the tmp-DB candle cache for one symbol."""
    upsert_ohlcv(SOURCE, symbol, TIMEFRAME, df)


def _closes_df(closes: list[float]) -> pd.DataFrame:
    """Minimal canonical frame from a close path (for ``_update_trailing``)."""
    n = len(closes)
    arr = np.asarray(closes, dtype=np.float64)
    index = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    return pd.DataFrame(
        {
            "open": arr,
            "high": arr,
            "low": arr,
            "close": arr,
            "volume": np.full(n, 250.0),
        },
        index=index,
    )


# --------------------------------------------------------------------------- #
# Bot helpers (same patterns as tests/test_auto_trader.py)                     #
# --------------------------------------------------------------------------- #


def _make_trader() -> tuple[PaperTradingEngine, AutoTrader]:
    """Build a fresh engine + trader pair against the per-test tmp database."""
    engine = PaperTradingEngine()
    return engine, AutoTrader(engine)


def _configure(trader: AutoTrader, **overrides: Any) -> BotConfig:
    """Apply the offline short-path test baseline config plus overrides.

    The baseline allows shorts and disables the gates that have their own
    suites elsewhere: the second judge (sockets are blocked here), the v3
    regime gate (tests/test_improvements_traders.py) and the v3 cost gate
    (0 disables it — the low-ATR bearish fixture would otherwise be
    cost-gated before the short behaviors under test run).
    """
    updates: dict[str, Any] = {
        "watchlist": [BEAR_SYMBOL],
        "source": SOURCE,
        "timeframe": TIMEFRAME,
        "use_ai": False,
        "use_second_judge": False,
        "min_vote": 2,
        "allow_short": True,
        "regime_gate_enabled": False,
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


class _AnalystStub:
    """Recording ``analyze_market`` replacement returning a fixed verdict."""

    def __init__(self, sentiment: str = "bearish", confidence: int = 90) -> None:
        self.sentiment = sentiment
        self.confidence = confidence
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
    monkeypatch: pytest.MonkeyPatch, sentiment: str, confidence: int
) -> _AnalystStub:
    """Replace ``analyze_market`` (wherever the bot looks it up) with a stub."""
    stub = _AnalystStub(sentiment=sentiment, confidence=confidence)
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
                "detail": detail if isinstance(detail, dict) else {},
            }
        )
    if action is not None:
        items = [item for item in items if item["action"] == action]
    return items


def _insert_sentiment(symbol: str, score: int, age_ms: int = 0) -> None:
    """Seed one FRESH ``coin_sentiment`` row (same shape as the intel layer)."""
    sentiment = "bearish" if score < 0 else "bullish" if score > 0 else "neutral"
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO coin_sentiment "
            "(ts, symbol, sentiment, score, confidence, summary, headlines, model) "
            "VALUES (?, ?, ?, ?, ?, ?, '[]', 'test')",
            (
                utc_now_ms() - int(age_ms),
                symbol,
                sentiment,
                int(score),
                80,
                f"seeded test sentiment for {symbol}",
            ),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _pin_contract_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every settings knob the fill/sizing assertions depend on.

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
# Fixture self-check — the bearish frame must vote the way the tests assume    #
# --------------------------------------------------------------------------- #


def test_bearish_frame_votes_short() -> None:
    """Pin the last-bar vote sum the bot-side short tests rely on."""
    breakdown = VotingStrategy().vote_breakdown(add_features(_bearish_df()))
    last = breakdown.iloc[-1]
    votes = {name: int(last[name]) for name in MEMBER_STRATEGIES}
    assert sum(votes.values()) == BEAR_VOTE_SUM, (
        f"bearish frame must vote sum {BEAR_VOTE_SUM} on its last bar, got {votes}"
    )
    # A strong down-move: no member may vote long except the mean-reverter.
    assert votes["trend_following"] == -1
    assert votes["breakout"] == -1
    assert votes["rsi_macd"] == -1


# --------------------------------------------------------------------------- #
# Engine short lifecycle                                                       #
# --------------------------------------------------------------------------- #


def test_short_open_marks_and_covers() -> None:
    """Open a short, mark it as price falls, then buy-to-cover at a profit."""
    _seed_flat_candles(SHORT_SYMBOL, n=30, close=100.0)
    engine = PaperTradingEngine()
    slip = settings.slippage_rate

    # A sell with no open long OPENS a short; stop ABOVE entry, target BELOW.
    order = engine.submit_order(
        SHORT_SYMBOL, "sell", "market", qty=2.0, stop_loss=110.0, take_profit=80.0
    )
    entry_fill = 100.0 * (1.0 - slip)  # sells pay adverse slippage DOWN
    assert order["status"] == "filled"
    assert order["fill_price"] == pytest.approx(entry_fill)

    positions = engine.get_positions("open")
    assert len(positions) == 1
    position = positions[0]
    assert position["side"] == "short"
    assert position["qty"] == pytest.approx(2.0)
    assert position["entry_price"] == pytest.approx(entry_fill)
    assert (
        float(position["take_profit"])
        < float(position["entry_price"])
        < float(position["stop_loss"])
    )
    portfolio_before = engine.get_portfolio()

    # Price falls to 95 without touching the 110 stop or the 80 target:
    # a short's unrealized PnL and the account equity must both RISE.
    _append_candle(SHORT_SYMBOL, "2024-01-02 06:00", 100.0, 100.5, 94.0, 95.0)
    result = engine.process_tick(SHORT_SYMBOL, source=SOURCE, timeframe=TIMEFRAME)
    assert result["closed"] == []

    marked = engine.get_positions("open")[0]
    expected_unrealized = (entry_fill - 95.0) * 2.0
    assert expected_unrealized > 0
    assert marked["unrealized_pnl"] == pytest.approx(expected_unrealized)
    portfolio = engine.get_portfolio()
    assert portfolio["unrealized_pnl"] == pytest.approx(expected_unrealized)
    assert portfolio["equity"] > portfolio_before["equity"]

    # Buy-to-cover the full quantity: nets FIFO against the short, opens nothing.
    exit_fill = 95.0 * (1.0 + slip)  # buys pay adverse slippage UP
    cover = engine.submit_order(SHORT_SYMBOL, "buy", "market", qty=2.0)
    assert cover["status"] == "filled"
    assert cover["fill_price"] == pytest.approx(exit_fill)
    assert cover["opened_position_id"] is None  # pure cover — no new long
    assert engine.get_positions("open") == []

    closed = engine.get_positions("closed")
    assert len(closed) == 1
    entry_commission = 2.0 * entry_fill * settings.commission_rate
    exit_commission = 2.0 * exit_fill * settings.commission_rate
    expected_pnl = (entry_fill - exit_fill) * 2.0 - entry_commission - exit_commission
    assert expected_pnl > 0
    assert closed[0]["pnl"] == pytest.approx(expected_pnl)
    assert closed[0]["exit_price"] == pytest.approx(exit_fill)

    final = engine.get_portfolio()
    assert final["realized_pnl_today"] == pytest.approx(expected_pnl)
    assert final["cash"] == pytest.approx(settings.initial_capital + expected_pnl)
    assert final["equity"] == pytest.approx(settings.initial_capital + expected_pnl)


def test_short_stop_loss_triggers_on_candle_high() -> None:
    """A short's stop (above entry) fires when a candle HIGH touches it."""
    _seed_flat_candles(SHORT_SYMBOL, n=30, close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(
        SHORT_SYMBOL, "sell", "market", qty=1.0, stop_loss=105.0
    )
    assert order["status"] == "filled"
    assert len(engine.get_positions("open")) == 1

    # Crafted candle whose HIGH (106) pierces the 105 stop.
    _append_candle(SHORT_SYMBOL, "2024-01-02 06:00", 100.0, 106.0, 99.5, 104.0)
    result = engine.process_tick(SHORT_SYMBOL, source=SOURCE, timeframe=TIMEFRAME)

    assert len(result["closed"]) == 1
    assert result["closed"][0]["exit_reason"] == "stop_loss"
    assert engine.get_positions("open") == []

    closed = engine.get_positions("closed")
    assert len(closed) == 1
    position = closed[0]
    # A short's stop is a buy-to-cover: adverse slippage pushes the fill UP.
    expected_exit = 105.0 * (1.0 + settings.slippage_rate)
    assert position["exit_price"] == pytest.approx(expected_exit)
    assert position["pnl"] < 0


def test_short_take_profit_triggers_on_candle_low() -> None:
    """A short's target (below entry) fires when a candle LOW touches it."""
    _seed_flat_candles(SHORT_SYMBOL, n=30, close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(
        SHORT_SYMBOL, "sell", "market", qty=1.0, stop_loss=110.0, take_profit=90.0
    )
    assert order["status"] == "filled"

    # LOW (88) pierces the 90 target; HIGH (100.5) stays under the 110 stop.
    _append_candle(SHORT_SYMBOL, "2024-01-02 06:00", 100.0, 100.5, 88.0, 92.0)
    result = engine.process_tick(SHORT_SYMBOL, source=SOURCE, timeframe=TIMEFRAME)

    assert len(result["closed"]) == 1
    assert result["closed"][0]["exit_reason"] == "take_profit"
    assert engine.get_positions("open") == []

    closed = engine.get_positions("closed")
    assert len(closed) == 1
    position = closed[0]
    # Take-profits fill AT the level (slippage applies to market/stop fills only).
    assert position["exit_price"] == pytest.approx(90.0)
    assert position["pnl"] > 0


# --------------------------------------------------------------------------- #
# AutoTrader AI gate — short direction                                         #
# --------------------------------------------------------------------------- #


def test_ai_gate_allows_bearish_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short candidate + high-confidence BEARISH verdict enters side 'short'."""
    _seed_symbol(BEAR_SYMBOL, _bearish_df())
    _patch_no_data_updates(monkeypatch)
    stub = _patch_analyst(monkeypatch, sentiment="bearish", confidence=90)

    engine, trader = _make_trader()
    _configure(trader, use_ai=True, min_ai_confidence=60)

    trader.run_cycle()

    assert stub.calls == [BEAR_SYMBOL]  # the AI was consulted exactly once
    positions = engine.get_positions("open")
    assert len(positions) == 1
    position = positions[0]
    assert position["symbol"] == BEAR_SYMBOL
    assert position["side"] == "short"
    # Short protective geometry: target BELOW entry BELOW stop.
    assert (
        float(position["take_profit"])
        < float(position["entry_price"])
        < float(position["stop_loss"])
    )

    enters = _bot_activity("enter")
    assert len(enters) == 1
    detail = enters[0]["detail"]
    assert detail["side"] == "short"
    assert detail["ai"]["sentiment"] == "bearish"
    assert int(detail["ai"]["confidence"]) == 90
    assert int(detail["vote_sum"]) == BEAR_VOTE_SUM


def test_ai_gate_blocks_short_on_bullish_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SAME short candidate with a BULLISH verdict is skipped ('ai_gate...')."""
    _seed_symbol(BEAR_SYMBOL, _bearish_df())
    _patch_no_data_updates(monkeypatch)
    stub = _patch_analyst(monkeypatch, sentiment="bullish", confidence=90)

    engine, trader = _make_trader()
    _configure(trader, use_ai=True, min_ai_confidence=60)

    trader.run_cycle()

    assert stub.calls == [BEAR_SYMBOL]
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []

    skips = [item for item in _bot_activity("skip") if item["symbol"] == BEAR_SYMBOL]
    assert skips, "the wrong-direction short candidate must be logged as 'skip'"
    reason = str(skips[-1]["detail"]["reason"])
    assert reason.startswith("ai_gate")


# --------------------------------------------------------------------------- #
# News veto — short side                                                       #
# --------------------------------------------------------------------------- #


def test_news_positive_blocks_short_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh sentiment >= +50 vetoes a short with skip reason 'news_positive'."""
    _seed_symbol(BEAR_SYMBOL, _bearish_df())
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)
    _insert_sentiment(BEAR_SYMBOL, score=75)  # fresh (< 3h) strongly positive

    engine, trader = _make_trader()
    _configure(trader, use_ai=False)

    trader.run_cycle()

    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []

    skips = [
        item
        for item in _bot_activity("skip")
        if item["detail"].get("reason") == "news_positive"
    ]
    assert len(skips) == 1
    assert skips[0]["symbol"] == BEAR_SYMBOL
    detail = skips[0]["detail"]
    assert int(detail["score"]) == 75
    assert isinstance(detail["explanation"], str) and detail["explanation"]


# --------------------------------------------------------------------------- #
# Short sizing with deployed cash (free-cash cap is long-only — regression)    #
# --------------------------------------------------------------------------- #


def test_short_sizing_not_capped_by_deployed_cash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With cash tied up in a long, a short still sizes by the equity caps.

    Regression for the free-cash-cap asymmetry: opening a short CREDITS the
    proceeds, so the 95%-of-free-cash cap applies to longs only. Before the
    fix, a short entered while cash was deployed in longs was shrunk to the
    tiny free-cash remainder instead of the max_position_fraction cap.
    """
    df = _bearish_df()
    _seed_symbol(BEAR_SYMBOL, df)
    _seed_flat_candles(CASH_SYMBOL, n=30, close=100.0)
    _patch_no_data_updates(monkeypatch)
    _forbid_analyst(monkeypatch)

    engine, trader = _make_trader()

    # Tie up ~85% of the cash in an open long elsewhere.
    engine.submit_order(CASH_SYMBOL, "buy", "market", qty=850.0)
    portfolio = engine.get_portfolio()
    cash_before = float(portfolio["cash"])
    equity_before = float(portfolio["equity"])
    assert cash_before < 0.2 * equity_before  # most cash genuinely deployed

    fraction = 0.25
    _configure(trader, use_ai=False, max_position_fraction=fraction)

    # Sanity: the pure risk-based size exceeds the fraction cap, so the
    # fraction cap is the binding constraint for a correctly sized short...
    featured = add_features(df)
    atr_value = float(featured["atr_14"].iloc[-1])
    price = float(featured["close"].iloc[-1])
    risk_qty = engine.risk.position_size(equity_before, atr_value, price)
    assert risk_qty * price > fraction * equity_before
    # ...while the old (buggy) free-cash cap would have crushed the size.
    assert 0.95 * cash_before < 0.7 * fraction * equity_before

    trader.run_cycle()

    shorts = [p for p in engine.get_positions("open") if p["symbol"] == BEAR_SYMBOL]
    assert len(shorts) == 1
    position = shorts[0]
    assert position["side"] == "short"
    notional = float(position["qty"]) * float(position["entry_price"])
    # Sized by equity/risk caps — NOT reduced to the free-cash remainder.
    assert notional == pytest.approx(fraction * equity_before, rel=0.02)
    assert notional > 0.95 * cash_before

    enters = _bot_activity("enter")
    assert len(enters) == 1
    detail = enters[0]["detail"]
    assert detail["side"] == "short"
    assert float(detail["notional"]) > 0.95 * cash_before


# --------------------------------------------------------------------------- #
# Trailing stop — short side (tighten-only: the stop is only ever LOWERED)     #
# --------------------------------------------------------------------------- #


def test_short_trailing_stop_lowers_and_never_raises() -> None:
    """As price falls past trail_activate_pct the stop is LOWERED, never raised."""
    _seed_flat_candles(SHORT_SYMBOL, n=30, close=100.0)
    engine = PaperTradingEngine()
    trader = AutoTrader(engine)

    order = engine.submit_order(
        SHORT_SYMBOL, "sell", "market", qty=1.0, stop_loss=105.0, take_profit=80.0
    )
    assert order["status"] == "filled"
    position = engine.get_positions("open")[0]
    assert position["side"] == "short"
    entry = float(position["entry_price"])  # ~99.95

    config = BotConfig(
        trailing_enabled=True, trail_activate_pct=0.02, trail_distance_pct=0.015
    )

    # 1) Price falls ~5% below entry → trailing activates and LOWERS the stop
    #    to last_close * (1 + trail_distance_pct); the take-profit is cleared.
    df_down = _closes_df([100.0] * 10 + [98.0, 96.0, 95.0])
    assert (entry - 95.0) / entry > config.trail_activate_pct  # genuinely active
    trader._update_trailing(config, position, df_down)

    trailed = engine.get_positions("open")[0]
    first_stop = float(trailed["stop_loss"])
    assert first_stop == pytest.approx(95.0 * (1.0 + config.trail_distance_pct))
    assert first_stop < 105.0  # lowered from the original stop
    assert trailed["take_profit"] is None  # removed so the winner can run

    trails = _bot_activity("trail")
    assert len(trails) == 1
    detail = trails[0]["detail"]
    assert detail["side"] == "short"
    assert detail["new_stop"] == pytest.approx(first_stop)
    assert "lowered" in detail["explanation"]
    assert "raised" not in detail["explanation"]

    # 2) Price bounces UP to 97 (gain still above activation): the candidate
    #    stop would be HIGHER — tighten-only must ignore it, and no new
    #    'trail' row may be logged.
    position = engine.get_positions("open")[0]
    df_bounce = _closes_df([100.0] * 10 + [98.0, 96.0, 95.0, 97.0])
    assert (entry - 97.0) / entry > config.trail_activate_pct
    trader._update_trailing(config, position, df_bounce)

    assert float(engine.get_positions("open")[0]["stop_loss"]) == pytest.approx(
        first_stop
    )
    assert len(_bot_activity("trail")) == 1  # no row for a non-move

    # 3) A deeper fall to 93 tightens the stop further DOWN.
    position = engine.get_positions("open")[0]
    df_deeper = _closes_df([100.0] * 10 + [98.0, 96.0, 95.0, 97.0, 93.0])
    trader._update_trailing(config, position, df_deeper)

    final_stop = float(engine.get_positions("open")[0]["stop_loss"])
    assert final_stop == pytest.approx(93.0 * (1.0 + config.trail_distance_pct))
    assert final_stop < first_stop
    assert len(_bot_activity("trail")) == 2


def test_update_protective_levels_short_tighten_only() -> None:
    """Engine guard: a short's stop may only move DOWN; up/equal is ignored."""
    _seed_flat_candles(SHORT_SYMBOL, n=30, close=100.0)
    engine = PaperTradingEngine()

    engine.submit_order(SHORT_SYMBOL, "sell", "market", qty=1.0, stop_loss=105.0)
    position_id = int(engine.get_positions("open")[0]["id"])

    # Raising the stop (widening a short's risk) must be silently ignored.
    result = engine.update_protective_levels(position_id, stop_loss=107.0)
    assert result["stop_moved"] is False
    assert float(result["stop_loss"]) == pytest.approx(105.0)

    # An equal stop is not a tighten either.
    result = engine.update_protective_levels(position_id, stop_loss=105.0)
    assert result["stop_moved"] is False
    assert float(result["stop_loss"]) == pytest.approx(105.0)

    # Lowering the stop tightens and is applied.
    result = engine.update_protective_levels(position_id, stop_loss=101.0)
    assert result["stop_moved"] is True
    assert float(result["stop_loss"]) == pytest.approx(101.0)
