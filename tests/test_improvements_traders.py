"""Improvement Pack v3 acceptance tests for BOTH traders (paper only).

Covers the CONTRACTS.md "## Improvement Pack v3" trader-side behaviours:

* AutoTrader regime gate — shorts blocked outside a BTC+symbol 4h
  double-downtrend, longs blocked ONLY inside it, ``regime_block`` skip rows
  with the full contract detail shape, per-cycle BTCUSDT regime caching and
  the ``regime_gate_enabled=False`` bypass.
* Cost gate (both traders) — ``cost_gate`` skip rows with
  ``expected_move_pct``/``needed_pct``.
* Shadow diagnostics + de-risk — ``shadow_flags``/``derisk_multiplier`` on
  every ``enter``/``scalp_enter`` detail.
* R-multiple logging — ``designed_r``/``realized_r`` on ``exit``/``scalp_exit``.
* Scalper ATR geometry — effective stops from ATR, always inside HARD_BOUNDS.
* Shrink-only tuner + martingale ban — clamp list and flag logged on
  ``scalp_tune``; the AI supervisor cannot touch the new v3 params.
* AI-offline behaviour — ONE ``ai_offline`` skip row per bot cycle and the
  no-op ``tune_with_ai`` skip, driven by ``account_state['ollama_status']``
  (missing/malformed key = online).
* Analyst plumbing — ``htf_context`` reaches ``analyze_market`` and the
  ``enter`` detail's ``ai`` payload carries ``opposing_case``.

Everything runs offline and deterministically: NO network (conftest blocks
sockets), NO Ollama (``analyze_market`` and ``OllamaClient.chat`` are always
stubbed), synthetic seeded candles only, and ``regime.get_regime`` is
monkeypatched to fixed :class:`RegimeInfo` values (its own computation is
covered by tests/test_regime.py). PAPER TRADING ONLY.
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
import backend.paper_trading.regime as regime_mod
import backend.paper_trading.scalper as scalper_mod
from backend.ai.analyst import MarketAnalysis
from backend.ai.ollama_client import OllamaClient
from backend.database.db import get_conn, upsert_ohlcv
from backend.indicators.features import add_features
from backend.paper_trading.auto_trader import AutoTrader, BotConfig
from backend.paper_trading.engine import PaperTradingEngine
from backend.paper_trading.regime import RegimeInfo
from backend.paper_trading.scalper import HARD_BOUNDS, Scalper, ScalperParams
from backend.strategies.voting import VotingStrategy
from config.settings import settings

SOURCE = "binance"
BOT_TIMEFRAME = "1h"
SCALP_TIMEFRAME = "15m"
SCALP_BAR_MS = 15 * 60 * 1000

TREND_SYMBOL = "TRENDUSDT"
TREND2_SYMBOL = "TRNDBUSDT"
BEAR_SYMBOL = "BEARUSDT"
SCLONG_SYMBOL = "SCLLONGUSDT"
SCSHORT_SYMBOL = "SCLSHORTUSDT"
SCVOL_SYMBOL = "SCLVOLUSDT"

#: The four member strategies whose last-bar votes the bot sums.
MEMBER_STRATEGIES = ("trend_following", "mean_reversion", "breakout", "rsi_macd")

#: Empirically pinned last-bar vote sums (re-derived in the self-check test).
TREND_VOTE_SUM = 2
BEAR_VOTE_SUM = -2

#: account_state key the watchdog persists (cross-module contract).
OLLAMA_STATUS_KEY = "ollama_status"

#: Mandatory shadow-diagnostics keys (CONTRACTS.md section 1).
SHADOW_KEYS = {"volume_ok", "atr_pctile", "atr_in_band", "dead_zone"}

#: Fields the shrink-only tuner may never RAISE before 100 closed trades.
SHRINK_ONLY_FIELDS = {"sl_pct", "max_positions", "position_fraction", "max_trades_per_day"}


# --------------------------------------------------------------------------- #
# Synthetic market data (seeded / arithmetic — fully deterministic)            #
# --------------------------------------------------------------------------- #


def _make_ohlcv(close: np.ndarray, volume: np.ndarray, freq: str) -> pd.DataFrame:
    """Wrap a close/volume path into a canonical UTC OHLCV frame.

    The frame is anchored so its last candle is the most recent fully CLOSED
    period of ``freq`` — the traders' unclosed-candle guards keep it and the
    staleness guards treat it as live data.

    Args:
        close: Closing prices (all positive).
        volume: Per-bar volumes (all positive).
        freq: pandas frequency string (``"1h"`` or ``"15min"``).

    Returns:
        Canonical OHLCV frame (ms-resolution index so DB roundtrips are exact).
    """
    n = len(close)
    bar = pd.Timedelta(freq)
    last_closed = pd.Timestamp.now(tz="UTC").floor(freq) - bar
    start = last_closed - (n - 1) * bar
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    index = pd.date_range(
        start, periods=n, freq=freq, tz="UTC", name="timestamp"
    ).as_unit("ms")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")
    assert (df["low"] > 0).all()
    return df


def _trending_df(n: int = 240) -> pd.DataFrame:
    """Up-trending 1h frame whose LAST bar draws a +2 strategy vote sum.

    Same construction as tests/test_auto_trader.py: flat warm-up base, then
    a steady climb ending in a decisive +2% breakout close on a 4x volume
    spike (votes pinned in ``test_synthetic_frames_have_expected_votes``).
    """
    rng = np.random.default_rng(7)
    i = np.arange(n, dtype=np.float64)
    base = 100.0 + 0.6 * np.sin(i[: n // 2] / 5.0)
    trend = base[-1] + 0.45 * (i[: n - n // 2] + 1) + 0.3 * np.sin(i[: n - n // 2] / 4.0)
    close = np.concatenate([base, trend]) + rng.normal(0.0, 0.05, n)
    close[-1] = close[-2] * 1.02  # decisive breakout close on the last bar
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    volume[-1] = 800.0  # volume spike confirms the breakout
    return _make_ohlcv(close, volume, "1h")


def _bearish_df(n: int = 240) -> pd.DataFrame:
    """Down-trending 1h frame whose LAST bar draws a -2 strategy vote sum.

    Exact mirror of :func:`_trending_df` (flat base, steady decline, -2%
    breakdown close on a volume spike); vote sum pinned in the self-check.
    """
    rng = np.random.default_rng(7)
    i = np.arange(n, dtype=np.float64)
    base = 200.0 + 0.6 * np.sin(i[: n // 2] / 5.0)
    trend = base[-1] - 0.45 * (i[: n - n // 2] + 1) - 0.3 * np.sin(i[: n - n // 2] / 4.0)
    close = np.concatenate([base, trend]) + rng.normal(0.0, 0.05, n)
    close[-1] = close[-2] * 0.98  # decisive breakdown close on the last bar
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    volume[-1] = 800.0
    return _make_ohlcv(close, volume, "1h")


def _scalp_long_df(n: int = 250, scale: float = 1.0) -> pd.DataFrame:
    """15m frame satisfying all three scalp LONG conditions (see test_scalper).

    A -0.3/+0.4 sawtooth uptrend ending on an up bar; the huge-volume last
    (decision) bar pins the day-cumulative VWAP just below its close, so the
    ``close > vwap`` condition is date-independent. ``scale`` multiplies the
    per-bar deltas ONLY (base price unchanged): RSI and the EMA/VWAP
    inequality signs are invariant under delta scaling, but ATR/price grows
    with it — ``scale=12`` pushes ``1.5 * atr/price`` past the 2% hard stop
    bound, which the ATR-geometry clamp test needs.
    """
    deltas = np.tile([-0.3, 0.4], n // 2) * scale
    close = 100.0 + np.cumsum(deltas)
    volume = np.full(n, 200.0)
    volume[-1] = 1e9  # pins the UTC-day VWAP to the decision bar
    return _make_ohlcv(close, volume, "15min")


def _scalp_short_df(n: int = 250) -> pd.DataFrame:
    """15m frame satisfying all three scalp SHORT conditions (mirror)."""
    deltas = np.tile([0.3, -0.4], n // 2)
    close = 150.0 + np.cumsum(deltas)
    volume = np.full(n, 200.0)
    volume[-1] = 1e9
    return _make_ohlcv(close, volume, "15min")


def _seed(symbol: str, df: pd.DataFrame, timeframe: str) -> None:
    """Pre-seed the tmp-DB candle cache for one symbol."""
    upsert_ohlcv(SOURCE, symbol, timeframe, df)


def _seed_stop_touch(symbol: str, df: pd.DataFrame, level: float) -> None:
    """Extend the cached 1h path so it dips through ``level`` after an entry.

    Overwrites the last closed candle AND appends the next (currently
    forming) candle so the touch is visible to the engine's ``process_tick``
    SL evaluation (new candle → new watermark). The bounce stays far below
    any take-profit so only the stop can fill.

    Args:
        symbol: Symbol whose 1h cache to extend.
        df: The frame originally seeded for the symbol.
        level: Price level to dip through (the position's stop_loss).
    """
    last = df.iloc[-1]
    t_last = df.index[-1]
    t_next = t_last + pd.Timedelta(hours=1)
    mod_close = level * 0.999
    modified = {
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": min(float(last["low"]), level * 0.997),
        "close": mod_close,
        "volume": float(last["volume"]),
    }
    nxt = {
        "open": mod_close,
        "high": float(last["high"]),
        "low": level * 0.996,
        "close": level * 0.9975,
        "volume": 200.0,
    }
    frame = pd.DataFrame(
        [modified, nxt],
        index=pd.DatetimeIndex([t_last, t_next], name="timestamp"),
    ).astype("float64")
    upsert_ohlcv(SOURCE, symbol, BOT_TIMEFRAME, frame)


# --------------------------------------------------------------------------- #
# DB / activity helpers                                                        #
# --------------------------------------------------------------------------- #


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


def _skips(action: str, reason: str) -> list[dict[str, Any]]:
    """Activity rows of ``action`` whose detail reason EQUALS ``reason``."""
    return [
        item
        for item in _bot_activity(action)
        if item["detail"].get("reason") == reason
    ]


def _write_state(key: str, value: Any) -> None:
    """JSON-encode + persist one ``account_state`` value."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()
    finally:
        conn.close()


def _write_raw_state(key: str, raw: str) -> None:
    """Persist a RAW (possibly malformed) ``account_state`` value."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
            (key, raw),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_scalp_exits(n: int, pnl: float, symbol: str = SCLONG_SYMBOL) -> None:
    """Fabricate ``n`` closed scalp trades (``scalp_exit`` rows) for stats().

    The scalper's trade ledger IS its ``scalp_exit`` activity rows, so this
    is the contract-shaped way to give the tuner a trade history.
    """
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    conn = get_conn()
    try:
        for i in range(n):
            detail = {"reason": "stop_loss", "pnl": pnl, "pnl_pct": -0.008}
            conn.execute(
                "INSERT INTO bot_activity (ts, symbol, action, detail) "
                "VALUES (?, ?, ?, ?)",
                (now_ms - (n - i) * 1000, symbol, "scalp_exit", json.dumps(detail)),
            )
        conn.commit()
    finally:
        conn.close()


def _backdate_position(position_id: int, bars: int, bar_ms: int = SCALP_BAR_MS) -> None:
    """Shift a position's ``opened_at`` back by ``bars`` bars (age it)."""
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE paper_positions SET opened_at = opened_at - ? WHERE id=?",
            (bars * bar_ms, int(position_id)),
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Trader / regime / analyst stubs                                              #
# --------------------------------------------------------------------------- #


def _make_trader() -> tuple[PaperTradingEngine, AutoTrader]:
    """Build a fresh engine + trader pair against the per-test tmp database."""
    engine = PaperTradingEngine()
    return engine, AutoTrader(engine)


def _make_scalper() -> tuple[PaperTradingEngine, Scalper]:
    """Build a fresh engine + scalper pair against the per-test tmp database."""
    engine = PaperTradingEngine()
    return engine, Scalper(engine)


def _configure_bot(trader: AutoTrader, **overrides: Any) -> BotConfig:
    """Apply the offline v3 test baseline config plus per-test overrides.

    The baseline DISABLES both new hard gates (each gate test re-enables its
    own) so entry/exit assertions never depend on the fixtures' natural
    ATR-to-price ratio or on the regime of an unseeded BTCUSDT.
    """
    updates: dict[str, Any] = {
        "watchlist": [TREND_SYMBOL],
        "source": SOURCE,
        "timeframe": BOT_TIMEFRAME,
        "use_ai": False,
        "use_second_judge": False,
        "min_vote": 2,
        "allow_short": False,
        "regime_gate_enabled": False,
        "cost_gate_multiple": 0.0,
    }
    updates.update(overrides)
    return trader.set_config(updates)


def _configure_scalper(
    engine: PaperTradingEngine,
    scalper: Scalper,
    watchlist: list[str],
    **overrides: Any,
) -> ScalperParams:
    """Persist the bot watchlist and enable the scalper with a v3 baseline.

    ``cost_gate_multiple`` is zeroed by default (its own test re-enables it)
    so entries never depend on the fixtures' ATR-to-price ratio.
    """
    AutoTrader(engine).set_config(
        {
            "watchlist": watchlist,
            "source": SOURCE,
            "timeframe": SCALP_TIMEFRAME,
            "use_ai": False,
        }
    )
    updates: dict[str, Any] = {"enabled": True, "cost_gate_multiple": 0.0}
    updates.update(overrides)
    return scalper.set_params(updates)


def _regime_info(
    regime: str,
    close: float | None = 100.0,
    ema50: float | None = 101.0,
    ema200: float | None = 102.0,
    bars_used: int = 300,
    pct_change_7d: float | None = 1.5,
    pct_change_30d: float | None = -3.2,
) -> RegimeInfo:
    """Build a fixed :class:`RegimeInfo` for the ``get_regime`` stub."""
    return RegimeInfo(
        regime=regime,
        close=close,
        ema50=ema50,
        ema200=ema200,
        bars_used=bars_used,
        pct_change_7d=pct_change_7d,
        pct_change_30d=pct_change_30d,
    )


def _patch_regime(
    monkeypatch: pytest.MonkeyPatch,
    mapping: dict[str, RegimeInfo],
    default: str = "neutral",
) -> list[str]:
    """Replace ``regime.get_regime`` with a fixed per-symbol mapping.

    Patched on the regime module itself (both traders call
    ``regime.get_regime`` through the module object) AND on any module that
    rebound the name, covering every import style. Returns the recorded list
    of symbols the traders asked about (per-cycle caching assertions).
    """
    calls: list[str] = []

    def _fake_get_regime(
        source: str, symbol: str, ref_timeframe: str = "1h", **kwargs: Any
    ) -> RegimeInfo:
        calls.append(str(symbol))
        info = mapping.get(str(symbol))
        return info if info is not None else _regime_info(default)

    monkeypatch.setattr(regime_mod, "get_regime", _fake_get_regime)
    for module in (auto_trader_mod, scalper_mod):
        if hasattr(module, "get_regime"):
            monkeypatch.setattr(module, "get_regime", _fake_get_regime)
    return calls


class _AnalystStub:
    """Recording ``analyze_market`` replacement returning a fixed verdict."""

    def __init__(
        self,
        sentiment: str = "bullish",
        confidence: int = 90,
        opposing_case: str = "synthetic opposing case",
    ) -> None:
        self.sentiment = sentiment
        self.confidence = confidence
        self.opposing_case = opposing_case
        self.calls: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(
        self,
        symbol: str,
        timeframe: str = BOT_TIMEFRAME,
        df: pd.DataFrame | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MarketAnalysis:
        self.calls.append(str(symbol))
        self.kwargs.append(dict(kwargs, model=model))
        return MarketAnalysis(
            sentiment=self.sentiment,  # type: ignore[arg-type]
            confidence=self.confidence,
            risk_commentary="synthetic stub commentary",
            key_indicators=[],
            reasoning="synthetic stub reasoning",
            model_used="stub-model",
            symbol=str(symbol),
            timeframe=str(timeframe),
            opposing_case=self.opposing_case,
        )


def _patch_analyst(
    monkeypatch: pytest.MonkeyPatch,
    sentiment: str = "bullish",
    confidence: int = 90,
    opposing_case: str = "synthetic opposing case",
) -> _AnalystStub:
    """Replace ``analyze_market`` (wherever the bot looks it up) with a stub."""
    stub = _AnalystStub(
        sentiment=sentiment, confidence=confidence, opposing_case=opposing_case
    )
    monkeypatch.setattr(analyst_mod, "analyze_market", stub, raising=True)
    if hasattr(auto_trader_mod, "analyze_market"):
        monkeypatch.setattr(auto_trader_mod, "analyze_market", stub)
    return stub


# --------------------------------------------------------------------------- #
# Autouse guards                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _pin_contract_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every settings knob the gate/sizing assertions depend on.

    A local ``.env`` may override any of these; the tests do exact math with
    the contract defaults (incl. the new v3 heat/direction knobs), so pin
    them per test (monkeypatch restores). ``max_open_positions`` is raised to
    10 so the global cap never masks the v3 gates under test.
    """
    monkeypatch.setattr(settings, "commission_rate", 0.001)
    monkeypatch.setattr(settings, "slippage_rate", 0.0005)
    monkeypatch.setattr(settings, "risk_per_trade", 0.01)
    monkeypatch.setattr(settings, "max_open_positions", 10)
    monkeypatch.setattr(settings, "daily_loss_limit", 0.03)
    monkeypatch.setattr(settings, "circuit_breaker_drawdown", 0.10)
    monkeypatch.setattr(settings, "atr_stop_multiplier", 2.0)
    monkeypatch.setattr(settings, "atr_take_profit_multiplier", 3.0)
    monkeypatch.setattr(settings, "heat_cap_fraction", 0.06)
    monkeypatch.setattr(settings, "max_same_direction", 5)


@pytest.fixture(autouse=True)
def _offline_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op every data refresh so passes run purely off the seeded cache."""

    def _noop_update(*args: Any, **kwargs: Any) -> int:
        return 0

    monkeypatch.setattr(data_service_mod, "update_symbol", _noop_update)
    for module in (scalper_mod, auto_trader_mod):
        if hasattr(module, "update_symbol"):
            monkeypatch.setattr(module, "update_symbol", _noop_update)


@pytest.fixture(autouse=True)
def _no_llm_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything reaches Ollama outside an explicit stub.

    Tune tests overwrite ``chat`` via the same function-scoped monkeypatch
    (the later setattr wins); AI-gate tests stub ``analyze_market`` instead.
    """

    def _forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError(
            "OllamaClient.chat must not be called here — stub it explicitly."
        )

    monkeypatch.setattr(OllamaClient, "chat", _forbidden)
    monkeypatch.setattr(OllamaClient, "is_available", lambda self: True)
    monkeypatch.setattr(
        OllamaClient, "list_models", lambda self: [settings.ollama_model]
    )


class _ChatStub:
    """Recording ``OllamaClient.chat`` replacement with a fixed JSON reply."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def __call__(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        self.calls.append(str(prompt))
        return self.response


def _patch_chat(monkeypatch: pytest.MonkeyPatch, response: str) -> _ChatStub:
    """Replace ``OllamaClient.chat`` for the tune tests (records prompts)."""
    stub = _ChatStub(response)
    monkeypatch.setattr(OllamaClient, "chat", stub)
    return stub


# --------------------------------------------------------------------------- #
# Fixture self-check — the synthetic frames must vote the way the tests assume #
# --------------------------------------------------------------------------- #


def test_synthetic_frames_have_expected_votes() -> None:
    """Pin the last-bar vote sums and scalp entry conditions the module uses."""
    voter = VotingStrategy()
    trend_last = voter.vote_breakdown(add_features(_trending_df())).iloc[-1]
    bear_last = voter.vote_breakdown(add_features(_bearish_df())).iloc[-1]
    assert int(sum(trend_last[name] for name in MEMBER_STRATEGIES)) == TREND_VOTE_SUM
    assert int(sum(bear_last[name] for name in MEMBER_STRATEGIES)) == BEAR_VOTE_SUM

    for scale in (1.0, 12.0):
        last = add_features(_scalp_long_df(scale=scale)).iloc[-1]
        assert last["ema_12"] > last["ema_26"]
        assert 50.0 < float(last["rsi_14"]) < 72.0
        assert last["close"] > last["vwap"]
    short_last = add_features(_scalp_short_df()).iloc[-1]
    assert short_last["ema_12"] < short_last["ema_26"]
    assert 28.0 < float(short_last["rsi_14"]) < 50.0
    assert short_last["close"] < short_last["vwap"]


# --------------------------------------------------------------------------- #
# BotConfig v3 fields                                                          #
# --------------------------------------------------------------------------- #


def test_botconfig_v3_defaults_and_clamps() -> None:
    """New config fields default per contract; cost_gate_multiple clamps [0, 10]."""
    cfg = BotConfig()
    assert cfg.regime_gate_enabled is True
    assert cfg.cost_gate_multiple == pytest.approx(4.0)

    _, trader = _make_trader()
    assert trader.set_config({"cost_gate_multiple": 99.0}).cost_gate_multiple == pytest.approx(10.0)
    assert trader.set_config({"cost_gate_multiple": -5.0}).cost_gate_multiple == pytest.approx(0.0)
    assert trader.set_config({"cost_gate_multiple": float("nan")}).cost_gate_multiple == pytest.approx(0.0)
    assert trader.set_config({"regime_gate_enabled": False}).regime_gate_enabled is False


# --------------------------------------------------------------------------- #
# Bot regime gate                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("btc_regime", ["uptrend", "neutral"])
def test_bot_regime_gate_blocks_short_outside_double_downtrend(
    monkeypatch: pytest.MonkeyPatch, btc_regime: str
) -> None:
    """A short candidate is blocked unless coin AND BTC are both in a downtrend."""
    _seed(BEAR_SYMBOL, _bearish_df(), BOT_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            BEAR_SYMBOL: _regime_info(
                "downtrend", close=140.0, ema50=145.0, ema200=150.0
            ),
            "BTCUSDT": _regime_info(btc_regime),
        },
    )
    engine, trader = _make_trader()
    _configure_bot(
        trader, watchlist=[BEAR_SYMBOL], allow_short=True, regime_gate_enabled=True
    )

    trader.run_cycle()

    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    blocks = _skips("skip", "regime_block")
    assert len(blocks) == 1
    assert blocks[0]["symbol"] == BEAR_SYMBOL
    detail = blocks[0]["detail"]
    assert detail["side"] == "short"
    assert detail["symbol_regime"] == "downtrend"
    assert detail["btc_regime"] == btc_regime
    # The SYMBOL's RegimeInfo numbers, JSON-rounded.
    assert detail["close"] == pytest.approx(140.0)
    assert detail["ema50"] == pytest.approx(145.0)
    assert detail["ema200"] == pytest.approx(150.0)
    assert int(detail["vote_sum"]) == BEAR_VOTE_SUM
    assert set(detail["votes"].keys()) == set(MEMBER_STRATEGIES)
    assert isinstance(detail["explanation"], str) and detail["explanation"]


def test_bot_regime_gate_allows_short_in_double_downtrend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coin AND BTC both in a 4h downtrend → the short entry goes through."""
    _seed(BEAR_SYMBOL, _bearish_df(), BOT_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            BEAR_SYMBOL: _regime_info("downtrend"),
            "BTCUSDT": _regime_info("downtrend"),
        },
    )
    engine, trader = _make_trader()
    _configure_bot(
        trader, watchlist=[BEAR_SYMBOL], allow_short=True, regime_gate_enabled=True
    )

    trader.run_cycle()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["symbol"] == BEAR_SYMBOL
    assert positions[0]["side"] == "short"
    assert _skips("skip", "regime_block") == []


def test_bot_regime_gate_blocks_long_in_double_downtrend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long candidate is blocked ONLY in the coin+BTC double-downtrend."""
    _seed(TREND_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            TREND_SYMBOL: _regime_info("downtrend"),
            "BTCUSDT": _regime_info("downtrend"),
        },
    )
    engine, trader = _make_trader()
    _configure_bot(trader, regime_gate_enabled=True)

    trader.run_cycle()

    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    blocks = _skips("skip", "regime_block")
    assert len(blocks) == 1
    assert blocks[0]["detail"]["side"] == "long"


@pytest.mark.parametrize(
    ("symbol_regime", "btc_regime"),
    [("downtrend", "neutral"), ("neutral", "downtrend"), ("downtrend", "uptrend")],
)
def test_bot_regime_gate_allows_long_outside_double_downtrend(
    monkeypatch: pytest.MonkeyPatch, symbol_regime: str, btc_regime: str
) -> None:
    """Every non-double-downtrend combination lets the long through."""
    _seed(TREND_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            TREND_SYMBOL: _regime_info(symbol_regime),
            "BTCUSDT": _regime_info(btc_regime),
        },
    )
    engine, trader = _make_trader()
    _configure_bot(trader, regime_gate_enabled=True)

    trader.run_cycle()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["side"] == "long"
    assert _skips("skip", "regime_block") == []


def test_bot_regime_gate_disabled_bypasses_the_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """regime_gate_enabled=False enters even in a double-downtrend."""
    _seed(TREND_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            TREND_SYMBOL: _regime_info("downtrend"),
            "BTCUSDT": _regime_info("downtrend"),
        },
    )
    engine, trader = _make_trader()
    _configure_bot(trader, regime_gate_enabled=False)

    trader.run_cycle()

    assert len(engine.get_positions("open")) == 1
    assert _skips("skip", "regime_block") == []


def test_bot_regime_btc_computed_at_most_once_per_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-cycle cache: ≤1 get_regime call per symbol, BTCUSDT included."""
    _seed(TREND_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    _seed(TREND2_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    calls = _patch_regime(monkeypatch, {}, default="uptrend")
    engine, trader = _make_trader()
    _configure_bot(
        trader, watchlist=[TREND_SYMBOL, TREND2_SYMBOL], regime_gate_enabled=True
    )

    trader.run_cycle()

    assert len(engine.get_positions("open")) == 2  # both candidates entered
    assert calls.count("BTCUSDT") <= 1
    assert calls.count(TREND_SYMBOL) <= 1
    assert calls.count(TREND2_SYMBOL) <= 1


# --------------------------------------------------------------------------- #
# Bot cost gate                                                                #
# --------------------------------------------------------------------------- #


def test_bot_cost_gate_skip_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ATR too small for fees logs a ``cost_gate`` skip with the gate math."""
    df = _trending_df()
    _seed(TREND_SYMBOL, df, BOT_TIMEFRAME)
    _patch_regime(monkeypatch, {}, default="uptrend")
    engine, trader = _make_trader()
    _configure_bot(trader, regime_gate_enabled=True, cost_gate_multiple=10.0)

    featured = add_features(df)
    expected_move = float(featured["atr_14"].iloc[-1]) / float(
        featured["close"].iloc[-1]
    )
    needed = 10.0 * 2.0 * (settings.commission_rate + settings.slippage_rate)
    assert expected_move < needed  # the gate genuinely binds on this frame

    trader.run_cycle()

    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    gates = _skips("skip", "cost_gate")
    assert len(gates) == 1
    detail = gates[0]["detail"]
    assert detail["expected_move_pct"] == pytest.approx(expected_move, rel=1e-3)
    assert detail["needed_pct"] == pytest.approx(needed, rel=1e-6)
    assert isinstance(detail["explanation"], str) and detail["explanation"]


# --------------------------------------------------------------------------- #
# Bot enter detail: shadow flags, de-risk multiplier, AI payload plumbing      #
# --------------------------------------------------------------------------- #


def test_bot_enter_detail_has_shadow_flags_and_derisk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ``enter`` row carries shadow_flags + derisk_multiplier (v3)."""
    _seed(TREND_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    engine, trader = _make_trader()
    _configure_bot(trader)

    trader.run_cycle()

    enters = _bot_activity("enter")
    assert len(enters) == 1
    detail = enters[0]["detail"]
    assert detail["derisk_multiplier"] == pytest.approx(1.0)  # fresh account

    shadow = detail["shadow_flags"]
    assert isinstance(shadow, dict)
    assert SHADOW_KEYS <= set(shadow.keys())
    assert isinstance(shadow["dead_zone"], bool)
    # The trending frame's decision bar is a 4x volume spike → volume_ok True.
    assert shadow["volume_ok"] is True
    pctile = shadow["atr_pctile"]
    if pctile is None:
        assert shadow["atr_in_band"] is None
    else:
        assert 0.0 <= float(pctile) <= 100.0
        assert shadow["atr_in_band"] == (20.0 <= float(pctile) <= 90.0)


def test_bot_ai_payload_carries_opposing_case_and_htf_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """analyze_market receives htf_context; enter.ai carries opposing_case."""
    _seed(TREND_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            TREND_SYMBOL: _regime_info(
                "uptrend",
                close=155.0,
                ema200=140.0,
                pct_change_7d=4.2,
                pct_change_30d=-11.0,
            ),
            "BTCUSDT": _regime_info("uptrend"),
        },
    )
    stub = _patch_analyst(
        monkeypatch,
        sentiment="bullish",
        confidence=85,
        opposing_case="Overbought RSI argues for a pullback first.",
    )
    engine, trader = _make_trader()
    _configure_bot(trader, use_ai=True, min_ai_confidence=60, regime_gate_enabled=True)

    trader.run_cycle()

    assert stub.calls == [TREND_SYMBOL]
    htf = stub.kwargs[0].get("htf_context")
    assert isinstance(htf, str) and htf
    assert "7-day price change: +4.2%." in htf
    assert "30-day price change: -11.0%." in htf
    # De-bias rule: facts only, no trend verdicts in the context string.
    assert "uptrend" not in htf and "downtrend" not in htf

    enters = _bot_activity("enter")
    assert len(enters) == 1
    ai_payload = enters[0]["detail"]["ai"]
    assert ai_payload["sentiment"] == "bullish"
    assert ai_payload["opposing_case"] == "Overbought RSI argues for a pullback first."


# --------------------------------------------------------------------------- #
# Bot AI-offline cycle skip (watchdog integration)                             #
# --------------------------------------------------------------------------- #


def test_bot_ai_offline_drops_shortlist_with_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ollama_status.available=False → ONE ai_offline skip, no analyst calls."""
    _seed(TREND_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    _seed(TREND2_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    stub = _patch_analyst(monkeypatch)  # must never be reached
    _write_state(
        OLLAMA_STATUS_KEY,
        {"available": False, "since_ms": 1712345678901, "restarts": 2},
    )
    engine, trader = _make_trader()
    _configure_bot(trader, watchlist=[TREND_SYMBOL, TREND2_SYMBOL], use_ai=True)

    summary = trader.run_cycle()

    assert stub.calls == []  # no candidate was touched
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    rows = _skips("skip", "ai_offline")
    assert len(rows) == 1  # ONE row for the whole shortlist, not one per coin
    assert rows[0]["symbol"] == ""
    detail = rows[0]["detail"]
    assert int(detail["since_ms"]) == 1712345678901
    assert int(detail["skipped_candidates"]) == 2
    assert isinstance(detail["explanation"], str) and detail["explanation"]
    assert summary["skipped"] == 2


def test_bot_ai_offline_malformed_status_means_online(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed ollama_status value must NOT block AI-gated trading."""
    _seed(TREND_SYMBOL, _trending_df(), BOT_TIMEFRAME)
    stub = _patch_analyst(monkeypatch, sentiment="bullish", confidence=90)
    _write_raw_state(OLLAMA_STATUS_KEY, "not-json{")
    engine, trader = _make_trader()
    _configure_bot(trader, use_ai=True, min_ai_confidence=60)

    trader.run_cycle()

    assert stub.calls == [TREND_SYMBOL]  # the gate ran normally
    assert len(engine.get_positions("open")) == 1
    assert _skips("skip", "ai_offline") == []


# --------------------------------------------------------------------------- #
# Bot exit detail: R multiples                                                 #
# --------------------------------------------------------------------------- #


def test_bot_exit_detail_has_designed_and_realized_r(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stop-loss close logs designed_r/realized_r per the contract formulas."""
    df = _trending_df()
    _seed(TREND_SYMBOL, df, BOT_TIMEFRAME)
    engine, trader = _make_trader()
    _configure_bot(trader)

    trader.run_cycle()
    positions = engine.get_positions("open")
    assert len(positions) == 1
    stop_loss = float(positions[0]["stop_loss"])

    # Dip the cached path through the stop, then run a manage-only cycle.
    _seed_stop_touch(TREND_SYMBOL, df, stop_loss)
    trader.set_config({"min_vote": 4})  # a +2 vote sum can no longer re-qualify
    trader.run_cycle()

    closed = [p for p in engine.get_positions("closed")]
    assert len(closed) == 1
    pos = closed[0]
    entry = float(pos["entry_price"])
    stop = float(pos["stop_loss"])
    take = float(pos["take_profit"])
    qty = float(pos["qty"])
    pnl = float(pos["pnl"])

    exits = _bot_activity("exit")
    assert len(exits) == 1
    detail = exits[0]["detail"]
    assert detail["reason"] == "stop_loss"
    expected_designed = abs(take - entry) / abs(entry - stop)
    expected_realized = pnl / (abs(entry - stop) * qty)
    assert detail["designed_r"] == pytest.approx(expected_designed, rel=1e-3)
    assert detail["realized_r"] == pytest.approx(expected_realized, rel=1e-3)
    assert float(detail["realized_r"]) < 0.0  # a stop-out loses ~1R


# --------------------------------------------------------------------------- #
# Scalper v3 params, bounds and module constants                               #
# --------------------------------------------------------------------------- #


def test_scalper_v3_defaults_bounds_and_constants() -> None:
    """New fields/constants match the contract; cost_gate_multiple clamps."""
    params = ScalperParams()
    assert params.use_atr_geometry is True
    assert params.cost_gate_multiple == pytest.approx(3.0)

    assert HARD_BOUNDS["cost_gate_multiple"] == (0.0, 10.0)
    assert scalper_mod.ATR_GEOMETRY_SL_MULT == pytest.approx(1.5)
    assert scalper_mod.ATR_GEOMETRY_TP_MULT == pytest.approx(2.0)
    assert scalper_mod.SHRINK_ONLY_MIN_TRADES == 100

    _, scalper = _make_scalper()
    assert scalper.set_params(
        {"cost_gate_multiple": 99.0}
    ).cost_gate_multiple == pytest.approx(10.0)
    assert scalper.set_params(
        {"cost_gate_multiple": -3.0}
    ).cost_gate_multiple == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Scalper regime gate                                                          #
# --------------------------------------------------------------------------- #


def test_scalp_regime_gate_blocks_short_outside_double_downtrend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalp short is blocked unless coin AND BTC are both in a downtrend."""
    _seed(SCSHORT_SYMBOL, _scalp_short_df(), SCALP_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            SCSHORT_SYMBOL: _regime_info(
                "downtrend", close=120.0, ema50=125.0, ema200=130.0
            ),
            "BTCUSDT": _regime_info("uptrend"),
        },
    )
    engine, scalper = _make_scalper()
    _configure_scalper(engine, scalper, [SCSHORT_SYMBOL])

    scalper.run_tick()

    assert engine.get_positions("open") == []
    blocks = _skips("scalp_skip", "regime_block")
    assert len(blocks) == 1
    assert blocks[0]["symbol"] == SCSHORT_SYMBOL
    detail = blocks[0]["detail"]
    assert detail["side"] == "short"
    assert detail["symbol_regime"] == "downtrend"
    assert detail["btc_regime"] == "uptrend"
    assert detail["close"] == pytest.approx(120.0)
    assert detail["ema50"] == pytest.approx(125.0)
    assert detail["ema200"] == pytest.approx(130.0)
    # The scalper's shape is the bot's MINUS the vote keys (contract).
    assert "vote_sum" not in detail and "votes" not in detail
    assert isinstance(detail["explanation"], str) and detail["explanation"]


def test_scalp_regime_gate_blocks_long_in_double_downtrend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scalp longs are likewise paused in a coin+BTC double-downtrend."""
    _seed(SCLONG_SYMBOL, _scalp_long_df(), SCALP_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            SCLONG_SYMBOL: _regime_info("downtrend"),
            "BTCUSDT": _regime_info("downtrend"),
        },
    )
    engine, scalper = _make_scalper()
    _configure_scalper(engine, scalper, [SCLONG_SYMBOL])

    scalper.run_tick()

    assert engine.get_positions("open") == []
    blocks = _skips("scalp_skip", "regime_block")
    assert len(blocks) == 1
    assert blocks[0]["detail"]["side"] == "long"


def test_scalp_regime_gate_allows_long_outside_double_downtrend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A downtrending coin with BTC not downtrending still scalps long."""
    _seed(SCLONG_SYMBOL, _scalp_long_df(), SCALP_TIMEFRAME)
    _patch_regime(
        monkeypatch,
        {
            SCLONG_SYMBOL: _regime_info("downtrend"),
            "BTCUSDT": _regime_info("neutral"),
        },
    )
    engine, scalper = _make_scalper()
    _configure_scalper(engine, scalper, [SCLONG_SYMBOL])

    scalper.run_tick()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["side"] == "long"
    assert _skips("scalp_skip", "regime_block") == []


# --------------------------------------------------------------------------- #
# Scalper cost gate                                                            #
# --------------------------------------------------------------------------- #


def test_scalp_cost_gate_skip_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 15m ATR too small for fees logs ``cost_gate`` with the gate math."""
    df = _scalp_long_df()
    _seed(SCLONG_SYMBOL, df, SCALP_TIMEFRAME)
    engine, scalper = _make_scalper()
    _configure_scalper(engine, scalper, [SCLONG_SYMBOL], cost_gate_multiple=10.0)

    featured = add_features(df)
    expected_move = float(featured["atr_14"].iloc[-1]) / float(
        featured["close"].iloc[-1]
    )
    needed = 10.0 * 2.0 * (settings.commission_rate + settings.slippage_rate)
    assert expected_move < needed  # the gate genuinely binds on this frame

    scalper.run_tick()

    assert engine.get_positions("open") == []
    gates = _skips("scalp_skip", "cost_gate")
    assert len(gates) == 1
    detail = gates[0]["detail"]
    assert detail["expected_move_pct"] == pytest.approx(expected_move, rel=1e-3)
    assert detail["needed_pct"] == pytest.approx(needed, rel=1e-6)
    assert isinstance(detail["explanation"], str) and detail["explanation"]


# --------------------------------------------------------------------------- #
# Scalper ATR geometry                                                         #
# --------------------------------------------------------------------------- #


def test_scalp_atr_geometry_prices_stops_from_atr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_atr_geometry=True prices SL/TP from the 15m ATR (unclamped case)."""
    df = _scalp_long_df()
    _seed(SCLONG_SYMBOL, df, SCALP_TIMEFRAME)
    engine, scalper = _make_scalper()
    _configure_scalper(engine, scalper, [SCLONG_SYMBOL], use_atr_geometry=True)

    last = add_features(df).iloc[-1]
    atr = float(last["atr_14"])
    price = float(last["close"])
    sl_expected = 1.5 * atr / price
    tp_expected = 2.0 * sl_expected
    lo_sl, hi_sl = HARD_BOUNDS["sl_pct"]
    lo_tp, hi_tp = HARD_BOUNDS["tp_pct"]
    assert lo_sl < sl_expected < hi_sl  # genuinely inside the bounds
    assert lo_tp < tp_expected < hi_tp

    scalper.run_tick()

    enters = _bot_activity("scalp_enter")
    assert len(enters) == 1
    detail = enters[0]["detail"]
    assert detail["atr_geometry"] is True
    assert detail["sl_pct"] == pytest.approx(sl_expected, rel=1e-3)
    assert detail["tp_pct"] == pytest.approx(tp_expected, rel=1e-3)
    # Stops placed off the sizing/fill-basis price at the effective percents.
    assert detail["stop_loss"] == pytest.approx(price * (1.0 - sl_expected), rel=1e-4)
    assert detail["take_profit"] == pytest.approx(price * (1.0 + tp_expected), rel=1e-4)
    # v3 transparency keys ride along on every scalp_enter.
    assert SHADOW_KEYS <= set(detail["shadow_flags"].keys())
    assert detail["derisk_multiplier"] == pytest.approx(1.0)


def test_scalp_atr_geometry_stays_inside_hard_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A violent frame clamps the geometry to sl 0.02 / tp 0.03 (never wider)."""
    df = _scalp_long_df(scale=12.0)
    _seed(SCVOL_SYMBOL, df, SCALP_TIMEFRAME)
    engine, scalper = _make_scalper()
    _configure_scalper(engine, scalper, [SCVOL_SYMBOL], use_atr_geometry=True)

    last = add_features(df).iloc[-1]
    raw_sl = 1.5 * float(last["atr_14"]) / float(last["close"])
    assert raw_sl > HARD_BOUNDS["sl_pct"][1]  # the clamp genuinely binds

    scalper.run_tick()

    enters = _bot_activity("scalp_enter")
    assert len(enters) == 1
    detail = enters[0]["detail"]
    assert detail["atr_geometry"] is True
    assert detail["sl_pct"] == pytest.approx(HARD_BOUNDS["sl_pct"][1])  # 0.02
    assert detail["tp_pct"] == pytest.approx(HARD_BOUNDS["tp_pct"][1])  # 0.03
    assert float(detail["sl_pct"]) < float(detail["tp_pct"])  # stops stay sane


def test_scalp_atr_geometry_disabled_uses_fixed_percents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_atr_geometry=False keeps the tuner-owned fixed tp/sl percents."""
    _seed(SCLONG_SYMBOL, _scalp_long_df(), SCALP_TIMEFRAME)
    engine, scalper = _make_scalper()
    params = _configure_scalper(
        engine, scalper, [SCLONG_SYMBOL], use_atr_geometry=False
    )

    scalper.run_tick()

    enters = _bot_activity("scalp_enter")
    assert len(enters) == 1
    detail = enters[0]["detail"]
    assert detail["atr_geometry"] is False
    assert detail["sl_pct"] == pytest.approx(params.sl_pct)
    assert detail["tp_pct"] == pytest.approx(params.tp_pct)


# --------------------------------------------------------------------------- #
# Scalper exit detail: R multiples                                             #
# --------------------------------------------------------------------------- #


def test_scalp_exit_detail_has_designed_and_realized_r(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A time-stop close logs designed_r/realized_r per the contract formulas."""
    _seed(SCLONG_SYMBOL, _scalp_long_df(), SCALP_TIMEFRAME)
    engine, scalper = _make_scalper()
    params = _configure_scalper(engine, scalper, [SCLONG_SYMBOL])

    scalper.run_tick()
    positions = engine.get_positions("open")
    assert len(positions) == 1
    _backdate_position(int(positions[0]["id"]), params.time_stop_bars + 2)

    scalper.run_tick()  # the manage pass time-stops the aged position

    closed = engine.get_positions("closed")
    assert len(closed) == 1
    pos = closed[0]
    entry = float(pos["entry_price"])
    stop = float(pos["stop_loss"])
    take = float(pos["take_profit"])
    qty = float(pos["qty"])
    pnl = float(pos["pnl"])

    exits = _bot_activity("scalp_exit")
    assert len(exits) == 1
    detail = exits[0]["detail"]
    assert detail["reason"] == "time_stop"
    expected_designed = abs(take - entry) / abs(entry - stop)
    expected_realized = pnl / (abs(entry - stop) * qty)
    assert detail["designed_r"] == pytest.approx(expected_designed, rel=1e-3)
    assert detail["realized_r"] == pytest.approx(expected_realized, rel=1e-3)


# --------------------------------------------------------------------------- #
# Shrink-only tuner + martingale ban                                           #
# --------------------------------------------------------------------------- #


def test_shrink_only_tuner_clamps_raises_and_logs_the_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below 100 trades no proposal may raise the four risk knobs."""
    engine, scalper = _make_scalper()
    scalper.set_params({"enabled": True})
    assert scalper.stats()["trades"] == 0  # fresh account → shrink-only active

    proposal = {
        "changes": {
            "sl_pct": 0.012,  # raise (banned)
            "max_positions": 9,  # raise (banned)
            "position_fraction": 0.08,  # raise (banned)
            "max_trades_per_day": 60,  # raise (banned)
            "tp_pct": 0.02,  # raise (allowed — not a shrink-only field)
            "cooldown_bars": 5,  # allowed
        },
        "disabled_symbols": [],
        "reasoning": "synthetic aggressive proposal",
    }
    _patch_chat(monkeypatch, json.dumps(proposal))

    result = scalper.tune_with_ai()

    params = scalper.get_params()
    assert params.sl_pct == pytest.approx(0.008)  # defaults kept
    assert params.max_positions == 8
    assert params.position_fraction == pytest.approx(0.06)
    assert params.max_trades_per_day == 40
    assert params.tp_pct == pytest.approx(0.02)  # allowed changes applied
    assert params.cooldown_bars == 5
    assert set(result["applied"].keys()) & SHRINK_ONLY_FIELDS == set()

    tunes = _bot_activity("scalp_tune")
    assert len(tunes) == 1
    detail = tunes[0]["detail"]
    assert set(detail["shrink_only_clamped"]) == SHRINK_ONLY_FIELDS
    assert detail["martingale_blocked"] is False  # flat PnL → no martingale


def test_shrink_only_tuner_allows_lowering_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lowering the guarded knobs is always allowed; the clamp list stays []."""
    engine, scalper = _make_scalper()
    scalper.set_params({"enabled": True})

    proposal = {
        "changes": {"sl_pct": 0.006, "position_fraction": 0.04},
        "disabled_symbols": [],
        "reasoning": "shrink risk",
    }
    _patch_chat(monkeypatch, json.dumps(proposal))

    result = scalper.tune_with_ai()

    params = scalper.get_params()
    assert params.sl_pct == pytest.approx(0.006)
    assert params.position_fraction == pytest.approx(0.04)
    assert {"sl_pct", "position_fraction"} <= set(result["applied"].keys())

    detail = _bot_activity("scalp_tune")[0]["detail"]
    assert detail["shrink_only_clamped"] == []
    assert detail["martingale_blocked"] is False


def test_martingale_ban_blocks_size_up_after_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even past 100 trades, sizing up into losses is dropped and flagged."""
    engine, scalper = _make_scalper()
    scalper.set_params({"enabled": True})
    _insert_scalp_exits(120, pnl=-5.0)
    stats = scalper.stats()
    assert stats["trades"] >= scalper_mod.SHRINK_ONLY_MIN_TRADES
    assert stats["pnl"] < 0  # the martingale precondition

    proposal = {
        "changes": {"position_fraction": 0.08, "cooldown_bars": 5},
        "disabled_symbols": [],
        "reasoning": "double down",
    }
    _patch_chat(monkeypatch, json.dumps(proposal))

    result = scalper.tune_with_ai()

    params = scalper.get_params()
    assert params.position_fraction == pytest.approx(0.06)  # size-up dropped
    assert params.cooldown_bars == 5  # unrelated change still applies
    assert "position_fraction" not in result["applied"]

    detail = _bot_activity("scalp_tune")[0]["detail"]
    assert detail["martingale_blocked"] is True
    assert detail["shrink_only_clamped"] == []  # >=100 trades: no shrink clamps


def test_tuner_cannot_touch_v3_gate_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_atr_geometry / cost_gate_multiple are user-only knobs (untunable)."""
    engine, scalper = _make_scalper()
    scalper.set_params({"enabled": True})

    proposal = {
        "changes": {
            "use_atr_geometry": False,
            "cost_gate_multiple": 9.0,
            "tp_pct": 0.014,
        },
        "disabled_symbols": [],
        "reasoning": "try to disable the safety gates",
    }
    _patch_chat(monkeypatch, json.dumps(proposal))

    result = scalper.tune_with_ai()

    params = scalper.get_params()
    assert params.use_atr_geometry is True  # untouched
    assert params.cost_gate_multiple == pytest.approx(3.0)  # untouched
    assert params.tp_pct == pytest.approx(0.014)  # tunable field applied
    assert "use_atr_geometry" not in result["applied"]
    assert "cost_gate_multiple" not in result["applied"]


# --------------------------------------------------------------------------- #
# Scalper AI-offline tune skip (watchdog integration)                          #
# --------------------------------------------------------------------------- #


def test_tune_ai_offline_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ollama_status.available=False → tune is a no-op with one skip row."""
    engine, scalper = _make_scalper()
    scalper.set_params({"enabled": True})
    before = scalper.get_params().model_dump()
    _write_state(
        OLLAMA_STATUS_KEY, {"available": False, "since_ms": 1, "restarts": 0}
    )
    # The autouse chat guard stays active: any LLM call fails the test.

    result = scalper.tune_with_ai()

    assert result == {"applied": {}, "reasoning": "", "status": "ai_offline"}
    assert scalper.get_params().model_dump() == before  # nothing changed
    rows = _skips("scalp_skip", "ai_offline")
    assert len(rows) == 1
    assert isinstance(rows[0]["detail"]["explanation"], str)
    assert rows[0]["detail"]["explanation"]
    assert _bot_activity("scalp_tune") == []  # no tune row for a skipped tune
